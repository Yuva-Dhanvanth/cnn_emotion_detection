from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import json
import torch
import torch.nn as nn

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

from f_model.extract_landmarks_runtime import get_landmarks, get_face_mesh_status
from gaze.gaze import get_gaze
from cnn_final.model import build_model


# ------------------ MODEL LOAD ------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = Path(__file__).resolve().parent

model = build_model(device)
model_path = BASE_DIR / "cnn_final" / "best_phase2.pt"
checkpoint = torch.load(model_path, map_location=device, weights_only=False)
state_dict = checkpoint.get("model_state_dict", checkpoint)
model.load_state_dict(state_dict)
model.eval()


# ------------------ PREPROCESS ------------------
def preprocess_image(frame):
    image = cv2.resize(frame, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = image / 255.0

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    image = (image - mean) / std

    image = np.transpose(image, (2, 0, 1))
    image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)

    return image


# ------------------ ENGAGEMENT HELPERS ------------------

def compute_emotion_score(outputs):
    probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

    # Order: ["confused", "frustrated", "happy", "neutral"]
    weights = np.array([0.4, 0.2, 1.0, 0.7])

    return float(np.dot(probs, weights)), probs


def compute_attention_score(focused):
    return float(focused)  # already 0 or 1


def compute_performance_score(metrics):
    accuracy = metrics.get("accuracy", None)

    if accuracy is not None:
        return float(accuracy) / 100.0

    return 0.5  # fallback


def compute_engagement(perf, attn, emo):
    return round(
        0.4 * perf +
        0.3 * attn +
        0.3 * emo,
        3
    )


# ------------------ API ------------------

@app.post("/analyze")
async def analyze(image: UploadFile = File(...), metrics: str = Form("{}")):
    try:
        metrics_data = json.loads(metrics)

        contents = await image.read()
        np_arr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return {"error": "invalid image"}

        # -------- EMOTION (CNN) --------
        input_tensor = preprocess_image(frame).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)

        emotion = int(torch.argmax(outputs, dim=1).item())
        emotion_map = ["confused", "frustrated", "happy", "neutral"]
        emotion_label = emotion_map[emotion]

        emotion_score, emotion_probs = compute_emotion_score(outputs)

        # -------- LANDMARKS --------
        landmarks = get_landmarks(frame)
        face_mesh_status = get_face_mesh_status()

        performance_score = compute_performance_score(metrics_data)

        if landmarks is None:
            attention_score = 0.0
            engagement = compute_engagement(
                performance_score,
                attention_score,
                emotion_score
            )

            result = {
                "focused": 0,
                "emotion": emotion,
                "emotion_label": emotion_label,
                "emotion_score": emotion_score,
                "emotion_probs": emotion_probs.tolist(),

                "performance_score": performance_score,
                "attention_score": attention_score,
                "engagement": engagement,

                "gaze_ratio": None,
                "yaw": None,
                "pitch": None,

                "metrics": metrics_data,
                "landmarks_detected": False,
                "face_mesh_available": face_mesh_status["available"],
                "face_mesh_error": face_mesh_status["error"],
                "error": "face not detected" if face_mesh_status["available"] else "face mesh unavailable",
                "landmarks": None
            }

            return result

        # -------- GAZE --------
        gaze_data = get_gaze(landmarks, frame.shape)

        attention_score = compute_attention_score(gaze_data["focused"])

        engagement = compute_engagement(
            performance_score,
            attention_score,
            emotion_score
        )

        result = {
            "focused": gaze_data["focused"],
            "gaze_ratio": gaze_data["gaze_ratio"],
            "yaw": gaze_data["yaw"],
            "pitch": gaze_data["pitch"],

            "emotion": emotion,
            "emotion_label": emotion_label,
            "emotion_score": emotion_score,
            "emotion_probs": emotion_probs.tolist(),

            "performance_score": performance_score,
            "attention_score": attention_score,
            "engagement": engagement,

            "metrics": metrics_data,

            "landmarks_detected": True,
            "face_mesh_available": face_mesh_status["available"],
            "face_mesh_error": face_mesh_status["error"],
            "landmarks": landmarks.tolist() if landmarks is not None else None
        }

        return result

    except Exception as e:
        return {
            "error": str(e)
        }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(device),
        "face_mesh": get_face_mesh_status(),
    }
