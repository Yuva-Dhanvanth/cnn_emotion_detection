import cv2
import mediapipe as mp

face_mesh = None
FACE_MESH_IMPORT_ERROR = None

try:
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
    )
except AttributeError as exc:
    FACE_MESH_IMPORT_ERROR = exc


def get_landmarks(frame):
    if face_mesh is None:
        return None

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    landmarks = face_mesh.process(rgb_frame)

    if not landmarks.multi_face_landmarks:
        return None

    return landmarks.multi_face_landmarks[0].landmark


def get_face_mesh_status():
    return {
        "available": face_mesh is not None,
        "error": None if FACE_MESH_IMPORT_ERROR is None else str(FACE_MESH_IMPORT_ERROR),
    }
