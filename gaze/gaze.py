import numpy as np
from collections import deque
import cv2

class GazeSmoother:
    def __init__(self,window_size=5):
        self.buffer = deque(maxlen=window_size)

    def update(self,value):
        self.buffer.append(value)
        return sum(self.buffer)/len(self.buffer)

class GazeCalibrator:
    def __init__(self,window_size=8):
        self.window_size = window_size
        self.gaze_values = deque(maxlen=window_size)
        self.yaw_values = deque(maxlen=window_size)
        self.pitch_values = deque(maxlen=window_size)

    def update(self,gaze_ratio,yaw,pitch):
        self.gaze_values.append(gaze_ratio)
        self.yaw_values.append(yaw)
        self.pitch_values.append(pitch)

    def ready(self):
        return len(self.gaze_values) >= self.window_size

    def baseline(self):
        if not self.gaze_values:
            return 0.5, 0.0, 0.0
        return (
            sum(self.gaze_values) / len(self.gaze_values),
            sum(self.yaw_values) / len(self.yaw_values),
            sum(self.pitch_values) / len(self.pitch_values),
        )
    
smoother=GazeSmoother(window_size=5)
calibrator=GazeCalibrator(window_size=8)


def normalize_angle(angle):
    return ((angle + 180.0) % 360.0) - 180.0


def angular_distance(a, b):
    return abs(normalize_angle(a - b))

def compute_eye_ratio(landmarks,left_idx,right_idx,pupil_idx):
    left=landmarks[left_idx]
    right=landmarks[right_idx]
    pupil=landmarks[pupil_idx]

    left_x = min(left.x,right.x)
    right_x = max(left.x,right.x)
    width=right_x-left_x
    if abs(width)<1e-6:
        return 0.5
    
    ratio = (pupil.x - left_x)/width
    return max(0.0,min(1.0,ratio))

def compute_gaze_ratio(landmarks):
    l_ratio = compute_eye_ratio(landmarks,33,133,468)
    r_ratio = compute_eye_ratio(landmarks,362,263,473)
    return (l_ratio+r_ratio)/2.0

def get_head_pose(landmarks, frame_shape):
    h, w, _ = frame_shape

    image_points = np.array([
        (landmarks[1].x * w, landmarks[1].y * h),     # Nose tip
        (landmarks[152].x * w, landmarks[152].y * h), # Chin
        (landmarks[33].x * w, landmarks[33].y * h),   # Left eye
        (landmarks[263].x * w, landmarks[263].y * h), # Right eye
        (landmarks[61].x * w, landmarks[61].y * h),   # Left mouth
        (landmarks[291].x * w, landmarks[291].y * h)  # Right mouth
    ], dtype="double")

    model_points = np.array([
        (0.0, 0.0, 0.0),        
        (0.0, -330.0, -65.0),   
        (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0), 
        (-150.0, -150.0, -125.0),
        (150.0, -150.0, -125.0) 
    ])

    focal_length = w
    center = (w / 2, h / 2)

    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")

    dist_coeffs = np.zeros((4, 1))

    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return 0, 0

    rmat, _ = cv2.Rodrigues(rotation_vector)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

    pitch, yaw, roll = angles

    return normalize_angle(yaw), normalize_angle(pitch)


def is_focused(gaze_ratio,yaw,pitch,gaze_thresh=0.25,yaw_thresh=35,pitch_thresh=30):
    baseline_gaze, baseline_yaw, baseline_pitch = calibrator.baseline()

    # Before calibration is full, use forgiving defaults so startup samples do not all become distracted.
    if not calibrator.ready():
        baseline_gaze = 0.5
        baseline_yaw = 0.0
        baseline_pitch = 0.0

    eye_centered = abs(gaze_ratio-baseline_gaze) < gaze_thresh
    head_forward = (
        angular_distance(yaw, baseline_yaw) < yaw_thresh
        and angular_distance(pitch, baseline_pitch) < pitch_thresh
    )

    return 1 if (eye_centered and head_forward) else 0

def get_gaze(landmarks,frame_shape):
    raw_ratio = compute_gaze_ratio(landmarks)
    smooth_ratio=smoother.update(raw_ratio)

    yaw,pitch = get_head_pose(landmarks,frame_shape)
    calibrator.update(smooth_ratio,yaw,pitch)
    focus=is_focused(smooth_ratio,yaw,pitch)
    baseline_gaze, baseline_yaw, baseline_pitch = calibrator.baseline()

    return {
        "gaze_ratio":float(smooth_ratio),
        "yaw":float(yaw),
        "pitch":float(pitch),
        "focused":int(focus),
        "calibrated":bool(calibrator.ready()),
        "baseline_gaze_ratio":float(baseline_gaze),
        "baseline_yaw":float(baseline_yaw),
        "baseline_pitch":float(baseline_pitch)
    }
