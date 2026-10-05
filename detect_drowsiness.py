"""
Real-time Driver Drowsiness Detection
-------------------------------------
Signals combined:
  1. CNN eye-state prediction (open/closed) on cropped eyes
  2. EAR  (Eye Aspect Ratio)   - geometric eye closure measure
  3. MAR  (Mouth Aspect Ratio) - yawning

Alarm triggers when eyes stay closed for ~CLOSED_SECONDS, or on repeated yawns.

Run:  python detect_drowsiness.py          (press 'q' to quit)
If eye_state_model.h5 is missing, it falls back to EAR only.
"""
import os
import time
import threading
import cv2
import numpy as np
import mediapipe as mp

# ---------------- Config ----------------
CLOSED_SECONDS = 1.5      # eyes closed this long -> alarm
EAR_THRESH = 0.21         # below = closed (tune per person)
MAR_THRESH = 0.60         # above = yawning
YAWN_FRAMES = 20          # consecutive frames of yawn
CNN_OPEN_THRESH = 0.5
IMG_SIZE = 64
MODEL_PATH = "eye_state_model.h5"

# MediaPipe FaceMesh landmark indices
LEFT_EYE = [362, 385, 387, 263, 373, 380]    # p1..p6
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH = [61, 291, 13, 14, 81, 178, 311, 402]  # corners, upper/lower lip points

# ---------------- Optional CNN ----------------
model = None
if os.path.exists(MODEL_PATH):
    import tensorflow as tf
    model = tf.keras.models.load_model(MODEL_PATH)
    print("CNN model loaded.")
else:
    print("No CNN model found - using EAR only.")


def dist(a, b):
    return np.linalg.norm(a - b)


def ear(pts):
    # (|p2-p6| + |p3-p5|) / (2*|p1-p4|)
    return (dist(pts[1], pts[5]) + dist(pts[2], pts[4])) / (2.0 * dist(pts[0], pts[3]))


def mar(m):
    horiz = dist(m[0], m[1])
    vert = (dist(m[4], m[5]) + dist(m[2], m[3]) + dist(m[6], m[7])) / 3.0
    return vert / horiz


def eye_crop(gray, pts, pad=0.35):
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    w, h = x2 - x1, y2 - y1
    x1, x2 = int(x1 - pad * w), int(x2 + pad * w)
    y1, y2 = int(y1 - pad * h * 2), int(y2 + pad * h * 2)
    x1, y1 = max(x1, 0), max(y1, 0)
    crop = gray[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (IMG_SIZE, IMG_SIZE)).astype("float32")


def cnn_open_prob(gray, left_pts, right_pts):
    crops = [c for c in (eye_crop(gray, left_pts), eye_crop(gray, right_pts)) if c is not None]
    if not crops or model is None:
        return None
    batch = np.stack(crops)[..., None]
    return float(model.predict(batch, verbose=0).mean())  # model rescales internally


# ---------------- Alarm (non-blocking) ----------------
alarm_on = False


def beep():
    global alarm_on
    if alarm_on:
        return
    alarm_on = True

    def _play():
        global alarm_on
        try:
            import winsound
            winsound.Beep(2500, 700)
        except ImportError:
            try:
                from playsound import playsound
                playsound("alarm.wav")
            except Exception:
                print("\a", end="", flush=True)  # terminal bell fallback
        time.sleep(0.3)
        alarm_on = False

    threading.Thread(target=_play, daemon=True).start()


# ---------------- Main loop ----------------
face_mesh = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1, refine_landmarks=True,
    min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)
closed_start = None
yawn_count = 0
yawn_frames = 0
total_yawns = 0
prev = time.time()

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    res = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    status, color = "No face", (200, 200, 200)

    if res.multi_face_landmarks:
        lm = res.multi_face_landmarks[0].landmark
        P = lambda idx: np.array([[lm[i].x * w, lm[i].y * h] for i in idx])
        left, right, mouth = P(LEFT_EYE), P(RIGHT_EYE), P(MOUTH)

        avg_ear = (ear(left) + ear(right)) / 2.0
        avg_mar = mar(mouth)
        p_open = cnn_open_prob(gray, left, right)

        # Eye closed decision: CNN + EAR vote (or EAR alone)
        ear_closed = avg_ear < EAR_THRESH
        cnn_closed = (p_open is not None) and (p_open < CNN_OPEN_THRESH)
        eyes_closed = (ear_closed and cnn_closed) if p_open is not None else ear_closed

        if eyes_closed:
            closed_start = closed_start or time.time()
            elapsed = time.time() - closed_start
        else:
            closed_start, elapsed = None, 0.0

        # Yawn detection
        if avg_mar > MAR_THRESH:
            yawn_frames += 1
        else:
            if yawn_frames >= YAWN_FRAMES:
                total_yawns += 1
            yawn_frames = 0

        drowsy = elapsed >= CLOSED_SECONDS or total_yawns >= 3
        if drowsy:
            status, color = "DROWSY! WAKE UP!", (0, 0, 255)
            beep()
            cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 8)
        elif eyes_closed:
            status, color = "Eyes closing...", (0, 165, 255)
        else:
            status, color = "Alert", (0, 200, 0)

        for pts in (left, right):
            cv2.polylines(frame, [pts.astype(np.int32)], True, (255, 255, 0), 1)
        cv2.putText(frame, f"EAR: {avg_ear:.2f}  MAR: {avg_mar:.2f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        if p_open is not None:
            cv2.putText(frame, f"CNN P(open): {p_open:.2f}", (10, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Yawns: {total_yawns}", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    else:
        closed_start = None

    now = time.time()
    fps = 1.0 / max(now - prev, 1e-6)
    prev = now
    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.putText(frame, f"FPS: {fps:.0f}", (w - 110, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Driver Drowsiness Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
