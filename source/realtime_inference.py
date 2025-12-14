#!/usr/bin/env python3
"""
Real-time valence/arousal/intensity estimation (process every frame).
Press q in the video window to quit.
"""

import argparse
import time
from pathlib import Path

import cv2
import dlib
import numpy as np

from emotions_dlib import EmotionsDlib


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-time valence/arousal/intensity estimation"
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--smooth", type=float, default=0.7,
                        help="EMA factor for smoothing (0 disables smoothing)")
    parser.add_argument("--no-display", action="store_true")
    return parser.parse_args()


def get_model_paths() -> dict:
    root = Path(__file__).resolve().parent.parent
    models = root / "models"
    return {
        "predictor": models / "shape_predictor_68_face_landmarks.dat",
        "frontalization": models / "model_frontalization.npy",
        "emotion": models / "model_emotion_pls=30_fullfeatures=False.joblib",
    }


def choose_largest_face(faces):
    if len(faces) <= 1:
        return faces[0]
    areas = [(f.right() - f.left()) * (f.bottom() - f.top()) for f in faces]
    return faces[areas.index(max(areas))]


def main():
    args = parse_args()
    paths = get_model_paths()

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(str(paths["predictor"]))
    estimator = EmotionsDlib(
        file_emotion_model=str(paths["emotion"]),
        file_frontalization_model=str(paths["frontalization"]),
        use_torch_mps=True,
    )

    def open_camera(index: int):
        cap_local = cv2.VideoCapture(index)
        if not cap_local.isOpened():
            cap_local = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
        return cap_local

    cap = open_camera(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {args.camera}")

    window = "Facial Expression Analysis (press q to quit)"

    if not args.no_display:
        # --- PRIME THE WINDOW WITH A REAL FRAME (macOS fix) ---
        ok, init_frame = cap.read()
        if not ok or init_frame is None:
            raise RuntimeError("Failed to read initial frame for window initialization")

        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.imshow(window, init_frame)
        cv2.waitKey(1)  # <-- THIS is the critical line

    disp_valence = disp_arousal = disp_intensity = 0.0
    alpha = max(0.0, min(args.smooth, 1.0))

    last_print = 0.0
    read_failures = 0
    max_failures_before_reopen = 15
    text = "Initializing..."

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                read_failures += 1
                if read_failures >= max_failures_before_reopen:
                    print("Camera read failed, attempting to reopen...", flush=True)
                    cap.release()
                    cap = open_camera(args.camera)
                    read_failures = 0
                    if not cap.isOpened():
                        print("Reopen failed; stopping.", flush=True)
                        break

                if not args.no_display:
                    status_frame = 255 * np.ones((540, 960, 3), dtype=np.uint8)
                    cv2.putText(
                        status_frame,
                        "Reconnecting camera...",
                        (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.imshow(window, status_frame)
                    cv2.waitKey(1)
                continue

            read_failures = 0
            now = time.time()

            # ✅ Process EVERY frame (interval logic removed)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)

            text = "No face detected"
            if len(faces) > 0:
                face = choose_largest_face(faces)
                landmarks_obj = predictor(gray, face)
                emotions = estimator.get_emotions(landmarks_obj)

                valence = emotions["emotions"]["valence"]
                arousal = emotions["emotions"]["arousal"]
                intensity = emotions["emotions"]["intensity"]
                name = emotions["emotions"]["name"]

                if alpha > 0:
                    disp_valence = alpha * valence + (1 - alpha) * disp_valence
                    disp_arousal = alpha * arousal + (1 - alpha) * disp_arousal
                    disp_intensity = alpha * intensity + (1 - alpha) * disp_intensity
                else:
                    disp_valence, disp_arousal, disp_intensity = valence, arousal, intensity

                text = f"V={disp_valence:.3f} | A={disp_arousal:.3f} | I={disp_intensity:.3f} | {name}"

                if not args.no_display:
                    cv2.rectangle(
                        frame,
                        (face.left(), face.top()),
                        (face.right(), face.bottom()),
                        (0, 255, 0),
                        2,
                    )

            if now - last_print > 0.2:
                print(text, flush=True)
                last_print = now

            if not args.no_display:
                cv2.putText(
                    frame, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0), 2, cv2.LINE_AA
                )
                cv2.imshow(window, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break

    except KeyboardInterrupt:
        print("Interrupted by user, closing gracefully.", flush=True)

    finally:
        cap.release()
        if not args.no_display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()