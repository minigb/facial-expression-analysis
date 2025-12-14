#!/usr/bin/env python3
"""
Minimal real-time inference loop that reads camera frames and returns
valence/arousal/intensity estimates using the provided DLIB-based models.

Dependencies (match the repo versions):
- python 3.6+
- opencv-python
- dlib
- joblib
- numpy

Run:
  python source/realtime_inference.py --camera 0
Press q in the video window to quit.
"""

import argparse
import time
from pathlib import Path

import cv2
import dlib

from emotions_dlib import EmotionsDlib


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-time valence/arousal/intensity estimation"
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index passed to cv2.VideoCapture",
    )
    parser.add_argument(
        "--smooth",
        type=float,
        default=0.7,
        help="EMA factor for smoothing (0 disables smoothing)",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Skip the OpenCV window and only print values",
    )
    parser.add_argument(
        "--no-mps",
        action="store_false",
        dest="mps",
        default=True,
        help="(Deprecated) kept for CLI compatibility; has no effect.",
    )  # retained to avoid breaking existing scripts
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
    )

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {args.camera}")

    disp_valence = 0.0
    disp_arousal = 0.0
    disp_intensity = 0.0
    alpha = max(0.0, min(args.smooth, 1.0))
    last_print = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

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
                    disp_valence, disp_arousal, disp_intensity = (
                        valence,
                        arousal,
                        intensity,
                    )

                text = (
                    f"V={disp_valence:.3f} | A={disp_arousal:.3f} | "
                    f"I={disp_intensity:.3f} | {name}"
                )

                now = time.time()
                if now - last_print > 0.2:
                    print(text, flush=True)
                    last_print = now

                if not args.no_display:
                    cv2.rectangle(
                        frame,
                        (face.left(), face.top()),
                        (face.right(), face.bottom()),
                        (0, 255, 0),
                        1,
                    )

            if not args.no_display:
                cv2.putText(
                    frame,
                    text,
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("Facial Expression Analysis (press q to quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        cap.release()
        if not args.no_display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

