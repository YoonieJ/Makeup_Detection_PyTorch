"""
Real-time webcam inference for the makeup/no-makeup classifier.

Captures frames from the webcam, aligns/crops the face using the exact same
pipeline as preprocess.py (align_and_crop_face), classifies with the trained
model, and overlays the prediction live.
"""

import argparse
from pathlib import Path

import cv2
import torch
from torchvision import transforms

from preprocess.preprocess import IMAGE_SIZE, align_and_crop_face
from train import CHECKPOINT_DIR, build_model, get_device

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

_eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def preprocess_frame(frame_bgr) -> torch.Tensor:
    # align_and_crop_face() is the exact function preprocess.py uses when
    # building the training set -- reusing it here (instead of writing a
    # second crop implementation) is what keeps train/inference crops
    # identical. It never returns None; a frame with no detectable face
    # falls back to a center crop rather than raising.
    crop_bgr = align_and_crop_face(frame_bgr, out_size=IMAGE_SIZE)
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    return _eval_transform(crop_rgb).unsqueeze(0)


def main():
    parser = argparse.ArgumentParser(description="Real-time webcam makeup/no-makeup classifier.")
    parser.add_argument("--checkpoint", type=str, default=str(CHECKPOINT_DIR / "best_model.pt"))
    # On macOS with Continuity Camera enabled, index 0 can resolve to a
    # nearby iPhone instead of the built-in camera (it'll visibly "ring" and
    # never connect). Run `system_profiler SPCameraDataType` to see the
    # registered devices and pass whichever index is the actual webcam.
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}. Run train.py first.")

    device = get_device()
    model = build_model().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    print(f"Loaded checkpoint {checkpoint_path} on {device}")

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera_index}")

    print("Press 'q' to quit.")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame from camera.")
                break

            display = frame.copy()
            try:
                tensor = preprocess_frame(frame).to(device)
                with torch.no_grad():
                    prob = torch.sigmoid(model(tensor).squeeze()).item()
                label = "MAKEUP" if prob >= args.threshold else "NO MAKEUP"
                color = (0, 0, 255) if label == "MAKEUP" else (0, 200, 0)
                text = f"{label} ({prob:.2f})"
            except Exception as e:
                # Broad on purpose: this is a live camera loop, and a single
                # bad/malformed frame (dropped USB frame, camera hiccup)
                # should show up on screen, not crash the whole app.
                text = f"detection error: {e}"
                color = (0, 165, 255)

            cv2.putText(display, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, color, 2, cv2.LINE_AA)
            cv2.imshow("Makeup Detector", display)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
