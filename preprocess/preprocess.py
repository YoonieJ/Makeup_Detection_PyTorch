"""
Makeup detection - data preprocessing pipeline.

Pipeline stages:
  1. Loaders (dataset-specific) -> produce raw records: (image_path, person_id, label, source)
  2. build_manifest()           -> combine all sources into one DataFrame
  3. detect_and_crop_face()     -> MediaPipe-based face detection + alignment + crop
  4. preprocess_dataset()       -> run cropping over the full manifest, save to disk
  5. split_by_identity()        -> group-aware train/val/test split (no identity leakage)

Run order: fill in the loader functions at the bottom for your actual downloaded
dataset structures, then run this as a script.
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit

# Optional MediaPipe support. If the installed package does not expose the
# legacy `mp.solutions` API, we fall back to OpenCV Haar cascades.
try:
    import mediapipe as mp
    if not hasattr(mp, "solutions"):
        mp = None
except Exception:
    mp = None

# Config

RAW_DATA_ROOT = Path("data/raw")          # where you extracted the downloaded datasets
PROCESSED_ROOT = Path("data/processed")   # where cropped/aligned faces will be written
MANIFEST_PATH = Path("data/manifest.csv") # combined record of every image + split assignment

IMAGE_SIZE = 224            # model input size
FACE_MARGIN = 0.25          # fraction of face bbox size to pad around the crop
RANDOM_SEED = 42

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15            # must sum to 1.0 with the above

# Face detection + alignment
FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
EYE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_eye.xml"
_face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
_eye_cascade = cv2.CascadeClassifier(EYE_CASCADE_PATH)
_cascades_available = not _face_cascade.empty() and not _eye_cascade.empty()

LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263

if mp is not None:
    try:
        _mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
        )
    except Exception:
        _mp_face_mesh = None
else:
    _mp_face_mesh = None


def align_and_crop_face(img: np.ndarray, out_size: int = IMAGE_SIZE, margin: float = FACE_MARGIN) -> np.ndarray:
    """
    Detect the face in an already-loaded BGR image, rotate to align the eyes
    horizontally, crop tightly to the face (excluding most hair/background),
    and resize. Always returns a crop: falls back to Haar cascades, then to
    a plain center crop, if no face landmarks are found.

    Shared by preprocess_dataset() (reading from disk) and the webcam
    inference script (reading live frames), so training and inference see
    identical crops.
    """
    h, w = img.shape[:2]
    if _mp_face_mesh is not None:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = _mp_face_mesh.process(rgb)

        if result and result.multi_face_landmarks:
            landmarks = result.multi_face_landmarks[0].landmark

            # Pixel coords of eye corners for alignment
            left_eye = np.array([landmarks[LEFT_EYE_OUTER].x * w, landmarks[LEFT_EYE_OUTER].y * h])
            right_eye = np.array([landmarks[RIGHT_EYE_OUTER].x * w, landmarks[RIGHT_EYE_OUTER].y * h])

            dy = right_eye[1] - left_eye[1]
            dx = right_eye[0] - left_eye[0]
            angle = np.degrees(np.arctan2(dy, dx))

            eyes_center = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)
            rot_mat = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)
            rotated = cv2.warpAffine(img, rot_mat, (w, h), flags=cv2.INTER_LINEAR)

            rgb_rot = cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)
            result_rot = _mp_face_mesh.process(rgb_rot)
            if result_rot and result_rot.multi_face_landmarks:
                landmarks_rot = result_rot.multi_face_landmarks[0].landmark
                xs = [lm.x * w for lm in landmarks_rot]
                ys = [lm.y * h for lm in landmarks_rot]
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)

                box_w = x_max - x_min
                box_h = y_max - y_min
                pad_w = box_w * margin
                pad_h = box_h * margin

                x1 = max(int(x_min - pad_w), 0)
                y1 = max(int(y_min - pad_h), 0)
                x2 = min(int(x_max + pad_w), w)
                y2 = min(int(y_max + pad_h), h)

                if x2 > x1 and y2 > y1:
                    crop = rotated[y1:y2, x1:x2]
                    return cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_AREA)

        # Fall back to OpenCV if MediaPipe did not detect a face

    if _cascades_available:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = _face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
        if len(faces) > 0:
            x, y, fw, fh = max(faces, key=lambda rect: rect[2] * rect[3])
            face_roi = gray[y : y + fh, x : x + fw]
            eyes = _eye_cascade.detectMultiScale(face_roi, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))

            angle = 0.0
            if len(eyes) >= 2:
                eyes = sorted(eyes, key=lambda e: e[0])[:2]
                left_eye_center = np.array([x + eyes[0][0] + eyes[0][2] / 2.0, y + eyes[0][1] + eyes[0][3] / 2.0])
                right_eye_center = np.array([x + eyes[1][0] + eyes[1][2] / 2.0, y + eyes[1][1] + eyes[1][3] / 2.0])
                dy = right_eye_center[1] - left_eye_center[1]
                dx = right_eye_center[0] - left_eye_center[0]
                angle = np.degrees(np.arctan2(dy, dx))

            eyes_center = (x + fw / 2.0, y + fh / 2.0)
            rot_mat = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)
            rotated = cv2.warpAffine(img, rot_mat, (w, h), flags=cv2.INTER_LINEAR)

            pad_w = int(fw * margin)
            pad_h = int(fh * margin)
            x1 = max(x - pad_w, 0)
            y1 = max(y - pad_h, 0)
            x2 = min(x + fw + pad_w, w)
            y2 = min(y + fh + pad_h, h)

            if x2 > x1 and y2 > y1:
                crop = rotated[y1:y2, x1:x2]
                return cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_AREA)

    # Fall back to a center crop if no face detection is available.
    min_dim = min(w, h)
    crop_size = int(min_dim * (1.0 - margin))
    x1 = max((w - crop_size) // 2, 0)
    y1 = max((h - crop_size) // 2, 0)
    x2 = min(x1 + crop_size, w)
    y2 = min(y1 + crop_size, h)
    crop = img[y1:y2, x1:x2]
    return cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_AREA)


def detect_and_crop_face(image_path: str, out_size: int = IMAGE_SIZE, margin: float = FACE_MARGIN):
    """
    Load image_path from disk and align/crop the face via align_and_crop_face().

    Returns:
        aligned_crop (np.ndarray, BGR, out_size x out_size x 3) on success
        None if the image cannot be read from disk -- caller should log and skip
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    return align_and_crop_face(img, out_size=out_size, margin=margin)


# Manifest construction

def build_manifest(loader_outputs: list) -> pd.DataFrame:
    """
    Combine records from all dataset loaders into a single manifest.

    loader_outputs: list of lists of dicts, each dict:
        {
            "image_path": str,        # absolute or repo-relative path to the ORIGINAL image
            "person_id": str,         # must be unique per person WITHIN a source dataset;
                                       # will be namespaced with source below to guarantee
                                       # global uniqueness across datasets
            "label": int,             # 1 = makeup, 0 = no makeup
            "source": str,            # e.g. "kaggle_mvnm", "tapakah68"
        }

    Returns a DataFrame with an added 'identity_key' column (source + person_id)
    which is what split_by_identity() groups on.
    """
    all_records = [rec for source_list in loader_outputs for rec in source_list]
    df = pd.DataFrame(all_records)

    required_cols = {"image_path", "person_id", "label", "source"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Manifest missing required columns: {missing}")

    df["identity_key"] = df["source"] + "::" + df["person_id"].astype(str)
    df["label"] = df["label"].astype(int)
    return df


# ---------------------------------------------------------------------------
# Identity-safe split
# ---------------------------------------------------------------------------

def split_by_identity(df: pd.DataFrame,
                       train_frac: float = TRAIN_FRAC,
                       val_frac: float = VAL_FRAC,
                       test_frac: float = TEST_FRAC,
                       seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Split the manifest into train/val/test by identity_key, so that no person
    (and no before/after pair) ever appears in more than one split.

    Adds a 'split' column ('train' / 'val' / 'test') to the returned DataFrame.
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, "Fractions must sum to 1.0"

    groups = df["identity_key"].values

    # First split off train vs (val+test)
    gss1 = GroupShuffleSplit(n_splits=1, train_size=train_frac, random_state=seed)
    train_idx, rest_idx = next(gss1.split(df, groups=groups))

    df_train = df.iloc[train_idx].copy()
    df_rest = df.iloc[rest_idx].copy()

    # Then split rest into val vs test
    rest_groups = df_rest["identity_key"].values
    relative_val_frac = val_frac / (val_frac + test_frac)
    gss2 = GroupShuffleSplit(n_splits=1, train_size=relative_val_frac, random_state=seed)
    val_idx, test_idx = next(gss2.split(df_rest, groups=rest_groups))

    df_val = df_rest.iloc[val_idx].copy()
    df_test = df_rest.iloc[test_idx].copy()

    df_train["split"] = "train"
    df_val["split"] = "val"
    df_test["split"] = "test"

    result = pd.concat([df_train, df_val, df_test], ignore_index=True)

    # Sanity check: verify zero identity overlap across splits
    train_ids = set(df_train["identity_key"])
    val_ids = set(df_val["identity_key"])
    test_ids = set(df_test["identity_key"])
    assert not (train_ids & val_ids), "Identity leakage between train and val!"
    assert not (train_ids & test_ids), "Identity leakage between train and test!"
    assert not (val_ids & test_ids), "Identity leakage between val and test!"

    return result


# Run preprocessing over the manifest (crop + save)

def preprocess_dataset(df: pd.DataFrame, processed_root: Path = PROCESSED_ROOT) -> pd.DataFrame:
    """
    For every row in df, detect/align/crop the face and save it under:
        processed_root/<split>/<label_name>/<source>__<person_id>__<orig_filename>

    Returns df with an added 'processed_path' column; rows where face detection
    failed are dropped (and reported).
    """
    processed_paths = []
    keep_mask = []
    failures = []

    for row in df.itertuples():
        crop = detect_and_crop_face(row.image_path)
        if crop is None:
            failures.append(row.image_path)
            keep_mask.append(False)
            processed_paths.append(None)
            continue

        label_name = "makeup" if row.label == 1 else "no_makeup"
        out_dir = processed_root / row.split / label_name
        out_dir.mkdir(parents=True, exist_ok=True)

        orig_name = Path(row.image_path).stem
        out_name = f"{row.source}__{row.person_id}__{orig_name}.jpg"
        out_path = out_dir / out_name

        cv2.imwrite(str(out_path), crop)
        processed_paths.append(str(out_path))
        keep_mask.append(True)

    df = df.copy()
    df["processed_path"] = processed_paths
    df = df[keep_mask].reset_index(drop=True)

    if failures:
        print(f"[WARNING] Face detection failed on {len(failures)} images (dropped):")
        for f in failures[:20]:
            print(f"  - {f}")
        if len(failures) > 20:
            print(f"  ... and {len(failures) - 20} more")

    return df


# Dataset loaders -- STUBS. Fill these in once folder structures are confirmed.


def load_kaggle_makeup_vs_nonmakeup(root: Path) -> list:
    """
    Load the petersunga Makeup-vs-No-Makeup dataset.

    The dataset is unpaired by identity, so each image is treated as its own
    identity for split grouping purposes.
    """
    records = []
    for label_name, label_value in [("makeup", 1), ("no_makeup", 0)]:
        folder = root / label_name
        if not folder.exists():
            raise FileNotFoundError(f"Expected folder not found: {folder}")

        for image_path in sorted(folder.iterdir()):
            if not image_path.is_file():
                continue
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue

            records.append({
                "image_path": str(image_path.resolve()),
                "person_id": image_path.stem,
                "label": label_value,
                "source": "kaggle_mvnm",
            })

    return records


def load_tapakah68(root: Path) -> list:
    """
    Load the tapakah68 paired makeup dataset.

    Each CSV row contains a matching no_makeup and with_makeup image for one person.
    The person_id is derived from the image stem so both records share the same identity.
    """
    csv_path = root / "make_up.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Expected CSV file not found: {csv_path}")

    records = []
    import csv

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            no_makeup_rel = row.get("no_makeup")
            with_makeup_rel = row.get("with_makeup")
            if not no_makeup_rel or not with_makeup_rel:
                continue

            no_path = (root / no_makeup_rel).resolve()
            with_path = (root / with_makeup_rel).resolve()
            if not no_path.exists() or not with_path.exists():
                raise FileNotFoundError(
                    f"Missing tapakah68 image pair: {no_path} / {with_path}"
                )

            person_id = no_path.stem
            records.append({
                "image_path": str(no_path),
                "person_id": person_id,
                "label": 0,
                "source": "tapakah68",
            })
            records.append({
                "image_path": str(with_path),
                "person_id": person_id,
                "label": 1,
                "source": "tapakah68",
            })

    return records


# Main

if __name__ == "__main__":
    loader_outputs = [
        load_kaggle_makeup_vs_nonmakeup(RAW_DATA_ROOT / "petersunga"),
        load_kaggle_makeup_vs_nonmakeup(RAW_DATA_ROOT / "petersunga"),
        load_tapakah68(RAW_DATA_ROOT / "tapakah68"),
    ]

    manifest = build_manifest(loader_outputs)
    manifest = split_by_identity(manifest)
    manifest = preprocess_dataset(manifest)

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(MANIFEST_PATH, index=False)

    print(manifest.groupby(["split", "label"]).size())
    print(f"Manifest written to {MANIFEST_PATH}")
