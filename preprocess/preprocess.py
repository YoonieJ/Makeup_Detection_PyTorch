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
import mediapipe as mp
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit

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

_mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.5,
)

# MediaPipe FaceMesh landmark indices for the outer corners of each eye.
# Used to compute the roll angle for alignment.
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263


def detect_and_crop_face(image_path: str, out_size: int = IMAGE_SIZE, margin: float = FACE_MARGIN):
    """
    Detect the face in image_path, rotate to align the eyes horizontally,
    crop tightly to the face (excluding most hair/background), and resize.

    Returns:
        aligned_crop (np.ndarray, BGR, out_size x out_size x 3) on success
        None on failure (no face detected) -- caller should log and skip
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return None

    h, w = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = _mp_face_mesh.process(rgb)

    if not result.multi_face_landmarks:
        return None

    landmarks = result.multi_face_landmarks[0].landmark

    # Pixel coords of eye corners for alignment
    left_eye = np.array([landmarks[LEFT_EYE_OUTER].x * w, landmarks[LEFT_EYE_OUTER].y * h])
    right_eye = np.array([landmarks[RIGHT_EYE_OUTER].x * w, landmarks[RIGHT_EYE_OUTER].y * h])

    # Roll angle between the eyes
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dy, dx))

    # Rotate the full image around the midpoint between the eyes
    eyes_center = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)
    rotated = cv2.warpAffine(img, rot_mat, (w, h), flags=cv2.INTER_LINEAR)

    # Re-run landmark detection on the rotated image to get an accurate face bbox
    rgb_rot = cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)
    result_rot = _mp_face_mesh.process(rgb_rot)
    if not result_rot.multi_face_landmarks:
        return None

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

    if x2 <= x1 or y2 <= y1:
        return None

    crop = rotated[y1:y2, x1:x2]
    crop_resized = cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_AREA)
    return crop_resized


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
    petersunga "Make-up vs No Make-up" dataset: root/makeup/*.jpeg and
    root/no_makeup/*.jpeg. Unpaired class data (no shared identities between
    or within classes), so person_id is just the image's own filename stem.
    Must return list of dicts: {"image_path", "person_id", "label", "source": "kaggle_mvnm"}
    """
    records = []
    for label, subdir in ((1, "makeup"), (0, "no_makeup")):
        for image_path in sorted((root / subdir).iterdir()):
            if not image_path.is_file():
                continue
            records.append({
                "image_path": str(image_path),
                "person_id": image_path.stem,
                "label": label,
                "source": "kaggle_mvnm",
            })
    return records


def load_tapakah68(root: Path) -> list:
    """
    tapakah68 "Makeup Detection Face Dataset": root/make_up.csv lists one row
    per person with matched no_makeup/with_makeup relative paths. Paired
    before/after data, so person_id is shared between the two rows emitted
    per person (here, the CSV row index).
    Must return list of dicts: {"image_path", "person_id", "label", "source": "tapakah68"}
    """
    df = pd.read_csv(root / "make_up.csv")

    records = []
    for idx, row in df.iterrows():
        person_id = str(idx)
        records.append({
            "image_path": str(root / row["no_makeup"]),
            "person_id": person_id,
            "label": 0,
            "source": "tapakah68",
        })
        records.append({
            "image_path": str(root / row["with_makeup"]),
            "person_id": person_id,
            "label": 1,
            "source": "tapakah68",
        })
    return records


# Main

if __name__ == "__main__":
    loader_outputs = [
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
