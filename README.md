# Makeup_Detection_PyTorch: Binary Classifier + Real-Time Webcam Overlay

Binary classification (makeup / no makeup) with a hard-negative angle:
full-glam images vs. very natural/subtle makeup. Trained model deployed as a real-time webcam classifier overlay.

## Project scope

- **Task**: binary classification, `makeup` (1) vs `no_makeup` (0).
- **Datasets**:
  Download both datasets directly from Kaggle and place the extracted files under
  the expected `data/raw/` paths. Raw dataset files are not tracked in Git.
  - petersunga
    ["Make-up vs No Make-up"](https://www.kaggle.com/datasets/petersunga/make-up-vs-no-make-up):
    larger, high-resolution makeup/no-makeup images. These are class examples,
    not before/after images of the same person.
  - tapakah68
    ["Makeup Detection Face Dataset"](https://www.kaggle.com/datasets/tapakah68/makeup-detection-dataset):
    smaller paired dataset with the same people photographed before and after
    makeup.
- **Target platform**: MacBook Pro (Apple Silicon, M5 Pro).
- **Framework**: PyTorch (MPS backend for training/inference on-device).
- **Real-time deployment**: webcam frame → face detect/crop → classify → label
  overlay (`webcam_inference.py`).

## Why identity-based splitting matters here

The tapakah68 dataset contains the same people photographed before and after
makeup. If a person's before image ends up in train and their after image ends up
in test, the model can learn to recognize the *person* rather than the *makeup*,
which silently inflates accuracy.

For paired data, splitting should be done by identity (`source::person_id`),
never by raw image, so no person appears in more than one split. The petersunga
dataset is treated as unpaired class data because its makeup/no-makeup images are
not before/after shots of the same people.

## Pipeline stages

1. **Loaders** (`load_kaggle_makeup_vs_nonmakeup`, `load_tapakah68` in
   `preprocess/preprocess.py`): dataset-specific, convert each source's raw
   folder structure into a common record format:
   `{"image_path", "person_id", "label", "source"}`.
   - `load_kaggle_makeup_vs_nonmakeup` reads `data/raw/petersunga/{makeup,no_makeup}/*`;
     since these are unpaired class examples, `person_id` is just each image's
     filename stem.
   - `load_tapakah68` reads `data/raw/tapakah68/make_up.csv`, which pairs each
     person's no_makeup/with_makeup images; the shared `person_id` keeps each
     before/after pair grouped together for splitting.
2. **`build_manifest()`**: merges all loader outputs into a single DataFrame,
   namespaces `person_id` by source to prevent cross-dataset ID collisions.
3. **`split_by_identity()`**: group-aware train/val/test split (70/15/15 default),
   with leakage assertions.
4. **`align_and_crop_face()` / `detect_and_crop_face()`**: face detection,
   eye-line alignment, tight crop (MediaPipe FaceMesh, falling back to Haar
   cascades, falling back to a plain center crop), resize to 224×224.
   `align_and_crop_face()` operates on an in-memory image array and is shared
   with `webcam_inference.py`, so training and live inference always see
   identical crops. `detect_and_crop_face()` is the thin disk-reading wrapper
   used by `preprocess_dataset()`; it only drops an image (returns `None`)
   if the file itself can't be read. A missing face never drops an image,
   it just falls back to a center crop, so the "N images dropped" warning
   only reflects corrupt/unreadable files, not detection quality.
5. **`preprocess_dataset()`**: runs the above over the full manifest, saves
   crops to `data/processed/<split>/<makeup|no_makeup>/`.
6. **`manifest.csv`**: final combined record of every processed image: source,
   person_id, label, split, processed_path.
7. **`dataset.py`**: `MakeupDataset` / `get_dataloaders()` read the manifest
   and serve `data/processed/` images as PyTorch tensors, split-aware
   (augmentation only on `train`; see the module docstring for why hue/
   saturation jitter is deliberately excluded — makeup is itself a color
   signal, and jittering it would blur out the thing the model needs to learn).
8. **`train.py`**: fine-tunes an ImageNet-pretrained ResNet18, with a
   class-weighted `BCEWithLogitsLoss` to correct for the makeup/no_makeup
   imbalance in the splits. Early-stops and checkpoints on best validation
   F1 (`checkpoints/best_model.pt`, gitignored), then reports test-set
   precision/recall/F1/AUC from that checkpoint.
9. **`webcam_inference.py`**: loads `checkpoints/best_model.pt`, runs live
   webcam frames through the same crop pipeline as training, and overlays a
   `MAKEUP`/`NO MAKEUP` + confidence label in real time.

## Setup

```bash
pip install -r requirements.txt
```

Version pins in `requirements.txt` matter, not just style:
- `mediapipe` must stay in `0.10.13`-`0.10.21`: `0.10.30+` dropped the legacy
  `mediapipe.solutions` API (including `face_mesh`) that `align_and_crop_face()`
  uses. `preprocess.py` falls back to Haar cascades automatically if a
  newer/solutions-less mediapipe is installed, but alignment quality is worse.
- Install `opencv-contrib-python` only, not `opencv-python` alongside it —
  installing both can clobber shared files under `site-packages/cv2` and break
  the `cv2` import. `opencv-contrib-python` is a superset, so nothing is lost.
  Also avoid `opencv-python(-contrib)==5.x`: that major version ships **no**
  Haar cascade XML files at all (`cv2.data.haarcascades` is empty); `4.11.0.86`
  is the latest 4.x release and has all 17.
- `mediapipe==0.10.21` requires `numpy<2`; verify whatever `opencv-contrib-python`
  version you use doesn't force `numpy>=2` before bumping it.

Verified working end-to-end on Apple Silicon (M-series, Python 3.12): loaders →
manifest → identity-split → face-crop → `dataset.py` → `train.py` →
`webcam_inference.py`, with zero face-detection failures on the full dataset
(1,558/1,558 raw images made it into the processed set).

## Directory layout expected

```
data/
  raw/
    petersunga/        # petersunga "Make-up vs No Make-up" (makeup/, no_makeup/)
    tapakah68/          # tapakah68 "Makeup Detection Face Dataset" (make_up.csv, no_makeup/, with_makeup/)
  processed/           # generated by preprocess.py
  manifest.csv          # generated by preprocess.py
checkpoints/            # generated by train.py (gitignored); best_model.pt lives here
```

## Usage

**1. Preprocess** builds the manifest and face-cropped image set:
```bash
python preprocess/preprocess.py
```
Prints a train/val/test x label count breakdown and writes `data/manifest.csv`
on completion. Safe to rerun on the same raw data (deterministic output
filenames just get overwritten); if the raw dataset itself changes between
runs, clear `data/processed/` first so stale files from a different split
assignment can't linger — the script does not do this automatically.

**2. Sanity-check the dataset/dataloaders**:
```bash
python dataset.py
```
Prints per-split class counts and one batch's tensor shape.

**3. Train**:
```bash
python train.py --epochs 20 --patience 5
```
See `python train.py --help` for all options (batch size, LR, weight decay,
num workers, manifest path). Saves the best validation-F1 checkpoint to
`checkpoints/best_model.pt` and reports final test-set metrics from it.

**4. Run live webcam inference**:
```bash
python webcam_inference.py
```
Requires `checkpoints/best_model.pt` to exist (run `train.py` first). Press
`q` to quit the window. Use `--camera-index N` to pick a different capture
device and `--threshold` to change the makeup/no-makeup decision cutoff
(default 0.5).

> **macOS + iPhone gotcha**: if you have Continuity Camera enabled and your
> iPhone is nearby, `--camera-index 0` may resolve to the iPhone (it'll
> visibly "ring" and never connect) instead of the built-in camera. Run
> `system_profiler SPCameraDataType` to see your registered cameras and try
> `--camera-index 1`, or turn off Continuity Camera on the iPhone (Settings →
> General → AirPlay & Handoff) to make index `0` reliably the built-in
> camera.

## Status / open items

- [x] Loaders implemented for both datasets (`load_kaggle_makeup_vs_nonmakeup`,
      `load_tapakah68`).
- [x] petersunga treated as unpaired class data, tapakah68 as paired
      before/after identity data.
- [x] Checked face-detection outcomes: 0 images dropped (all 1,558 raw images
      made it into `manifest.csv`). Note this only measures unreadable files,
      not detection quality — see the pipeline-stage note above on
      `align_and_crop_face()`'s fallback chain.
- [x] `requirements.txt` added, pinning the versions under Setup above.
- [x] Model training script (`train.py`).
- [x] Real-time webcam inference script (`webcam_inference.py`).
- [ ] Revisit the 70/15/15 split ratio now that dataset size is known
      (1,532 unique identities total, 1,072/230/230 across train/val/test) —
      current split has zero identity leakage (verified) but hasn't been
      tuned beyond the default. tapakah68 has only 26 identities, so val/test
      carry very few of them (~4 each).
- [ ] Instrument `align_and_crop_face()` to report which detection path
      (MediaPipe / Haar cascade / center-crop fallback) was used per image,
      so image quality issues are visible instead of silently absorbed by
      the fallback chain.
- [ ] Run a full training pass and record baseline metrics here (only a short
      smoke-test run has been done so far, to verify the script itself works).
