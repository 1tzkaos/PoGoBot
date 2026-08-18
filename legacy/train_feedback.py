#!/usr/bin/env python3
"""
Active Learning Dataset Ingestion & Automated Retraining Pipeline for PoGoBot.

Capabilities:
  1. Ingests user-provided raw images from `datasets/user_images/`
  2. Inspects & visualizes generated bounding boxes (`--inspect` or `--export-previews`)
  3. Merges self-supervised active learning buffer (`datasets/active_feedback/`) with base datasets
  4. Manages 5-class detection dataset: ['pokemon', 'pokestop', 'rocket', 'gym', 'raid']
  5. One-command MPS-accelerated fine-tuning and evaluation
"""

import argparse
import os
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

# Core Paths
BASE_DIR = Path(__file__).parent.resolve()
DATASETS_DIR = BASE_DIR / "datasets"
USER_IMAGES_DIR = DATASETS_DIR / "user_images"
ACTIVE_FEEDBACK_DIR = DATASETS_DIR / "active_feedback"
MERGED_DATASET_DIR = DATASETS_DIR / "merged_pokemongo"
PREVIEWS_DIR = ACTIVE_FEEDBACK_DIR / "previews"

MODELS_DIR = BASE_DIR / "models"
DET_MODEL_PATH = MODELS_DIR / "pokemongo_yolov8n.pt"

CLASS_NAMES = ["pokemon", "pokestop", "pokestop_rocket"]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}
ID_TO_CLASS = {i: name for i, name in enumerate(CLASS_NAMES)}

CLASS_COLORS = {
    "pokemon": (0, 255, 0),             # Green
    "pokestop": (255, 200, 0),          # Yellow
    "pokestop_rocket": (200, 50, 255),  # Purple
}


def get_default_device(user_device: str = "auto"):
    if user_device != "auto":
        return user_device
    try:
        import torch
        if torch.cuda.is_available():
            return 0
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def init_directories():
    """Initializes standard dataset folders."""
    USER_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    (ACTIVE_FEEDBACK_DIR / "train" / "images").mkdir(parents=True, exist_ok=True)
    (ACTIVE_FEEDBACK_DIR / "train" / "labels").mkdir(parents=True, exist_ok=True)
    (ACTIVE_FEEDBACK_DIR / "val" / "images").mkdir(parents=True, exist_ok=True)
    (ACTIVE_FEEDBACK_DIR / "val" / "labels").mkdir(parents=True, exist_ok=True)
    (ACTIVE_FEEDBACK_DIR / "hard_negatives").mkdir(parents=True, exist_ok=True)
    PREVIEWS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def write_yolo_yaml(dest_yaml: Path, dataset_path: Path):
    """Writes standard YOLOv8 data.yaml for 5 classes."""
    data = {
        "path": str(dataset_path.resolve()),
        "train": "train/images",
        "val": "val/images",
        "nc": len(CLASS_NAMES),
        "names": CLASS_NAMES,
    }
    with open(dest_yaml, "w") as f:
        yaml.dump(data, f, default_flow_style=False)
    print(f"[dataset] Written configuration to {dest_yaml}")


def show_stats():
    """Displays dataset statistics across base, user, and active feedback buffers."""
    init_directories()
    user_imgs = list(USER_IMAGES_DIR.glob("*.png")) + list(USER_IMAGES_DIR.glob("*.jpg"))
    active_train_imgs = list((ACTIVE_FEEDBACK_DIR / "train" / "images").glob("*.png")) + list((ACTIVE_FEEDBACK_DIR / "train" / "images").glob("*.jpg"))
    active_val_imgs = list((ACTIVE_FEEDBACK_DIR / "val" / "images").glob("*.png")) + list((ACTIVE_FEEDBACK_DIR / "val" / "images").glob("*.jpg"))
    hard_negatives = list((ACTIVE_FEEDBACK_DIR / "hard_negatives").glob("*.png")) + list((ACTIVE_FEEDBACK_DIR / "hard_negatives").glob("*.jpg"))

    print("\n" + "=" * 58)
    print("           POGOBOT DATASET & ACTIVE LEARNING STATS")
    print("=" * 58)
    print(f"  * User Drop Folder (`{USER_IMAGES_DIR.name}/`):       {len(user_imgs):5d} raw images")
    print(f"  * Active Feedback Buffer (Train):                 {len(active_train_imgs):5d} verified samples")
    print(f"  * Active Feedback Buffer (Val):                   {len(active_val_imgs):5d} verified samples")
    print(f"  * Hard Negative Log (`hard_negatives/`):          {len(hard_negatives):5d} false positive clips")
    print("=" * 58 + "\n")


def ingest_user_images(confidence: float = 0.25, device: str = "auto"):
    """
    Runs pre-annotation on raw images in `datasets/user_images/` using the current model,
    converting them into YOLO-formatted training samples in `datasets/active_feedback/train/`.
    """
    from ultralytics import YOLO

    init_directories()
    user_imgs = sorted(list(USER_IMAGES_DIR.glob("*.png")) + list(USER_IMAGES_DIR.glob("*.jpg")))
    if not user_imgs:
        print(f"[ingest] No raw images found in {USER_IMAGES_DIR}.")
        print("  -> Drop your screenshots (*.png, *.jpg) into datasets/user_images/ and rerun.")
        return

    if not DET_MODEL_PATH.exists():
        print(f"[ingest] Base model {DET_MODEL_PATH} not found. Using pretrained yolov8n.pt for bootstrap.")
        model = YOLO("yolov8n.pt")
    else:
        print(f"[ingest] Loading detector model: {DET_MODEL_PATH}")
        model = YOLO(str(DET_MODEL_PATH))

    selected_device = get_default_device(device)
    print(f"[ingest] Pre-annotating {len(user_imgs)} user images on {selected_device}...")

    train_img_dir = ACTIVE_FEEDBACK_DIR / "train" / "images"
    train_lbl_dir = ACTIVE_FEEDBACK_DIR / "train" / "labels"

    processed_count = 0
    box_count = 0

    for img_path in user_imgs:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        results = model.predict(source=img, conf=confidence, device=selected_device, verbose=False)
        label_lines = []

        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id = int(box.cls[0].item())
                if hasattr(r, "names") and cls_id in r.names:
                    cname = r.names[cls_id].lower()
                else:
                    cname = "pokemon"

                target_id = CLASS_TO_ID.get(cname, 0)
                xywhn = box.xywhn[0].tolist()
                label_lines.append(f"{target_id} {xywhn[0]:.6f} {xywhn[1]:.6f} {xywhn[2]:.6f} {xywhn[3]:.6f}")
                box_count += 1

        dest_img_path = train_img_dir / img_path.name
        dest_lbl_path = train_lbl_dir / f"{img_path.stem}.txt"

        shutil.copy2(img_path, dest_img_path)
        dest_lbl_path.write_text("\n".join(label_lines) + ("\n" if label_lines else ""))
        processed_count += 1

    print(f"\n[ingest] Successfully ingested {processed_count} images ({box_count} detected bounding boxes) into {train_img_dir}")
    print(f"[ingest] To visually inspect annotations: python train_feedback.py --inspect")


def render_annotated_image(img_path: Path, lbl_path: Path):
    """Renders bounding boxes on an image based on its YOLO .txt label."""
    img = cv2.imread(str(img_path))
    if img is None:
        return None, 0

    h, w = img.shape[:2]
    boxes_drawn = 0

    if lbl_path.exists():
        lines = [line.strip() for line in lbl_path.read_text().splitlines() if line.strip()]
        for line in lines:
            parts = line.split()
            if len(parts) != 5:
                continue
            cls_id = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:])

            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)

            cname = ID_TO_CLASS.get(cls_id, f"class_{cls_id}")
            color = CLASS_COLORS.get(cname, (0, 255, 0))

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            label = f"{cname}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img, (x1, max(0, y1 - th - 10)), (x1 + tw + 10, y1), color, -1)
            cv2.putText(img, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            boxes_drawn += 1

    return img, boxes_drawn


def inspect_dataset_interactive():
    """
    Opens an interactive OpenCV window to visually browse through all pre-annotated images.
    Keys:
      - 'd' / Right Arrow / Space: Next Image
      - 'a' / Left Arrow: Previous Image
      - 'q' / ESC: Quit
    """
    train_img_dir = ACTIVE_FEEDBACK_DIR / "train" / "images"
    train_lbl_dir = ACTIVE_FEEDBACK_DIR / "train" / "labels"

    img_files = sorted(list(train_img_dir.glob("*.png")) + list(train_img_dir.glob("*.jpg")))
    if not img_files:
        print(f"[inspect] No images found in {train_img_dir}. Run `python train_feedback.py --ingest` first.")
        return

    print(f"\n[inspect] Opening visual inspector ({len(img_files)} images)...")
    print("  -> Press 'd' / Right Arrow / Space for NEXT image")
    print("  -> Press 'a' / Left Arrow for PREVIOUS image")
    print("  -> Press 'q' or ESC to EXIT\n")

    window_name = "PoGoBot Dataset Inspector (Press 'd': Next, 'a': Prev, 'q': Quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 540, 960)

    idx = 0
    while 0 <= idx < len(img_files):
        img_path = img_files[idx]
        lbl_path = train_lbl_dir / f"{img_path.stem}.txt"

        annotated_img, num_boxes = render_annotated_image(img_path, lbl_path)
        if annotated_img is None:
            idx += 1
            continue

        h, w = annotated_img.shape[:2]

        # Render Header Bar
        cv2.rectangle(annotated_img, (0, 0), (w, 80), (20, 20, 20), -1)
        header_text = f"[{idx + 1}/{len(img_files)}] {img_path.name}"
        stats_text = f"Boxes: {num_boxes} | Class: pokemon/pokestop | ('d': Next, 'a': Prev, 'q': Quit)"
        cv2.putText(annotated_img, header_text, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(annotated_img, stats_text, (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        cv2.imshow(window_name, annotated_img)
        key = cv2.waitKey(0) & 0xFF

        if key in [ord("q"), 27]:  # 'q' or ESC
            break
        elif key in [ord("d"), ord(" "), 83]:  # 'd', Space, or Right
            idx = min(len(img_files) - 1, idx + 1)
        elif key in [ord("a"), 81]:  # 'a' or Left
            idx = max(0, idx - 1)

    cv2.destroyAllWindows()


def export_previews():
    """
    Renders bounding boxes on all training images and saves them to `datasets/active_feedback/previews/`
    so you can view them as thumbnails directly in macOS Finder or Preview!
    """
    train_img_dir = ACTIVE_FEEDBACK_DIR / "train" / "images"
    train_lbl_dir = ACTIVE_FEEDBACK_DIR / "train" / "labels"
    PREVIEWS_DIR.mkdir(parents=True, exist_ok=True)

    img_files = sorted(list(train_img_dir.glob("*.png")) + list(train_img_dir.glob("*.jpg")))
    if not img_files:
        print(f"[previews] No images found in {train_img_dir}. Run `python train_feedback.py --ingest` first.")
        return

    print(f"[previews] Rendering bounding box overlays for {len(img_files)} images into {PREVIEWS_DIR}...")
    for idx, img_path in enumerate(img_files):
        lbl_path = train_lbl_dir / f"{img_path.stem}.txt"
        annotated_img, num_boxes = render_annotated_image(img_path, lbl_path)
        if annotated_img is not None:
            out_path = PREVIEWS_DIR / f"preview_{img_path.name}"
            cv2.imwrite(str(out_path), annotated_img)

    print(f"[previews] Exported {len(img_files)} preview overlays to: {PREVIEWS_DIR}")
    print(f"[previews] Tip: Run `open {PREVIEWS_DIR}` in terminal to browse thumbnails in Finder!")


def merge_datasets():
    """Merges base pokemongo dataset with active feedback buffer into merged_pokemongo/."""
    MERGED_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    m_train_imgs = MERGED_DATASET_DIR / "train" / "images"
    m_train_lbls = MERGED_DATASET_DIR / "train" / "labels"
    m_val_imgs = MERGED_DATASET_DIR / "val" / "images"
    m_val_lbls = MERGED_DATASET_DIR / "val" / "labels"

    for d in [m_train_imgs, m_train_lbls, m_val_imgs, m_val_lbls]:
        d.mkdir(parents=True, exist_ok=True)

    # 1. Copy base dataset (prioritizing pokemongo_v2 from latest Roboflow export)
    base_dir = DATASETS_DIR / "pokemongo_v2" if (DATASETS_DIR / "pokemongo_v2").exists() else DATASETS_DIR / "pokemongo"
    if (base_dir / "train" / "images").exists():
        print(f"[merge] Copying base training dataset from {base_dir.name}...")
        for f in (base_dir / "train" / "images").glob("*.*"):
            im = cv2.imread(str(f))
            if im is not None:
                h, w = im.shape[:2]
                # If image was squashed to square in Roboflow, restore vertical proportions
                if abs(w - h) < 10:
                    target_h = int(w / 0.4615)
                    im = cv2.resize(im, (w, target_h), interpolation=cv2.INTER_LANCZOS4)
                cv2.imwrite(str(m_train_imgs / f.name), im)

        for f in (base_dir / "train" / "labels").glob("*.txt"):
            shutil.copy2(f, m_train_lbls / f.name)

    if (base_dir / "valid" / "images").exists():
        for f in (base_dir / "valid" / "images").glob("*.*"):
            shutil.copy2(f, m_val_imgs / f.name)
        for f in (base_dir / "valid" / "labels").glob("*.txt"):
            shutil.copy2(f, m_val_lbls / f.name)

    # 2. Merge active feedback buffer
    if (ACTIVE_FEEDBACK_DIR / "train" / "images").exists():
        print("[merge] Merging self-supervised active learning buffer...")
        for f in (ACTIVE_FEEDBACK_DIR / "train" / "images").glob("*.*"):
            shutil.copy2(f, m_train_imgs / f.name)
        for f in (ACTIVE_FEEDBACK_DIR / "train" / "labels").glob("*.txt"):
            shutil.copy2(f, m_train_lbls / f.name)
    if (ACTIVE_FEEDBACK_DIR / "val" / "images").exists():
        for f in (ACTIVE_FEEDBACK_DIR / "val" / "images").glob("*.*"):
            shutil.copy2(f, m_val_imgs / f.name)
        for f in (ACTIVE_FEEDBACK_DIR / "val" / "labels").glob("*.txt"):
            shutil.copy2(f, m_val_lbls / f.name)

    # Ensure validation set has at least 1 image
    val_files = list(m_val_imgs.glob("*.*"))
    if not val_files:
        train_files = list(m_train_imgs.glob("*.*"))
        if train_files:
            split_point = max(1, int(len(train_files) * 0.15))
            for f in train_files[:split_point]:
                shutil.move(f, m_val_imgs / f.name)
                lbl = m_train_lbls / f"{f.stem}.txt"
                if lbl.exists():
                    shutil.move(lbl, m_val_lbls / lbl.name)

    yaml_path = MERGED_DATASET_DIR / "data.yaml"
    write_yolo_yaml(yaml_path, MERGED_DATASET_DIR)
    return yaml_path


def retrain_model(epochs: int = 50, imgsz: int = 640, device: str = "auto"):
    """Fine-tunes YOLOv8n object detector using the merged active learning dataset."""
    from ultralytics import YOLO

    yaml_path = merge_datasets()
    selected_device = get_default_device(device)

    print(f"\n[retrain] Starting YOLOv8n training on {selected_device} for {epochs} epochs (rect=True)...")
    base_weights = str(DET_MODEL_PATH) if DET_MODEL_PATH.exists() else "yolov8n.pt"
    model = YOLO(base_weights)

    run_name = f"retrain_{int(time.time())}"
    result = model.train(
        data=str(yaml_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=16,
        rect=True,
        device=selected_device,
        project=str(MODELS_DIR / "runs"),
        name=run_name,
        exist_ok=True,
        verbose=True,
    )

    trained_best = Path(result.save_dir) / "weights" / "best.pt"
    if trained_best.exists():
        shutil.copy2(trained_best, DET_MODEL_PATH)
        print(f"\n[retrain] Retraining complete! Updated model saved to: {DET_MODEL_PATH}\n")


# -------------------------------------------------------------
# CLI Entry Point
# -------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PoGoBot Active Learning & Dataset Training Pipeline")
    parser.add_argument("--stats", action="store_true", help="Display dataset statistics.")
    parser.add_argument("--ingest", action="store_true", help="Pre-annotate and ingest images from datasets/user_images/.")
    parser.add_argument("--inspect", action="store_true", help="Open interactive visual inspector GUI to review annotations.")
    parser.add_argument("--export-previews", action="store_true", help="Render and export annotated preview images to datasets/active_feedback/previews/ for Finder.")
    parser.add_argument("--retrain", action="store_true", help="Merge active feedback buffer and fine-tune YOLO model.")
    parser.add_argument("--epochs", type=int, default=40, help="Training epochs (default: 40).")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size (default: 640).")
    parser.add_argument("--confidence", type=float, default=0.25, help="Confidence for pre-annotation (default: 0.25).")
    parser.add_argument("--device", default="auto", help="Inference/training device ('auto', 'mps', 'cuda', 'cpu').")
    args = parser.parse_args()

    init_directories()

    if args.ingest:
        ingest_user_images(confidence=args.confidence, device=args.device)
    elif args.inspect:
        inspect_dataset_interactive()
    elif args.export_previews:
        export_previews()
    elif args.retrain:
        retrain_model(epochs=args.epochs, imgsz=args.imgsz, device=args.device)
    else:
        show_stats()
