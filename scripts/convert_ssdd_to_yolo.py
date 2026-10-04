"""Convert the SSDD dataset (BBox, VOC style) to the YOLO (Ultralytics) format.

Creates data/ssdd_yolo/ with parallel images/ and labels/ folders for the
splits: train, val, test, test_inshore and test_offshore.

The validation set is carved out of the official train split, so the
official test split stays untouched until the final evaluation.

Run from the project root:
    uv run python scripts/convert_ssdd_to_yolo.py
"""

import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


def convert_to_yolo_format(xmin, ymin, xmax, ymax, width, height):
    """Convert a VOC box (pixel corners) to YOLO format.

    YOLO expects the box center and size, normalized by the image size
    to the 0-1 range, so labels stay valid when images are resized.

    Returns:
        (x_center, y_center, box_width, box_height), all normalized.
    """
    x_center = (xmin + xmax) / 2.0 / width
    y_center = (ymin + ymax) / 2.0 / height
    box_width = (xmax - xmin) / width
    box_height = (ymax - ymin) / height
    return x_center, y_center, box_width, box_height


def convert_xml_file(xml_path, output_dir):
    """Convert one VOC XML annotation into a YOLO .txt label file.

    The label file gets the same name as the XML (e.g. 000002.txt),
    because YOLO matches images and labels by file name.

    Returns:
        Number of boxes written to the label file.
    """
    root = ET.parse(xml_path).getroot()

    width = int(root.find("size/width").text)
    height = int(root.find("size/height").text)

    lines = []
    # One <object> block per ship; an image can contain any number of them.
    for obj in root.findall("object"):
        box = obj.find("bndbox")
        xmin = int(box.find("xmin").text)
        ymin = int(box.find("ymin").text)
        xmax = int(box.find("xmax").text)
        ymax = int(box.find("ymax").text)
        x_center, y_center, box_width, box_height = convert_to_yolo_format(
            xmin, ymin, xmax, ymax, width, height
        )
        # Class id 0 = "ship" (the dataset has a single class).
        lines.append(f"0 {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}")

    label_path = output_dir / f"{xml_path.stem}.txt"
    label_path.write_text("\n".join(lines), encoding="utf-8")

    return len(lines)


def copy_image(stem, img_dir, images_out):
    """Copy the image matching a given file stem into the output folder.

    The image is looked up by file name rather than the <filename> field
    in the XML, because the XML metadata is outdated in SSDD.

    Raises:
        FileNotFoundError: If the image does not exist. Failing loudly is
            better than silently creating a label without an image.
    """
    img_path = img_dir / f"{stem}.jpg"
    if not img_path.exists():
        raise FileNotFoundError(f"File not found: {img_path}")
    shutil.copy2(img_path, images_out / img_path.name)


def split_train_val(files, val_ratio=0.1, seed=42):
    """Randomly split a list of files into train and validation subsets.

    A fixed seed makes the split reproducible across runs and machines.

    Returns:
        (train_files, val_files), both sorted.
    """
    # Copy the list: shuffle() works in place and must not modify the caller's list.
    shuffled = list(files)
    # Local generator, so the split doesn't depend on (or affect) global random state.
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    n_val = round(len(shuffled) * val_ratio)
    # Shuffling only decides which files go where; sort back for a predictable order.
    val_files = sorted(shuffled[:n_val])
    train_files = sorted(shuffled[n_val:])
    return train_files, val_files


def process_split(xml_files, img_dir, images_out, labels_out):
    """Copy images and write YOLO labels for one dataset split.

    Returns:
        (total_boxes, empty_images) statistics for the split.
    """
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    total_boxes = 0
    empty_images = 0
    for xml_path in xml_files:
        # Copy the image first: if it's missing, we stop before writing an orphan label.
        copy_image(xml_path.stem, img_dir, images_out)
        n_boxes = convert_xml_file(xml_path, labels_out)
        total_boxes += n_boxes
        if n_boxes == 0:
            empty_images += 1

    return total_boxes, empty_images


if __name__ == "__main__":
    src = Path("data/SSDD/BBox_SSDD/voc_style")
    out = Path("data/ssdd_yolo")

    # Start from a clean folder. Leftovers from earlier runs could leak
    # validation images into train. Only the generated folder is removed,
    # never the original dataset.
    if out.exists():
        shutil.rmtree(out)

    xml_files = sorted((src / "Annotations_train").glob("*.xml"))
    train_files, val_files = split_train_val(xml_files)

    # Each job: (split name, list of XML files, folder with matching images).
    # Train and val both come from the official train split.
    jobs = [
        ("train", train_files, src / "JPEGImages_train"),
        ("val", val_files, src / "JPEGImages_train"),
    ]

    # Test splits follow SSDD's naming pattern: Annotations_<name> / JPEGImages_<name>.
    # Inshore/offshore are subsets of test, used for a per-scene-type evaluation.
    for name in ["test", "test_inshore", "test_offshore"]:
        files = sorted((src / f"Annotations_{name}").glob("*.xml"))
        jobs.append((name, files, src / f"JPEGImages_{name}"))

    for split, files, img_dir in jobs:
        n_boxes, n_empty = process_split(
            files,
            img_dir,
            out / "images" / split,
            out / "labels" / split,
        )
        print(f"{split}: {len(files)} images, {n_boxes} ships, {n_empty} empty")