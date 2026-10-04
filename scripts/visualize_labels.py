"""Draw YOLO labels on images to visually verify the dataset conversion."""

from pathlib import Path
import random
import cv2


def yolo_to_pixel_box(x_center, y_center, box_width, box_height, img_width, img_height):
    """Convert a normalized YOLO box back to pixel corners.

    Inverse of convert_to_yolo_format() from convert_ssdd_to_yolo.py.

    Returns:
        (xmin, ymin, xmax, ymax) as integers, ready for drawing.
    """
    xmin = (x_center - box_width / 2) * img_width
    ymin = (y_center - box_height / 2) * img_height
    xmax = (x_center + box_width / 2) * img_width
    ymax = (y_center + box_height / 2) * img_height
    # Drawing functions need whole pixels.
    return round(xmin), round(ymin), round(xmax), round(ymax)


def read_label_file(label_path):
    """Read a YOLO label file into a list of (class_id, x, y, w, h) tuples."""
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        class_id, x_center, y_center, box_width, box_height = line.split()
        boxes.append((int(class_id), float(x_center), float(y_center),
                      float(box_width), float(box_height)))
    return boxes

def draw_labels(img_path, label_path):
    """Load an image and draw its YOLO boxes on it.

    Returns:
        (image with boxes drawn, number of boxes).
    """
    # cv2 doesn't raise on a bad path, it silently returns None, so we check ourselves.
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")

    # NumPy arrays are (rows, columns, channels), i.e. height comes first.
    img_height, img_width = img.shape[:2]

    boxes = read_label_file(label_path)
    for class_id, x_c, y_c, w, h in boxes:
        xmin, ymin, xmax, ymax = yolo_to_pixel_box(x_c, y_c, w, h, img_width, img_height)
        # Color is BGR in OpenCV: (0, 255, 0) = green. Thickness 1 px, ships are small.
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)

    return img, len(boxes)

if __name__ == "__main__":
    dataset = Path("data/ssdd_yolo")
    out_dir = Path("results/label_check")
    out_dir.mkdir(parents=True, exist_ok=True)

    samples_per_split = 4
    # Fixed seed: the same images are drawn each run, so results are comparable.
    rng = random.Random(0)

    for split in ["train", "val", "test_inshore", "test_offshore"]:
        img_paths = sorted((dataset / "images" / split).glob("*.jpg"))
        for img_path in rng.sample(img_paths, samples_per_split):
            label_path = dataset / "labels" / split / f"{img_path.stem}.txt"
            img, n_boxes = draw_labels(img_path, label_path)

            out_path = out_dir / f"{split}_{img_path.stem}.jpg"
            cv2.imwrite(str(out_path), img)
            print(f"{out_path.name}: {n_boxes} ships")