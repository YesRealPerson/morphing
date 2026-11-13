"""Build Beier-Neely feature lines for face images.

For each image:
  1. Get the 68 face keypoints from a Sapiens COCO-WholeBody model (or from an existing
     sapiens_pose.py JSON via --pose-json).
  2. Ask you to click points along both ears and the outline of the hair.
  3. Save <name>_lines.json with every point plus the feature lines, and <name>_lines.png.

Lines are listed in the same order for every image, so line i in image A corresponds to
line i in image B. Click the same number of points, in the same order, on every image.

Clicking: left-click adds a point, right-click (or Backspace) removes the last one.
The window moves on once you've placed the requested number of points.

Examples:
  python face_lines.py examples/pitt.jpg examples/norton.jpg --out examples/lines
  python face_lines.py examples/pitt.jpg --pose-json examples/pose/pitt.json
  python face_lines.py examples/norton.jpg --reference examples/lines/pitt_lines.json
  python face_lines.py examples/pitt.png examples/norton.png --skip ear_image_right

--skip drops a section from every image in the run, so the line lists still correspond.
If a feature is hidden in any one image (say an ear behind hair), skip it for all of them.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

# COCO-WholeBody keypoints 23..90 are the 68 face points in the 300-W layout:
#   0-16 jaw (image-left -> image-right), 17-21 / 22-26 brows, 27-30 nose bridge,
#   31-35 bottom of nose, 36-41 / 42-47 eyes (loops), 48-59 outer lips (loop), 60-67 inner lips.
FACE_OFFSET = 23
NUM_FACE = 68

# Face polylines (face indices 0-67). Consecutive points become one line each.
# Every other jaw/lip point is used so lines aren't too short. Edit to taste.
FACE_POLYLINES = {
    "jaw": [0, 2, 4, 6, 8, 10, 12, 14, 16],
    "brow_a": [17, 19, 21],
    "brow_b": [22, 24, 26],
    "nose_bridge": [27, 30],
    "nose_bottom": [31, 33, 35],
    "eye_a": [36, 37, 38, 39, 40, 41, 36],
    "eye_b": [42, 43, 44, 45, 46, 47, 42],
    "mouth": [48, 50, 52, 54, 56, 58, 48],
}


MANUAL_NAMES = ["ear_image_left", "hair", "ear_image_right", "hairline"]


def manual_sections(ear_points: int, hair_points: int, hairline_points: int, skip=()):
    """(name, number of points, instructions) for each hand-clicked polyline not in `skip`."""
    sections = [
        ("ear_image_left", ear_points,
         "Ear on the LEFT side of the image: click from the TOP of the ear down to the LOBE"),
        ("hair", hair_points,
         "Hair outline: start where the hair meets the top of the LEFT-side ear, "
         "go over the top of the head, end at the top of the RIGHT-side ear"),
        ("ear_image_right", ear_points,
         "Ear on the RIGHT side of the image: click from the TOP of the ear down to the LOBE"),
    ]
    if hairline_points > 0:
        sections.append(("hairline", hairline_points,
                         "Hairline (where forehead skin meets hair): click from LEFT temple to RIGHT temple"))
    return [s for s in sections if s[1] > 0 and s[0] not in skip]


# --------------------------------------------------------------------------------------
# Face keypoints
# --------------------------------------------------------------------------------------
def face_from_pose_json(path: Path):
    record = json.loads(path.read_text())
    inst = largest_instance(record["instances"])
    kpts = np.asarray(inst["keypoints"], dtype=np.float32)
    if len(kpts) != 133:
        raise SystemExit(f"{path} has {len(kpts)} keypoints; need a COCO-WholeBody (133) model.")
    scores = np.asarray(inst["keypoint_scores"], dtype=np.float32)
    return kpts[FACE_OFFSET:FACE_OFFSET + NUM_FACE], scores[FACE_OFFSET:FACE_OFFSET + NUM_FACE]


def largest_instance(instances):
    def area(inst):
        x1, y1, x2, y2 = inst["bbox"]
        return (x2 - x1) * (y2 - y1)
    return max(instances, key=area)


class FaceKeypointModel:
    """Runs sapiens_pose.py's pipeline and returns the 68 face points of the largest person."""

    def __init__(self, model_key, checkpoint, device, use_detector, det_thr):
        import torch
        import sapiens_pose as sp

        self.sp = sp
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ckpt = sp.resolve_checkpoint(model_key, checkpoint)
        print(f"Loading {ckpt.name} on {self.device}...")
        self.model = sp.load_pose_model(ckpt, self.device)
        self.detector = sp.PersonDetector(self.device, det_thr) if use_detector else None

    def __call__(self, img_bgr):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        bboxes = self.detector(img_rgb) if self.detector else np.empty((0, 4), np.float32)
        if len(bboxes) == 0:
            bboxes = np.array([[0, 0, w - 1, h - 1]], dtype=np.float32)
        results = self.sp.estimate_pose(self.model, img_rgb, bboxes, self.device, batch_size=4)
        inst = largest_instance(results)
        kpts = np.asarray(inst["keypoints"], dtype=np.float32)
        if len(kpts) != 133:
            raise SystemExit(f"Model returned {len(kpts)} keypoints; use a *-wholebody model.")
        scores = np.asarray(inst["keypoint_scores"], dtype=np.float32)
        return kpts[FACE_OFFSET:FACE_OFFSET + NUM_FACE], scores[FACE_OFFSET:FACE_OFFSET + NUM_FACE]


# --------------------------------------------------------------------------------------
# Clicking
# --------------------------------------------------------------------------------------
def click_points(img_rgb, face_pts, done_polylines, name, count, instructions, reference=None):
    """Show the image (and optional reference) and collect exactly `count` clicks."""
    import matplotlib.pyplot as plt

    while True:
        if reference is not None:
            fig, (ax, ax_ref) = plt.subplots(1, 2, figsize=(16, 9))
            ax_ref.imshow(reference["image"])
            ref_pts = np.asarray(reference["points"].get(name, []))
            if len(ref_pts):
                ax_ref.plot(ref_pts[:, 0], ref_pts[:, 1], "-o", color="orange", ms=5)
                for i, (x, y) in enumerate(ref_pts):
                    ax_ref.annotate(str(i + 1), (x, y), color="yellow", fontsize=11,
                                    xytext=(4, -4), textcoords="offset points")
            ax_ref.set_title(f"Reference: {reference['label']}", fontsize=10)
            ax_ref.axis("off")
        else:
            fig, ax = plt.subplots(figsize=(10, 10))

        ax.imshow(img_rgb)
        ax.scatter(face_pts[:, 0], face_pts[:, 1], s=8, c="cyan")
        for pts in done_polylines.values():
            pts = np.asarray(pts)
            ax.plot(pts[:, 0], pts[:, 1], "-o", color="lime", ms=3, lw=1)
        ax.set_title(f"{instructions}\n{count} points. Left-click to add, right-click/Backspace to undo.",
                     fontsize=10)
        ax.axis("off")
        fig.tight_layout()

        pts = plt.ginput(n=count, timeout=0, show_clicks=True,
                         mouse_add=1, mouse_pop=3, mouse_stop=None)
        plt.close(fig)
        if len(pts) == count:
            return [[float(x), float(y)] for x, y in pts]
        if not plt.get_fignums() and len(pts) == 0:
            raise SystemExit("Window closed; aborting.")
        print(f"  Got {len(pts)} of {count} points for {name}; try again.")


# --------------------------------------------------------------------------------------
# Lines + output
# --------------------------------------------------------------------------------------
def build_lines(face_pts, manual, face_polylines):
    """Return (lines, names). lines: (N, 2, 2) array of [[x1, y1], [x2, y2]]."""
    lines, names = [], []
    for poly_name, idx in face_polylines.items():
        for j, (a, b) in enumerate(zip(idx[:-1], idx[1:])):
            lines.append([face_pts[a].tolist(), face_pts[b].tolist()])
            names.append(f"{poly_name}_{j}")
    for poly_name, pts in manual.items():
        for j, (p, q) in enumerate(zip(pts[:-1], pts[1:])):
            lines.append([p, q])
            names.append(f"{poly_name}_{j}")
    return lines, names


def draw_lines(img_bgr, face_pts, lines):
    vis = img_bgr.copy()
    s = max(1, round(max(vis.shape[:2]) / 600))
    for x, y in face_pts:  # all 68 detected points; skipped ones just have no lines
        cv2.circle(vis, (round(x), round(y)), 2 * s, (255, 255, 0), -1, cv2.LINE_AA)
    for i, (p, q) in enumerate(lines):
        p = tuple(round(v) for v in p)
        q = tuple(round(v) for v in q)
        cv2.line(vis, p, q, (0, 255, 0), s, cv2.LINE_AA)
        cv2.circle(vis, p, 2 * s, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(vis, q, 2 * s, (0, 0, 255), -1, cv2.LINE_AA)
        mid = ((p[0] + q[0]) // 2, (p[1] + q[1]) // 2)
        cv2.putText(vis, str(i), mid, cv2.FONT_HERSHEY_SIMPLEX, 0.35 * s, (0, 255, 255), max(1, s // 2), cv2.LINE_AA)
    return vis


def load_reference(path: Path):
    rec = json.loads(path.read_text())
    img = cv2.cvtColor(cv2.imread(rec["image"]), cv2.COLOR_BGR2RGB)
    return {"image": img, "points": rec["points"], "label": Path(rec["image"]).name,
            "skipped": rec.get("skipped", [])}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+", help="face images, annotated in the order given")
    ap.add_argument("--out", default="lines_out", help="output folder")
    ap.add_argument("--pose-json", nargs="+",
                    help="sapiens_pose.py JSONs (WholeBody model), one per image, instead of running the model")
    ap.add_argument("--model", default="2b-wholebody", help="sapiens_pose.py --model key (must be *-wholebody)")
    ap.add_argument("--checkpoint", help="local WholeBody .pt2 checkpoint (overrides --model)")
    ap.add_argument("--device", help="cuda / cpu (default: cuda if available)")
    ap.add_argument("--detector", default="torchvision", choices=["torchvision", "none"],
                    help="'none' uses the whole image as the person box")
    ap.add_argument("--det-thr", type=float, default=0.5)
    ap.add_argument("--ear-points", type=int, default=4, help="points to click per ear (0 to skip ears)")
    ap.add_argument("--hair-points", type=int, default=9, help="points to click along the hair outline")
    ap.add_argument("--hairline-points", type=int, default=0,
                    help="points along the forehead hairline (0 = skip)")
    ap.add_argument("--reference", help="a previous *_lines.json to show beside each image while clicking; "
                                        "defaults to the first image annotated in this run")
    ap.add_argument("--skip", nargs="+", default=[], metavar="SECTION",
                    choices=MANUAL_NAMES + list(FACE_POLYLINES),
                    help="sections to leave out of every image in this run, e.g. an occluded ear: "
                         "--skip ear_image_right. Choices: " + ", ".join(MANUAL_NAMES + list(FACE_POLYLINES)))
    args = ap.parse_args()

    if args.pose_json and len(args.pose_json) != len(args.images):
        raise SystemExit("--pose-json needs exactly one JSON per image.")
    skip = sorted(set(args.skip))
    sections = manual_sections(args.ear_points, args.hair_points, args.hairline_points, skip)
    face_polylines = {k: v for k, v in FACE_POLYLINES.items() if k not in skip}
    if skip:
        print(f"Skipping: {', '.join(skip)}")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = None
    if not args.pose_json:
        model = FaceKeypointModel(args.model, args.checkpoint, args.device,
                                  args.detector == "torchvision", args.det_thr)

    reference = load_reference(Path(args.reference)) if args.reference else None
    if reference is not None and sorted(reference["skipped"]) != skip:
        print(f"WARNING: {args.reference} skipped {reference['skipped'] or 'nothing'} but this run skips "
              f"{skip or 'nothing'}; the line lists won't correspond. Use the same --skip for both.")

    for i, img_path in enumerate(map(Path, args.images)):
        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise SystemExit(f"Could not read {img_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]

        if args.pose_json:
            face_pts, face_scores = face_from_pose_json(Path(args.pose_json[i]))
        else:
            face_pts, face_scores = model(img_bgr)
        low = int((face_scores < 0.3).sum())
        print(f"{img_path.name}: 68 face points ({low} with score < 0.3)")

        manual = {}
        for name, count, instructions in sections:
            manual[name] = click_points(img_rgb, face_pts, manual, name, count, instructions, reference)

        lines, line_names = build_lines(face_pts, manual, face_polylines)
        record = {
            "image": str(img_path),
            "width": w,
            "height": h,
            "points": {"face": face_pts.tolist(), "face_scores": face_scores.tolist(), **manual},
            "face_polylines": face_polylines,
            "skipped": skip,
            "line_names": line_names,
            "lines": lines,
        }
        json_path = out_dir / f"{img_path.stem}_lines.json"
        json_path.write_text(json.dumps(record, indent=1))
        cv2.imwrite(str(out_dir / f"{img_path.stem}_lines.png"), draw_lines(img_bgr, face_pts, lines))
        print(f"  saved {json_path} ({len(lines)} lines)")

        if reference is None:
            reference = {"image": img_rgb, "points": manual, "label": img_path.name, "skipped": skip}


if __name__ == "__main__":
    main()
