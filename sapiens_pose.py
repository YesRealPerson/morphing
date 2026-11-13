"""Run Sapiens pose estimation (facebookresearch/sapiens, "lite" TorchScript models) on images.

Pipeline (top-down, mirrors lite/demo/vis_pose.py):
  1. Find people: torchvision Faster R-CNN person detector, a user-supplied --bbox,
     or the whole image (--detector none).
  2. Crop each person with a 1.25x padded, 3:4 affine warp to 1024x768.
  3. Run the Sapiens TorchScript model -> heatmaps (K x 256 x 192).
  4. Decode heatmaps with UDP + DARK sub-pixel refinement, map back to image coords.
  5. Write <name>.json (keypoints, scores, bboxes) and <name>_pose.png (visualization).

Install:
  pip install torch torchvision opencv-python numpy huggingface_hub

Examples:
  python sapiens_pose.py images/ --model 1b-goliath --out out/
  python sapiens_pose.py face.jpg --model 0.3b-coco --detector none
  python sapiens_pose.py photo.jpg --checkpoint path/to/model_torchscript.pt2
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from sapiens_keypoints import KEYPOINT_SETS

# --------------------------------------------------------------------------------------
# Checkpoints (Hugging Face). Goliath = 308 keypoints (body + hands + dense face),
# COCO = 17 body keypoints, COCO-WholeBody = 133 (body, feet, 68 face, hands).
# --------------------------------------------------------------------------------------
_COCO_REPO = "noahcao/sapiens-pose-coco"
_COCO_DIR = "sapiens_lite_host/torchscript/pose/checkpoints"
CHECKPOINTS = {
    "0.3b-goliath": ("facebook/sapiens-pose-0.3b-torchscript", "sapiens_0.3b_goliath_best_goliath_AP_573_torchscript.pt2"),
    "0.6b-goliath": ("facebook/sapiens-pose-0.6b-torchscript", "sapiens_0.6b_goliath_best_goliath_AP_609_torchscript.pt2"),
    "1b-goliath": ("facebook/sapiens-pose-1b-torchscript", "sapiens_1b_goliath_best_goliath_AP_639_torchscript.pt2"),
    "0.3b-coco": (_COCO_REPO, f"{_COCO_DIR}/sapiens_0.3b/sapiens_0.3b_coco_best_coco_AP_796_torchscript.pt2"),
    "0.6b-coco": (_COCO_REPO, f"{_COCO_DIR}/sapiens_0.6b/sapiens_0.6b_coco_best_coco_AP_812_torchscript.pt2"),
    "1b-coco": (_COCO_REPO, f"{_COCO_DIR}/sapiens_1b/sapiens_1b_coco_best_coco_AP_821_torchscript.pt2"),
    "2b-coco": (_COCO_REPO, f"{_COCO_DIR}/sapiens_2b/sapiens_2b_coco_best_coco_AP_822_torchscript.pt2"),
    "0.3b-wholebody": (_COCO_REPO, f"{_COCO_DIR}/sapiens_0.3b/sapiens_0.3b_coco_wholebody_best_coco_wholebody_AP_620_torchscript.pt2"),
    "0.6b-wholebody": (_COCO_REPO, f"{_COCO_DIR}/sapiens_0.6b/sapiens_0.6b_coco_wholebody_best_coco_wholebody_AP_695_torchscript.pt2"),
    "1b-wholebody": (_COCO_REPO, f"{_COCO_DIR}/sapiens_1b/sapiens_1b_coco_wholebody_best_coco_wholebody_AP_727_torchscript.pt2"),
    "2b-wholebody": (_COCO_REPO, f"{_COCO_DIR}/sapiens_2b/sapiens_2b_coco_wholebody_best_coco_wholebody_AP_745_torchscript.pt2"),
}

INPUT_H, INPUT_W = 1024, 768  # model input (H, W)
MEAN = np.array([123.5, 116.5, 103.5], dtype=np.float32)  # RGB, 0-255 space
STD = np.array([58.5, 57.0, 57.5], dtype=np.float32)
BBOX_PADDING = 1.25
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


# --------------------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------------------
def resolve_checkpoint(model_key: str | None, checkpoint: str | None) -> Path:
    if checkpoint:
        return Path(checkpoint)
    from huggingface_hub import hf_hub_download

    repo_id, filename = CHECKPOINTS[model_key]
    print(f"Fetching {repo_id}/{filename} (cached after first download)...")
    return Path(hf_hub_download(repo_id=repo_id, filename=filename))


def load_pose_model(path: Path, device: torch.device):
    if "_torchscript" in path.name:
        model = torch.jit.load(str(path), map_location=device)
    else:  # bfloat16 .pt2 exports (torch.export); meant for recent CUDA GPUs
        model = torch.export.load(str(path)).module().to(device)
    model.eval()
    return model


# --------------------------------------------------------------------------------------
# Person detection
# --------------------------------------------------------------------------------------
class PersonDetector:
    """torchvision Faster R-CNN (COCO) restricted to the 'person' class."""

    def __init__(self, device: torch.device, score_thr: float):
        from torchvision.models.detection import (
            FasterRCNN_ResNet50_FPN_V2_Weights,
            fasterrcnn_resnet50_fpn_v2,
        )

        self.model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
        self.model.eval().to(device)
        self.device = device
        self.score_thr = score_thr

    @torch.inference_mode()
    def __call__(self, img_rgb: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(img_rgb).permute(2, 0, 1).float().div(255).to(self.device)
        out = self.model([x])[0]
        keep = (out["labels"] == 1) & (out["scores"] > self.score_thr)  # COCO label 1 = person
        return out["boxes"][keep].cpu().numpy().reshape(-1, 4)


# --------------------------------------------------------------------------------------
# Crop (UDP affine) and heatmap decoding
# --------------------------------------------------------------------------------------
def bbox_to_center_scale(bbox: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """xyxy -> (center, scale) padded by 1.25 and expanded to the model's 3:4 aspect."""
    x1, y1, x2, y2 = bbox
    center = np.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype=np.float32)
    w, h = (x2 - x1) * BBOX_PADDING, (y2 - y1) * BBOX_PADDING
    aspect = INPUT_W / INPUT_H
    if w > h * aspect:
        h = w / aspect
    else:
        w = h * aspect
    return center, np.array([w, h], dtype=np.float32)


def crop_matrix(center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """UDP warp: maps the box [c - s/2, c + s/2] onto [0, W-1] x [0, H-1]."""
    sx = (INPUT_W - 1) / scale[0]
    sy = (INPUT_H - 1) / scale[1]
    return np.array(
        [[sx, 0, sx * (0.5 * scale[0] - center[0])],
         [0, sy, sy * (0.5 * scale[1] - center[1])]],
        dtype=np.float32,
    )


def preprocess(img_rgb: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    crop = cv2.warpAffine(img_rgb, crop_matrix(center, scale), (INPUT_W, INPUT_H), flags=cv2.INTER_LINEAR)
    crop = (crop.astype(np.float32) - MEAN) / STD
    return crop.transpose(2, 0, 1)  # CHW


def _gaussian_blur(heatmaps: np.ndarray, kernel: int) -> np.ndarray:
    """Blur each heatmap while preserving its original peak value (as in mmpose)."""
    border = (kernel - 1) // 2
    out = np.empty_like(heatmaps)
    for k, hm in enumerate(heatmaps):
        peak = hm.max()
        padded = cv2.copyMakeBorder(hm, border, border, border, border, cv2.BORDER_CONSTANT, value=0)
        blurred = cv2.GaussianBlur(padded, (kernel, kernel), 0)[border:-border, border:-border]
        out[k] = blurred * (peak / max(blurred.max(), 1e-12))
    return out


def decode_heatmaps(heatmaps: np.ndarray, blur_kernel: int = 11) -> tuple[np.ndarray, np.ndarray]:
    """UDP + DARK decoding. heatmaps: (K, h, w) -> keypoints (K, 2) in heatmap pixels, scores (K,)."""
    K, H, W = heatmaps.shape
    flat = heatmaps.reshape(K, -1)
    idx = flat.argmax(axis=1)
    scores = flat[np.arange(K), idx]
    kpts = np.stack([idx % W, idx // W], axis=1).astype(np.float32)

    # DARK: second-order Taylor refinement on the log of the blurred heatmap.
    hm = np.log(np.clip(_gaussian_blur(heatmaps.copy(), blur_kernel), 1e-3, 50.0))
    hm = np.pad(hm, ((0, 0), (1, 1), (1, 1)), mode="edge")
    x = kpts[:, 0].astype(int) + 1
    y = kpts[:, 1].astype(int) + 1
    k = np.arange(K)
    c = hm[k, y, x]
    dx = 0.5 * (hm[k, y, x + 1] - hm[k, y, x - 1])
    dy = 0.5 * (hm[k, y + 1, x] - hm[k, y - 1, x])
    dxx = hm[k, y, x + 1] - 2 * c + hm[k, y, x - 1]
    dyy = hm[k, y + 1, x] - 2 * c + hm[k, y - 1, x]
    dxy = 0.25 * (hm[k, y + 1, x + 1] - hm[k, y + 1, x - 1] - hm[k, y - 1, x + 1] + hm[k, y - 1, x - 1])
    hess = np.stack([np.stack([dxx, dxy], -1), np.stack([dxy, dyy], -1)], -2)
    hess += np.finfo(np.float32).eps * np.eye(2)
    grad = np.stack([dx, dy], -1)[..., None]
    offset = (np.linalg.inv(hess) @ grad)[..., 0]
    valid = np.isfinite(offset).all(axis=1) & (np.abs(offset) < 1).all(axis=1)
    kpts[valid] -= offset[valid]
    return kpts, scores


def heatmap_to_image_coords(kpts_hm: np.ndarray, hm_shape, center, scale) -> np.ndarray:
    hm_h, hm_w = hm_shape
    # heatmap pixel -> model-input pixel (UDP: corners align)
    x_in = kpts_hm[:, 0] * (INPUT_W - 1) / (hm_w - 1)
    y_in = kpts_hm[:, 1] * (INPUT_H - 1) / (hm_h - 1)
    # model-input pixel -> original image (inverse of crop_matrix)
    x = x_in * scale[0] / (INPUT_W - 1) + center[0] - 0.5 * scale[0]
    y = y_in * scale[1] / (INPUT_H - 1) + center[1] - 0.5 * scale[1]
    return np.stack([x, y], axis=1)


# --------------------------------------------------------------------------------------
# Inference + visualization
# --------------------------------------------------------------------------------------
@torch.inference_mode()
def estimate_pose(model, img_rgb: np.ndarray, bboxes: np.ndarray, device, batch_size: int):
    centers_scales = [bbox_to_center_scale(b) for b in bboxes]
    crops = np.stack([preprocess(img_rgb, c, s) for c, s in centers_scales])
    heatmaps = []
    for i in range(0, len(crops), batch_size):
        batch = torch.from_numpy(crops[i : i + batch_size]).to(device)
        heatmaps.append(model(batch).float().cpu().numpy())
    heatmaps = np.concatenate(heatmaps)  # (N, K, h, w)

    results = []
    for hm, (c, s), bbox in zip(heatmaps, centers_scales, bboxes):
        kpts_hm, scores = decode_heatmaps(hm)
        kpts = heatmap_to_image_coords(kpts_hm, hm.shape[1:], c, s)
        results.append({"bbox": bbox.tolist(), "keypoints": kpts.tolist(), "keypoint_scores": scores.tolist()})
    return results


def draw(img_bgr: np.ndarray, results, skeleton, kpt_thr: float, draw_bbox: bool) -> np.ndarray:
    vis = img_bgr.copy()
    radius = max(2, round(max(vis.shape[:2]) / 400))
    thickness = max(1, radius // 2)
    rng = np.random.default_rng(0)
    link_colors = [tuple(int(v) for v in rng.integers(64, 256, 3)) for _ in skeleton]
    for r in results:
        kpts = np.asarray(r["keypoints"])
        scores = np.asarray(r["keypoint_scores"])
        if draw_bbox:
            x1, y1, x2, y2 = map(int, r["bbox"])
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), thickness)
        for (a, b), color in zip(skeleton, link_colors):
            if scores[a] > kpt_thr and scores[b] > kpt_thr:
                pa, pb = kpts[a].round().astype(int), kpts[b].round().astype(int)
                cv2.line(vis, tuple(pa), tuple(pb), color, thickness, cv2.LINE_AA)
        for (x, y), s in zip(kpts, scores):
            if s > kpt_thr:
                cv2.circle(vis, (round(x), round(y)), radius, (0, 0, 255), -1, cv2.LINE_AA)
    return vis


def collect_images(inputs: list[str]) -> list[Path]:
    paths = []
    for p in map(Path, inputs):
        if p.is_dir():
            paths += sorted(q for q in p.iterdir() if q.suffix.lower() in IMAGE_EXTS)
        else:
            paths.append(p)
    return paths


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="image files and/or directories")
    ap.add_argument("--model", default="1b-goliath", choices=sorted(CHECKPOINTS), help="checkpoint to download from Hugging Face")
    ap.add_argument("--checkpoint", help="local .pt2 path (overrides --model)")
    ap.add_argument("--out", default="pose_out", help="output directory")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--detector", default="torchvision", choices=["torchvision", "none"],
                    help="'none' treats the whole image as one person")
    ap.add_argument("--bbox", type=float, nargs=4, metavar=("X1", "Y1", "X2", "Y2"),
                    help="use this person box for every image (skips the detector)")
    ap.add_argument("--det-thr", type=float, default=0.5, help="person detection score threshold")
    ap.add_argument("--max-people", type=int, default=0, help="keep only the N largest people (0 = all)")
    ap.add_argument("--kpt-thr", type=float, default=0.3, help="min keypoint score to draw")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--no-vis", action="store_true", help="skip writing visualization images")
    args = ap.parse_args()

    device = torch.device(args.device)
    ckpt = resolve_checkpoint(args.model, args.checkpoint)
    print(f"Loading {ckpt.name} on {device}...")
    model = load_pose_model(ckpt, device)

    detector = None
    if args.bbox is None and args.detector == "torchvision":
        detector = PersonDetector(device, args.det_thr)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    images = collect_images(args.inputs)
    if not images:
        raise SystemExit("No images found.")

    for path in images:
        img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            print(f"[skip] could not read {path}")
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]

        if args.bbox is not None:
            bboxes = np.array([args.bbox], dtype=np.float32)
        elif detector is not None:
            bboxes = detector(img_rgb)
        else:
            bboxes = np.empty((0, 4), dtype=np.float32)
        if len(bboxes) == 0:  # nothing detected (or detector off): use the whole image
            bboxes = np.array([[0, 0, w - 1, h - 1]], dtype=np.float32)
        if args.max_people > 0:
            areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
            bboxes = bboxes[np.argsort(-areas)[: args.max_people]]

        results = estimate_pose(model, img_rgb, bboxes, device, args.batch_size)
        num_kpts = len(results[0]["keypoints"])
        names, skeleton = KEYPOINT_SETS.get(num_kpts, (None, []))

        record = {
            "image": str(path),
            "width": w,
            "height": h,
            "checkpoint": ckpt.name,
            "keypoint_names": names,
            "instances": results,
        }
        (out_dir / f"{path.stem}.json").write_text(json.dumps(record))
        if not args.no_vis:
            vis = draw(img_bgr, results, skeleton, args.kpt_thr, draw_bbox=detector is not None)
            cv2.imwrite(str(out_dir / f"{path.stem}_pose.png"), vis)
        print(f"{path.name}: {len(results)} person(s), {num_kpts} keypoints")


if __name__ == "__main__":
    main()
