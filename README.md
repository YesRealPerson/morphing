# AI Usage
Claude made this readme and the pose runners because all it does is run the model and run a UI so I can click points to generate vectors around the head

morphing code in morph.ipynb is human besides debugging and speed optimizations

## Install

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
```

The `--extra-index-url` gets the CUDA build of PyTorch. Remove it for CPU pytorch

# Running morphing
Assuming you've made the appropriate json files, load them into python with something like the following code.
```
norton = open("./examples/lines2/norton_lines.json", 'r')
pitt = open("./examples/lines2/pitt_lines.json", 'r')
norton = json.load(norton)
pitt = json.load(pitt)
nortonimg = cv2.cvtColor(cv2.imread(norton["image"]), cv2.COLOR_BGR2RGB)
pittimg = cv2.cvtColor(cv2.imread(pitt["image"]), cv2.COLOR_BGR2RGB)
```
Then call morph with the desired `t` parameter.
```
frame = morph(nortonimg, norton["lines"], pittimg, pitt["lines"], t=0.5)
```
Do this over a loop to generate a video or something
<img src="./example1.gif">
<img src="./example2.gif">
# Running pose generation (Claude)

```
python sapiens_pose.py <file.jpg> --model 0.3b-coco
```

## Arguments

| Argument | Default | Description |
|---|---|---|
| `inputs` | (required) | One or more image files and/or folders. Folders are scanned for `.jpg .jpeg .png .bmp .webp .tif .tiff`; subfolders are not searched. |
| `--model` | `1b-goliath` | Checkpoint to download from Hugging Face; see [Models](#models). It is cached after the first download. |
| `--checkpoint` | none | Path to a local `.pt2` checkpoint. Overrides `--model`. Files with `_torchscript` in the name are loaded as TorchScript; other files are loaded with `torch.export`, which is meant for the bfloat16 exports. |
| `--out` | `pose_out` | Output folder. Created if it doesn't exist. |
| `--device` | `cuda` if available, else `cpu` | Device for the pose model and the detector, e.g. `cuda`, `cuda:1`, `cpu`. |
| `--bbox X1 Y1 X2 Y2` | none | Bound detections to a bounding box. |
| `--det-thr` | `0.5` | Minimum detector confidence to count a box as a person. |
| `--max-people` | `0` | Keep only the N largest people per image. `0` keeps everyone. |
| `--kpt-thr` | `0.3` | Minimum keypoint score to draw in the visualization. The JSON always contains every keypoint, whatever its score. |
| `--batch-size` | `4` | Person crops per forward pass. Lower it if you run out of GPU memory. |
| `--no-vis` | off | Skip writing the `_pose.png` visualization. |

If the detector finds nobody, the whole image is used as one person.

## Models

| `--model` | Keypoints | Notes |
|---|---|---|
| `0.3b-goliath`, `0.6b-goliath`, `1b-goliath` | 308 | Body, hands, feet and dense face (eyebrows, eyelids, irises, nose, lips, ears). |
| `0.3b-coco`, `0.6b-coco`, `1b-coco`, `2b-coco` | 17 | Standard COCO body keypoints. |
| `0.3b-wholebody`, `0.6b-wholebody`, `1b-wholebody`, `2b-wholebody` | 133 | Body, feet, 68 face points, and hands. |

## Output

For each input image `<name>`, the output folder gets:

- `<name>.json`: the results.
- `<name>_pose.png`: the image with keypoints and skeleton drawn on it, plus detector boxes when the detector was used.

JSON layout:

```json
{
  "image": "images/face.jpg",
  "width": 640,
  "height": 480,
  "checkpoint": "sapiens_1b_goliath_best_goliath_AP_639_torchscript.pt2",
  "keypoint_names": ["nose", "left_eye", "..."],
  "instances": [
    {
      "bbox": [x1, y1, x2, y2],
      "keypoints": [[x, y], "..."],
      "keypoint_scores": [0.92, "..."]
    }
  ]
}
```

`keypoints[i]` is in original-image pixel coordinates and matches `keypoint_names[i]`. Scores are raw heatmap peaks, roughly 0 to 1. Keypoint names and skeleton links for each model family are in [sapiens_keypoints.py](sapiens_keypoints.py).

# Face line annotation (Claude)

`face_lines.py` gets the 68 face keypoints from a COCO-WholeBody model, then asks you to click points along both ears and the hair outline. It saves matching feature lines for each image.

```
python face_lines.py examples/pitt.jpg examples/norton.jpg --out examples/lines
```

- Left-click adds a point; right-click or Backspace removes the last one. Each window moves on once you've placed the requested number of points.
- Click in the same order on every image. From the second image on, the first image's clicks are shown beside it, numbered, as a guide.
- Output: `<name>_lines.json` (`lines` is a list of `[[x1, y1], [x2, y2]]`, in the same order for every image, so line *i* matches across images) and `<name>_lines.png`.

| Argument | Default | Description |
|---|---|---|
| `images` | (required) | Face images, annotated in the order given. |
| `--out` | `lines_out` | Output folder. |
| `--pose-json` | none | One `sapiens_pose.py` JSON per image (from a WholeBody model), used instead of running the model. |
| `--model` | `2b-wholebody` | Pose model; must be a `*-wholebody` model. |
| `--checkpoint` | none | Local WholeBody `.pt2` checkpoint. Overrides `--model`. |
| `--device` | `cuda` if available | Device to run on. |
| `--detector` | `torchvision` | `none` uses the whole image as the person box. |
| `--det-thr` | `0.5` | Person detector confidence threshold. |
| `--ear-points` | `4` | Points to click per ear, top to lobe. `0` skips the ears. |
| `--hair-points` | `9` | Points along the hair outline, from the left-side ear over the head to the right-side ear. |
| `--hairline-points` | `0` | Points along the forehead hairline. `0` skips it. |
| `--reference` | first image of the run | A previous `*_lines.json` to show beside each image while you click. |
| `--skip` | none | Sections to leave out, for features that are hidden in any of the images (e.g. `--skip ear_image_right`). Applies to every image in the run so lines still match. Names: `ear_image_left`, `ear_image_right`, `hair`, `hairline`, or any `FACE_POLYLINES` key (`jaw`, `brow_a`, `eye_b`, ...). |

Which face lines are used is set by `FACE_POLYLINES` at the top of `face_lines.py`.
