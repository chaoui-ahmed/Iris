# Training the iris models

Two small models, both run in the browser (ONNX Runtime Web), so inference costs nothing.

| Model | What it does | Status |
|---|---|---|
| `models/iris-seg.onnx` | Labels every pixel: background, white of the eye, iris, pupil. The site fits the iris and pupil circles to that. | Trained on synthetic eyes, works now. Retrain with real photos to make it solid. |
| `models/iris-enhance.onnx` | Turns a soft phone-quality iris into a sharp macro one. | Script ready, needs your macro photos. Not wired into the site yet. |

If `iris-seg.onnx` is missing or fails to load, the site falls back to the hand-written detection.

## Setup

```bash
cd training
pip install -r requirements.txt
```

Or use `train_colab.ipynb` on Google Colab (free GPU): it clones the repo, trains, and lets you download the model.

## 1. Iris segmenter

```bash
python synth_eyes.py --out preview.png --n 24     # look at what the synthetic data looks like
python train.py                                    # synthetic only
python predict.py ~/Pictures/eyes/*.jpg --out preds/   # check it on your own photos
```

`train.py` writes `../models/iris-seg.onnx`. Commit it and push: the site picks it up.

### Adding real data (what will make it reliable)

The synthetic eyes teach the shapes; real photos teach what real eyes, lids, lashes and flash look like.
Put real data in this layout:

```
data/real/images/eye001.jpg
data/real/masks/eye001.png      # same name, PNG, one value per pixel:
                                # 0 background, 1 white of the eye, 2 iris, 3 pupil, 255 unknown (ignored)
```

Then:

```bash
python train.py --real data/real --real-val data/real_val --epochs 30 --resume ../models/iris-seg.pt
```

Synthetic data stays mixed in (keeps it robust to skin tones, colours and lighting). Use `--synthetic 0` to train on real data only.

Where to get it:

- **Your own photos.** Take 100–300 shots with the site's camera (different people, eye colours, lighting), and draw the masks with a free tool like CVAT or Label Studio (polygon for the iris, the pupil and the visible white). Even 100 masks on top of the synthetic data should make a big difference, because they're exactly what the site sees. This is also the cleanest option if you ever want to make money from the site.
- **UBIRIS.v2**: 11,102 visible-light colour eye photos from 261 people (University of Beira Interior). Free for research after signing their licence form. Iris masks for part of it come from the NICE challenges. Check the licence: research only, no commercial use.
- **MOBIUS / MICHE**: research datasets of eyes photographed with phones, closest to this app's conditions. Also request-only, research licences.

Most datasets ship binary masks (iris = white). Convert them to the format above: iris pixels → 2, pupil → 3 if given, everything else → 0, or 255 where you don't know (for example the white of the eye, if the dataset doesn't label it).

## 2. Detail enhancer (needs data)

```bash
python train_enhancer.py --data data/macro --epochs 40
```

`data/macro/` = a few hundred to a few thousand sharp iris macro photos you have the rights to (own shots with a clip-on macro lens, licensed or free-licence stock). No phone versions needed: the script degrades each photo to look like a phone shot and learns to undo it. Don't use photos scraped from iris-photography studios: they're copyrighted, and they're photos of real people's eyes.

## 3. Texture library (what makes the render look real)

```bash
python build_library.py --n 80 --out ../lib/iris
```

Downloads irises from the Synthetic Iris Image Dataset (diffusion-generated from DSLR photos, nobody's real eye), unwraps each into a 1024x160 strip and writes `lib/iris/NNN.jpg` + `index.json` (colours per ring, used to pick the closest texture to the user's eye). More textures = more variety; each is about 20 KB.

## Files

- `synth_eyes.py`: procedural eye images + masks (incl. images with no eye, so the model learns to say "nothing here")
- `model.py`: the network (tiny U-Net, ~60k parameters, ~250 KB)
- `train.py`: training + ONNX export
- `predict.py`: same detection logic as the site, on a folder of photos, with overlays
- `train_enhancer.py`: detail enhancer, degradation pipeline, export
- `build_library.py`: builds the iris texture library used by the renderer
- `train_colab.ipynb`: all of the above on Colab

## Privacy

Iris images are biometric data (GDPR). The site processes everything on the phone and uploads nothing. Keep it that way, and don't collect users' scans for training without explicit consent.
