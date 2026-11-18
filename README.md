# Iris

Scan your eye with your phone and get a detailed macro picture of your iris.

Everything runs on the device: no server, no AI model, no API. Hosting is just static files.

## How it works

0. **Capture.** "Scan your eye" opens the back camera with the flash on (where the browser allows it). You can't see the screen while the phone faces your eye, so it captures on its own: once it sees the same iris in a few frames in a row, it takes a short burst, keeps the sharpest frame, then chimes and buzzes. Tapping the screen captures right away, and "Choose a photo" uses an existing picture.
   "Front camera" flips to the selfie camera: you can see yourself, line your iris up with the circle, and the screen turns into a ring light around it.
   The app only captures on its own when it's sure it sees an iris: a dark-to-bright edge all the way round, brighter white of the eye just outside, and a pupil darker than the iris. If it isn't sure (manual capture, or a photo), it asks you to place the circle instead of guessing.

1. **Find the eye.** The app searches the photo for the strongest circular dark-to-bright edge (iris to white of the eye), then for the pupil inside it. Edges from single bright spots, like a flash reflection, are ignored.
2. **Unwrap the iris.** The ring is turned into a flat rectangle (angle × radius). Eyelids, lashes and glare are detected per ring and replaced.
3. **Keep what's real.** The colours, their layout and the fine texture the camera actually caught come from the photo.
4. **Add the detail a phone can't capture.** Fibres, the collarette ring, crypts, furrows and the dark outer ring are generated with noise. The noise is seeded from the photo, so the same eye always gives the same result.
5. **Render.** The image is rendered in parallel Web Workers (one per CPU core), so the UI stays smooth.

If the detection misses, "Adjust" lets you place the circle by hand.

## Fusion

"Fuse with another eye" scans a second eye (a friend's or your other one) and merges both into one iris: each eye takes one half, meeting along a soft, wavy seam. "Swap sides" flips them.

## Sharing

"Save" exports a 9:16 story card (iris, a short caption, the IRIS wordmark). On phones it opens the share sheet, so it goes straight to Photos, Instagram or TikTok. On desktop it downloads.

## The iris model

Detection uses a small neural network (`models/iris-seg.onnx`, ~250 KB) that labels every pixel as background, white of the eye, iris or pupil; the circles are fitted to that. It runs in the browser with ONNX Runtime Web (self-hosted in `vendor/ort/`), so it costs nothing to run. If it can't load, the hand-written detection takes over.

The first model is trained on procedurally generated eyes. `training/` has everything to retrain it with real photos (much more reliable), plus a detail-enhancer model to train on macro photos. See `training/README.md`.

## Run locally

Any static server works (the worker doesn't load from `file://`):

```bash
python3 -m http.server 8000
# open http://localhost:8000
```

## Deploy

Vercel: import the repo, framework preset "Other", no build command. That's it.

## Files

- `index.html`: UI, eye detection, worker orchestration
- `render-worker.js`: iris unwrapping and rendering
- `models/`: trained models loaded by the site
- `vendor/ort/`: ONNX Runtime Web
- `training/`: data generator, training, export, Colab notebook
- `manifest.webmanifest`, `icon.svg`, `icon-512.png`: installable as a home-screen app
