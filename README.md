# Iris

Scan your eye with your phone and get a detailed macro picture of your iris.

Everything runs on the device: no server, no AI model, no API. Hosting is just static files.

## How it works

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
- `manifest.webmanifest`, `icon.svg`, `icon-512.png`: installable as a home-screen app
