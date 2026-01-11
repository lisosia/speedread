# Page Extraction

## Current Implementation (LLM + Similarity)

This pipeline prioritizes page text consistency over motion peaks. It samples frames at a fixed base FPS, runs local LLM transcription, and segments the timeline by text-change peaks. Each segment yields one representative frame.

### Change of Plan: Tesseract -> Local LLM

- OCR now uses a local LLM for transcription instead of Tesseract.
- Tesseract (including `tessdata_best` with `jpn_vert`) was tested but produced poor accuracy on real videos, so it was dropped.

### Local LLM Defaults

- Base URL: `http://127.0.0.1:1234`
- Model: `qwen/qwen3-vl-8b`

### Steps

1. User selects rotation (0/90/180/270) before extraction.
2. Sample the video at base FPS (default 1 fps) for segmentation.
3. Run per-frame transcription using the local LLM (LM Studio) on high-res frames.
4. Compute text similarity between frames (sim(i,j)) from transcriptions.
5. Segment the timeline using the two-reference method (jump search + |g(t)| minimum) with min/max intervals.
6. For each segment, pick the middle frame as the representative page.
7. Decode the high-res frame, apply rotation and optional perspective warp, and export.

Each exported page JSON includes the transcription text captured for the selected frame.

### Parameters

- Rotation: 0/90/180/270 degrees (user selection in GUI)
- Base FPS: default 1 fps
- Max interval: 5.0 sec
- LLM base URL and model (default LM Studio endpoint and `qwen/qwen3-vl-8b`)

### Requirements

- Local LLM server running (LM Studio).

## Legacy Implementation (Motion-Peak Based)

The previous method used motion energy to detect page turns:

- Downsample the video to low FPS and low resolution.
- Compute frame-to-frame motion blobs (absdiff + morph + OR accumulation).
- Pick motion peaks as page-turn events.
- Within a short window after each peak, select the sharpest frame (Laplacian variance) with low residual motion.
- Optionally detect page quadrilaterals for perspective warp; otherwise fall back to the original frame.

This approach was sensitive to how strongly motion peaks appeared and often missed pages in steady flipping videos.

## Future Work

- Use motionless windows to refine the best frame within a segment (optional quality booster).
- Improve text-change scoring with language-specific tokenization.
- Add transcription confidence tracking to reject low-signal segments.
