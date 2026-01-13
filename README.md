# Speedread Extractor

Desktop app to extract page-like frames from a reading video, transcribe them via a local
LLM (LM Studio), remove duplicates, and summarize.

See `APP_PROPOSAL.md` for the original proposal and scope.

## Quick Start

1. Install Python deps: `pip install -r requirements.txt`
2. Start LM Studio with `qwen/qwen3-vl-8b` at `http://127.0.0.1:1234`
3. Run the app: `python main.py`

## Core Workflow (Steps)

1. Extract frames
2. Transcribe frames (LLM)
3. Mark duplication
4. Create summary

You can run each step independently (Actions panel). "Clear" deletes outputs for that
step only (with confirmation).

## Outputs

Each run creates a folder under the output root:

- `pages/` page images and transcription text files
- `pages_raw.json` all frames + OCR text + similarity
- `pages_selected.json` selected frames
- `final_summary.txt` summary
- `session.json` video path + base interval + crop settings

## Notes

- Cropping is optional and applied before frame extraction/transcription.
- Summary and transcription support right-click copy.

## Known Issues

- Qwen3-VL may produce repeated output on transcription. https://github.com/QwenLM/Qwen3-VL/issues/1611
