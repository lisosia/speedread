# Speedread Extractor (MVP)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

## Tests

```bash
pytest
```

## Notes

- Exported assets are written to the selected output folder under `pages/`.
- PDF export requires Pillow; disable the checkbox if you do not want it.
- Extraction requires a local LLM server (LM Studio) running at `http://127.0.0.1:1234` with the model `qwen/qwen3-vl-8b`.
