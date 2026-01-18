Authors: Benjamin, Juno, Kelly, Evelyn

# TranSlaDoorss
This repository contains the work-in-progress code for my Fall 2025 CS 222 project. Its is a web-based ASL letter prediction using a CLIP-based model, Django REST API, and a Bootstrap/vanilla JS frontend.

## Minimum Requirements
* Python 3.10–3.12
* Conda (for `env/environment_unified.yaml`) or virtualenv
* Node.js (for frontend Jest tests)

## Other Requirements
* Optional GPU support for PyTorch (`cuda`/`mps`) if available
* If running ML fine-tuning/eval: internet access to pull Hugging Face models/datasets

## Usage

### Setup
```bash
conda env create -f env/environment_unified.yaml
conda activate transladoorss
```
Set env vars (via `python-decouple`):
* `SECRET_KEY` (required)
* `DEBUG` (`true` for local dev)
* `ALLOWED_HOSTS` (comma-separated, e.g., `localhost,127.0.0.1`)
* Optional ML: `ASL_MODEL_ID` (e.g., `ml/saved_weights/epoch_7`), `ASL_MODEL_DEVICE` (`cuda`/`mps`/`cpu`)

### Run the server
```bash
cd backend
python manage.py migrate
python manage.py runserver   # http://localhost:8000/ serves frontend/index.html
```

### API: POST /img_in/translate/
- Form-data:
  - `image` (file, required)
  - `top_k` (int, 1–26, default 1)
- Success:
  - `{"translation": "A", "confidence": 0.97, "prediction": {"letter": "A", "confidence": 0.97}}` (top_k=1)
  - `{"translation": "A", "top_predictions": [...], "prediction": {"top_predictions": [...]}}` (top_k>1; translation mirrors best letter)
- Errors: 400 invalid/missing image; 500 wraps inference exceptions.

### Frontend
* UI lives in `frontend/index.html`; logic in `frontend/js/scripts.js`.
* Served by Django in DEBUG at `http://localhost:8000/`.

### ML utilities
* Fine-tune: `python -m ml.development.clip_fine_tuned --config <optional.json> --output-dir ml/saved_weights`
* Select best checkpoint: `python -m ml.development.export_best_checkpoint --output-dir ml/saved_weights`
* Inference helper: `python -m ml.inference.service --model-id <ckpt_or_hf_id> --image path/to/img --top-k 3`
* Evaluator: `python -m ml.testing.evaluator --model-id <ckpt_or_hf_id> --num-samples 100 --split test`

### Testing
* Backend (pytest): `pytest -c backend/pytest.ini` (tests in `backend/tests/`)
* Frontend (Jest): `cd frontend/js && npm test` (tests in `frontend/tests/`)
* Keep API contract in sync across both suites if you change response shapes.

## Project Structure
```
transladoorss-model-dev-env/
├── backend/               # Django project (settings, img_in app, backend tests)
│   ├── base/              # Settings, URLs, WSGI/ASGI
│   ├── img_in/            # Translation API
│   ├── manage.py
│   └── tests/             # Backend tests + fixtures
├── frontend/              # Frontend entry + static assets + JS tests
│   ├── index.html
│   ├── assets/
│   ├── css/
│   └── js/                # JS + package.json; Jest tests in ../tests/
├── ml/                    # ML training/inference/eval code
│   ├── inference/
│   ├── development/
│   └── testing/
└── env/
    └── environment_unified.yaml
```

## Current Gaps
* Feedback storage/loop is a work-in-progress (`backend/img_in/models.py` placeholder; no persistence). More changes to be released in future versions as the RL pipeline is being added.
