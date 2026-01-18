Authors: Benjamin, Juno, Kelly, Evelyn

# TranSlaDoorss
Web-based ASL letter prediction using a CLIP-based model, Django REST API, and a Bootstrap/vanilla JS frontend. Includes a feedback loop to collect labeled frames for future retraining.

## Minimum Requirements
- Python 3.10–3.12
- Conda (for `env/environment_unified.yaml`) or virtualenv
- Node.js (frontend tests)
- Optional GPU: CUDA/MPS for faster PyTorch inference/fine-tuning

## Other Requirements
- Internet access for Hugging Face models/datasets when fine-tuning/evaluating
- Pillow/libjpeg/libpng for image handling

## Usage

### Setup
```bash
conda env create -f env/environment_unified.yaml
conda activate transladoorss
```
Set env vars (via `python-decouple`):
- `SECRET_KEY` (required)
- `DEBUG` (`true` for local dev)
- `ALLOWED_HOSTS` (comma-separated, e.g., `localhost,127.0.0.1`)
- Optional ML: `ASL_MODEL_ID` (e.g., `ml/saved_weights/epoch_7`), `ASL_MODEL_DEVICE` (`cuda`/`mps`/`cpu`)

### Run the server
```bash
cd backend
python manage.py migrate
python manage.py runserver   # http://localhost:8000/ serves frontend/index.html
```

### API
- **POST** `/img_in/translate/`
  - Form-data: `image` (file, required), `top_k` (int, 1–26, default 1)
  - Success: `{"translation": "A", "confidence": 0.97, "prediction": {...}}` (top_k=1) or `{"translation": "A", "top_predictions": [...], "prediction": {...}}` (top_k>1)
  - Errors: 400 invalid/missing image; 500 inference failure
- **POST** `/img_in/feedback/`
  - Form-data: `image` (file, required), `predicted_label` (A–Z), `correct_label` (A–Z)
  - Stores uploaded frame + labels in `TrainingFeedback` (media under `backend/feedback_images` in dev)

### Frontend
- UI in `frontend/index.html`, logic in `frontend/js/scripts.js` (webcam capture, translate, feedback).
- Served by Django in DEBUG at `http://localhost:8000/`.

### ML utilities
- Fine-tune: `python -m ml.development.clip_fine_tuned --config <optional.json> --output-dir ml/saved_weights`
- Select best checkpoint: `python -m ml.development.export_best_checkpoint --output-dir ml/saved_weights`
- Inference helper: `python -m ml.inference.service --model-id <ckpt_or_hf_id> --image path/to/img --top-k 3`
- Evaluator: `python -m ml.testing.evaluator --model-id <ckpt_or_hf_id> --num-samples 100 --split test`

### Testing
- Backend: `pytest -c backend/pytest.ini` (tests in `backend/tests/`)
- Frontend: `cd frontend/js && npm test` (tests in `frontend/tests/`)
- Keep API contracts in sync across both suites if you change responses.

## Project Structure
```
transladoorss-model-dev-env/
├── backend/               # Django project (settings, img_in app, backend tests)
│   ├── base/              # Settings, URLs, WSGI/ASGI
│   ├── img_in/            # Translation + feedback API
│   ├── manage.py
│   └── tests/             # Backend tests + fixtures
├── frontend/              # Frontend entry + static assets + JS tests
│   ├── index.html
│   ├── assets/
│   ├── css/
│   ├── js/                # JS + package.json; Jest tests in ../tests/
├── ml/                    # ML training/inference/eval code
│   ├── inference/
│   ├── development/
│   └── testing/
└── env/
    └── environment_unified.yaml
```

## Current Gaps
- Feedback is stored, but no pipeline yet to retrain models on it. Upcoming feature with the RL pipline in future releases.
