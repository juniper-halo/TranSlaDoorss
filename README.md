# TranSlaDoorss

**Team 075**: Benjamin, Juno, Kelly, Evelyn

## Overview
TranSlaDoorss is a web-based accessibility tool designed to bridge communication gaps by providing real-time, bidirectional translation between American Sign Language (ASL) and text. The system leverages a fine-tuned CLIP-based machine learning model to interpret hand signs captured via a webcam.

## Features
* **Real-time ASL Translation**: Captures video frames from the user's webcam and predicts the corresponding ASL letter.
* **Confidence Scoring**: Displays the model's confidence level for each prediction to assist users in gauging accuracy.
* **Feedback Loop**: Users can validate predictions or provide corrections, which are stored to retrain and improve the model.
* **Responsive UI**: A user-friendly interface built with Bootstrap and vanilla JavaScript.

## Tech Stack
* **Frontend**: HTML5, CSS3, JavaScript (Vanilla), Bootstrap, Jest (Testing).
* **Backend**: Django 5.2, Django REST Framework, SQLite.
* **Machine Learning**: PyTorch, Transformers (Hugging Face), CLIP Model, PIL (Pillow).
* **Dev Tools**: Prettier, ESLint, Black, Pytest.

## Project Structure
```text
fa25-fa25-team075-transladoorss/
├── assets/                 # Static assets (images, icons)
├── css/                    # Global styles
├── django_server/          # Django backend project
│   ├── base/               # Project settings and configuration
│   ├── feedback_images/    # Storage for user-submitted feedback images
│   ├── img_in/             # Main app for image translation & feedback APIs
│   └── manage.py           # Django management script
├── js/                     # Frontend logic and tests
│   ├── scripts.js          # Main frontend functionality (webcam, API calls)
│   └── scripts.test.js     # Frontend unit tests
├── ml_dev/                 # Machine Learning development & inference
│   ├── inference/          # Service layer for model predictions
│   └── development/        # Training and preprocessing scripts
└── index.html              # Main entry point for the web interface
