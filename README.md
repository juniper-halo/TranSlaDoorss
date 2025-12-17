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
```
## Testing

The project includes automated tests for both the Django backend and the JavaScript frontend.

### Backend Tests (Python/Django)
The backend uses **pytest** to verify API endpoints, image processing, and database models.

1.  Navigate to the server directory:
    ```bash
    cd django_server
    ```

2.  Run the full test suite:
    ```bash
    pytest
    ```

**(e.g.):**
* Run tests with detailed output: `pytest -v`
* Run a specific test file: `pytest img_in/tests/test_basic.py`
* Stop on the first failure: `pytest -x`

### Frontend Tests (JavaScript)
The frontend uses **Jest** to test DOM interactions and API logic (mocking the webcam and fetch calls).

1.  Navigate to the frontend directory:
    ```bash
    cd js
    ```

2.  Run the tests:
    ```bash
    npm test
    ```

## Usage

### Running the Server
1.  **Apply Migrations:**
    Initialize the database for the Feedback model.
    ```bash
    cd django_server
    python manage.py migrate
    ```

2.  **Start the Server:**
    Run the Django development server.
    ```bash
    python manage.py runserver
    ```

3.  **Launch the App:**
    Open `index.html` in your browser.
    * *Recommended:* Serve it via a simple HTTP server:
        ```bash
        # From the project root
        python -m http.server 8000
        ```
    * Visit `http://localhost:8000`

### Using the App
1.  Click **"Start Camera"** to enable webcam access.
2.  Make an ASL sign hand gesture.
3.  Click **"Translate"** to send the frame to the backend.
4.  View the prediction and confidence score.
5.  Use the **Feedback Section** to confirm if the prediction was correct or provide the correct label.
