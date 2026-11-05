# Willow — Member 5

Willow is a compassionate, AI-assisted interface for exploring text that may indicate suicidal ideation. i am responsible for developing the landing page and interactive user interface and connects backend.

## Languages and tools

- **JavaScript (React JSX)** — interface and application logic
- **CSS** — responsive styling, light/dark themes, and landing-page visuals
- **HTML** — Vite app entry point
- **Python (Flask)** — backend API integration example
- **Vite** — local development and production builds

## Pages and features

- **Landing page:** illustrated hero and an **Enter Willow** button.
- **Prediction interface:** enter text (up to 1,000 characters), submit it for prediction, and view the latest result, confidence, and recent session history.
- **CSV tools:** upload or drag in a CSV, select its text column, process rows, and download the results.
- **Interface controls:** responsive navigation, light/dark mode, return-to-landing control, supportive daily messages, and responsible-use guidance.

## Run locally

From the project folder, install dependencies and start the frontend:

```powershell
npm install
npm run dev
```

Open the URL Vite prints (usually `http://localhost:8080`). Run checks and create a production build with:

```powershell
npm test
npm run build
```

## Backend connection

Start a Flask backend separately on `127.0.0.1:5000`. The frontend sends **POST `/api/predict`** with `{"text":"..."}` and expects a JSON response containing `prediction`. `confidence`, `model_prediction`, and `prediction_source` are optional.

Vite proxies `/api` requests to the Flask server. The project includes a backend route contract example, but does not include the Flask app or trained model.

## Responsible use

Willow is an educational support tool—not a diagnosis or substitute for professional care. Predictions can be wrong and should not be the sole basis for safety decisions. If someone may be in immediate danger, contact local emergency services or a trusted person.
