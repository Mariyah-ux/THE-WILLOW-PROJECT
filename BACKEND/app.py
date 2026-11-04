
import uuid
import sqlite3
import sys
from pathlib import Path
from io import BytesIO

import pandas as pd

from flask import Flask, jsonify, request, send_file

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from BACKEND.model_handler import predict_text

from BACKEND.database.db_handler import (
    initialize_database,
    save_prediction,
    save_predictions,
    get_prediction_history,
    get_prediction_count
)


# --------------------------------------------------
# FLASK CONFIGURATION
# --------------------------------------------------

app = Flask(__name__)
DISCLAIMER = (
    "Experimental text classification only. This output is not a clinical "
    "assessment, diagnosis, or determination of personal safety. "
    "model_score is the classifier probability for the dataset suicide class, "
    "not an individual suicide-risk estimate. context_review_required means the text needs contextual review; its model_score is withheld and it does not establish personal safety."
)

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

# Initialize SQLite database
initialize_database()


# --------------------------------------------------
# HOME ROUTE
# --------------------------------------------------

@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "project": "Elaria",
        "message": "Backend is running successfully!",
        "status": "active"
    })


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.route("/health", methods=["GET"])
def health():

    return jsonify({
        "status": "healthy"
    })


# --------------------------------------------------
# SINGLE TEXT PREDICTION
# --------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({
            "error": "Request must contain valid JSON."
        }), 400

    text = data.get("text")

    if not isinstance(text, str) or not text.strip():
        return jsonify({
            "error": "Please provide non-empty text."
        }), 400

    try:

        result = predict_text(text)

        # Save prediction to SQLite
        record_id = save_prediction(
            prediction=result["prediction"],
            model_score=result["model_score"],
            source="single_text"
        )

        return jsonify({
            "status": "success",
            "record_id": record_id,
            "prediction": result["prediction"],
            "model_score": result["model_score"],
            "disclaimer": DISCLAIMER
        }), 200

    except ValueError as error:

        return jsonify({
            "error": "Input contains no usable text or invalid values."
        }), 400

    except Exception as error:

        app.logger.error("Single prediction failed (%s)", type(error).__name__)

        return jsonify({
            "error": "Prediction failed. Check backend logs."
        }), 500


# --------------------------------------------------
# CSV BATCH PREDICTION
# --------------------------------------------------

@app.route("/predict-csv", methods=["POST"])
def predict_csv():

    if "file" not in request.files:

        return jsonify({
            "error": "No CSV file was uploaded."
        }), 400

    file = request.files["file"]

    if not file.filename:

        return jsonify({
            "error": "Please select a CSV file."
        }), 400

    if Path(file.filename).suffix.lower() != ".csv":

        return jsonify({
            "error": "Only CSV files are accepted."
        }), 400

    try:

        data = pd.read_csv(file)

        if "text" not in data.columns:

            return jsonify({
                "error": "CSV must contain a 'text' column."
            }), 400

        if data.empty:

            return jsonify({
                "error": "The uploaded CSV is empty."
            }), 400

        if data["text"].isna().any():

            return jsonify({
                "error": "The text column contains missing values."
            }), 400

        if not data["text"].map(
            lambda value: isinstance(value, str)
            and bool(value.strip())
        ).all():

            return jsonify({
                "error": "Every row must contain non-empty text."
            }), 400

        # Generate unique batch ID
        batch_id = str(uuid.uuid4())

        predictions = []
        model_scores = []

        records = []

        # Validate/predict all rows before persisting any batch metadata
        for text in data["text"]:

            result = predict_text(text)

            predictions.append(result["prediction"])
            model_scores.append(result["model_score"])

            records.append({
                "prediction": result["prediction"],
                "model_score": result["model_score"],
                "source": "csv",
                "batch_id": batch_id,
            })

        # Add prediction columns
        data["prediction"] = predictions
        data["model_score"] = model_scores
        data["disclaimer"] = DISCLAIMER

        # Create downloadable CSV in memory
        output = BytesIO()

        data.to_csv(
            output,
            index=False,
            encoding="utf-8-sig"
        )

        output.seek(0)

        save_predictions(records)

        return send_file(
            output,
            mimetype="text/csv",
            as_attachment=True,
            download_name="elaria_predictions.csv"
        )

    except (pd.errors.ParserError, UnicodeError):
        return jsonify({"error": "Upload a valid UTF-8 CSV file."}), 400

    except pd.errors.EmptyDataError:

        return jsonify({
            "error": "The uploaded CSV contains no data."
        }), 400

    except ValueError as error:

        return jsonify({
            "error": "Input contains no usable text or invalid values."
        }), 400

    except Exception as error:

        app.logger.error("CSV prediction failed (%s)", type(error).__name__)

        return jsonify({
            "error": "CSV processing failed. Check backend logs."
        }), 500


# --------------------------------------------------
# PREDICTION HISTORY
# --------------------------------------------------

@app.route("/history", methods=["GET"])
def prediction_history():

    records = get_prediction_history()

    return jsonify({
        "status": "success",
        "total_records": len(records),
        "records": records
    })


# --------------------------------------------------
# PREDICTION STATISTICS
# --------------------------------------------------

@app.route("/statistics", methods=["GET"])
def statistics():

    return jsonify({
        "status": "success",
        "total_predictions": get_prediction_count()
    })


# --------------------------------------------------
# FILE SIZE ERROR HANDLER
# --------------------------------------------------

@app.errorhandler(413)
def file_too_large(error):

    return jsonify({
        "error": "File exceeds the maximum upload size of 10 MB."
    }), 413


# --------------------------------------------------
# RUN APPLICATION
# --------------------------------------------------

@app.errorhandler(sqlite3.Error)
def database_error(error):
    app.logger.error("Database operation failed (%s)", type(error).__name__)
    return jsonify({"error": "Database operation failed."}), 500


if __name__ == "__main__":

    app.run(host="127.0.0.1", debug=False)
