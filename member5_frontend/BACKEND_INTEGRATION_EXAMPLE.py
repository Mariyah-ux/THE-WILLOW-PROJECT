"""
Example API contract for the Willow React frontend.

Do not replace your existing app.py blindly. Merge this route into it and
replace `your_existing_prediction_function` with the REAL function in your
model_handler.py. The model/vectorizer artifacts should be loaded by your
existing backend, not by React.
"""

from flask import request, jsonify

# Add this route to your existing Flask app:
#
# @app.post("/api/predict")
# def api_predict():
#     data = request.get_json(silent=True) or {}
#     text = str(data.get("text", "")).strip()
#     if not text:
#         return jsonify({"error": "Text is required"}), 400
#
#     try:
#         # CHANGE THIS LINE to match your model_handler.py function/signature.
#         result = your_existing_prediction_function(text)
#
#         # Return this standard shape to the React frontend.
#         # If your function returns a dict, map its actual keys here.
#         return jsonify({
#             "prediction": result["prediction"],
#             "confidence": result.get("confidence"),
#             "model_prediction": result.get("model_prediction"),
#             "prediction_source": result.get("prediction_source", "model")
#         })
#     except Exception as exc:
#         app.logger.exception("Prediction failed")
#         return jsonify({"error": "Prediction could not be completed"}), 500
#
# Enable CORS if frontend and backend run on different origins:
# from flask_cors import CORS
# CORS(app, resources={r"/api/*": {"origins": "http://localhost:8080"}})
