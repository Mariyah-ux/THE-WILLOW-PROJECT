const API_BASE = import.meta.env.VITE_API_BASE_URL || "";

export async function predictText(text) {
  const response = await fetch(`${API_BASE}/api/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text })
  });

  let payload;
  try {
    payload = await response.json();
  } catch {
    throw new Error(
      response.ok
        ? "The prediction service returned invalid JSON."
        : `Prediction failed (${response.status})`
    );
  }

  if (!response.ok) {
    const message =
      payload && typeof payload === "object"
        ? payload.error || payload.message
        : null;
    throw new Error(message || `Prediction failed (${response.status})`);
  }

  if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
    throw new Error("The prediction service returned an invalid response.");
  }

  // Supports common response names from Flask backends.
  const prediction =
    payload.prediction ?? payload.label ?? payload.result ?? payload.class;
  if (prediction === null || prediction === undefined || String(prediction).trim() === "") {
    throw new Error("The prediction service response did not include a prediction.");
  }

  const confidenceRaw =
    payload.confidence ?? payload.probability ?? payload.score ?? null;

  let confidence = null;
  if (confidenceRaw !== null && confidenceRaw !== undefined && confidenceRaw !== "") {
    confidence = Number(confidenceRaw);
    if (Number.isFinite(confidence)) {
      if (confidence > 1 && confidence <= 100) confidence /= 100;
      confidence = Math.max(0, Math.min(1, confidence));
    } else {
      confidence = null;
    }
  }

  return {
    prediction: String(prediction),
    modelPrediction: String(payload.model_prediction ?? prediction),
    predictionSource: String(payload.prediction_source ?? "model"),
    confidence,
    raw: payload
  };
}