export function normalizePrediction(prediction) {
  const label = String(prediction ?? "").trim();
  const normalized = label.toLowerCase().replace(/[\s_]+/g, "-");

  if (
    /^(non|not|no)-/.test(normalized) ||
    /\b(non|not|no)\s+suicid/i.test(label)
  ) {
    return "Non-suicide";
  }

  if (/(suicid|risk|ideation)/.test(normalized)) {
    return "Suicide-related";
  }

  return label || "Unknown";
}
