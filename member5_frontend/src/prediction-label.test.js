import assert from "node:assert/strict";
import test from "node:test";
import { normalizePrediction } from "./prediction-label.js";

test("keeps non-suicidal labels out of the suicide-related category", () => {
  for (const label of ["non-suicide", "non suicidal", "not suicidal", "no-suicide"]) {
    assert.equal(normalizePrediction(label), "Non-suicide");
  }
});

test("normalizes suicidal labels", () => {
  for (const label of ["suicide", "suicide-related", "suicidal", "suicidal ideation"]) {
    assert.equal(normalizePrediction(label), "Suicide-related");
  }
});

test("preserves other labels and handles missing labels", () => {
  assert.equal(normalizePrediction("neutral"), "neutral");
  assert.equal(normalizePrediction(""), "Unknown");
});
