import { render, screen } from "@testing-library/react";
import { expect, test } from "vitest";
import ResultPanel from "./components/ResultPanel.jsx";

test("renders empty result state", () => {
  render(<ResultPanel result={null} />);
  expect(screen.getByText(/Run an analysis/i)).toBeInTheDocument();
});

test("renders quality label and score", () => {
  render(
    <ResultPanel
      result={{
        quality_label: "DEGRADED",
        quality_score: 54,
        quality_confidence: 0.81,
        issues: [{ type: "blur", severity: "high", confidence: 0.9 }],
        statistics: { sharpness: 12.5, brightness: 120 },
        explanation: { summary: "Blurry image.", contributing_factors: ["Low sharpness"] },
      }}
    />
  );
  expect(screen.getByText("DEGRADED")).toBeInTheDocument();
  expect(screen.getByText("54")).toBeInTheDocument();
  expect(screen.getByText("blur")).toBeInTheDocument();
});
