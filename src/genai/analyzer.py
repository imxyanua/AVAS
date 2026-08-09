"""Turn structured model findings into an operator-facing incident report.

The deep learning pipeline owns the predictions. This module only receives the
frozen findings and writes a narrative around them. The model findings section
of every report is copied from the analysis unchanged, so a generative model
cannot quietly revise an action label or anomaly score.

When no API key is available the same facts are rendered by a deterministic
template, which keeps reports available offline and keeps the module testable
without calling a remote service.

Example:
    python -m src.genai.analyzer --input outputs/predictions/sample.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger("avas.genai")

DEFAULT_MODEL = "gemini-2.0-flash"
SYSTEM_RULES = """\
You explain video behaviour analysis results for a human operator.
You must not change, invent, or contradict any of the model findings.
Use only the facts in the JSON. If the findings are empty, say so.
Do not claim certainty beyond the reported confidence and anomaly scores.
Always recommend manual review before any consequential action.
Respond with JSON only, using exactly these keys:
  analysis: short paragraph summarising what the model found
  risk_assessment: short paragraph about severity using the reported risk levels
  recommended_action: short paragraph telling the operator what to do next
"""


class GenAIError(RuntimeError):
    """Raised when a generative explanation cannot be produced."""


class TextGenerator(Protocol):
    """Anything that turns a prompt into a string, usually a Gemini client."""

    def generate(self, prompt: str) -> str: ...


@dataclass(frozen=True)
class IncidentReport:
    """Operator-facing explanation grounded in frozen model findings."""

    model_findings: dict[str, Any]
    analysis: str
    risk_assessment: str
    recommended_action: str
    source: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_findings": self.model_findings,
            "analysis": self.analysis,
            "risk_assessment": self.risk_assessment,
            "recommended_action": self.recommended_action,
            "source": self.source,
        }

    def format_text(self) -> str:
        """Render a plain-text report for logs and terminals."""
        lines = [
            "Model findings (unchanged):",
            json.dumps(self.model_findings, indent=2, ensure_ascii=False),
            "",
            f"Analysis ({self.source}):",
            self.analysis.strip(),
            "",
            "Risk assessment:",
            self.risk_assessment.strip(),
            "",
            "Recommended action:",
            self.recommended_action.strip(),
        ]
        return "\n".join(lines)


class GoogleGenAIClient:
    """Thin wrapper around the Google GenAI SDK."""

    def __init__(self, api_key: str, *, model: str = DEFAULT_MODEL) -> None:
        if not api_key:
            raise ValueError("api_key must not be empty")
        if not model:
            raise ValueError("model must not be empty")

        from google import genai

        self.model = model
        self._client = genai.Client(api_key=api_key)

    def generate(self, prompt: str) -> str:
        response = self._client.models.generate_content(model=self.model, contents=prompt)
        text = getattr(response, "text", None)
        if not text or not str(text).strip():
            raise GenAIError("Google GenAI returned an empty response")
        return str(text)


def format_timestamp(seconds: float) -> str:
    """Format seconds as ``HH:MM:SS`` for operator-facing windows."""
    if seconds < 0:
        raise ValueError(f"seconds must be non-negative, got {seconds}")
    total = round(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def findings_from_analysis(analysis: Mapping[str, Any] | Any) -> dict[str, Any]:
    """Extract the frozen findings dict from a VideoAnalysis or a raw mapping."""
    if hasattr(analysis, "to_dict"):
        payload = analysis.to_dict()
    elif isinstance(analysis, Mapping):
        payload = dict(analysis)
    else:
        raise TypeError(
            f"analysis must be a VideoAnalysis or mapping, got {type(analysis).__name__}"
        )

    people = [
        {
            "person_id": person["person_id"],
            "action": person["action"],
            "confidence": person["confidence"],
            "status": person["status"],
            "anomaly_score": person["anomaly_score"],
            "risk_level": person["risk_level"],
            "start_time": person["start_time"],
            "end_time": person["end_time"],
            "peak_window": person.get("peak_window", {}),
        }
        for person in payload.get("people", [])
    ]

    return {
        "video": payload.get("video"),
        "duration_seconds": payload.get("duration_seconds"),
        "people_detected": payload.get("people_detected", len(people)),
        "abnormal_people": payload.get(
            "abnormal_people", sum(1 for person in people if person["status"] == "abnormal")
        ),
        "max_anomaly_score": payload.get(
            "max_anomaly_score",
            max((person["anomaly_score"] for person in people), default=0.0),
        ),
        "people": people,
    }


def build_prompt(findings: Mapping[str, Any]) -> str:
    """Build the user prompt that carries only the frozen model findings."""
    return (
        f"{SYSTEM_RULES}\n\n"
        "Model findings JSON:\n"
        f"{json.dumps(findings, indent=2, ensure_ascii=False)}\n"
    )


def render_template_report(findings: Mapping[str, Any]) -> IncidentReport:
    """Write a deterministic report from the findings without calling GenAI."""
    people = list(findings.get("people", []))
    video = findings.get("video") or "the uploaded video"
    duration = findings.get("duration_seconds")
    duration_text = (
        f" over {format_timestamp(float(duration))}" if isinstance(duration, (int, float)) else ""
    )

    if not people:
        analysis = (
            f"The model analysed {video}{duration_text} and did not report any "
            "tracked people with a usable motion history."
        )
        risk = "No person-level risk was assessed because no people were reported."
        action = (
            "Confirm that the video contains visible people and that detection "
            "thresholds are appropriate, then re-run the analysis if needed."
        )
        return IncidentReport(
            model_findings=dict(findings),
            analysis=analysis,
            risk_assessment=risk,
            recommended_action=action,
            source="template",
        )

    abnormal = [person for person in people if person.get("status") == "abnormal"]
    focus = max(people, key=lambda person: float(person.get("anomaly_score", 0.0)))
    peak = focus.get("peak_window") or {}
    window = _window_text(peak.get("start_time"), peak.get("end_time"))

    if abnormal:
        analysis = (
            f"The model analysed {video}{duration_text} and reported "
            f"{len(people)} tracked people, of whom {len(abnormal)} were labelled "
            f"abnormal. The strongest signal is person {focus['person_id']}, "
            f"predicted as {focus['action']} with confidence "
            f"{100 * float(focus['confidence']):.1f}%{window}."
        )
    else:
        analysis = (
            f"The model analysed {video}{duration_text} and reported "
            f"{len(people)} tracked people, all labelled normal. "
            f"The highest anomaly score was {float(findings.get('max_anomaly_score', 0.0)):.3f}."
        )

    risk = (
        f"Person {focus['person_id']} carries risk level "
        f"{focus.get('risk_level', 'unknown')} with anomaly score "
        f"{float(focus.get('anomaly_score', 0.0)):.3f}. "
        "Risk levels come from the anomaly score thresholds and can disagree "
        "with the predicted class when the probability mass is not concentrated "
        "on abnormal behaviours."
    )
    action = (
        "Review the reported peak window in the video and verify the person "
        "identity before taking any operational decision. Treat the model output "
        "as decision support, not as a final determination."
    )

    return IncidentReport(
        model_findings=dict(findings),
        analysis=analysis,
        risk_assessment=risk,
        recommended_action=action,
        source="template",
    )


def parse_generated_report(text: str, findings: Mapping[str, Any]) -> IncidentReport:
    """Parse a GenAI JSON response and attach the frozen findings unchanged."""
    payload = _extract_json_object(text)
    required = ("analysis", "risk_assessment", "recommended_action")
    missing = [key for key in required if key not in payload or not str(payload[key]).strip()]
    if missing:
        raise GenAIError(f"generated report is missing fields: {missing}")

    return IncidentReport(
        model_findings=dict(findings),
        analysis=str(payload["analysis"]).strip(),
        risk_assessment=str(payload["risk_assessment"]).strip(),
        recommended_action=str(payload["recommended_action"]).strip(),
        source="google-genai",
    )


def explain(
    analysis: Mapping[str, Any] | Any,
    *,
    client: TextGenerator | None = None,
    api_key: str | None = None,
    model: str = DEFAULT_MODEL,
    prefer_template: bool = False,
) -> IncidentReport:
    """Explain model findings with GenAI when available, otherwise a template.

    ``model_findings`` in the returned report is always a copy of the input
    findings. The generative path only fills the narrative fields.
    """
    findings = findings_from_analysis(analysis)

    if prefer_template:
        return render_template_report(findings)

    generator = client
    if generator is None:
        key = api_key if api_key is not None else os.environ.get("GOOGLE_API_KEY", "").strip()
        if not key:
            logger.info("GOOGLE_API_KEY not set; using the template report")
            return render_template_report(findings)
        generator = GoogleGenAIClient(key, model=model)

    prompt = build_prompt(findings)
    try:
        text = generator.generate(prompt)
        return parse_generated_report(text, findings)
    except Exception as error:
        logger.warning("GenAI explanation failed (%s); falling back to the template", error)
        report = render_template_report(findings)
        return IncidentReport(
            model_findings=report.model_findings,
            analysis=report.analysis,
            risk_assessment=report.risk_assessment,
            recommended_action=report.recommended_action,
            source="template-fallback",
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="JSON produced by src.inference.predict",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="report path (default: outputs/reports/<input stem>_incident.json)",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("GOOGLE_GENAI_MODEL", DEFAULT_MODEL),
        help=f"Google GenAI model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--template-only",
        action="store_true",
        help="skip GenAI and write the deterministic template report",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)

    if not args.input.is_file():
        raise SystemExit(f"analysis file not found: {args.input}")

    findings = findings_from_analysis(json.loads(args.input.read_text(encoding="utf-8")))
    report = explain(findings, model=args.model, prefer_template=args.template_only)

    output = args.output or Path("outputs/reports") / f"{args.input.stem}_incident.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info("incident report written to %s (source=%s)", output, report.source)
    print(report.format_text())
    return 0


def _window_text(start: Any, end: Any) -> str:
    if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
        return ""
    return f", strongest window {format_timestamp(float(start))} to {format_timestamp(float(end))}"


def _extract_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL)
    if fenced:
        cleaned = fenced.group(1)
    else:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            cleaned = cleaned[start : end + 1]

    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as error:
        raise GenAIError(f"generated report is not valid JSON: {error}") from error

    if not isinstance(payload, dict):
        raise GenAIError("generated report root must be a JSON object")
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
