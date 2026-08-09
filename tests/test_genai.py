import json

import pytest

from src.genai.analyzer import (
    GenAIError,
    IncidentReport,
    build_prompt,
    explain,
    findings_from_analysis,
    format_timestamp,
    main,
    parse_generated_report,
    render_template_report,
)

SAMPLE = {
    "video": "data/raw/demo/street.avi",
    "duration_seconds": 3.0,
    "people_detected": 2,
    "abnormal_people": 1,
    "max_anomaly_score": 0.383,
    "people": [
        {
            "person_id": 2,
            "action": "fighting",
            "confidence": 0.3223,
            "status": "abnormal",
            "anomaly_score": 0.383,
            "risk_level": "normal",
            "start_time": 0.0,
            "end_time": 2.933,
            "peak_window": {"start_time": 0.933, "end_time": 2.933, "anomaly_score": 0.3853},
            "clips": [{"action": "fighting"}],
        },
        {
            "person_id": 1,
            "action": "sitting",
            "confidence": 0.299,
            "status": "normal",
            "anomaly_score": 0.338,
            "risk_level": "normal",
            "start_time": 0.0,
            "end_time": 2.933,
            "peak_window": {"start_time": 0.933, "end_time": 2.933},
        },
    ],
}


class FakeClient:
    def __init__(self, text: str):
        self.text = text
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.text


class FailingClient:
    def generate(self, prompt: str) -> str:
        raise RuntimeError("network down")


def test_format_timestamp_rounds_to_whole_seconds():
    assert format_timestamp(0) == "00:00:00"
    assert format_timestamp(62.4) == "00:01:02"
    assert format_timestamp(3661) == "01:01:01"


def test_format_timestamp_rejects_negative_values():
    with pytest.raises(ValueError, match="non-negative"):
        format_timestamp(-1)


def test_findings_strip_clip_details_and_keep_operator_fields():
    findings = findings_from_analysis(SAMPLE)

    assert findings["video"] == SAMPLE["video"]
    assert findings["people_detected"] == 2
    assert findings["abnormal_people"] == 1
    assert "clips" not in findings["people"][0]
    assert set(findings["people"][0]) == {
        "person_id",
        "action",
        "confidence",
        "status",
        "anomaly_score",
        "risk_level",
        "start_time",
        "end_time",
        "peak_window",
    }


def test_findings_accept_an_object_with_to_dict():
    class Wrapper:
        def to_dict(self):
            return SAMPLE

    findings = findings_from_analysis(Wrapper())

    assert findings["people_detected"] == 2


def test_findings_reject_unsupported_types():
    with pytest.raises(TypeError, match="VideoAnalysis or mapping"):
        findings_from_analysis(["not", "valid"])


def test_prompt_forbids_changing_findings_and_embeds_the_json():
    findings = findings_from_analysis(SAMPLE)
    prompt = build_prompt(findings)

    assert "must not change" in prompt
    assert '"action": "fighting"' in prompt
    assert "Respond with JSON only" in prompt


def test_template_report_keeps_model_findings_unchanged():
    findings = findings_from_analysis(SAMPLE)

    report = render_template_report(findings)

    assert report.model_findings == findings
    assert report.source == "template"
    assert "fighting" in report.analysis
    assert "person 2" in report.analysis
    assert "00:00:01" in report.analysis
    assert "verify" in report.recommended_action.lower()


def test_template_report_handles_an_empty_video():
    findings = {
        "video": "empty.mp4",
        "duration_seconds": 1.0,
        "people_detected": 0,
        "abnormal_people": 0,
        "max_anomaly_score": 0.0,
        "people": [],
    }

    report = render_template_report(findings)

    assert "did not report any tracked people" in report.analysis
    assert report.model_findings == findings


def test_template_report_for_all_normal_people():
    findings = findings_from_analysis(
        {
            "video": "park.mp4",
            "duration_seconds": 10,
            "people_detected": 1,
            "abnormal_people": 0,
            "max_anomaly_score": 0.1,
            "people": [
                {
                    "person_id": 1,
                    "action": "walking",
                    "confidence": 0.9,
                    "status": "normal",
                    "anomaly_score": 0.1,
                    "risk_level": "normal",
                    "start_time": 0,
                    "end_time": 5,
                    "peak_window": {"start_time": 1, "end_time": 3},
                }
            ],
        }
    )

    report = render_template_report(findings)

    assert "all labelled normal" in report.analysis
    assert "0.100" in report.analysis


def test_parse_generated_report_reads_json_and_freezes_findings():
    findings = findings_from_analysis(SAMPLE)
    text = json.dumps(
        {
            "analysis": "Person 2 appears to be fighting.",
            "risk_assessment": "Risk remains below the suspicious threshold.",
            "recommended_action": "Review the peak window manually.",
        }
    )

    report = parse_generated_report(text, findings)

    assert report.source == "google-genai"
    assert report.model_findings == findings
    assert report.analysis.startswith("Person 2")


def test_parse_generated_report_accepts_fenced_json():
    findings = findings_from_analysis(SAMPLE)
    text = """Here you go:
```json
{"analysis": "A", "risk_assessment": "B", "recommended_action": "C"}
```
"""

    report = parse_generated_report(text, findings)

    assert report.analysis == "A"
    assert report.risk_assessment == "B"


def test_parse_generated_report_rejects_missing_fields():
    with pytest.raises(GenAIError, match="missing fields"):
        parse_generated_report('{"analysis": "only this"}', findings_from_analysis(SAMPLE))


def test_parse_generated_report_rejects_invalid_json():
    with pytest.raises(GenAIError, match="not valid JSON"):
        parse_generated_report("not json at all", findings_from_analysis(SAMPLE))


def test_explain_uses_a_provided_client():
    client = FakeClient(
        json.dumps(
            {
                "analysis": "Generated analysis",
                "risk_assessment": "Generated risk",
                "recommended_action": "Generated action",
            }
        )
    )

    report = explain(SAMPLE, client=client)

    assert report.source == "google-genai"
    assert report.analysis == "Generated analysis"
    assert len(client.prompts) == 1
    assert report.model_findings == findings_from_analysis(SAMPLE)


def test_explain_falls_back_when_the_client_fails():
    report = explain(SAMPLE, client=FailingClient())

    assert report.source == "template-fallback"
    assert report.model_findings == findings_from_analysis(SAMPLE)
    assert "fighting" in report.analysis


def test_explain_uses_the_template_when_requested():
    client = FakeClient("{}")

    report = explain(SAMPLE, client=client, prefer_template=True)

    assert report.source == "template"
    assert client.prompts == []


def test_explain_without_api_key_uses_the_template(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    report = explain(SAMPLE)

    assert report.source == "template"


def test_incident_report_format_text_includes_all_sections():
    report = IncidentReport(
        model_findings={"people": []},
        analysis="A",
        risk_assessment="B",
        recommended_action="C",
        source="template",
    )

    text = report.format_text()

    assert "Model findings" in text
    assert "Analysis (template)" in text
    assert "Risk assessment" in text
    assert "Recommended action" in text


def test_cli_writes_a_template_report(tmp_path):
    input_path = tmp_path / "street.json"
    input_path.write_text(json.dumps(SAMPLE), encoding="utf-8")
    output_path = tmp_path / "incident.json"

    exit_code = main(["--input", str(input_path), "--output", str(output_path), "--template-only"])
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert payload["source"] == "template"
    assert payload["model_findings"]["people"][0]["action"] == "fighting"
    assert "clips" not in payload["model_findings"]["people"][0]


def test_cli_reports_a_missing_input(tmp_path):
    with pytest.raises(SystemExit, match="not found"):
        main(["--input", str(tmp_path / "absent.json"), "--template-only"])


def test_generated_report_cannot_replace_model_findings():
    findings = findings_from_analysis(SAMPLE)
    text = json.dumps(
        {
            "analysis": "Rewritten as walking",
            "risk_assessment": "Safe",
            "recommended_action": "Ignore",
            "model_findings": {"people": [{"action": "walking"}]},
        }
    )

    report = parse_generated_report(text, findings)

    assert report.model_findings == findings
    assert report.model_findings["people"][0]["action"] == "fighting"
