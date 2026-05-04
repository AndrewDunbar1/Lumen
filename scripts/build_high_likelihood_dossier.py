#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import html
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from matcher.trial_normalization import normalize_trial_record


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def read_trials(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["nct_id"]: row for row in csv.DictReader(handle) if row.get("nct_id")}


def esc(value: object) -> str:
    return html.escape(str(value or ""))


def render_list(items: list[str], empty_text: str = "None recorded") -> str:
    if not items:
        return f"<div class='empty'>{esc(empty_text)}</div>"
    return "<ul>" + "".join(f"<li>{esc(item)}</li>" for item in items) + "</ul>"


def render_kv(label: str, value: object) -> str:
    return f"<div><span class='k'>{esc(label)}:</span> {esc(value)}</div>"


def render_criteria(criteria: list, empty_text: str) -> str:
    if not criteria:
        return f"<div class='empty'>{esc(empty_text)}</div>"
    return (
        "<ol class='criteria'>"
        + "".join(
            "<li>"
            f"<div class='criterion-text'>{esc(item.text)}</div>"
            f"<div class='criterion-meta'>priority={esc(item.priority)}"
            f" | admin={esc(item.is_administrative)}"
            f" | or_group={esc(item.is_or_group)}</div>"
            "</li>"
            for item in criteria
        )
        + "</ol>"
    )


def pair_band(pair_index: int) -> tuple[str, str]:
    if pair_index <= 39:
        return "band-1", "Pairs 1-39"
    if pair_index <= 79:
        return "band-2", "Pairs 40-79"
    if pair_index <= 119:
        return "band-3", "Pairs 80-119"
    return "band-4", "Pairs 120-159+"


def build_pair_section(patient_entry: dict, high_trial: dict, trial_row: dict[str, str], pair_index: int) -> str:
    profile = patient_entry["profile"]
    normalized_trial = normalize_trial_record(trial_row)
    evidence = profile.get("evidence", [])[:40]
    band_class, band_label = pair_band(pair_index)

    overview = [
        render_kv("Pair number", pair_index),
        render_kv("Patient ID", patient_entry.get("patient_id")),
        render_kv("Best trial", patient_entry.get("best_trial_nct_id")),
        render_kv("Best label", patient_entry.get("best_trial_label")),
        render_kv("Trial NCT", high_trial.get("trial_nct_id")),
        render_kv("Trial title", high_trial.get("trial_title")),
        render_kv("LLM confidence", high_trial.get("confidence")),
        render_kv("Eligibility probability", high_trial.get("eligibility_probability")),
        render_kv("Match recommendation", high_trial.get("match_recommendation")),
    ]

    patient_blocks = [
        "<div class='subsection'><h4>Demographics</h4>"
        + render_kv("Age", profile.get("age"))
        + render_kv("Sex", profile.get("sex"))
        + render_kv("Matcher demographics text", profile.get("demographics_text"))
        + "</div>",
        "<div class='subsection'><h4>Diagnoses</h4>" + render_list(profile.get("diagnoses", [])) + "</div>",
        "<div class='subsection'><h4>Normalized Diagnoses</h4>" + render_list(profile.get("normalized_diagnoses", [])) + "</div>",
        "<div class='subsection'><h4>Active Diagnoses</h4>" + render_list(profile.get("active_diagnoses", [])) + "</div>",
        "<div class='subsection'><h4>Procedures</h4>" + render_list(profile.get("procedures", [])) + "</div>",
        "<div class='subsection'><h4>Medications</h4>" + render_list(profile.get("medications", [])) + "</div>",
        "<div class='subsection'><h4>Labs</h4>" + render_list(profile.get("labs", [])) + "</div>",
        "<div class='subsection'><h4>Note Signals</h4>" + render_list(profile.get("note_signals", [])) + "</div>",
        "<div class='subsection'><h4>Clinical Tags</h4>" + render_list(profile.get("clinical_tags", [])) + "</div>",
        "<div class='subsection'><h4>Matcher Narrative</h4><pre>"
        + esc(profile.get("narrative_text", ""))
        + "</pre></div>",
        "<div class='subsection'><h4>Evidence Snippets</h4>"
        + (
            "<ul>"
            + "".join(
                f"<li><span class='src'>[{esc(item.get('source_ref'))}]</span> {esc(item.get('text'))}</li>"
                for item in evidence
            )
            + "</ul>"
            if evidence
            else "<div class='empty'>No evidence captured</div>"
        )
        + "</div>",
    ]

    trial_blocks = [
        "<div class='subsection'><h4>Trial Overview</h4>"
        + render_kv("NCT ID", normalized_trial.nct_id)
        + render_kv("Title", normalized_trial.title)
        + render_kv("Overall status", normalized_trial.overall_status)
        + render_kv("Sex gate", normalized_trial.sex)
        + render_kv("Min age", normalized_trial.age_min)
        + render_kv("Max age", normalized_trial.age_max)
        + render_kv("Conditions", trial_row.get("conditions", ""))
        + render_kv("Condition tags", ", ".join(normalized_trial.condition_tags))
        + render_kv("Intervention tags", ", ".join(normalized_trial.intervention_tags))
        + render_kv("Disease tags", ", ".join(normalized_trial.disease_tags))
        + render_kv("Required patient signals", ", ".join(normalized_trial.required_patient_signals))
        + "</div>",
        "<div class='subsection'><h4>Inclusion Criteria</h4>"
        + render_criteria(normalized_trial.inclusion_criteria, "No inclusion criteria parsed")
        + "</div>",
        "<div class='subsection'><h4>Exclusion Criteria</h4>"
        + render_criteria(normalized_trial.exclusion_criteria, "No exclusion criteria parsed")
        + "</div>",
        "<div class='subsection'><h4>Raw Eligibility Text</h4><pre>"
        + esc(trial_row.get("eligibility_criteria", ""))
        + "</pre></div>",
    ]

    return f"""
    <section class="pair {band_class}">
      <div class="pair-header">
        <div>
          <div class="eyebrow">High-Likelihood Patient-Trial Match</div>
          <div class="band-chip">{esc(band_label)}</div>
          <h1>{esc(patient_entry.get("patient_id"))} -> {esc(high_trial.get("trial_nct_id"))}</h1>
          <div class="subtitle">{esc(high_trial.get("trial_title"))}</div>
        </div>
        <div class="score-box">
          <div><span class="k">Confidence:</span> {esc(high_trial.get("confidence"))}</div>
          <div><span class="k">Patient-level reason:</span> {esc(patient_entry.get("patient_level_reason"))}</div>
          <div><span class="k">Pair reason:</span> {esc(high_trial.get("reason"))}</div>
        </div>
      </div>
      <div class="grid two">
        <div class="panel">
          <h2>Match Overview</h2>
          {''.join(overview)}
        </div>
        <div class="panel">
          <h2>Source</h2>
          {render_kv("Patient JSON", patient_entry.get("source_patient_json"))}
        </div>
      </div>
      <div class="grid two">
        <div class="panel">
          <h2>Patient Profile</h2>
          {''.join(patient_blocks)}
        </div>
        <div class="panel">
          <h2>Trial Criteria</h2>
          {''.join(trial_blocks)}
        </div>
      </div>
    </section>
    """


def build_html(profile_json: Path, trials_csv: Path, output_html: Path) -> tuple[int, int]:
    profile_payload = load_json(profile_json)
    trials = read_trials(trials_csv)
    patients = profile_payload.get("patients", [])
    sections: list[str] = []
    pair_count = 0

    for patient_entry in patients:
        for high_trial in patient_entry.get("high_likelihood_trials", []):
            nct_id = str(high_trial.get("trial_nct_id") or "").strip()
            trial_row = trials.get(nct_id)
            if not trial_row:
                continue
            pair_count += 1
            sections.append(build_pair_section(patient_entry, high_trial, trial_row, pair_count))

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>High-Likelihood Trial Match Dossier</title>
  <style>
    @page {{
      size: 8.5in 11in;
      margin: 0.45in;
    }}
    body {{
      font-family: Georgia, 'Times New Roman', serif;
      color: #1a1a1a;
      background: #f3efe7;
      margin: 0;
      line-height: 1.35;
    }}
    .cover {{
      padding: 0.8in;
      background: linear-gradient(180deg, #f9f5ea 0%, #efe6d2 100%);
      min-height: 100vh;
      page-break-after: always;
    }}
    .cover h1 {{
      font-size: 34px;
      margin: 0 0 10px 0;
    }}
    .cover p {{
      max-width: 9in;
      font-size: 16px;
    }}
    .pair {{
      page-break-before: always;
      padding: 0.2in 0.28in 0.3in 0.28in;
      background: #fcfaf5;
    }}
    .pair.band-1 {{
      background: linear-gradient(180deg, #f9fbff 0%, #edf4ff 100%);
    }}
    .pair.band-2 {{
      background: linear-gradient(180deg, #f7fcf6 0%, #ebf7e7 100%);
    }}
    .pair.band-3 {{
      background: linear-gradient(180deg, #fffaf3 0%, #fff0d9 100%);
    }}
    .pair.band-4 {{
      background: linear-gradient(180deg, #fff6f8 0%, #ffe8ef 100%);
    }}
    .pair-header {{
      display: flex;
      justify-content: space-between;
      gap: 18px;
      align-items: flex-start;
      border-bottom: 2px solid #b79e6a;
      padding-bottom: 10px;
      margin-bottom: 14px;
    }}
    .eyebrow {{
      text-transform: uppercase;
      letter-spacing: 0.08em;
      font-size: 11px;
      color: #6f5a34;
      margin-bottom: 6px;
    }}
    h1 {{
      font-size: 26px;
      margin: 0 0 4px 0;
    }}
    h2 {{
      font-size: 18px;
      margin: 0 0 10px 0;
      border-bottom: 1px solid #d6c3a0;
      padding-bottom: 4px;
    }}
    h4 {{
      font-size: 14px;
      margin: 12px 0 6px 0;
      color: #5e4f31;
    }}
    .subtitle {{
      font-size: 15px;
      color: #4c4c4c;
    }}
    .band-chip {{
      display: inline-block;
      margin-bottom: 8px;
      padding: 4px 8px;
      border-radius: 999px;
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      border: 1px solid #c8b896;
      background: rgba(255,255,255,0.75);
      color: #5f5134;
    }}
    .score-box {{
      width: 38%;
      min-width: 2.8in;
      background: #efe4ca;
      border: 1px solid #d6c3a0;
      padding: 10px 12px;
      border-radius: 10px;
      font-size: 13px;
    }}
    .grid.two {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 14px;
      align-items: start;
    }}
    .panel {{
      background: white;
      border: 1px solid #decfb3;
      border-radius: 12px;
      padding: 12px 14px;
      box-shadow: 0 1px 0 rgba(0,0,0,0.03);
    }}
    .subsection {{
      page-break-inside: avoid;
    }}
    .k {{
      font-weight: 700;
      color: #3e341f;
    }}
    .empty {{
      color: #777;
      font-style: italic;
    }}
    ul, ol {{
      margin: 6px 0 10px 18px;
      padding: 0;
    }}
    li {{
      margin-bottom: 4px;
    }}
    .criteria li {{
      margin-bottom: 8px;
    }}
    .criterion-meta {{
      font-size: 12px;
      color: #6e6e6e;
    }}
    pre {{
      white-space: pre-wrap;
      word-break: break-word;
      background: #faf7f1;
      border: 1px solid #ece1ce;
      border-radius: 8px;
      padding: 10px;
      font-size: 12px;
      line-height: 1.32;
      max-height: none;
    }}
    .src {{
      font-weight: 700;
      color: #6a5731;
    }}
    @media print {{
      body {{
        background: white;
      }}
      .pair, .cover, .panel {{
        box-shadow: none;
      }}
    }}
  </style>
</head>
<body>
  <section class="cover">
    <div class="eyebrow">Clinical Trial Matching Dossier</div>
    <h1>High-Likelihood Patient-Trial Match Report</h1>
    <p>This dossier contains the full matcher patient profile and the parsed trial inclusion and exclusion criteria for every patient-trial pair labeled <strong>high_likelihood</strong> by the final top-5 LLM review.</p>
    <p><span class="k">Unique matched patients:</span> {len(patients)}<br>
    <span class="k">High-likelihood pairs:</span> {pair_count}</p>
    <p>Open this file in a browser and use Print -> Save as PDF if you want a PDF export.</p>
  </section>
  {''.join(sections)}
</body>
</html>
"""
    output_html.write_text(html_doc, encoding="utf-8")
    return len(patients), pair_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a print-ready dossier for high-likelihood trial matches.")
    parser.add_argument("--profile-json", required=True)
    parser.add_argument("--trials-csv", required=True)
    parser.add_argument("--output-html", required=True)
    args = parser.parse_args()

    patient_count, pair_count = build_html(
        profile_json=Path(args.profile_json),
        trials_csv=Path(args.trials_csv),
        output_html=Path(args.output_html),
    )
    print(f"patients={patient_count}")
    print(f"pairs={pair_count}")
    print(args.output_html)


if __name__ == "__main__":
    main()
