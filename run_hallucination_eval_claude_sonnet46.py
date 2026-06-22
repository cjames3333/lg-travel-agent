"""
Hallucination eval using Claude Sonnet 4.6 as the judge.
Extracts preamble+CSV scenarios from the GPT-5.5 analysis doc and re-evaluates
each with claude-sonnet-4-6 using the Okahu hallucination template.

Usage:
    python3 run_hallucination_eval_claude_sonnet46.py
    (ANTHROPIC_API_KEY loaded from .env automatically)
"""

import json
import os
import re
import sys
import time
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv()

DOC_PATH = "docs/hallucination_eval_matrix_gpt55_preamble_csv_2026-04-27T20-22-37.md"
TEMPLATE_PATH = "/Users/careyjames/Documents/GitHub/observability/azure-fn/eval-api/core/templates/hallucination.json"
OUTPUT_PATH = "docs/hallucination_eval_matrix_claude_sonnet46_preamble_csv.json"
MODEL = "claude-sonnet-4-6"

LABEL_MAP = {
    "no": "no_hallucination",
    "minor": "minor_hallucination",
    "major": "major_hallucination",
    "no_hallucination": "no_hallucination",
    "minor_hallucination": "minor_hallucination",
    "major_hallucination": "major_hallucination",
}

EVAL_TOOL = {
    "name": "hallucination_eval",
    "description": "Return the hallucination evaluation result as structured JSON",
    "input_schema": {
        "type": "object",
        "properties": {
            "label": {
                "type": "string",
                "enum": ["no_hallucination", "minor_hallucination", "major_hallucination"],
                "description": "Hallucination classification of the response",
            },
            "explanation": {
                "type": "string",
                "description": "Detailed analysis of the hallucination evaluation",
            },
            "hallucination_score": {
                "type": "number",
                "description": "0.0 = no hallucination, 1.0 = severe hallucination",
            },
            "factual_alignments": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Statements factually aligned with the context",
            },
            "contradictions": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Statements that contradict or are unsupported by the context",
            },
            "hallucination_types": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Types: factual_inaccuracy, unsupported_claim, contradiction, fabrication, exaggeration",
            },
            "context_coverage": {
                "type": "string",
                "enum": ["excellent", "good", "fair", "poor"],
            },
            "factual_accuracy": {
                "type": "string",
                "enum": ["accurate", "mostly_accurate", "partially_accurate", "inaccurate"],
            },
            "verification_status": {
                "type": "string",
                "enum": ["fully_verified", "partially_verified", "unverified", "contradicted"],
            },
            "confidence_level": {
                "type": "number",
                "description": "0.0–1.0 confidence in the assessment",
            },
        },
        "required": [
            "label", "explanation", "hallucination_score",
            "factual_alignments", "contradictions", "hallucination_types",
            "context_coverage", "factual_accuracy", "verification_status", "confidence_level",
        ],
    },
}


def load_template():
    with open(TEMPLATE_PATH) as f:
        return json.load(f)


def extract_scenarios(doc_text):
    scenarios = []

    section_pattern = re.compile(
        r"^### ((?:CC|FS|LGS)-T\d+) — (.+?) — `([0-9a-f]+)`",
        re.MULTILINE,
    )

    expected_pattern = re.compile(r"\|\s*\*\*Expected\*\*\s*\|\s*(\S+)\s*\|")

    trace_pattern = re.compile(
        r"\*\*Trace content used \(preamble \+ CSV\):\*\*\s*\n+```\n(.*?)```",
        re.DOTALL,
    )

    matches = list(section_pattern.finditer(doc_text))
    for i, m in enumerate(matches):
        scenario_id = m.group(1)
        scenario_name = m.group(2).strip()
        trace_id = m.group(3)

        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(doc_text)
        section = doc_text[start:end]

        exp_match = expected_pattern.search(section)
        expected_raw = exp_match.group(1).strip() if exp_match else "unknown"
        expected = LABEL_MAP.get(expected_raw, expected_raw)

        trace_match = trace_pattern.search(section)
        trace_content = trace_match.group(1).strip() if trace_match else None

        if trace_content:
            scenarios.append({
                "id": scenario_id,
                "name": scenario_name,
                "trace_id": trace_id,
                "expected": expected,
                "trace_content": trace_content,
            })
        else:
            print(f"  [WARN] No trace content found for {scenario_id}")

    return scenarios


def evaluate_scenario(client, template, scenario):
    eval_prompt = f"""{template['eval_prompt']}

The conversation below is a full multi-agent conversation tree in preamble+CSV format.
Evaluate whether the final_response row contains hallucinated content relative to what the
tool calls and sub-agents actually returned during the conversation.

--- CONVERSATION TREE (preamble + CSV) ---
{scenario['trace_content']}
"""

    response = client.messages.create(
        model=MODEL,
        max_tokens=2048,
        tools=[EVAL_TOOL],
        tool_choice={"type": "tool", "name": "hallucination_eval"},
        messages=[
            {
                "role": "user",
                "content": eval_prompt,
            }
        ],
    )

    for block in response.content:
        if block.type == "tool_use" and block.name == "hallucination_eval":
            return block.input, response.usage

    raise ValueError(f"No tool_use block in response for {scenario['id']}")


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set")
        sys.exit(1)

    template = load_template()
    doc_text = Path(DOC_PATH).read_text()

    print(f"Model: {MODEL}")
    print(f"Extracting scenarios from doc...")
    scenarios = extract_scenarios(doc_text)
    print(f"Found {len(scenarios)} scenarios\n")

    client = anthropic.Anthropic(api_key=api_key)

    results = []
    correct = 0
    totals = {"no_hallucination": 0, "minor_hallucination": 0, "major_hallucination": 0}
    correct_by_label = {"no_hallucination": 0, "minor_hallucination": 0, "major_hallucination": 0}

    for i, sc in enumerate(scenarios, 1):
        print(f"[{i:2d}/{len(scenarios)}] {sc['id']} (expected: {sc['expected']}) ... ", end="", flush=True)
        try:
            eval_result, usage = evaluate_scenario(client, template, sc)
            label = eval_result.get("label", "unknown")
            score = eval_result.get("hallucination_score", 0)
            match = label == sc["expected"]
            mark = "✓" if match else "✗"

            print(f"{label} {mark}  (score={score:.2f})")

            if match:
                correct += 1
            totals[sc["expected"]] = totals.get(sc["expected"], 0) + 1
            if match:
                correct_by_label[sc["expected"]] = correct_by_label.get(sc["expected"], 0) + 1

            results.append({
                "id": sc["id"],
                "name": sc["name"],
                "trace_id": sc["trace_id"],
                "expected": sc["expected"],
                "model": MODEL,
                "eval_result": eval_result,
                "correct": match,
                "usage": {
                    "input_tokens": usage.input_tokens,
                    "output_tokens": usage.output_tokens,
                },
            })

            time.sleep(0.5)

        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                "id": sc["id"],
                "name": sc["name"],
                "trace_id": sc["trace_id"],
                "expected": sc["expected"],
                "model": MODEL,
                "error": str(e),
                "correct": False,
            })

    total = len(scenarios)
    print(f"\n{'='*60}")
    print(f"RESULTS — {MODEL}")
    print(f"{'='*60}")
    print(f"Overall accuracy: {correct}/{total} = {correct/total*100:.1f}%")
    for lbl in ["no_hallucination", "minor_hallucination", "major_hallucination"]:
        t = totals.get(lbl, 0)
        c = correct_by_label.get(lbl, 0)
        if t:
            print(f"  {lbl}: {c}/{t}")
    print()

    # Count false positives and false negatives
    false_positives = sum(
        1 for r in results
        if r.get("expected") == "no_hallucination" and r.get("eval_result", {}).get("label") != "no_hallucination"
    )
    false_negatives = sum(
        1 for r in results
        if r.get("expected") != "no_hallucination" and r.get("eval_result", {}).get("label") == "no_hallucination"
    )
    print(f"  False positives (clean flagged as hal): {false_positives}")
    print(f"  False negatives (hal missed): {false_negatives}")

    output = {
        "model": MODEL,
        "eval_template": "hallucination",
        "format": "preamble+csv",
        "total_scenarios": total,
        "correct": correct,
        "accuracy": round(correct / total, 4) if total else 0,
        "by_label": {
            lbl: {"correct": correct_by_label.get(lbl, 0), "total": totals.get(lbl, 0)}
            for lbl in ["no_hallucination", "minor_hallucination", "major_hallucination"]
        },
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "results": results,
    }

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
