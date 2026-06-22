"""
Local hallucination eval using Anthropic SDK directly.
Usage:
    ANTHROPIC_API_KEY=your-key python3 run_hallucination_eval.py
"""

import json
import os
import re
import sys
from docx import Document
import anthropic

TEMPLATE_PATH = "hallucination.json"
DOCX_PATH = "hallucination eval output_31352875cbf7baf3f8143fa8a4a2cb74.docx"
OUTPUT_PATH = "hallucination_eval_result.json"


def load_template():
    with open(TEMPLATE_PATH) as f:
        return json.load(f)


def extract_docx_content():
    doc = Document(DOCX_PATH)
    raw = doc.paragraphs[0].text
    parts = raw.split("|", 4)

    trace_id = parts[0].strip()
    start_time = parts[1].strip()
    end_time = parts[2].strip()
    workflow = parts[3].strip()
    conversation_raw = parts[4]

    # Extract user input
    user_input_match = re.search(r'\[\["([^"]+)"\]\]', conversation_raw)
    user_input = user_input_match.group(1) if user_input_match else "book a hotel in Paris, Tennessee"

    # Extract tool response (the simulated mismatch)
    tool_resp_match = re.search(
        r'hotel_name.*?Hotel de la Seine \(Paris, France\).*?simulated_location_mismatch[^}]*\}',
        conversation_raw
    )

    # Extract final AI responses
    final_responses = re.findall(r'Your hotel is booked[^"\\]+', conversation_raw)
    final_response = final_responses[-1].strip().rstrip("\\").strip() if final_responses else ""

    return {
        "trace_id": trace_id,
        "start_time": start_time,
        "end_time": end_time,
        "workflow": workflow,
        "user_input": user_input,
        "final_response": final_response,
        "tool_context": (
            '{"hotel_name": "Hotel de la Seine (Paris, France)", '
            '"city": "Paris", "country": "France", "note": "simulated_location_mismatch"}'
        ),
    }


def build_eval_schema():
    return {
        "type": "object",
        "properties": {
            "label": {
                "type": "string",
                "enum": ["no_hallucination", "minor_hallucination", "major_hallucination"]
            },
            "explanation": {"type": "string"},
            "hallucination_score": {"type": "number"},
            "factual_alignments": {
                "type": "array",
                "items": {"type": "string"}
            },
            "contradictions": {
                "type": "array",
                "items": {"type": "string"}
            },
            "hallucination_types": {
                "type": "array",
                "items": {"type": "string"}
            },
            "context_coverage": {
                "type": "string",
                "enum": ["excellent", "good", "fair", "poor"]
            },
            "factual_accuracy": {
                "type": "string",
                "enum": ["accurate", "mostly_accurate", "partially_accurate", "inaccurate"]
            },
            "verification_status": {
                "type": "string",
                "enum": ["fully_verified", "partially_verified", "unverified", "contradicted"]
            },
            "confidence_level": {"type": "number"},
        },
        "required": [
            "label", "explanation", "hallucination_score",
            "factual_alignments", "contradictions", "hallucination_types",
            "context_coverage", "factual_accuracy", "verification_status", "confidence_level"
        ],
        "additionalProperties": False,
    }


def run_eval():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY environment variable not set.")
        print("Run with: ANTHROPIC_API_KEY=your-key python3 run_hallucination_eval.py")
        sys.exit(1)

    template = load_template()
    content = extract_docx_content()

    print(f"Trace ID:      {content['trace_id']}")
    print(f"Workflow:      {content['workflow']}")
    print(f"User input:    {content['user_input']}")
    print(f"AI response:   {content['final_response']}")
    print(f"Tool context:  {content['tool_context']}")
    print()
    print("Running eval via Anthropic API...")

    client = anthropic.Anthropic(api_key=api_key)

    eval_message = f"""{template['eval_prompt']}

--- CONTEXT (what the tool actually returned) ---
{content['tool_context']}

--- USER INPUT ---
{content['user_input']}

--- AI ASSISTANT RESPONSE ---
{content['final_response']}
"""

    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=4096,
        thinking={"type": "adaptive"},
        messages=[{"role": "user", "content": eval_message}],
        output_config={
            "format": {
                "type": "json_schema",
                "schema": build_eval_schema(),
            }
        },
    )

    result_text = next(b.text for b in response.content if b.type == "text")
    result = json.loads(result_text)

    output = {
        "trace_id": content["trace_id"],
        "workflow": content["workflow"],
        "eval_template": template["name"],
        "user_input": content["user_input"],
        "ai_response": content["final_response"],
        "tool_context": content["tool_context"],
        "eval_result": result,
        "usage": {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "cache_read_input_tokens": getattr(response.usage, "cache_read_input_tokens", 0),
        },
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n=== EVAL RESULT ===")
    print(f"Label:              {result['label']}")
    print(f"Hallucination Score:{result['hallucination_score']}")
    print(f"Factual Accuracy:   {result['factual_accuracy']}")
    print(f"Verification:       {result['verification_status']}")
    print(f"Context Coverage:   {result['context_coverage']}")
    print(f"Confidence Level:   {result['confidence_level']}")
    print(f"\nExplanation:\n{result['explanation']}")
    print(f"\nContradictions:")
    for c in result["contradictions"]:
        print(f"  - {c}")
    print(f"\nHallucination Types: {result['hallucination_types']}")
    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    run_eval()
