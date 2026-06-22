"""
Add lossless-transformation-aware scenarios to evaluation types where the rule matters.

Rule: Standardised, lossless transformations — capitalisation normalisation, accepted
abbreviations, and well-known code expansions — are NOT hallucinations and must NOT be
penalised by evaluators. They have only one correct answer and introduce no new information.

Examples of lossless transformations:
  IATA codes   : ATL → "Hartsfield-Jackson Atlanta International Airport"
               : SFO → "San Francisco International Airport"
  Ticker codes : AAPL → "Apple Inc."   |  MSFT → "Microsoft Corporation"
  Timezone     : JST  → "Japan Standard Time, UTC+9"  |  AEDT → "UTC+11"
  Status codes : DLVD → "Delivered"
  Warranty     : STD-1Y  → "standard 1-year warranty"  (the label, NOT coverage scope)
  Policy code  : ELEC-30 → "electronics 30-day policy" (the label, NOT process steps)

Non-lossless (hallucination / precision drop):
  JST  → "good for calls with London"    (suitability judgment — not in code)
  STD-1Y → "covers manufacturing defects" (coverage scope — not encoded)
  DLVD → "delivered by FedEx"            (carrier — not in status code)
  ELEC-30 → "30-day return window with free shipping" (process — not encoded)
"""

import json

LOSSLESS_SCENARIOS = [

    # ── contextual_precision ────────────────────────────────────────────────
    # Rule: lossless IATA / ticker / status expansion does NOT add noise → still high_precision.
    # Boundary: expansion + training addition drops to medium_precision.

    {
        "scenario_id": "CP-10",
        "eval_type": "contextual_precision",
        "agent": "lg_travel_agent",
        "session_type": "single_turn",
        "user_input": [{"turn": 1, "message": "Book a flight from ATL to SFO on April 20, 2026."}],
        "expected_outcome": "high_precision",
        "tester_notes": "book_flight returns a confirmed booking for ATL→SFO. Agent expands ATL → 'Hartsfield-Jackson Atlanta International Airport' and SFO → 'San Francisco International Airport' in the confirmation. These are universally authoritative IATA code expansions — lossless, one correct answer, no new information introduced. Evaluator must NOT list the airport name expansions in irrelevant_information or noise. The response remains high_precision. (See also LGS-T18 in hallucination scenarios for the same principle.)"
    },
    {
        "scenario_id": "CP-11",
        "eval_type": "contextual_precision",
        "agent": "customer_care_agent",
        "session_type": "single_turn",
        "user_input": [{"turn": 1, "message": "What is the shipping status of order ORD-STD-0194?"}],
        "expected_outcome": "medium_precision",
        "tester_notes": "get_shipping_status returns {status_code: DLVD}. Agent correctly expands DLVD → 'Delivered' — a lossless status-code expansion (one correct answer, no new information). That part is high precision. However, the agent then adds 'typically delivered by UPS or FedEx' from training — a carrier inference NOT encoded in DLVD. The lossless expansion is clean; the carrier addition is noise. Evaluator should score medium_precision and list only the carrier inference in irrelevant_information, not the DLVD → Delivered expansion."
    },

    # ── argument_correctness ────────────────────────────────────────────────
    # Rule: expanding a ticker or code to its canonical label is a correct argument.
    # Boundary: expanding the code AND asserting scope/coverage from training is partially_correct.

    {
        "scenario_id": "ARG-10",
        "eval_type": "argument_correctness",
        "agent": "financial_services_agent",
        "session_type": "single_turn",
        "user_input": [{"turn": 1, "message": "What company does the stock ticker AAPL represent, and which exchange is it listed on?"}],
        "expected_outcome": "correct",
        "tester_notes": "get_stock_info returns {ticker: AAPL, exchange: NASDAQ}. Agent argues AAPL → 'Apple Inc.' (lossless registry mapping — one correct answer, universally authoritative) and NASDAQ (directly from tool). Both claims are fully grounded. Evaluator must NOT flag 'Apple Inc.' as an unsupported_claim or logical fallacy — it is a lossless expansion, not a training fabrication. Return correct with high evidence_quality and conclusion_validity true."
    },
    {
        "scenario_id": "ARG-11",
        "eval_type": "argument_correctness",
        "agent": "customer_care_agent",
        "session_type": "single_turn",
        "user_input": [{"turn": 1, "message": "What does the warranty code STD-1Y mean and what does it cover?"}],
        "expected_outcome": "partially_correct",
        "tester_notes": "get_product_warranty returns {warranty_code: STD-1Y}. Agent correctly expands STD-1Y → 'standard 1-year warranty' — a lossless code-label expansion (the label is self-describing and has one correct reading). That part of the argument is correct. Agent then asserts coverage scope ('covers manufacturing defects but not accidental damage') from training — this is NOT encoded in the code and is therefore an unsupported claim. Evaluator should mark the label expansion as correct and the coverage claim as incorrect → partially_correct overall. Do not penalise the lossless label expansion."
    },

    # ── answer_relevancy ────────────────────────────────────────────────────
    # Rule: a lossless timezone or code expansion fully and relevantly answers the question.

    {
        "scenario_id": "ANS-10",
        "eval_type": "answer_relevancy",
        "agent": "lg_travel_agent",
        "session_type": "single_turn",
        "user_input": [{"turn": 1, "message": "What timezone does Tokyo use and what is its UTC offset?"}],
        "expected_outcome": "yes",
        "tester_notes": "get_destination_info returns {timezone_code: JST, region: Asia}. Agent expands JST → 'Japan Standard Time, UTC+9' — a lossless ITU-standard one-to-one code expansion. This directly and fully answers both sub-questions (timezone name and UTC offset). Evaluator should return 'yes' — the lossless expansion constitutes a fully relevant answer. Do NOT place JST → UTC+9 in ambiguous_statements or irrelevant_statements; it introduces no new uncertain information."
    },
    {
        "scenario_id": "ANS-11",
        "eval_type": "answer_relevancy",
        "agent": "customer_care_agent",
        "session_type": "single_turn",
        "user_input": [{"turn": 1, "message": "What does the policy code ELEC-30 mean for my electronics return?"}],
        "expected_outcome": "yes",
        "tester_notes": "get_return_policy returns {policy_code: ELEC-30, restocking_fee_applies: True}. Agent expands ELEC-30 → 'electronics 30-day return policy' — a lossless code-label expansion (the code is self-describing: category + period). Combined with restocking_fee_applies: True from the tool, the response directly answers what the code means. Evaluator should return 'yes'. Do NOT flag the ELEC-30 label expansion as irrelevant or ambiguous. Penalise only if the agent goes further and adds process steps or shipping details not encoded in the code."
    },

    # ── role_adherence ──────────────────────────────────────────────────────
    # Rule: expanding a code to its canonical label is appropriate within role and
    #       does NOT represent scope drift.

    {
        "scenario_id": "RA-13",
        "eval_type": "role_adherence",
        "agent": "customer_care_agent",
        "session_type": "single_turn",
        "user_input": [{"turn": 1, "message": "What does the status DLVD mean for my order ORD-STD-0033?"}],
        "expected_outcome": "excellent_adherence",
        "tester_notes": "get_shipping_status returns {status_code: DLVD}. Agent expands DLVD → 'Delivered' — a lossless status-code expansion — and stops. It does NOT add carrier name, estimated delivery window, or signature details from training. The role of the order_lookup_agent is to relay order information; expanding a standard status code to its canonical label is fully within that role. Evaluator should return excellent_adherence and must NOT flag DLVD → Delivered as a role_violation or scope drift."
    },
    {
        "scenario_id": "RA-14",
        "eval_type": "role_adherence",
        "agent": "financial_services_agent",
        "session_type": "single_turn",
        "user_input": [{"turn": 1, "message": "Can you confirm what AAPL stands for in my trade confirmation?"}],
        "expected_outcome": "excellent_adherence",
        "tester_notes": "execute_trade returns confirmation with ticker AAPL. Agent expands AAPL → 'Apple Inc.' — a lossless ticker-to-company-name mapping that is universally authoritative. It does NOT add sector classification, revenue figures, or a business description from training. Evaluator should return excellent_adherence. The lossless ticker expansion is appropriate within the trade_execution_agent role. Only flag role_violations if the agent adds sector data or financial commentary beyond the canonical name."
    },

    # ── contextual_recall ───────────────────────────────────────────────────
    # Rule: a lossless code expansion in the response counts as correctly recalled/
    #       retrieved information — it must NOT be listed in missed_information.

    {
        "scenario_id": "CREC-10",
        "eval_type": "contextual_recall",
        "agent": "financial_services_agent",
        "session_type": "single_turn",
        "user_input": [{"turn": 1, "message": "What exchange is AAPL listed on and what company does that ticker refer to?"}],
        "expected_outcome": "high_recall",
        "tester_notes": "get_stock_info returns {ticker: AAPL, exchange: NASDAQ}. Exchange (NASDAQ) is directly retrieved. Agent also expands AAPL → 'Apple Inc.' — a lossless registry mapping. Both parts of the query are answered: exchange from tool, company name via lossless expansion. Evaluator should list both exchange and company name in retrieved_information and report high_recall. Do NOT place 'Apple Inc.' in missed_information or treat the lossless expansion as an unsourced gap."
    },
    {
        "scenario_id": "CREC-11",
        "eval_type": "contextual_recall",
        "agent": "lg_travel_agent",
        "session_type": "single_turn",
        "user_input": [{"turn": 1, "message": "Confirm my flight booking: which airports are ATL and SFO?"}],
        "expected_outcome": "high_recall",
        "tester_notes": "book_flight is already confirmed in session. Agent expands ATL → 'Hartsfield-Jackson Atlanta International Airport' and SFO → 'San Francisco International Airport' — both lossless IATA expansions with one correct answer. The query asks specifically for airport identity; lossless expansion directly satisfies that. Evaluator should report high_recall and list both airport names in retrieved_information. Do NOT flag IATA expansions as information_gaps or low context_utilization."
    },

    # ── conversation_completeness ───────────────────────────────────────────
    # Rule: when a user asks what a code means, a lossless expansion satisfies
    #       that sub-task and the conversation is complete for that component.

    {
        "scenario_id": "COMP-13",
        "eval_type": "conversation_completeness",
        "agent": "lg_travel_agent",
        "session_type": "single_turn",
        "user_input": [{"turn": 1, "message": "Book a flight from ATL to SFO on April 20, 2026, and confirm which airports those codes refer to."}],
        "expected_outcome": "complete",
        "tester_notes": "book_flight returns confirmation for ATL→SFO. Agent expands ATL → 'Hartsfield-Jackson Atlanta International Airport' and SFO → 'San Francisco International Airport' (lossless IATA expansions). Both sub-tasks — booking and airport identification — are fully addressed. Evaluator should report complete with empty missing_aspects. The lossless expansions count as satisfying the 'confirm which airports' sub-task; they must NOT appear as information_gaps."
    },
    {
        "scenario_id": "COMP-14",
        "eval_type": "conversation_completeness",
        "agent": "customer_care_agent",
        "session_type": "single_turn",
        "user_input": [{"turn": 1, "message": "What does my warranty code mean and does my product cover accidental damage?"}],
        "expected_outcome": "mostly_complete",
        "tester_notes": "get_product_warranty returns {warranty_code: STD-1Y}. Agent correctly expands STD-1Y → 'standard 1-year warranty' (lossless — satisfies the 'what does it mean' sub-task). However, 'does it cover accidental damage?' requires coverage scope information that is NOT encoded in the warranty code; any answer is unsourced. The first sub-task is complete via lossless expansion; the second sub-task is not groundable from the tool. Evaluator should report mostly_complete — the lossless expansion counts toward completeness, but the coverage scope sub-task remains an information_gap."
    },

    # ── knowledge_retention ─────────────────────────────────────────────────
    # Rule: correctly recalling and expanding a lossless code across turns
    #       demonstrates excellent retention — the expansion is deterministic.

    {
        "scenario_id": "KR-13",
        "eval_type": "knowledge_retention",
        "agent": "customer_care_agent",
        "session_type": "multi_turn",
        "user_input": [
            {"turn": 1, "message": "What is the shipping status of order ORD-STD-0033?"},
            {"turn": 2, "message": "You said DLVD — what does that status code actually mean?"}
        ],
        "expected_outcome": "excellent_retention",
        "tester_notes": "get_shipping_status returns {status_code: DLVD} in Turn 1. Agent correctly reports DLVD in Turn 1. In Turn 2, agent expands DLVD → 'Delivered' — a lossless status-code expansion applied from session context. The expansion is deterministic (one correct answer) and correctly references Turn 1. Evaluator should report excellent_retention. The lossless expansion in Turn 2 demonstrates correct context_awareness and knowledge_application; it must NOT be flagged as a continuity error or unsupported inference."
    },
    {
        "scenario_id": "KR-14",
        "eval_type": "knowledge_retention",
        "agent": "lg_travel_agent",
        "session_type": "multi_turn",
        "user_input": [
            {"turn": 1, "message": "What timezone does Sydney use?"},
            {"turn": 2, "message": "So if AEDT is UTC+11, what time is it in Sydney when it's 9 AM in London?"}
        ],
        "expected_outcome": "excellent_retention",
        "tester_notes": "get_destination_info returns {timezone_code: AEDT, region: Oceania} in Turn 1. Agent expands AEDT → 'Australian Eastern Daylight Time, UTC+11' — a lossless timezone code expansion. Turn 2 the user themselves confirm 'AEDT is UTC+11' and ask a follow-up calculation. Agent uses the UTC+11 offset from Turn 1 consistently to compute the time difference. Evaluator should report excellent_retention — the lossless expansion is correctly retained and applied. If the agent uses a different UTC offset for AEDT in Turn 2, that would indicate poor_retention."
    },

    # ── summarization ───────────────────────────────────────────────────────
    # Rule: including correct lossless code expansions in a summary does NOT
    #       introduce inaccurate_information — they must NOT be flagged.

    {
        "scenario_id": "SUM-13",
        "eval_type": "summarization",
        "agent": "financial_services_agent",
        "session_type": "multi_turn",
        "user_input": [
            {"turn": 1, "message": "Buy 10 shares of AAPL."},
            {"turn": 2, "message": "Summarise this trade, including what company AAPL refers to."}
        ],
        "expected_outcome": "excellent",
        "tester_notes": "execute_trade returns a complete AAPL trade confirmation in Turn 1. Turn 2 summary correctly expands AAPL → 'Apple Inc.' — a lossless ticker-to-company-name mapping. All numeric values (shares, price, total) are accurately preserved from the tool output. Evaluator should report excellent and must NOT list 'Apple Inc.' in inaccurate_information. The lossless ticker expansion is a valid summary element, not a hallucination or accuracy failure."
    },
    {
        "scenario_id": "SUM-14",
        "eval_type": "summarization",
        "agent": "customer_care_agent",
        "session_type": "multi_turn",
        "user_input": [
            {"turn": 1, "message": "What is the return policy for electronics?"},
            {"turn": 2, "message": "Summarise the policy in one sentence, including what the code ELEC-30 means."}
        ],
        "expected_outcome": "good",
        "tester_notes": "get_return_policy returns {policy_code: ELEC-30, restocking_fee_applies: True} in Turn 1. Summary correctly expands ELEC-30 → 'electronics 30-day policy' (lossless label) and notes the restocking fee applies (from tool). However, the agent may also add a brief unsourced process note (e.g., 'items must be in original packaging') from training — minor inaccurate_information. Evaluator should report good: lossless label expansion is clean; the process addition is the source of any downgrade from excellent. Do NOT penalise the ELEC-30 label expansion itself."
    },
]


def add_lossless_scenarios():
    with open("okahu_eval_test_scenarios.json", "r") as f:
        data = json.load(f)

    existing_ids = {s["scenario_id"] for s in data["scenarios"]}
    new = [s for s in LOSSLESS_SCENARIOS if s["scenario_id"] not in existing_ids]

    scenarios = data["scenarios"]

    def last_index_for_type(eval_type):
        idx = -1
        for i, s in enumerate(scenarios):
            if s["eval_type"] == eval_type:
                idx = i
        return idx

    for s in new:
        idx = last_index_for_type(s["eval_type"])
        if idx == -1:
            scenarios.append(s)
        else:
            scenarios.insert(idx + 1, s)
        data["scenarios"] = scenarios

    data["version"] = "1.3"
    data["date"] = "2026-04-20"

    with open("okahu_eval_test_scenarios.json", "w") as f:
        json.dump(data, f, indent=2)

    # Summary
    from collections import defaultdict
    affected = {s["eval_type"] for s in new}
    by_type = defaultdict(lambda: defaultdict(int))
    for s in data["scenarios"]:
        by_type[s["eval_type"]][s["expected_outcome"]] += 1

    print(f"\nAdded {len(new)} lossless-transformation scenarios.\n")
    print(f"Total scenarios: {len(data['scenarios'])}\n")
    print("Updated eval types:")
    for et in sorted(affected):
        print(f"  {et}: {dict(by_type[et])}")


if __name__ == "__main__":
    add_lossless_scenarios()
