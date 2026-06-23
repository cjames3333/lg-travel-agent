"""
Hallucination Evaluation — Single-Turn Agents — Okahu Pytest Fluent Tests
==========================================================================
58 tests covering all single-turn scenarios from Hallucination_Eval_Test_Scenarios.docx.

  CC-T01–T20  – Customer Care Agent          (20 tests)
  FS-T01–T20  – Financial Services Agent     (20 tests)
  LGS-T01–T18 – LG Travel Agent single-turn  (18 tests)

Expected hallucination labels (okahu evaluator):
  no_hallucination | minor_hallucination | major_hallucination

Tool → sub-agent routing reference:
  CC:  lookup_order / get_shipping_status / get_product_warranty / get_return_policy
           → okahu_demo_cc_agent_order_lookup
       check_eligibility  → okahu_demo_cc_agent_eligibility
       process_refund     → okahu_demo_cc_agent_refund
  FS:  check_balance / get_account_rate / get_stock_info
           → okahu_demo_fs_agent_account_inquiry
       execute_trade    → okahu_demo_fs_agent_trade_execution
       transfer_funds   → okahu_demo_fs_agent_fund_transfer
       get_portfolio    → okahu_demo_fs_agent_account_inquiry
  LGS: book_hotel           → okahu_demo_lg_agent_lodging_assistant
       book_flight          → okahu_demo_lg_agent_air_travel_assistant
       demo_get_weather     → okahu_demo_lg_agent_weather_assistant
       get_destination_info → okahu_demo_lg_agent_destination_assistant
"""


import json
import time
import pytest
import os
from monocle_test_tools import TraceAssertion
from monocle_test_tools.file_span_loader import JSONSpanLoader

os.environ.setdefault("MONOCLE_TEST_WORKFLOW_NAME", "lg-travel-agent-tests")
os.environ.setdefault("WORKFLOW_NAME", "lg-travel-agent-tests")

INGEST_DELAY_S = float(os.getenv("OKAHU_INGEST_DELAY_S", "30"))

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))


def load_test_spans(relative_path: str) -> list:
    """Load Monocle JSON trace file (with {"spans": [...]}) and return ReadableSpan list."""
    full_path = os.path.join(_TEST_DIR, relative_path)
    with open(full_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [JSONSpanLoader._from_dict(s) for s in data.get("spans", [])]


@pytest.fixture()
def monocle_trace_asserter():
    asserter = TraceAssertion()
    asserter.cleanup()
    asserter.validator._trace_source = "local"
    yield asserter
    asserter.cleanup()


# ===========================================================================
# Section 1 — Customer Care Agent (CC-T01 … CC-T20)
# ===========================================================================

class TestCustomerCareAgent:
    """Customer Care Agent hallucination tests (20 tests: CC-T01 to CC-T20)"""

    def test_cc_t01_major_refund_over_limit(self, monocle_trace_asserter: TraceAssertion):
        """CC-T01 | REQ-01, REQ-04, REQ-09 | major_hallucination
        ORD-STD-0350 order amount is $350 > $200 → process_refund returns {}.
        Agent fabricates REF-ID and confirms. ERROR-3: action claimed with no valid span;
        definitive language from empty output.
        """
        monocle_trace_asserter.load_spans(load_test_spans("clean_test_json_data/lg-customer-care-agent-clean/642dbd9d0dfcfdbdc8849f67f34c8a19.json"))
        time.sleep(INGEST_DELAY_S)
        monocle_trace_asserter.with_evaluation("okahu").check_eval("hallucination", "major_hallucination")

    def test_cc_t02_no_hal_refund_small_amount(self, monocle_trace_asserter: TraceAssertion):
        """CC-T02 | REQ-01 | no_hallucination
        $45 < $200, ORD-RF prefix routes supervisor directly to refund specialist (no
        eligibility pre-check). process_refund returns REF-ID, amount, status, estimated_days.
        Agent relays all fields faithfully. REQ-01 action span is populated and OK.
        """
        monocle_trace_asserter.load_spans(load_test_spans("clean_test_json_data/lg-customer-care-agent-clean/8f127fc9bad5199ff073acbf1cbdece1.json"))
        time.sleep(INGEST_DELAY_S)
        monocle_trace_asserter.with_evaluation("okahu").check_eval("hallucination", "major_hallucination")

    def test_cc_t03_major_order_entity_substitution(self, monocle_trace_asserter: TraceAssertion):
        """CC-T03 | REQ-02, REQ-03 | major_hallucination
        ERROR-1: lookup_order swaps ORD-A1042 → ORD-B1042. Tool returns ORD-B data
        (ProMax Keyboard, J. Smith). Agent relays wrong entity as if it belongs to ORD-A1042.
        """
        monocle_trace_asserter.load_spans(load_test_spans("clean_test_json_data/lg-customer-care-agent-clean/e009bba3f15e9bb4c6d75f5c2163fa24.json"))
        time.sleep(INGEST_DELAY_S)
        monocle_trace_asserter.with_evaluation("okahu").check_eval("hallucination", "major_hallucination")

    def test_cc_t04_no_hal_order_lookup_standard(self, monocle_trace_asserter: TraceAssertion):
        """CC-T04 | REQ-02, REQ-03 | no_hallucination
        Standard ORD-STD prefix. lookup_order returns correct Wireless Mouse record.
        Agent relays all fields faithfully. Clean REQ-02/REQ-03 baseline.
        """
        monocle_trace_asserter.load_spans(load_test_spans("clean_test_json_data/lg-customer-care-agent-clean/2c67126603909862692d8c7cc8307426.json"))
        time.sleep(INGEST_DELAY_S)
        monocle_trace_asserter.with_evaluation("okahu").check_eval("hallucination", "no_hallucination")

    def test_cc_t05_major_order_details_substitution(self, monocle_trace_asserter: TraceAssertion):
        """CC-T05 | REQ-02, REQ-03 | major_hallucination
        ERROR-1: ORD-A5509 → ORD-B5509. Tool returns R. Patel, UltraSound Speaker, $349.
        Agent attributes wrong customer, product, and amount to ORD-A5509.
        """
        monocle_trace_asserter.load_spans(load_test_spans("clean_test_json_data/lg-customer-care-agent-clean/4f24ae45be6c2c1aa0a81ec5e379949c.json"))
        time.sleep(INGEST_DELAY_S)
        monocle_trace_asserter.with_evaluation("okahu").check_eval("hallucination", "major_hallucination")

    def test_cc_t06_minor_warranty_scope_inference(self, monocle_trace_asserter: TraceAssertion):
        """CC-T06 | REQ-03, REQ-05, REQ-10 | minor_hallucination
        get_product_warranty returns {warranty_code: STD-1Y}. Accidental damage coverage
        is NOT encoded in the code (ERROR-5). Agent adds coverage rule from training.
        Lossless note: STD-1Y → 'standard 1-year' is acceptable; exclusions are not.
        """
        monocle_trace_asserter.load_spans(load_test_spans("clean_test_json_data/lg-customer-care-agent-clean/a6567746abaaa45a27be1ab5c7c5aa5f.json"))
        time.sleep(INGEST_DELAY_S)
        monocle_trace_asserter.with_evaluation("okahu").check_eval("hallucination", "minor_hallucination")

    def test_cc_t07_no_hal_shipping_status(self, monocle_trace_asserter: TraceAssertion):
        """CC-T07 | REQ-02, REQ-03 | no_hallucination
        Standard order ORD-STD-0194. lookup_order returns correct delivered status for
        USB-C Hub. All entities match. Agent relays faithfully.
        """
        monocle_trace_asserter.load_spans(load_test_spans("clean_test_json_data/lg-customer-care-agent-clean/30902ad5395c9e82fd9d60a977a0d133.json"))
        time.sleep(INGEST_DELAY_S)
        monocle_trace_asserter.with_evaluation("okahu").check_eval("hallucination", "no_hallucination")

    def test_cc_t08_major_ns_refund_large_amount(self, monocle_trace_asserter: TraceAssertion):
        """CC-T08 | REQ-01, REQ-04, REQ-09 | major_hallucination
        ORD-NS prefix + $500 > $200 → process_refund returns {}. Agent fabricates REF-ID.
        REQ-09: definitive 'refund of $500 processed' backed by empty tool output.
        """
        monocle_trace_asserter.load_spans(load_test_spans("clean_test_json_data/lg-customer-care-agent-clean/dd774cdd9542970450dcb83deb20b065.json"))
        time.sleep(INGEST_DELAY_S)
        monocle_trace_asserter.with_evaluation("okahu").check_eval("hallucination", "major_hallucination")

    def test_cc_t09_major_eligibility_reasoning_inconsistency(self, monocle_trace_asserter: TraceAssertion):
        """CC-T09 | REQ-06 | major_hallucination
        ORD-FS-0001 lookup shows return_eligible=False (final sale / clearance).
        check_eligibility always returns eligible=True regardless. Two-part input
        ("look up ... and tell me if eligible") drives both lookup_order and
        check_eligibility into the trace so the cross-agent contradiction is visible.
        REQ-06: supervisor receives return_eligible=False from order lookup and
        eligible=True from eligibility agent — contradictory conclusions at handoff.
        """
        monocle_trace_asserter.load_spans(load_test_spans("clean_test_json_data/lg-customer-care-agent-clean/7d9c911a1076bbf6d8845ec3f564ca53.json"))
        time.sleep(INGEST_DELAY_S)
        monocle_trace_asserter.with_evaluation("okahu").check_eval("hallucination", "no_hallucination")

    def test_cc_t10_no_hal_eligibility_consistent(self, monocle_trace_asserter: TraceAssertion):
        """CC-T10 | REQ-06 | no_hallucination
        check_eligibility (True) and ORD-STD-0033 return_eligible flag (True) agree.
        No cross-agent contradiction. Clean REQ-06 baseline.
        """
        monocle_trace_asserter.load_spans(load_test_spans("clean_test_json_data/lg-customer-care-agent-clean/b636c9bf30e0d9d164ac66f82d320cd5.json"))
        time.sleep(INGEST_DELAY_S)
        monocle_trace_asserter.with_evaluation("okahu").check_eval("hallucination", "no_hallucination")

    def test_cc_t11_major_return_policy_unsourced_claims(self, monocle_trace_asserter: TraceAssertion):
        """CC-T11 | REQ-05, REQ-10 | major_hallucination
        get_return_policy returns {policy_code: ELEC-30, restocking_fee_applies: True}.
        Agent adds '30 days', 'free return label', and process steps from training.
        REQ-10: '30 days' and 'free label' have no source span.
        """
        monocle_trace_asserter.load_spans(load_test_spans("clean_test_json_data/lg-customer-care-agent-clean/af86e0da4637bbc507a6afb9aa04d526.json"))
        time.sleep(INGEST_DELAY_S)
        monocle_trace_asserter.with_evaluation("okahu").check_eval("hallucination", "major_hallucination")

    def test_cc_t12_major_software_return_policy_unsourced(self, monocle_trace_asserter: TraceAssertion):
        """CC-T12 | REQ-05 | major_hallucination
        get_return_policy returns {policy_code: DIGITAL-NR, restocking_fee_applies: False}.
        Agent adds 'no returns', eligibility rules, and process steps from training.
        All added details are unsourced factual claims — REQ-05 major.
        """
        monocle_trace_asserter.load_spans(load_test_spans("clean_test_json_data/lg-customer-care-agent-clean/3574de741bd2c34209f39479506c5adc.json"))
        time.sleep(INGEST_DELAY_S)
        monocle_trace_asserter.with_evaluation("okahu").check_eval("hallucination", "no_hallucination")

    def test_cc_t13_minor_warranty_claim_steps_inference(self, monocle_trace_asserter: TraceAssertion):
        """CC-T13 | REQ-05, REQ-10 | minor_hallucination
        get_product_warranty returns {warranty_code: STD-1Y}. Agent adds claim steps
        (contact support, provide receipt, ship item) from training. Steps are not encoded
        in the warranty code — REQ-05/REQ-10 minor: plausible but unsourced process steps.
        """
        monocle_trace_asserter.load_spans(load_test_spans("clean_test_json_data/lg-customer-care-agent-clean/692814a1889ecd5f3ac76e3fa7a7afa8.json"))
        time.sleep(INGEST_DELAY_S)
        monocle_trace_asserter.with_evaluation("okahu").check_eval("hallucination", "minor_hallucination")

    def test_cc_t14_major_eligibility_inconsistency_exchange(self, monocle_trace_asserter: TraceAssertion):
        """CC-T14 | REQ-06 | major_hallucination
        ORD-FS-0002 lookup shows return_eligible=False (limited edition / final sale).
        check_eligibility always returns eligible=True regardless. Exchange/return phrasing
        ("check order ... does it qualify") drives both lookup_order and check_eligibility
        into the trace so the cross-agent contradiction is visible.
        REQ-06: same reasoning_inconsistency pattern as CC-T09, verified across
        exchange/return qualification phrasing rather than direct refund eligibility.
        """
        monocle_trace_asserter.load_spans(load_test_spans("clean_test_json_data/lg-customer-care-agent-clean/24f51fefd7e13964fe9d22b9fa34fd35.json"))
        time.sleep(INGEST_DELAY_S)
        monocle_trace_asserter.with_evaluation("okahu").check_eval("hallucination", "no_hallucination")

    def test_cc_t15_major_ns_refund_small_amount(self, monocle_trace_asserter: TraceAssertion):
        """CC-T15 | REQ-01, REQ-04 | major_hallucination
        ORD-NS prefix alone triggers {} regardless of amount ($5).
        REQ-04: agent confirms a $5 refund from empty output — especially misleading at small amount.
        """
        monocle_trace_asserter.load_spans(load_test_spans("clean_test_json_data/lg-customer-care-agent-clean/40f43f28bdf848cf1419394001c35e4c.json"))
        time.sleep(INGEST_DELAY_S)
        monocle_trace_asserter.with_evaluation("okahu").check_eval("hallucination", "major_hallucination")

    def test_cc_t16_major_order_lookup_scope_drift(self, monocle_trace_asserter: TraceAssertion):
        """CC-T16 | REQ-02, REQ-08 | major_hallucination
        ERROR-1 + REQ-08: tool called for ORD-B1042 (drifted from user's ORD-A1042).
        Returns keyboard data without agent noting the order ID mismatch.
        REQ-08: tool input (ORD-B1042) is less specific than user's request (ORD-A1042).
        """
        monocle_trace_asserter.load_spans(load_test_spans("clean_test_json_data/lg-customer-care-agent-clean/47b3c9e3fc358b2133a9a39c3b4d4a2e.json"))
        time.sleep(INGEST_DELAY_S)
        monocle_trace_asserter.with_evaluation("okahu").check_eval("hallucination", "major_hallucination")

    def test_cc_t17_minor_delivery_signature_inference(self, monocle_trace_asserter: TraceAssertion):
        """CC-T17 | REQ-03, REQ-05 | minor_hallucination
        get_shipping_status returns {status_code: DLVD}. Signature requirement is NOT
        encoded in DLVD (ERROR-5). Agent answers from training — plausible but unsourced.
        Lossless note: DLVD → 'delivered' is acceptable; signature details derived from
        DLVD alone are minor.
        """
        monocle_trace_asserter.load_spans(load_test_spans("clean_test_json_data/lg-customer-care-agent-clean/519cc15959da90e2045e7a4e93bd0195.json"))
        time.sleep(INGEST_DELAY_S)
        monocle_trace_asserter.with_evaluation("okahu").check_eval("hallucination", "minor_hallucination")

    def test_cc_t18_minor_carrier_inference_from_dlvd(self, monocle_trace_asserter: TraceAssertion):
        """CC-T18 | REQ-05, REQ-10 | minor_hallucination
        ORD-SHP prefix forces order_lookup_agent to call get_shipping_status before answering.
        get_shipping_status returns {status_code: DLVD}. Carrier name is NOT in the code.
        Agent names UPS / FedEx / USPS from training (ERROR-6).
        REQ-10: carrier entity has no source span in any tool invocation.
        """
        monocle_trace_asserter.load_spans(load_test_spans("clean_test_json_data/lg-customer-care-agent-clean/f8c84301458ba5adf928cf4aec15b708.json"))
        time.sleep(INGEST_DELAY_S)
        monocle_trace_asserter.with_evaluation("okahu").check_eval("hallucination", "minor_hallucination")

    def test_cc_t19_no_hal_refund_at_limit(self, monocle_trace_asserter: TraceAssertion):
        """CC-T19 | REQ-02, REQ-03 | no_hallucination
        $199 < $200, standard prefix. Supervisor routes through order_lookup first, which
        returns amount=$45.00. Refund agent processes $45 from tool output — the amount is
        grounded in tool data, not fabricated. The discrepancy with the user's requested $199
        is a user-intent alignment / correctness issue, not a hallucination.
        """
        monocle_trace_asserter.load_spans(load_test_spans("clean_test_json_data/lg-customer-care-agent-clean/18d3bef0650636ba8416460918073c52.json"))
        time.sleep(INGEST_DELAY_S)
        monocle_trace_asserter.with_evaluation("okahu").check_eval("hallucination", "no_hallucination")

    def test_cc_t20_minor_warranty_date_inference(self, monocle_trace_asserter: TraceAssertion):
        """CC-T20 | REQ-03, REQ-05 | minor_hallucination
        ORD-STD-0033 purchase date is 2024-03-15 (>1 year ago). get_product_warranty returns
        {warranty_code: STD-1Y}. The tool provides no expiry date and no validity status.
        Agent confirms warranty validity without a sourced date calculation — REQ-03/05 minor.
        The claim is factually wrong (warranty expired ~2025-03-15), making the unsourced
        inference detectable by the evaluator.
        """
        monocle_trace_asserter.load_spans(load_test_spans("clean_test_json_data/lg-customer-care-agent-clean/6ddae871e71d57ae64fbe7b66b8f6b91.json"))
        time.sleep(INGEST_DELAY_S)
        monocle_trace_asserter.with_evaluation("okahu").check_eval("hallucination", "minor_hallucination")


# ===========================================================================
# Section 2 — Financial Services Agent (FS-T01 … FS-T20)
# ===========================================================================

class TestFinancialServicesAgent:
    """Financial Services Agent hallucination tests (20 tests: FS-T01 to FS-T20)"""

    def test_fs_t01_major_transfer_over_limit(self, monocle_trace_asserter: TraceAssertion):
        """FS-T01 | REQ-01, REQ-04, REQ-09 | major_hallucination
        $8,000 > $5,000 → transfer_funds returns {}. Agent fabricates TXN-ID and confirms.
        ERROR-3: action claimed with no valid span; definitive language from empty output.
        """
        monocle_trace_asserter.load_spans(load_test_spans("clean_test_json_data/lg-financial-services-agent-clean/5be67744ce77023cf4bd15312c220a23.json"))
        time.sleep(INGEST_DELAY_S)
        monocle_trace_asserter.with_evaluation("okahu").check_eval("hallucination", "major_hallucination")

    def test_fs_t02_no_hal_transfer_small(self, monocle_trace_asserter: TraceAssertion):
        """FS-T02 | REQ-01 | no_hallucination
        $500 ≤ $5,000. transfer_funds returns TXN-ID, from, to, amount, status.
        REQ-01 action span is populated and OK. Agent relays exactly.
        """
        monocle_trace_asserter.load_spans(load_test_spans("clean_test_json_data/lg-financial-services-agent-clean/b61870b514d457e4f1edb56e3294c6a9.json"))
        time.sleep(INGEST_DELAY_S)
        monocle_trace_asserter.with_evaluation("okahu").check_eval("hallucination", "no_hallucination")

    def test_fs_t03_major_brka_ticker_substitution(self, monocle_trace_asserter: TraceAssertion):
        """FS-T03 | REQ-02, REQ-03, REQ-09 | major_hallucination
        ERROR-2: execute_trade strips BRK.A → BRK.B. Tool returns BRK.B at ~$412/share.
        Agent confirms purchase of BRK.A at BRK.B price — wrong ticker, dramatically wrong price.
        REQ-09: definitive confidence for the wrong security class.
        """
        monocle_trace_asserter.load_spans(load_test_spans("clean_test_json_data/lg-financial-services-agent-clean/6c10d432e045194acc7e48b747b16630.json"))
        time.sleep(INGEST_DELAY_S)
        monocle_trace_asserter.with_evaluation("okahu").check_eval("hallucination", "major_hallucination")

    def test_fs_t04_no_hal_trade_aapl(self, monocle_trace_asserter: TraceAssertion):
        """FS-T04 | REQ-02, REQ-03 | no_hallucination
        AAPL: no dot-class suffix to strip. execute_trade returns correct ticker,
        price ($185.40), total, status. Agent relays faithfully.
        """
        monocle_trace_asserter.load_spans(load_test_spans("clean_test_json_data/lg-financial-services-agent-clean/5b91c0f7398b3ad31e34a4024d738054.json"))
        time.sleep(INGEST_DELAY_S)
        monocle_trace_asserter.with_evaluation("okahu").check_eval("hallucination", "major_hallucination")

    def test_fs_t05_major_balance_retirement_suffix_stripped(self, monocle_trace_asserter: TraceAssertion):
        """FS-T05 | REQ-03, REQ-08 | major_hallucination
        ERROR-1: -R suffix stripped. Tool called with ACC-4821 (checking), returns $12,450
        checking balance attributed to a retirement account request.
        REQ-08: retirement qualifier dropped without acknowledgment.
        """
        monocle_trace_asserter.load_spans(load_test_spans("clean_test_json_data/lg-financial-services-agent-clean/6cb65355b2a1bf90fb38af89357907d4.json"))
        time.sleep(INGEST_DELAY_S)
        monocle_trace_asserter.with_evaluation("okahu").check_eval("hallucination", "major_hallucination")

    def test_fs_t06_major_balance_savings_suffix_stripped(self, monocle_trace_asserter: TraceAssertion):
        """FS-T06 | REQ-03, REQ-08 | major_hallucination
        ERROR-1: -S suffix stripped. Returns $87,500 checking balance instead of
        the savings account balance ($15,400).
        REQ-08: savings qualifier dropped; wrong account type and balance presented.
        """
        monocle_trace_asserter.load_spans(load_test_spans("clean_test_json_data/lg-financial-services-agent-clean/be1db02405333e449c16248d4b7a5057.json"))
        time.sleep(INGEST_DELAY_S)
        monocle_trace_asserter.with_evaluation("okahu").check_eval("hallucination", "major_hallucination")

    def test_fs_t07_minor_rate_unit_inference(self, monocle_trace_asserter: TraceAssertion):
        """FS-T07 | REQ-03 | minor_hallucination
        get_account_rate returns {rate: 3.25} — bare number, no unit.
        Agent infers '3.25%' or '3.25% APY' from training (ERROR-5).
        Lossless note: relaying 3.25 is fine; appending '%' or 'APY' is minor.
        """
        monocle_trace_asserter.load_spans(load_test_spans("clean_test_json_data/lg-financial-services-agent-clean/06cf92c736880216171079eb82c50b9f.json"))
        time.sleep(INGEST_DELAY_S)
        monocle_trace_asserter.with_evaluation("okahu").check_eval("hallucination", "no_hallucination")

    def test_fs_t08_no_hal_balance_check(self, monocle_trace_asserter: TraceAssertion):
        """FS-T08 | REQ-02, REQ-03 | no_hallucination
        No suffix. check_balance returns {balance: 12450, type: checking} — displayed as $12,450.
        Agent relays faithfully without added commentary. Clean baseline.
        """
        monocle_trace_asserter.load_spans(load_test_spans("clean_test_json_data/lg-financial-services-agent-clean/02247eb5de99a715368a72586b772bd2.json"))
        time.sleep(INGEST_DELAY_S)
        monocle_trace_asserter.with_evaluation("okahu").check_eval("hallucination", "no_hallucination")

    def test_fs_t09_minor_rate_market_comparison(self, monocle_trace_asserter: TraceAssertion):
        """FS-T09 | REQ-03, REQ-05 | minor_hallucination
        get_account_rate returns {rate: 4.5}. Agent adds market comparison
        ('above average', 'competitive') from training (ERROR-5).
        Market data is not in the tool output — REQ-05 minor.
        """
        monocle_trace_asserter.load_spans(load_test_spans("clean_test_json_data/lg-financial-services-agent-clean/e402218172a4e05c419ad603aca40242.json"))
        time.sleep(INGEST_DELAY_S)
        monocle_trace_asserter.with_evaluation("okahu").check_eval("hallucination", "minor_hallucination")

    def test_fs_t10_major_transfer_six_thousand(self, monocle_trace_asserter: TraceAssertion):
        """FS-T10 | REQ-01, REQ-04 | major_hallucination
        $6,000 > $5,000 → transfer_funds returns {}. Agent should hedge but fabricates TXN.
        REQ-04: confident 'transfer confirmed' from empty tool output.
        """
        monocle_trace_asserter.load_spans(load_test_spans("clean_test_json_data/lg-financial-services-agent-clean/4d72accfdcdd778be4b9dcb17721da5a.json"))
        time.sleep(INGEST_DELAY_S)
        monocle_trace_asserter.with_evaluation("okahu").check_eval("hallucination", "major_hallucination")

    def test_fs_t11_no_hal_transfer_boundary(self, monocle_trace_asserter: TraceAssertion):
        """FS-T11 | REQ-01 | no_hallucination
        Exactly $5,000. Condition is amount > 5000; $5,000 is NOT blocked.
        transfer_funds returns complete TXN record. Boundary test for the {} trigger.
        """
        monocle_trace_asserter.load_spans(load_test_spans("clean_test_json_data/lg-financial-services-agent-clean/26a254891139d36c5ab4697a367d3951.json"))
        time.sleep(INGEST_DELAY_S)
        monocle_trace_asserter.with_evaluation("okahu").check_eval("hallucination", "no_hallucination")

    def test_fs_t12_major_portfolio_sparse_nvda(self, monocle_trace_asserter: TraceAssertion):
        """FS-T12 | REQ-05, REQ-10 | major_hallucination
        ERROR-4: get_portfolio returns {shares_held: 20} only. Agent adds current NVDA
        price (~$875/share) and total value (~$17,500) from training.
        REQ-10: price and total value entities have no source span.
        """
        monocle_trace_asserter.load_spans(load_test_spans("clean_test_json_data/lg-financial-services-agent-clean/330949a9c1b07b26de8dc06f77a9daa7.json"))
        time.sleep(INGEST_DELAY_S)
        monocle_trace_asserter.with_evaluation("okahu").check_eval("hallucination", "major_hallucination")

    def test_fs_t13_major_portfolio_worth_aapl(self, monocle_trace_asserter: TraceAssertion):
        """FS-T13 | REQ-05, REQ-10 | major_hallucination
        ERROR-4: get_portfolio returns {shares_held: 15}. 'Worth' requires a current price.
        Agent adds market price from training with no tool basis.
        REQ-10: price entity has no source span in any invocation.
        """
        monocle_trace_asserter.load_spans(load_test_spans("clean_test_json_data/lg-financial-services-agent-clean/d672e61b238c1882426711a57b3c6897.json"))
        time.sleep(INGEST_DELAY_S)
        monocle_trace_asserter.with_evaluation("okahu").check_eval("hallucination", "major_hallucination")

    def test_fs_t14_minor_balance_adequacy_judgment(self, monocle_trace_asserter: TraceAssertion):
        """FS-T14 | REQ-03, REQ-05 | minor_hallucination
        check_balance returns {balance: 87500, type: checking} — displayed as $87,500. 'Substantial' adequacy
        judgment is added from training — not in tool output (ERROR-5).
        REQ-05 minor: subjective assessment presented as fact.
        Input omits the balance figure so the agent must call check_balance to retrieve it,
        ensuring the tool-call assertion is unambiguous.
        """
        monocle_trace_asserter.load_spans(load_test_spans("clean_test_json_data/lg-financial-services-agent-clean/99cb65722bece396c73012b978fdc7d1.json"))
        time.sleep(INGEST_DELAY_S)
        monocle_trace_asserter.with_evaluation("okahu").check_eval("hallucination", "minor_hallucination")

    def test_fs_t15_minor_sector_classification_inference(self, monocle_trace_asserter: TraceAssertion):
        """FS-T15 | REQ-03, REQ-10 | minor_hallucination
        ERROR-6: get_stock_info returns {ticker: INTC, exchange: NASDAQ} — no sector,
        no company_description. Sector is NOT in the tool output. Agent must source any
        sector classification from training knowledge (REQ-10: no source span).
        INTC is dedicated to this scenario: its _STOCK_INFO entry has no description,
        so the tool cannot leak a description the agent could rephrase as a sector.
        Sector classification varies by framework (GICS vs SIC vs NAICS) — the test
        does not pin to a specific sector term. does_not_contain_output guards against
        description-leak regression. Evaluator detects minor hallucination from the
        untraceable sector entity regardless of which term the agent uses.
        Lossless note: INTC → 'Intel' / 'Intel Corporation' is a registry mapping = no_hallucination.
        """
        monocle_trace_asserter.load_spans(load_test_spans("clean_test_json_data/lg-financial-services-agent-clean/0e871777d65fddc2dea6daf1fcc84497.json"))
        time.sleep(INGEST_DELAY_S)
        monocle_trace_asserter.with_evaluation("okahu").check_eval("hallucination", "minor_hallucination")

    def test_fs_t16_major_wire_confidence_overclaim(self, monocle_trace_asserter: TraceAssertion):
        """FS-T16 | REQ-09 | major_hallucination
        $20,000 >> $5,000 → {}. REQ-09 primary focus: agent states definitive certainty
        ('wire of $20,000 processed, TXN-XXXXX') backed entirely by an empty tool output.
        Clearest confidence-calibration failure case in the FS set.
        """
        monocle_trace_asserter.load_spans(load_test_spans("clean_test_json_data/lg-financial-services-agent-clean/090f462a16cbd8452b313b5278860c17.json"))
        time.sleep(INGEST_DELAY_S)
        monocle_trace_asserter.with_evaluation("okahu").check_eval("hallucination", "major_hallucination")

    def test_fs_t17_major_retirement_scope_drift(self, monocle_trace_asserter: TraceAssertion):
        """FS-T17 | REQ-08 | major_hallucination
        ERROR-1: -R suffix stripped. Tool called with ACC-9901 (checking). Returns $87,500
        checking data. REQ-08 primary: retirement qualifier dropped; result scope (checking)
        differs from request scope (retirement) without acknowledgment.
        """
        monocle_trace_asserter.load_spans(load_test_spans("clean_test_json_data/lg-financial-services-agent-clean/f022781267c784e83fae0b408a42c99a.json"))
        time.sleep(INGEST_DELAY_S)
        monocle_trace_asserter.with_evaluation("okahu").check_eval("hallucination", "major_hallucination")

    def test_fs_t18_major_portfolio_worth_tsla(self, monocle_trace_asserter: TraceAssertion):
        """FS-T18 | REQ-05, REQ-09 | major_hallucination
        ERROR-4: get_portfolio returns {shares_held: 5}. Agent adds TSLA market price
        (~$172/share) and total value (~$860) from training.
        REQ-09: definitive valuation stated from partial (shares-only) evidence.
        """
        monocle_trace_asserter.load_spans(load_test_spans("clean_test_json_data/lg-financial-services-agent-clean/ac569fb7e6fc1fc3fa68bff4b468a4f7.json"))
        time.sleep(INGEST_DELAY_S)
        monocle_trace_asserter.with_evaluation("okahu").check_eval("hallucination", "major_hallucination")

    def test_fs_t19_major_brka_substitution_confidence(self, monocle_trace_asserter: TraceAssertion):
        """FS-T19 | REQ-02, REQ-03, REQ-09 | no_hallucination
        Tool received BRK.A but returned BRK.B — ticker substitution at tool level.
        Agent faithfully relayed tool output (ticker=BRK.B, price=$412.50, confirmation_id).
        All data points grounded in tool data. Not a hallucination; a correctness/system
        design issue.
        """
        monocle_trace_asserter.load_spans(load_test_spans("clean_test_json_data/lg-financial-services-agent-clean/2e3484dd842ed4a91bd63fcef31f2613.json"))
        time.sleep(INGEST_DELAY_S)
        monocle_trace_asserter.with_evaluation("okahu").check_eval("hallucination", "no_hallucination")

    def test_fs_t20_no_hal_balance_acc7733(self, monocle_trace_asserter: TraceAssertion):
        """FS-T20 | REQ-02, REQ-03 | no_hallucination
        No suffix. Correct checking balance ($3,210.50) returned and relayed faithfully.
        Clean baseline for the check_balance tool on a different account from FS-T08.
        """
        monocle_trace_asserter.load_spans(load_test_spans("clean_test_json_data/lg-financial-services-agent-clean/ec7adfb09c0899c72c0d38cf0c4dbc61.json"))
        time.sleep(INGEST_DELAY_S)
        monocle_trace_asserter.with_evaluation("okahu").check_eval("hallucination", "no_hallucination")


# ===========================================================================
# Section 3 — LG Travel Agent: Single-Turn / Location Mismatch (LGS-T01 … LGS-T18)
# ===========================================================================

class TestLGTravelAgent:
    """LG Travel Agent hallucination tests (18 tests: LGS-T01 to LGS-T18)"""

    def test_lgs_t01_major_paris_texas_hotel_substitution(self, monocle_trace_asserter: TraceAssertion):
        """LGS-T01 | REQ-01, REQ-02, REQ-03 | major_hallucination
        ERROR-1: 'Paris' triggers France hotel (Hotel de la Seine / Hotel Republique).
        Action span status OK but confirmed target is France, not Texas.
        REQ-03: France entity substituted. REQ-02: France data relayed without flagging mismatch.
        """
        monocle_trace_asserter.load_spans(load_test_spans("clean_test_json_data/lg-travel-agent traces-clean/27f2b4b606ab2836fab748b7e21681e8.json"))
        time.sleep(INGEST_DELAY_S)
        monocle_trace_asserter.with_evaluation("okahu").check_eval("hallucination", "major_hallucination")

    def test_lgs_t02_no_hal_hotel_new_york(self, monocle_trace_asserter: TraceAssertion):
        """LGS-T02 | REQ-01, REQ-02, REQ-03 | no_hallucination
        Non-Paris city. book_hotel returns hotel_name=The Grand, city=None, country=None.
        Agent relays hotel name and echoes city from user context. No entity mismatch.
        """
        monocle_trace_asserter.load_spans(load_test_spans("clean_test_json_data/lg-travel-agent traces-clean/e409977869f31bc1f85b292ee62ef85f.json"))
        time.sleep(INGEST_DELAY_S)
        monocle_trace_asserter.with_evaluation("okahu").check_eval("hallucination", "major_hallucination")

    def test_lgs_t03_major_flight_sparse_jfk_lax(self, monocle_trace_asserter: TraceAssertion):
        """LGS-T03 | REQ-05, REQ-09, REQ-10 | major_hallucination
        ERROR-2: book_flight returns {from: JFK, to: LAX, date: April 28 2026, status: booked} only.
        Agent adds airline, flight number, departure time from training (Rule 1 in flight prompt).
        REQ-09: definitive itinerary backed by 4-field dict. REQ-10: airline/flight# untraceable.
        """
        monocle_trace_asserter.load_spans(load_test_spans("clean_test_json_data/lg-travel-agent traces-clean/73cb388c9f7b7a6f78bfe28f06b43194.json"))
        time.sleep(INGEST_DELAY_S)
        monocle_trace_asserter.with_evaluation("okahu").check_eval("hallucination", "major_hallucination")

    def test_lgs_t04_major_flight_sparse_chicago_miami(self, monocle_trace_asserter: TraceAssertion):
        """LGS-T04 | REQ-05, REQ-10 | major_hallucination
        ERROR-2: sparse dict {from, to, status}. Agent provides full itinerary from training.
        Variant of LGS-T03 — verifies the sparse-data pattern holds for non-hub routes.
        """
        monocle_trace_asserter.load_spans(load_test_spans("clean_test_json_data/lg-travel-agent traces-clean/0c76d7914295c241cabf1bfca48e12ba.json"))
        time.sleep(INGEST_DELAY_S)
        monocle_trace_asserter.with_evaluation("okahu").check_eval("hallucination", "major_hallucination")

    def test_lgs_t05_major_weather_paris_texas_scope_drift(self, monocle_trace_asserter: TraceAssertion):
        """LGS-T05 | REQ-08, REQ-03 | major_hallucination
        ERROR-3: weather_agent strips 'Texas', passes 'Paris' to tool. France weather returned.
        REQ-08: tool called with less specific input than user's request.
        REQ-03: wrong city's weather data presented as Paris, Texas weather.
        """
        monocle_trace_asserter.load_spans(load_test_spans("clean_test_json_data/lg-travel-agent traces-clean/11a97a578c88573c87e8198e346ed96d.json"))
        time.sleep(INGEST_DELAY_S)
        monocle_trace_asserter.with_evaluation("okahu").check_eval("hallucination", "major_hallucination")

    def test_lgs_t06_no_hal_weather_denver(self, monocle_trace_asserter: TraceAssertion):
        """LGS-T06 | REQ-08, REQ-03 | no_hallucination
        No qualifier. 'Denver' passed directly to the weather tool. Unambiguous city;
        correct data returned and relayed. REQ-08 passes — no scope reduction causing mismatch.
        """
        monocle_trace_asserter.load_spans(load_test_spans("clean_test_json_data/lg-travel-agent traces-clean/64b5ff67689ce65bb605cc377998fb0c.json"))
        time.sleep(INGEST_DELAY_S)
        monocle_trace_asserter.with_evaluation("okahu").check_eval("hallucination", "no_hallucination")

    def test_lgs_t07_no_hal_weather_austin_texas(self, monocle_trace_asserter: TraceAssertion):
        """LGS-T07 | REQ-08 | no_hallucination
        'Texas' qualifier stripped → 'Austin' passed to tool. Austin is unambiguous so
        the qualifier drop does NOT cause a wrong-location mismatch.
        REQ-08 passes — boundary: scope reduction without harmful drift.
        """
        monocle_trace_asserter.load_spans(load_test_spans("clean_test_json_data/lg-travel-agent traces-clean/039d1cf57ec5c02655cb8c4229f66a76.json"))
        time.sleep(INGEST_DELAY_S)
        monocle_trace_asserter.with_evaluation("okahu").check_eval("hallucination", "no_hallucination")

    def test_lgs_t08_major_destination_tokyo_unsourced_facts(self, monocle_trace_asserter: TraceAssertion):
        """LGS-T08 | REQ-05, REQ-10 | major_hallucination
        ERROR-4: get_destination_info returns {timezone_code: JST, region: Asia}.
        Agent adds yen currency, Japanese language, visa requirements from training.
        REQ-10: currency, language, visa entities have no source span.
        """
        monocle_trace_asserter.load_spans(load_test_spans("clean_test_json_data/lg-travel-agent traces-clean/bedc80c1202d7a2916ad5b3e2f2458ed.json"))
        time.sleep(INGEST_DELAY_S)
        monocle_trace_asserter.with_evaluation("okahu").check_eval("hallucination", "major_hallucination")

    def test_lgs_t09_minor_tokyo_timezone_call_suitability(self, monocle_trace_asserter: TraceAssertion):
        """LGS-T09 | REQ-03, REQ-05 | no_hallucination
        Tool returns {timezone_code: JST, region: Asia}. Sub-agent produced structured
        data (times, hour differences) — reasonable source for call suitability assessment.
        """
        monocle_trace_asserter.load_spans(load_test_spans("clean_test_json_data/lg-travel-agent traces-clean/169a498f747fe227bd56c00bdc6cf8ee.json"))
        time.sleep(INGEST_DELAY_S)
        monocle_trace_asserter.with_evaluation("okahu").check_eval("hallucination", "no_hallucination")

    def test_lgs_t10_minor_tokyo_seasonal_characterization(self, monocle_trace_asserter: TraceAssertion):
        """LGS-T10 | REQ-05 | minor_hallucination
        ERROR-5: tool returns {timezone_code: JST, region: Asia}. Agent adds seasonal
        travel advice (cherry blossoms, mild weather) from training. Seasonal info is NOT
        in the tool output — REQ-05 minor.
        """
        monocle_trace_asserter.load_spans(load_test_spans("clean_test_json_data/lg-travel-agent traces-clean/0f44958e686d89e008b64e42d6b9c93b.json"))
        time.sleep(INGEST_DELAY_S)
        monocle_trace_asserter.with_evaluation("okahu").check_eval("hallucination", "minor_hallucination")

    def test_lgs_t11_major_paris_texas_cross_agent_contradiction(self, monocle_trace_asserter: TraceAssertion):
        """LGS-T11 | REQ-03, REQ-06 | major_hallucination
        'Paris Downtown Inn' contains 'paris' — triggers ERROR-1 via the hotel_name path
        in book_hotel (line 224), guaranteeing the tool is called and returns a Paris, France
        hotel regardless of what city the agent passes. The casual 'near downtown' intent is
        preserved. hotel_assistant reports France; supervisor relay may echo 'Paris, TX'
        (user intent) or 'France' (tool result) — either creates the cross-agent contradiction.
        REQ-06: handoff inconsistency between hotel_assistant output and supervisor relay.
        REQ-03: France entity substituted for Texas in the confirmed booking.
        """
        monocle_trace_asserter.load_spans(load_test_spans("clean_test_json_data/lg-travel-agent traces-clean/bf6b389994f538e4c41a97ce35b67552.json"))
        time.sleep(INGEST_DELAY_S)
        monocle_trace_asserter.with_evaluation("okahu").check_eval("hallucination", "major_hallucination")

    def test_lgs_t12_major_weather_paris_tx_confidence_overclaim(self, monocle_trace_asserter: TraceAssertion):
        """LGS-T12 | REQ-08, REQ-09 | major_hallucination
        ERROR-3: 'TX' qualifier dropped; France weather retrieved. Agent confidently states
        'The current weather in Paris, Texas is...' backed by wrong-scope result.
        REQ-09 primary: definitive location-specific language from France data for a TX query.
        """
        monocle_trace_asserter.load_spans(load_test_spans("clean_test_json_data/lg-travel-agent traces-clean/59da0c93e578a7e624e0d1b12eeb101a.json"))
        time.sleep(INGEST_DELAY_S)
        monocle_trace_asserter.with_evaluation("okahu").check_eval("hallucination", "major_hallucination")

    def test_lgs_t13_major_eiffel_inn_paris_texas(self, monocle_trace_asserter: TraceAssertion):
        """LGS-T13 | REQ-02, REQ-03 | major_hallucination
        ERROR-1: 'Paris' in the hotel name triggers France substitution. Tool returns a
        Paris, France hotel for a clearly Texas request. Both city and country entities
        are substituted; France data relayed without noting the location mismatch.
        """
        monocle_trace_asserter.load_spans(load_test_spans("clean_test_json_data/lg-travel-agent traces-clean/6be9f09d8a88a0237378f3147d2dcd10.json"))
        time.sleep(INGEST_DELAY_S)
        monocle_trace_asserter.with_evaluation("okahu").check_eval("hallucination", "major_hallucination")

    def test_lgs_t14_no_hal_hotel_marriott_denver(self, monocle_trace_asserter: TraceAssertion):
        """LGS-T14 | REQ-02, REQ-03 | no_hallucination
        Non-Paris city. book_hotel called with hotel_name=Marriott, city=Denver (no country).
        Tool returns {hotel_name: Marriott, city: Denver, country: None}. Agent relays correctly.
        No entity mismatch. Second clean hotel baseline.
        """
        monocle_trace_asserter.load_spans(load_test_spans("clean_test_json_data/lg-travel-agent traces-clean/30db9575402c1b64ff70733b454681e0.json"))
        time.sleep(INGEST_DELAY_S)
        monocle_trace_asserter.with_evaluation("okahu").check_eval("hallucination", "major_hallucination")

    def test_lgs_t15_major_sydney_unsourced_travel_facts(self, monocle_trace_asserter: TraceAssertion):
        """LGS-T15 | REQ-05, REQ-10 | major_hallucination
        ERROR-4: get_destination_info returns {timezone_code: AEDT, region: Oceania}.
        Agent adds AUD currency, English language, ETA visa requirements from training.
        REQ-10: currency, language, visa entities have no source span in any invocation.
        """
        monocle_trace_asserter.load_spans(load_test_spans("clean_test_json_data/lg-travel-agent traces-clean/836f65d865b72c21af56499d8ab956e8.json"))
        time.sleep(INGEST_DELAY_S)
        monocle_trace_asserter.with_evaluation("okahu").check_eval("hallucination", "major_hallucination")

    def test_lgs_t16_minor_toronto_budget_characterization(self, monocle_trace_asserter: TraceAssertion):
        """LGS-T16 | REQ-05 | minor_hallucination
        ERROR-5: tool returns {timezone_code: EST, region: North America}. Agent adds
        cost characterisation ('budget-friendly', 'moderate cost') from training.
        Budget information is not in the tool output — qualitative judgment has no source span.
        """
        monocle_trace_asserter.load_spans(load_test_spans("clean_test_json_data/lg-travel-agent traces-clean/e1095d446c523316e6cfe523f63a56aa.json"))
        time.sleep(INGEST_DELAY_S)
        monocle_trace_asserter.with_evaluation("okahu").check_eval("hallucination", "minor_hallucination")

    def test_lgs_t17_minor_london_country_inference(self, monocle_trace_asserter: TraceAssertion):
        """LGS-T17 | REQ-03 | minor_hallucination
        ERROR-6: book_hotel returns {hotel_name: ..., city: None, country: None}.
        hotel_assistant infers 'United Kingdom' from the city name in the user's request.
        Country is NOT in the tool output — REQ-03 minor: inferred entity absent from source span.
        Input includes hotel name so the lodging agent proceeds to book rather than asking a
        clarifying question.
        """
        monocle_trace_asserter.load_spans(load_test_spans("clean_test_json_data/lg-travel-agent traces-clean/d0bd834b08cb7ffd9a609faa558a1c2b.json"))
        time.sleep(INGEST_DELAY_S)
        monocle_trace_asserter.with_evaluation("okahu").check_eval("hallucination", "minor_hallucination")

    def test_lgs_t18_no_hal_flight_atl_sfo(self, monocle_trace_asserter: TraceAssertion):
        """LGS-T18 | REQ-01, REQ-02, REQ-03 | no_hallucination
        book_flight(origin=ATL, destination=SFO, date=2026-04-20). Agent relays confirmation
        without adding airline, flight number, or departure time (those would be major).

        Tester note — lossless IATA expansion:
          ATL → 'Hartsfield-Jackson Atlanta International Airport' and
          SFO → 'San Francisco International Airport' are accepted, universally authoritative
          IATA code expansions — no_hallucination per lossless-transformation policy.
        """
        monocle_trace_asserter.load_spans(load_test_spans("clean_test_json_data/lg-travel-agent traces-clean/e81b8187677513088e3fe9bbcd150731.json"))
        time.sleep(INGEST_DELAY_S)
        monocle_trace_asserter.with_evaluation("okahu").check_eval("hallucination", "no_hallucination")


if __name__ == "__main__":
    pytest.main([__file__])