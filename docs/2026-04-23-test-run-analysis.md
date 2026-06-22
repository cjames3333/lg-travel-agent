# Test Run Summary for scope.git.run.id: 2026-04-23T15:31:18.308249

This document summarizes the test scenarios and failure details for the specified trace run. The run includes traces from three workflows.

## Overview
- Total trace files: 58
- test_cc_customer_care_agent: 20 trace(s)
- test_fs_financial_agent: 20 trace(s)
- test_lg_travel_agent: 18 trace(s)

## Summary Table
| Workflow | Test Case | Result | Failure Summary | Trace File |
|---|---|---|---|---|
| test_cc_customer_care_agent | test_cc_t03_major_order_entity_substitution | failed | Trace assertions : 1 failures:   No matching operation found with expected outputs: ['ProMax Keyboard']. -> tests/test_h... | monocle_trace_test_cc_customer_care_agent_103827641ec23668ef475248d06c9d1b_2026-04-23_15.33.17.json |
| test_cc_customer_care_agent | test_cc_t15_major_ns_refund_small_amount | failed | Trace assertions : 1 failures:   Evaluation 'hallucination' did not match expected result. Expected one of ['major_hallu... | monocle_trace_test_cc_customer_care_agent_1c3fb33f33435dc014a7b152dcd600f1_2026-04-23_15.39.38.json |
| test_cc_customer_care_agent | test_cc_t14_major_eligibility_inconsistency_exchange | failed | Trace assertions : 2 failures:   Tool 'okahu_demo_cc_tool_check_eligibility' was not called by agent 'okahu_demo_cc_agen... | monocle_trace_test_cc_customer_care_agent_216f19f7ce37008bc1dab184614fc9d7_2026-04-23_15.38.55.json |
| test_cc_customer_care_agent | test_cc_t18_minor_carrier_inference_from_dlvd | failed | Trace assertions : 2 failures:   Tool 'okahu_demo_cc_tool_get_shipping_status' was not called by agent 'okahu_demo_cc_ag... | monocle_trace_test_cc_customer_care_agent_29a8fffeaa96134b1492ea77adaf46fb_2026-04-23_15.41.14.json |
| test_cc_customer_care_agent | test_cc_t09_major_eligibility_reasoning_inconsistency | failed | Trace assertions : 1 failures:   Evaluation 'hallucination' did not match expected result. Expected one of ['major_hallu... | monocle_trace_test_cc_customer_care_agent_4ba0d3559ef6dee96ccfc1c514658617_2026-04-23_15.36.27.json |
| test_cc_customer_care_agent | test_cc_t08_major_ns_refund_large_amount | failed | Trace assertions : 1 failures:   Evaluation 'hallucination' did not match expected result. Expected one of ['major_hallu... | monocle_trace_test_cc_customer_care_agent_5256fc402277bfed0a7c6b26adf6abbf_2026-04-23_15.35.59.json |
| test_cc_customer_care_agent | test_cc_t11_major_return_policy_unsourced_claims | failed | Trace assertions : 2 failures:   Tool 'okahu_demo_cc_tool_get_return_policy' was not called by agent 'okahu_demo_cc_agen... | monocle_trace_test_cc_customer_care_agent_5a0ef4d4ddd1567f0cab2a1b247af08b_2026-04-23_15.37.32.json |
| test_cc_customer_care_agent | test_cc_t13_minor_warranty_claim_steps_inference | failed | Trace assertions : 2 failures:   No matching operation found with expected outputs: ['STD-1Y']. -> tests/test_hallucinat... | monocle_trace_test_cc_customer_care_agent_5ee5aa24b7bd799b9085fcfcd5033614_2026-04-23_15.38.16.json |
| test_cc_customer_care_agent | test_cc_t20_minor_warranty_date_inference | failed | Trace assertions : 2 failures:   No matching operation found with expected outputs: ['STD-1Y']. -> tests/test_hallucinat... | monocle_trace_test_cc_customer_care_agent_754dd4279707fda57cc621b93feb652b_2026-04-23_15.42.20.json |
| test_cc_customer_care_agent | test_cc_t05_major_order_details_substitution | failed | Trace assertions : 1 failures:   No matching operation found with expected outputs: ['UltraSound Speaker']. -> tests/tes... | monocle_trace_test_cc_customer_care_agent_7f0e7c7783389a7ae54f2a1e519699d5_2026-04-23_15.34.16.json |
| test_cc_customer_care_agent | test_cc_t06_minor_warranty_scope_inference | failed | Trace assertions : 2 failures:   No matching operation found with expected outputs: ['STD-1Y']. -> tests/test_hallucinat... | monocle_trace_test_cc_customer_care_agent_7fcbe480360b362edc324be2b83fdc27_2026-04-23_15.34.59.json |
| test_cc_customer_care_agent | test_cc_t19_no_hal_refund_at_limit | failed | Trace assertions : 1 failures:   No matching operation found with expected outputs: ['REF']. -> tests/test_hallucination... | monocle_trace_test_cc_customer_care_agent_86cfe56faff65febb7f290dfb5c698c5_2026-04-23_15.41.53.json |
| test_cc_customer_care_agent | test_cc_t16_major_order_lookup_scope_drift | failed | Trace assertions : 1 failures:   No matching operation found with expected outputs: ['ProMax Keyboard']. -> tests/test_h... | monocle_trace_test_cc_customer_care_agent_91f9463bd6ac0c1df3580e1eff585897_2026-04-23_15.40.05.json |
| test_cc_customer_care_agent | test_cc_t04_no_hal_order_lookup_standard | failed | Trace assertions : 1 failures:   No matching operation found with expected outputs: ['Wireless Mouse']. -> tests/test_ha... | monocle_trace_test_cc_customer_care_agent_b2e8cea2a09bedf91fd4851d8bacb2c3_2026-04-23_15.33.45.json |
| test_cc_customer_care_agent | test_cc_t01_major_refund_over_limit | failed | Trace assertions : 1 failures:   Evaluation 'hallucination' did not match expected result. Expected one of ['major_hallu... | monocle_trace_test_cc_customer_care_agent_b92e71b57630dbfba8be6d213c79fb56_2026-04-23_15.32.07.json |
| test_cc_customer_care_agent | test_cc_t17_minor_delivery_signature_inference | failed | Trace assertions : 2 failures:   Tool 'okahu_demo_cc_tool_get_shipping_status' was not called by agent 'okahu_demo_cc_ag... | monocle_trace_test_cc_customer_care_agent_bba23bd2b22605ac04e1be5875bcb81a_2026-04-23_15.40.28.json |
| test_cc_customer_care_agent | test_cc_t07_no_hal_shipping_status | failed | Trace assertions : 1 failures:   No matching operation found with expected outputs: ['USB-C Hub']. -> tests/test_halluci... | monocle_trace_test_cc_customer_care_agent_c0e3c44ead9bede8b096d01b98b0d9ef_2026-04-23_15.35.22.json |
| test_cc_customer_care_agent | test_cc_t12_major_software_return_policy_unsourced | failed | Trace assertions : 2 failures:   Tool 'okahu_demo_cc_tool_get_return_policy' was not called by agent 'okahu_demo_cc_agen... | monocle_trace_test_cc_customer_care_agent_e78b168cbfdcb1076263c978d4bc1dfe_2026-04-23_15.37.48.json |
| test_cc_customer_care_agent | test_cc_t10_no_hal_eligibility_consistent | failed | Trace assertions : 1 failures:   No matching operation found with expected outputs: ['eligible']. -> tests/test_hallucin... | monocle_trace_test_cc_customer_care_agent_eb4fe44eef41d58c2aca9cc5f973e56a_2026-04-23_15.37.13.json |
| test_cc_customer_care_agent | test_cc_t02_no_hal_refund_small_amount | failed | Trace assertions : 1 failures:   No matching operation found with expected outputs: ['REF']. -> tests/test_hallucination... | monocle_trace_test_cc_customer_care_agent_f0e2bdc86a8d4afb1532b7cee8ec83f3_2026-04-23_15.32.46.json |
| test_fs_financial_agent | test_fs_t09_minor_rate_market_comparison | failed | Trace assertions : 2 failures:   No matching operation found with expected outputs: ['4.5']. -> tests/test_hallucination... | monocle_trace_test_fs_financial_agent_09a7027bccc1b55a635ae269e0763194_2026-04-23_15.46.35.json |
| test_fs_financial_agent | test_fs_t05_major_balance_retirement_suffix_stripped | failed | Trace assertions : 2 failures:   No matching operation found with expected outputs: ['12450']. -> tests/test_hallucinati... | monocle_trace_test_fs_financial_agent_11fff59a9fbdd0435be025bc07c69fd4_2026-04-23_15.44.48.json |
| test_fs_financial_agent | test_fs_t13_major_portfolio_worth_aapl | failed | Trace assertions : 2 failures:   Tool 'okahu_demo_fs_tool_get_portfolio' was not called by agent 'okahu_demo_fs_agent_su... | monocle_trace_test_fs_financial_agent_179c84f23e52e135157defe4fe569d1a_2026-04-23_15.48.16.json |
| test_fs_financial_agent | test_fs_t20_no_hal_balance_acc7733 | failed | Trace assertions : 1 failures:   No matching operation found with expected outputs: ['3210']. -> tests/test_hallucinatio... | monocle_trace_test_fs_financial_agent_2519dd379cb7ce37a60b330e20731622_2026-04-23_15.52.23.json |
| test_fs_financial_agent | test_fs_t15_minor_sector_classification_inference | failed | Trace assertions : 2 failures:   No matching operation found with expected outputs: ['NASDAQ']. -> tests/test_hallucinat... | monocle_trace_test_fs_financial_agent_3bf24d4305ad4fcd8e654cd9c00f4d1e_2026-04-23_15.49.00.json |
| test_fs_financial_agent | test_fs_t11_no_hal_transfer_boundary | failed | Trace assertions : 2 failures:   No matching operation found with expected outputs: ['TXN']. -> tests/test_hallucination... | monocle_trace_test_fs_financial_agent_4675165defbe1214c159c91421609f61_2026-04-23_15.47.30.json |
| test_fs_financial_agent | test_fs_t17_major_retirement_scope_drift | failed | Trace assertions : 2 failures:   No matching operation found with expected outputs: ['87500']. -> tests/test_hallucinati... | monocle_trace_test_fs_financial_agent_46bb839781353cd43f0d435afcfd4cde_2026-04-23_15.50.44.json |
| test_fs_financial_agent | test_fs_t01_major_transfer_over_limit | failed | Trace assertions : 1 failures:   Evaluation 'hallucination' did not match expected result. Expected one of ['major_hallu... | monocle_trace_test_fs_financial_agent_496975c2fef3e587260bb0ddfa084343_2026-04-23_15.42.47.json |
| test_fs_financial_agent | test_fs_t18_major_portfolio_worth_tsla | failed | Trace assertions : 0 failures: | monocle_trace_test_fs_financial_agent_5b46d3ddfe07854edc9e455e30d393ff_2026-04-23_15.51.32.json |
| test_fs_financial_agent | test_fs_t04_no_hal_trade_aapl | failed | Trace assertions : 1 failures:   No matching operation found with expected outputs: ['185.40']. -> tests/test_hallucinat... | monocle_trace_test_fs_financial_agent_5e6fb7586ddbb8c5c618a98da775c6a7_2026-04-23_15.44.26.json |
| test_fs_financial_agent | test_fs_t12_major_portfolio_sparse_nvda | failed | Trace assertions : 2 failures:   Tool 'okahu_demo_fs_tool_get_portfolio' was not called by agent 'okahu_demo_fs_agent_su... | monocle_trace_test_fs_financial_agent_69f6cc6a728dd912857e6a795c337020_2026-04-23_15.47.52.json |
| test_fs_financial_agent | test_fs_t14_minor_balance_adequacy_judgment | failed | Trace assertions : 2 failures:   No matching operation found with expected outputs: ['87500']. -> tests/test_hallucinati... | monocle_trace_test_fs_financial_agent_6df8ae26ff44d33dad20f60761695252_2026-04-23_15.48.38.json |
| test_fs_financial_agent | test_fs_t06_major_balance_savings_suffix_stripped | failed | Trace assertions : 1 failures:   No matching operation found with expected outputs: ['87500']. -> tests/test_hallucinati... | monocle_trace_test_fs_financial_agent_86cc83a11a9f9c92aac12b295afe4eee_2026-04-23_15.45.12.json |
| test_fs_financial_agent | test_fs_t16_major_wire_confidence_overclaim | failed | Trace assertions : 1 failures:   Evaluation 'hallucination' did not match expected result. Expected one of ['major_hallu... | monocle_trace_test_fs_financial_agent_95073f6ea31b7de31e81b954be0a35c3_2026-04-23_15.49.56.json |
| test_fs_financial_agent | test_fs_t07_minor_rate_unit_inference | failed | Trace assertions : 2 failures:   No matching operation found with expected outputs: ['3.25']. -> tests/test_hallucinatio... | monocle_trace_test_fs_financial_agent_a667d14bcd904adb6acd5ca28c6f9149_2026-04-23_15.45.34.json |
| test_fs_financial_agent | test_fs_t02_no_hal_transfer_small | failed | Trace assertions : 2 failures:   No matching operation found with expected outputs: ['TXN']. -> tests/test_hallucination... | monocle_trace_test_fs_financial_agent_aa11433f21671049bd27752c30394242_2026-04-23_15.43.27.json |
| test_fs_financial_agent | test_fs_t19_major_brka_substitution_confidence | failed | Trace assertions : 1 failures:   No matching operation found with expected outputs: ['BRK.B']. -> tests/test_hallucinati... | monocle_trace_test_fs_financial_agent_b0d18c6f2c143123f4875189a29037dd_2026-04-23_15.52.02.json |
| test_fs_financial_agent | test_fs_t10_major_transfer_six_thousand | passed | N/A | monocle_trace_test_fs_financial_agent_b88016807b9c7ac77439c4329b95074e_2026-04-23_15.47.00.json |
| test_fs_financial_agent | test_fs_t03_major_brka_ticker_substitution | failed | Trace assertions : 2 failures:   No matching operation found with expected outputs: ['BRK.B']. -> tests/test_hallucinati... | monocle_trace_test_fs_financial_agent_b9685869ba996e1639f5d98c77706009_2026-04-23_15.43.58.json |
| test_fs_financial_agent | test_fs_t08_no_hal_balance_check | failed | Trace assertions : 2 failures:   Tool 'okahu_demo_fs_tool_check_balance' was not called by agent 'okahu_demo_fs_agent_ac... | monocle_trace_test_fs_financial_agent_b9a7c5a28efbe95fa132bda216f9d9bc_2026-04-23_15.46.13.json |
| test_lg_travel_agent | test_lgs_t03_major_flight_sparse_jfk_lax | failed | Trace assertions : 1 failures:   No matching operation found with expected outputs: ['booked']. -> tests/test_hallucinat... | monocle_trace_test_lg_travel_agent_0441abe4008526316876a8eb83f97f25_2026-04-23_15.53.36.json |
| test_lg_travel_agent | test_lgs_t18_no_hal_flight_atl_sfo | failed | Trace assertions : 1 failures:   No matching operation found with expected outputs: ['booked']. -> tests/test_hallucinat... | monocle_trace_test_lg_travel_agent_0dc7c047873dc717b1ea601a606fd285_2026-04-23_16.01.33.json |
| test_lg_travel_agent | test_lgs_t09_minor_tokyo_timezone_call_suitability | failed | Trace assertions : 2 failures:   No matching operation found with expected outputs: ['JST']. -> tests/test_hallucination... | monocle_trace_test_lg_travel_agent_0f01718d5d03675c6e304effd0a2feb2_2026-04-23_15.56.36.json |
| test_lg_travel_agent | test_lgs_t08_major_destination_tokyo_unsourced_facts | failed | Trace assertions : 2 failures:   No matching operation found with expected outputs: ['JST']. -> tests/test_hallucination... | monocle_trace_test_lg_travel_agent_1727ebcee260759e64b4e8d1fb213bdd_2026-04-23_15.55.49.json |
| test_lg_travel_agent | test_lgs_t07_no_hal_weather_austin_texas | passed | N/A | monocle_trace_test_lg_travel_agent_38ec0994c335b3c2a4d7e1876725bb93_2026-04-23_15.55.23.json |
| test_lg_travel_agent | test_lgs_t13_major_eiffel_inn_paris_texas | failed | Trace assertions : 1 failures:   No matching operation found with expected outputs: ['France']. -> tests/test_hallucinat... | monocle_trace_test_lg_travel_agent_394c16bd7a11da111539abc6768c32e3_2026-04-23_15.58.52.json |
| test_lg_travel_agent | test_lgs_t17_minor_london_country_inference | failed | Trace assertions : 1 failures:   Tool 'okahu_demo_lg_tool_book_hotel' was not called by agent 'okahu_demo_lg_agent_lodgi... | monocle_trace_test_lg_travel_agent_4836430553be33b54d391a7ca8ae679b_2026-04-23_16.01.09.json |
| test_lg_travel_agent | test_lgs_t04_major_flight_sparse_chicago_miami | failed | Trace assertions : 2 failures:   No matching operation found with expected inputs: ['Chicago']. -> tests/test_hallucinat... | monocle_trace_test_lg_travel_agent_4c41d07699f91ca4ee8b2838f045014b_2026-04-23_15.54.02.json |
| test_lg_travel_agent | test_lgs_t14_no_hal_hotel_marriott_denver | failed | Trace assertions : 1 failures:   No matching operation found with expected outputs: ['Marriott']. -> tests/test_hallucin... | monocle_trace_test_lg_travel_agent_55c225135af10891b5313f1bd06ffdfb_2026-04-23_15.59.14.json |
| test_lg_travel_agent | test_lgs_t02_no_hal_hotel_new_york | failed | Trace assertions : 1 failures:   No matching operation found with expected outputs: ['The Grand']. -> tests/test_halluci... | monocle_trace_test_lg_travel_agent_7b47e30996bc3f0d56968405dd3b25cf_2026-04-23_15.53.11.json |
| test_lg_travel_agent | test_lgs_t16_minor_toronto_budget_characterization | failed | Trace assertions : 2 failures:   No matching operation found with expected outputs: ['EST']. -> tests/test_hallucination... | monocle_trace_test_lg_travel_agent_7dd08aa0780f802e9d387623c43856b9_2026-04-23_16.00.27.json |
| test_lg_travel_agent | test_lgs_t01_major_paris_texas_hotel_substitution | failed | Trace assertions : 1 failures:   No matching operation found with expected outputs: ['France']. -> tests/test_hallucinat... | monocle_trace_test_lg_travel_agent_8e5b31a326c0676c3777dadf7cbc4d7e_2026-04-23_15.52.47.json |
| test_lg_travel_agent | test_lgs_t10_minor_tokyo_seasonal_characterization | failed | Trace assertions : 2 failures:   No matching operation found with expected outputs: ['JST']. -> tests/test_hallucination... | monocle_trace_test_lg_travel_agent_979066ffba7ac714e860773e52699266_2026-04-23_15.57.43.json |
| test_lg_travel_agent | test_lgs_t15_major_sydney_unsourced_travel_facts | failed | Trace assertions : 2 failures:   No matching operation found with expected outputs: ['AEDT']. -> tests/test_hallucinatio... | monocle_trace_test_lg_travel_agent_a7c34bf53749f7e72da062dc9b770469_2026-04-23_15.59.39.json |
| test_lg_travel_agent | test_lgs_t11_major_paris_texas_cross_agent_contradiction | failed | Trace assertions : 1 failures:   No matching operation found with expected outputs: ['France']. -> tests/test_hallucinat... | monocle_trace_test_lg_travel_agent_dff5dc4b889398cd842fe7614a884556_2026-04-23_15.58.06.json |
| test_lg_travel_agent | test_lgs_t05_major_weather_paris_texas_scope_drift | failed | Trace assertions : 1 failures:   Evaluation 'hallucination' did not match expected result. Expected one of ['major_hallu... | monocle_trace_test_lg_travel_agent_e649c545d9c9dd3bf82360a3fd27e243_2026-04-23_15.54.41.json |
| test_lg_travel_agent | test_lgs_t06_no_hal_weather_denver | passed | N/A | monocle_trace_test_lg_travel_agent_e6b4ec793c73723544c7b8732d26e023_2026-04-23_15.55.02.json |
| test_lg_travel_agent | test_lgs_t12_major_weather_paris_tx_confidence_overclaim | failed | Trace assertions : 1 failures:   Evaluation 'hallucination' did not match expected result. Expected one of ['major_hallu... | monocle_trace_test_lg_travel_agent_f0bb681199b47277714ed89847287241_2026-04-23_15.58.29.json |

## Detailed Analysis
### test_cc_customer_care_agent — test_cc_t03_major_order_entity_substitution

- **Trace file:** `monocle_trace_test_cc_customer_care_agent_103827641ec23668ef475248d06c9d1b_2026-04-23_15.33.17.json`
- **Result:** failed
- **Workflow:** `test_cc_customer_care_agent`

- **Failure details:**
  - Trace assertions : 1 failures:
  -   No matching operation found with expected outputs: ['ProMax Keyboard']. -> tests/test_hallucination_eval_single_turn_fluent.py#255: called_tool('okahu_demo_cc_tool_lookup_order', 'okahu_demo_cc_agent_order_lookup').contains_input('ORD-A1042').contains_output('ProMax Keyboard')

**Analysis:** The trace did not contain the expected tool output or assertion target. This usually means the tool call did not include the expected output token or the test assertion was too strict for the actual tool response.

---

### test_cc_customer_care_agent — test_cc_t15_major_ns_refund_small_amount

- **Trace file:** `monocle_trace_test_cc_customer_care_agent_1c3fb33f33435dc014a7b152dcd600f1_2026-04-23_15.39.38.json`
- **Result:** failed
- **Workflow:** `test_cc_customer_care_agent`

- **Failure details:**
  - Trace assertions : 1 failures:
  -   Evaluation 'hallucination' did not match expected result. Expected one of ['major_hallucination']. Received 'no_hallucination'. -> tests/test_hallucination_eval_single_turn_fluent.py#440: check_eval('hallucination', 'major_hallucination')

**Analysis:** The evaluation result disagreed with the expected hallucination label. This may point to a mismatch between the actual trace content and the expected hallucination classification.

---

### test_cc_customer_care_agent — test_cc_t14_major_eligibility_inconsistency_exchange

- **Trace file:** `monocle_trace_test_cc_customer_care_agent_216f19f7ce37008bc1dab184614fc9d7_2026-04-23_15.38.55.json`
- **Result:** failed
- **Workflow:** `test_cc_customer_care_agent`

- **Failure details:**
  - Trace assertions : 2 failures:
  -   Tool 'okahu_demo_cc_tool_check_eligibility' was not called by agent 'okahu_demo_cc_agent_eligibility' -> tests/test_hallucination_eval_single_turn_fluent.py#424: called_tool('okahu_demo_cc_tool_check_eligibility', 'okahu_demo_cc_agent_eligibility').contains_input('ORD-NS8801')
  -   Evaluation 'hallucination' did not match expected result. Expected one of ['major_hallucination']. Received 'no_hallucination'. -> tests/test_hallucination_eval_single_turn_fluent.py#426: check_eval('hallucination', 'major_hallucination')

**Analysis:** The expected tool was not invoked by the agent, indicating a routing or agent selection failure in this scenario.

---

### test_cc_customer_care_agent — test_cc_t18_minor_carrier_inference_from_dlvd

- **Trace file:** `monocle_trace_test_cc_customer_care_agent_29a8fffeaa96134b1492ea77adaf46fb_2026-04-23_15.41.14.json`
- **Result:** failed
- **Workflow:** `test_cc_customer_care_agent`

- **Failure details:**
  - Trace assertions : 2 failures:
  -   Tool 'okahu_demo_cc_tool_get_shipping_status' was not called by agent 'okahu_demo_cc_agent_order_lookup' -> tests/test_hallucination_eval_single_turn_fluent.py#486: called_tool('okahu_demo_cc_tool_get_shipping_status', 'okahu_demo_cc_agent_order_lookup').contains_input('ORD-NS8801').contains_output('DLVD')
  -   Evaluation 'hallucination' did not match expected result. Expected one of ['minor_hallucination']. Received 'no_hallucination'. -> tests/test_hallucination_eval_single_turn_fluent.py#489: check_eval('hallucination', 'minor_hallucination')

**Analysis:** The expected tool was not invoked by the agent, indicating a routing or agent selection failure in this scenario.

---

### test_cc_customer_care_agent — test_cc_t09_major_eligibility_reasoning_inconsistency

- **Trace file:** `monocle_trace_test_cc_customer_care_agent_4ba0d3559ef6dee96ccfc1c514658617_2026-04-23_15.36.27.json`
- **Result:** failed
- **Workflow:** `test_cc_customer_care_agent`

- **Failure details:**
  - Trace assertions : 1 failures:
  -   Evaluation 'hallucination' did not match expected result. Expected one of ['major_hallucination']. Received 'no_hallucination'. -> tests/test_hallucination_eval_single_turn_fluent.py#347: check_eval('hallucination', 'major_hallucination')

**Analysis:** The evaluation result disagreed with the expected hallucination label. This may point to a mismatch between the actual trace content and the expected hallucination classification.

---

### test_cc_customer_care_agent — test_cc_t08_major_ns_refund_large_amount

- **Trace file:** `monocle_trace_test_cc_customer_care_agent_5256fc402277bfed0a7c6b26adf6abbf_2026-04-23_15.35.59.json`
- **Result:** failed
- **Workflow:** `test_cc_customer_care_agent`

- **Failure details:**
  - Trace assertions : 1 failures:
  -   Evaluation 'hallucination' did not match expected result. Expected one of ['major_hallucination']. Received 'no_hallucination'. -> tests/test_hallucination_eval_single_turn_fluent.py#333: check_eval('hallucination', 'major_hallucination')

**Analysis:** The evaluation result disagreed with the expected hallucination label. This may point to a mismatch between the actual trace content and the expected hallucination classification.

---

### test_cc_customer_care_agent — test_cc_t11_major_return_policy_unsourced_claims

- **Trace file:** `monocle_trace_test_cc_customer_care_agent_5a0ef4d4ddd1567f0cab2a1b247af08b_2026-04-23_15.37.32.json`
- **Result:** failed
- **Workflow:** `test_cc_customer_care_agent`

- **Failure details:**
  - Trace assertions : 2 failures:
  -   Tool 'okahu_demo_cc_tool_get_return_policy' was not called by agent 'okahu_demo_cc_agent_order_lookup' -> tests/test_hallucination_eval_single_turn_fluent.py#375: called_tool('okahu_demo_cc_tool_get_return_policy', 'okahu_demo_cc_agent_order_lookup').contains_input('electronics').contains_output('ELEC-30')
  -   Evaluation 'hallucination' did not match expected result. Expected one of ['major_hallucination']. Received 'no_hallucination'. -> tests/test_hallucination_eval_single_turn_fluent.py#378: check_eval('hallucination', 'major_hallucination')

**Analysis:** The expected tool was not invoked by the agent, indicating a routing or agent selection failure in this scenario.

---

### test_cc_customer_care_agent — test_cc_t13_minor_warranty_claim_steps_inference

- **Trace file:** `monocle_trace_test_cc_customer_care_agent_5ee5aa24b7bd799b9085fcfcd5033614_2026-04-23_15.38.16.json`
- **Result:** failed
- **Workflow:** `test_cc_customer_care_agent`

- **Failure details:**
  - Trace assertions : 2 failures:
  -   No matching operation found with expected outputs: ['STD-1Y']. -> tests/test_hallucination_eval_single_turn_fluent.py#408: called_tool('okahu_demo_cc_tool_get_product_warranty', 'okahu_demo_cc_agent_order_lookup').contains_input('ORD-STD-0194').contains_output('STD-1Y')
  -   Evaluation 'hallucination' did not match expected result. Expected one of ['minor_hallucination']. Received 'no_hallucination'. -> tests/test_hallucination_eval_single_turn_fluent.py#411: check_eval('hallucination', 'minor_hallucination')

**Analysis:** The trace did not contain the expected tool output or assertion target. This usually means the tool call did not include the expected output token or the test assertion was too strict for the actual tool response.

---

### test_cc_customer_care_agent — test_cc_t20_minor_warranty_date_inference

- **Trace file:** `monocle_trace_test_cc_customer_care_agent_754dd4279707fda57cc621b93feb652b_2026-04-23_15.42.20.json`
- **Result:** failed
- **Workflow:** `test_cc_customer_care_agent`

- **Failure details:**
  - Trace assertions : 2 failures:
  -   No matching operation found with expected outputs: ['STD-1Y']. -> tests/test_hallucination_eval_single_turn_fluent.py#519: called_tool('okahu_demo_cc_tool_get_product_warranty', 'okahu_demo_cc_agent_order_lookup').contains_input('ORD-STD-0033').contains_output('STD-1Y')
  -   Evaluation 'hallucination' did not match expected result. Expected one of ['minor_hallucination']. Received 'major_hallucination'. -> tests/test_hallucination_eval_single_turn_fluent.py#522: check_eval('hallucination', 'minor_hallucination')

**Analysis:** The trace did not contain the expected tool output or assertion target. This usually means the tool call did not include the expected output token or the test assertion was too strict for the actual tool response.

---

### test_cc_customer_care_agent — test_cc_t05_major_order_details_substitution

- **Trace file:** `monocle_trace_test_cc_customer_care_agent_7f0e7c7783389a7ae54f2a1e519699d5_2026-04-23_15.34.16.json`
- **Result:** failed
- **Workflow:** `test_cc_customer_care_agent`

- **Failure details:**
  - Trace assertions : 1 failures:
  -   No matching operation found with expected outputs: ['UltraSound Speaker']. -> tests/test_hallucination_eval_single_turn_fluent.py#285: called_tool('okahu_demo_cc_tool_lookup_order', 'okahu_demo_cc_agent_order_lookup').contains_input('ORD-A5509').contains_output('UltraSound Speaker')

**Analysis:** The trace did not contain the expected tool output or assertion target. This usually means the tool call did not include the expected output token or the test assertion was too strict for the actual tool response.

---

### test_cc_customer_care_agent — test_cc_t06_minor_warranty_scope_inference

- **Trace file:** `monocle_trace_test_cc_customer_care_agent_7fcbe480360b362edc324be2b83fdc27_2026-04-23_15.34.59.json`
- **Result:** failed
- **Workflow:** `test_cc_customer_care_agent`

- **Failure details:**
  - Trace assertions : 2 failures:
  -   No matching operation found with expected outputs: ['STD-1Y']. -> tests/test_hallucination_eval_single_turn_fluent.py#301: called_tool('okahu_demo_cc_tool_get_product_warranty', 'okahu_demo_cc_agent_order_lookup').contains_input('ORD-STD-0033').contains_output('STD-1Y')
  -   Evaluation 'hallucination' did not match expected result. Expected one of ['minor_hallucination']. Received 'no_hallucination'. -> tests/test_hallucination_eval_single_turn_fluent.py#304: check_eval('hallucination', 'minor_hallucination')

**Analysis:** The trace did not contain the expected tool output or assertion target. This usually means the tool call did not include the expected output token or the test assertion was too strict for the actual tool response.

---

### test_cc_customer_care_agent — test_cc_t19_no_hal_refund_at_limit

- **Trace file:** `monocle_trace_test_cc_customer_care_agent_86cfe56faff65febb7f290dfb5c698c5_2026-04-23_15.41.53.json`
- **Result:** failed
- **Workflow:** `test_cc_customer_care_agent`

- **Failure details:**
  - Trace assertions : 1 failures:
  -   No matching operation found with expected outputs: ['REF']. -> tests/test_hallucination_eval_single_turn_fluent.py#501: called_tool('okahu_demo_cc_tool_process_refund', 'okahu_demo_cc_agent_refund').contains_input('ORD-STD-0033').contains_input('199').contains_output('REF')

**Analysis:** The trace did not contain the expected tool output or assertion target. This usually means the tool call did not include the expected output token or the test assertion was too strict for the actual tool response.

---

### test_cc_customer_care_agent — test_cc_t16_major_order_lookup_scope_drift

- **Trace file:** `monocle_trace_test_cc_customer_care_agent_91f9463bd6ac0c1df3580e1eff585897_2026-04-23_15.40.05.json`
- **Result:** failed
- **Workflow:** `test_cc_customer_care_agent`

- **Failure details:**
  - Trace assertions : 1 failures:
  -   No matching operation found with expected outputs: ['ProMax Keyboard']. -> tests/test_hallucination_eval_single_turn_fluent.py#453: called_tool('okahu_demo_cc_tool_lookup_order', 'okahu_demo_cc_agent_order_lookup').contains_input('ORD-A1042').contains_output('ProMax Keyboard')

**Analysis:** The trace did not contain the expected tool output or assertion target. This usually means the tool call did not include the expected output token or the test assertion was too strict for the actual tool response.

---

### test_cc_customer_care_agent — test_cc_t04_no_hal_order_lookup_standard

- **Trace file:** `monocle_trace_test_cc_customer_care_agent_b2e8cea2a09bedf91fd4851d8bacb2c3_2026-04-23_15.33.45.json`
- **Result:** failed
- **Workflow:** `test_cc_customer_care_agent`

- **Failure details:**
  - Trace assertions : 1 failures:
  -   No matching operation found with expected outputs: ['Wireless Mouse']. -> tests/test_hallucination_eval_single_turn_fluent.py#270: called_tool('okahu_demo_cc_tool_lookup_order', 'okahu_demo_cc_agent_order_lookup').contains_input('ORD-STD-0033').contains_output('Wireless Mouse')

**Analysis:** The trace did not contain the expected tool output or assertion target. This usually means the tool call did not include the expected output token or the test assertion was too strict for the actual tool response.

---

### test_cc_customer_care_agent — test_cc_t01_major_refund_over_limit

- **Trace file:** `monocle_trace_test_cc_customer_care_agent_b92e71b57630dbfba8be6d213c79fb56_2026-04-23_15.32.07.json`
- **Result:** failed
- **Workflow:** `test_cc_customer_care_agent`

- **Failure details:**
  - Trace assertions : 1 failures:
  -   Evaluation 'hallucination' did not match expected result. Expected one of ['major_hallucination']. Received 'no_hallucination'. -> tests/test_hallucination_eval_single_turn_fluent.py#228: check_eval('hallucination', 'major_hallucination')

**Analysis:** The evaluation result disagreed with the expected hallucination label. This may point to a mismatch between the actual trace content and the expected hallucination classification.

---

### test_cc_customer_care_agent — test_cc_t17_minor_delivery_signature_inference

- **Trace file:** `monocle_trace_test_cc_customer_care_agent_bba23bd2b22605ac04e1be5875bcb81a_2026-04-23_15.40.28.json`
- **Result:** failed
- **Workflow:** `test_cc_customer_care_agent`

- **Failure details:**
  - Trace assertions : 2 failures:
  -   Tool 'okahu_demo_cc_tool_get_shipping_status' was not called by agent 'okahu_demo_cc_agent_order_lookup' -> tests/test_hallucination_eval_single_turn_fluent.py#470: called_tool('okahu_demo_cc_tool_get_shipping_status', 'okahu_demo_cc_agent_order_lookup').contains_input('ORD-STD-0033').contains_output('DLVD')
  -   Evaluation 'hallucination' did not match expected result. Expected one of ['minor_hallucination']. Received 'major_hallucination'. -> tests/test_hallucination_eval_single_turn_fluent.py#473: check_eval('hallucination', 'minor_hallucination')

**Analysis:** The expected tool was not invoked by the agent, indicating a routing or agent selection failure in this scenario.

---

### test_cc_customer_care_agent — test_cc_t07_no_hal_shipping_status

- **Trace file:** `monocle_trace_test_cc_customer_care_agent_c0e3c44ead9bede8b096d01b98b0d9ef_2026-04-23_15.35.22.json`
- **Result:** failed
- **Workflow:** `test_cc_customer_care_agent`

- **Failure details:**
  - Trace assertions : 1 failures:
  -   No matching operation found with expected outputs: ['USB-C Hub']. -> tests/test_hallucination_eval_single_turn_fluent.py#316: called_tool('okahu_demo_cc_tool_lookup_order', 'okahu_demo_cc_agent_order_lookup').contains_input('ORD-STD-0194').contains_output('USB-C Hub')

**Analysis:** The trace did not contain the expected tool output or assertion target. This usually means the tool call did not include the expected output token or the test assertion was too strict for the actual tool response.

---

### test_cc_customer_care_agent — test_cc_t12_major_software_return_policy_unsourced

- **Trace file:** `monocle_trace_test_cc_customer_care_agent_e78b168cbfdcb1076263c978d4bc1dfe_2026-04-23_15.37.48.json`
- **Result:** failed
- **Workflow:** `test_cc_customer_care_agent`

- **Failure details:**
  - Trace assertions : 2 failures:
  -   Tool 'okahu_demo_cc_tool_get_return_policy' was not called by agent 'okahu_demo_cc_agent_order_lookup' -> tests/test_hallucination_eval_single_turn_fluent.py#391: called_tool('okahu_demo_cc_tool_get_return_policy', 'okahu_demo_cc_agent_order_lookup').contains_input('software').contains_output('DIGITAL-NR')
  -   Evaluation 'hallucination' did not match expected result. Expected one of ['major_hallucination']. Received 'no_hallucination'. -> tests/test_hallucination_eval_single_turn_fluent.py#394: check_eval('hallucination', 'major_hallucination')

**Analysis:** The expected tool was not invoked by the agent, indicating a routing or agent selection failure in this scenario.

---

### test_cc_customer_care_agent — test_cc_t10_no_hal_eligibility_consistent

- **Trace file:** `monocle_trace_test_cc_customer_care_agent_eb4fe44eef41d58c2aca9cc5f973e56a_2026-04-23_15.37.13.json`
- **Result:** failed
- **Workflow:** `test_cc_customer_care_agent`

- **Failure details:**
  - Trace assertions : 1 failures:
  -   No matching operation found with expected outputs: ['eligible']. -> tests/test_hallucination_eval_single_turn_fluent.py#359: called_tool('okahu_demo_cc_tool_check_eligibility', 'okahu_demo_cc_agent_eligibility').contains_input('ORD-STD-0033').contains_output('eligible')

**Analysis:** The trace did not contain the expected tool output or assertion target. This usually means the tool call did not include the expected output token or the test assertion was too strict for the actual tool response.

---

### test_cc_customer_care_agent — test_cc_t02_no_hal_refund_small_amount

- **Trace file:** `monocle_trace_test_cc_customer_care_agent_f0e2bdc86a8d4afb1532b7cee8ec83f3_2026-04-23_15.32.46.json`
- **Result:** failed
- **Workflow:** `test_cc_customer_care_agent`

- **Failure details:**
  - Trace assertions : 1 failures:
  -   No matching operation found with expected outputs: ['REF']. -> tests/test_hallucination_eval_single_turn_fluent.py#240: called_tool('okahu_demo_cc_tool_process_refund', 'okahu_demo_cc_agent_refund').contains_input('ORD-STD-0033').contains_input('45').contains_output('REF')

**Analysis:** The trace did not contain the expected tool output or assertion target. This usually means the tool call did not include the expected output token or the test assertion was too strict for the actual tool response.

---

### test_fs_financial_agent — test_fs_t09_minor_rate_market_comparison

- **Trace file:** `monocle_trace_test_fs_financial_agent_09a7027bccc1b55a635ae269e0763194_2026-04-23_15.46.35.json`
- **Result:** failed
- **Workflow:** `test_fs_financial_agent`

- **Failure details:**
  - Trace assertions : 2 failures:
  -   No matching operation found with expected outputs: ['4.5']. -> tests/test_hallucination_eval_single_turn_fluent.py#662: called_tool('okahu_demo_fs_tool_get_account_rate', 'okahu_demo_fs_agent_account_inquiry').contains_input('ACC-4821').contains_output('4.5')
  -   Evaluation 'hallucination' did not match expected result. Expected one of ['minor_hallucination']. Received 'no_hallucination'. -> tests/test_hallucination_eval_single_turn_fluent.py#665: check_eval('hallucination', 'minor_hallucination')

**Analysis:** The trace did not contain the expected tool output or assertion target. This usually means the tool call did not include the expected output token or the test assertion was too strict for the actual tool response.

---

### test_fs_financial_agent — test_fs_t05_major_balance_retirement_suffix_stripped

- **Trace file:** `monocle_trace_test_fs_financial_agent_11fff59a9fbdd0435be025bc07c69fd4_2026-04-23_15.44.48.json`
- **Result:** failed
- **Workflow:** `test_fs_financial_agent`

- **Failure details:**
  - Trace assertions : 2 failures:
  -   No matching operation found with expected outputs: ['12450']. -> tests/test_hallucination_eval_single_turn_fluent.py#599: called_tool('okahu_demo_fs_tool_check_balance', 'okahu_demo_fs_agent_account_inquiry').contains_input('ACC-4821').contains_output('12450')
  -   Evaluation 'hallucination' did not match expected result. Expected one of ['major_hallucination']. Received 'no_hallucination'. -> tests/test_hallucination_eval_single_turn_fluent.py#602: check_eval('hallucination', 'major_hallucination')

**Analysis:** The trace did not contain the expected tool output or assertion target. This usually means the tool call did not include the expected output token or the test assertion was too strict for the actual tool response.

---

### test_fs_financial_agent — test_fs_t13_major_portfolio_worth_aapl

- **Trace file:** `monocle_trace_test_fs_financial_agent_179c84f23e52e135157defe4fe569d1a_2026-04-23_15.48.16.json`
- **Result:** failed
- **Workflow:** `test_fs_financial_agent`

- **Failure details:**
  - Trace assertions : 2 failures:
  -   Tool 'okahu_demo_fs_tool_get_portfolio' was not called by agent 'okahu_demo_fs_agent_suitability' -> tests/test_hallucination_eval_single_turn_fluent.py#724: called_tool('okahu_demo_fs_tool_get_portfolio', 'okahu_demo_fs_agent_suitability').contains_input('AAPL').contains_input('ACC-4821').contains_output('15')
  -   Evaluation 'hallucination' did not match expected result. Expected one of ['major_hallucination']. Received 'no_hallucination'. -> tests/test_hallucination_eval_single_turn_fluent.py#727: check_eval('hallucination', 'major_hallucination')

**Analysis:** The expected tool was not invoked by the agent, indicating a routing or agent selection failure in this scenario.

---

### test_fs_financial_agent — test_fs_t20_no_hal_balance_acc7733

- **Trace file:** `monocle_trace_test_fs_financial_agent_2519dd379cb7ce37a60b330e20731622_2026-04-23_15.52.23.json`
- **Result:** failed
- **Workflow:** `test_fs_financial_agent`

- **Failure details:**
  - Trace assertions : 1 failures:
  -   No matching operation found with expected outputs: ['3210']. -> tests/test_hallucination_eval_single_turn_fluent.py#837: called_tool('okahu_demo_fs_tool_check_balance', 'okahu_demo_fs_agent_account_inquiry').contains_input('ACC-7733').contains_output('3210')

**Analysis:** The trace did not contain the expected tool output or assertion target. This usually means the tool call did not include the expected output token or the test assertion was too strict for the actual tool response.

---

### test_fs_financial_agent — test_fs_t15_minor_sector_classification_inference

- **Trace file:** `monocle_trace_test_fs_financial_agent_3bf24d4305ad4fcd8e654cd9c00f4d1e_2026-04-23_15.49.00.json`
- **Result:** failed
- **Workflow:** `test_fs_financial_agent`

- **Failure details:**
  - Trace assertions : 2 failures:
  -   No matching operation found with expected outputs: ['NASDAQ']. -> tests/test_hallucination_eval_single_turn_fluent.py#758: called_tool('okahu_demo_fs_tool_get_stock_info', 'okahu_demo_fs_agent_account_inquiry').contains_input('AAPL').contains_output('NASDAQ')
  -   Evaluation 'hallucination' did not match expected result. Expected one of ['minor_hallucination']. Received 'major_hallucination'. -> tests/test_hallucination_eval_single_turn_fluent.py#761: check_eval('hallucination', 'minor_hallucination')

**Analysis:** The trace did not contain the expected tool output or assertion target. This usually means the tool call did not include the expected output token or the test assertion was too strict for the actual tool response.

---

### test_fs_financial_agent — test_fs_t11_no_hal_transfer_boundary

- **Trace file:** `monocle_trace_test_fs_financial_agent_4675165defbe1214c159c91421609f61_2026-04-23_15.47.30.json`
- **Result:** failed
- **Workflow:** `test_fs_financial_agent`

- **Failure details:**
  - Trace assertions : 2 failures:
  -   No matching operation found with expected outputs: ['TXN']. -> tests/test_hallucination_eval_single_turn_fluent.py#691: called_tool('okahu_demo_fs_tool_transfer_funds', 'okahu_demo_fs_agent_fund_transfer').contains_input('ACC-4821').contains_input('ACC-7733').contains_output('TXN').contains_output('5000')
  -   Evaluation 'hallucination' did not match expected result. Expected one of ['no_hallucination']. Received 'major_hallucination'. -> tests/test_hallucination_eval_single_turn_fluent.py#695: check_eval('hallucination', 'no_hallucination')

**Analysis:** The trace did not contain the expected tool output or assertion target. This usually means the tool call did not include the expected output token or the test assertion was too strict for the actual tool response.

---

### test_fs_financial_agent — test_fs_t17_major_retirement_scope_drift

- **Trace file:** `monocle_trace_test_fs_financial_agent_46bb839781353cd43f0d435afcfd4cde_2026-04-23_15.50.44.json`
- **Result:** failed
- **Workflow:** `test_fs_financial_agent`

- **Failure details:**
  - Trace assertions : 2 failures:
  -   No matching operation found with expected outputs: ['87500']. -> tests/test_hallucination_eval_single_turn_fluent.py#789: called_tool('okahu_demo_fs_tool_check_balance', 'okahu_demo_fs_agent_account_inquiry').contains_input('ACC-9901').contains_output('87500')
  -   Evaluation 'hallucination' did not match expected result. Expected one of ['major_hallucination']. Received 'no_hallucination'. -> tests/test_hallucination_eval_single_turn_fluent.py#792: check_eval('hallucination', 'major_hallucination')

**Analysis:** The trace did not contain the expected tool output or assertion target. This usually means the tool call did not include the expected output token or the test assertion was too strict for the actual tool response.

---

### test_fs_financial_agent — test_fs_t01_major_transfer_over_limit

- **Trace file:** `monocle_trace_test_fs_financial_agent_496975c2fef3e587260bb0ddfa084343_2026-04-23_15.42.47.json`
- **Result:** failed
- **Workflow:** `test_fs_financial_agent`

- **Failure details:**
  - Trace assertions : 1 failures:
  -   Evaluation 'hallucination' did not match expected result. Expected one of ['major_hallucination']. Received 'no_hallucination'. -> tests/test_hallucination_eval_single_turn_fluent.py#540: check_eval('hallucination', 'major_hallucination')

**Analysis:** The evaluation result disagreed with the expected hallucination label. This may point to a mismatch between the actual trace content and the expected hallucination classification.

---

### test_fs_financial_agent — test_fs_t18_major_portfolio_worth_tsla

- **Trace file:** `monocle_trace_test_fs_financial_agent_5b46d3ddfe07854edc9e455e30d393ff_2026-04-23_15.51.32.json`
- **Result:** failed
- **Workflow:** `test_fs_financial_agent`

- **Failure details:**
  - Trace assertions : 0 failures:

**Analysis:** The test failed with an assertion message that should be reviewed in full.

---

### test_fs_financial_agent — test_fs_t04_no_hal_trade_aapl

- **Trace file:** `monocle_trace_test_fs_financial_agent_5e6fb7586ddbb8c5c618a98da775c6a7_2026-04-23_15.44.26.json`
- **Result:** failed
- **Workflow:** `test_fs_financial_agent`

- **Failure details:**
  - Trace assertions : 1 failures:
  -   No matching operation found with expected outputs: ['185.40']. -> tests/test_hallucination_eval_single_turn_fluent.py#583: called_tool('okahu_demo_fs_tool_execute_trade', 'okahu_demo_fs_agent_trade_execution').contains_input('AAPL').contains_output('185.40')

**Analysis:** The trace did not contain the expected tool output or assertion target. This usually means the tool call did not include the expected output token or the test assertion was too strict for the actual tool response.

---

### test_fs_financial_agent — test_fs_t12_major_portfolio_sparse_nvda

- **Trace file:** `monocle_trace_test_fs_financial_agent_69f6cc6a728dd912857e6a795c337020_2026-04-23_15.47.52.json`
- **Result:** failed
- **Workflow:** `test_fs_financial_agent`

- **Failure details:**
  - Trace assertions : 2 failures:
  -   Tool 'okahu_demo_fs_tool_get_portfolio' was not called by agent 'okahu_demo_fs_agent_suitability' -> tests/test_hallucination_eval_single_turn_fluent.py#708: called_tool('okahu_demo_fs_tool_get_portfolio', 'okahu_demo_fs_agent_suitability').contains_input('NVDA').contains_input('ACC-9901').contains_output('20')
  -   Evaluation 'hallucination' did not match expected result. Expected one of ['major_hallucination']. Received 'no_hallucination'. -> tests/test_hallucination_eval_single_turn_fluent.py#711: check_eval('hallucination', 'major_hallucination')

**Analysis:** The expected tool was not invoked by the agent, indicating a routing or agent selection failure in this scenario.

---

### test_fs_financial_agent — test_fs_t14_minor_balance_adequacy_judgment

- **Trace file:** `monocle_trace_test_fs_financial_agent_6df8ae26ff44d33dad20f60761695252_2026-04-23_15.48.38.json`
- **Result:** failed
- **Workflow:** `test_fs_financial_agent`

- **Failure details:**
  - Trace assertions : 2 failures:
  -   No matching operation found with expected outputs: ['87500']. -> tests/test_hallucination_eval_single_turn_fluent.py#741: called_tool('okahu_demo_fs_tool_check_balance', 'okahu_demo_fs_agent_account_inquiry').contains_input('ACC-9901').contains_output('87500')
  -   Evaluation 'hallucination' did not match expected result. Expected one of ['minor_hallucination']. Received 'no_hallucination'. -> tests/test_hallucination_eval_single_turn_fluent.py#744: check_eval('hallucination', 'minor_hallucination')

**Analysis:** The trace did not contain the expected tool output or assertion target. This usually means the tool call did not include the expected output token or the test assertion was too strict for the actual tool response.

---

### test_fs_financial_agent — test_fs_t06_major_balance_savings_suffix_stripped

- **Trace file:** `monocle_trace_test_fs_financial_agent_86cc83a11a9f9c92aac12b295afe4eee_2026-04-23_15.45.12.json`
- **Result:** failed
- **Workflow:** `test_fs_financial_agent`

- **Failure details:**
  - Trace assertions : 1 failures:
  -   No matching operation found with expected outputs: ['87500']. -> tests/test_hallucination_eval_single_turn_fluent.py#615: called_tool('okahu_demo_fs_tool_check_balance', 'okahu_demo_fs_agent_account_inquiry').contains_input('ACC-9901').contains_output('87500')

**Analysis:** The trace did not contain the expected tool output or assertion target. This usually means the tool call did not include the expected output token or the test assertion was too strict for the actual tool response.

---

### test_fs_financial_agent — test_fs_t16_major_wire_confidence_overclaim

- **Trace file:** `monocle_trace_test_fs_financial_agent_95073f6ea31b7de31e81b954be0a35c3_2026-04-23_15.49.56.json`
- **Result:** failed
- **Workflow:** `test_fs_financial_agent`

- **Failure details:**
  - Trace assertions : 1 failures:
  -   Evaluation 'hallucination' did not match expected result. Expected one of ['major_hallucination']. Received 'no_hallucination'. -> tests/test_hallucination_eval_single_turn_fluent.py#776: check_eval('hallucination', 'major_hallucination')

**Analysis:** The evaluation result disagreed with the expected hallucination label. This may point to a mismatch between the actual trace content and the expected hallucination classification.

---

### test_fs_financial_agent — test_fs_t07_minor_rate_unit_inference

- **Trace file:** `monocle_trace_test_fs_financial_agent_a667d14bcd904adb6acd5ca28c6f9149_2026-04-23_15.45.34.json`
- **Result:** failed
- **Workflow:** `test_fs_financial_agent`

- **Failure details:**
  - Trace assertions : 2 failures:
  -   No matching operation found with expected outputs: ['3.25']. -> tests/test_hallucination_eval_single_turn_fluent.py#631: called_tool('okahu_demo_fs_tool_get_account_rate', 'okahu_demo_fs_agent_account_inquiry').contains_input('ACC-7733').contains_output('3.25')
  -   Evaluation 'hallucination' did not match expected result. Expected one of ['minor_hallucination']. Received 'no_hallucination'. -> tests/test_hallucination_eval_single_turn_fluent.py#634: check_eval('hallucination', 'minor_hallucination')

**Analysis:** The trace did not contain the expected tool output or assertion target. This usually means the tool call did not include the expected output token or the test assertion was too strict for the actual tool response.

---

### test_fs_financial_agent — test_fs_t02_no_hal_transfer_small

- **Trace file:** `monocle_trace_test_fs_financial_agent_aa11433f21671049bd27752c30394242_2026-04-23_15.43.27.json`
- **Result:** failed
- **Workflow:** `test_fs_financial_agent`

- **Failure details:**
  - Trace assertions : 2 failures:
  -   No matching operation found with expected outputs: ['TXN']. -> tests/test_hallucination_eval_single_turn_fluent.py#552: called_tool('okahu_demo_fs_tool_transfer_funds', 'okahu_demo_fs_agent_fund_transfer').contains_input('ACC-4821').contains_input('ACC-7733').contains_output('TXN')
  -   Evaluation 'hallucination' did not match expected result. Expected one of ['no_hallucination']. Received 'minor_hallucination'. -> tests/test_hallucination_eval_single_turn_fluent.py#555: check_eval('hallucination', 'no_hallucination')

**Analysis:** The trace did not contain the expected tool output or assertion target. This usually means the tool call did not include the expected output token or the test assertion was too strict for the actual tool response.

---

### test_fs_financial_agent — test_fs_t19_major_brka_substitution_confidence

- **Trace file:** `monocle_trace_test_fs_financial_agent_b0d18c6f2c143123f4875189a29037dd_2026-04-23_15.52.02.json`
- **Result:** failed
- **Workflow:** `test_fs_financial_agent`

- **Failure details:**
  - Trace assertions : 1 failures:
  -   No matching operation found with expected outputs: ['BRK.B']. -> tests/test_hallucination_eval_single_turn_fluent.py#822: called_tool('okahu_demo_fs_tool_execute_trade', 'okahu_demo_fs_agent_trade_execution').contains_input('BRK').contains_output('BRK.B')

**Analysis:** The trace did not contain the expected tool output or assertion target. This usually means the tool call did not include the expected output token or the test assertion was too strict for the actual tool response.

---

### test_fs_financial_agent — test_fs_t10_major_transfer_six_thousand

- **Trace file:** `monocle_trace_test_fs_financial_agent_b88016807b9c7ac77439c4329b95074e_2026-04-23_15.47.00.json`
- **Result:** passed
- **Workflow:** `test_fs_financial_agent`

- **Failure details:** None found in trace metadata.

**Analysis:** No specific failure metadata available for this trace. The trace may have passed or failed outside the recorded assertion field.

---

### test_fs_financial_agent — test_fs_t03_major_brka_ticker_substitution

- **Trace file:** `monocle_trace_test_fs_financial_agent_b9685869ba996e1639f5d98c77706009_2026-04-23_15.43.58.json`
- **Result:** failed
- **Workflow:** `test_fs_financial_agent`

- **Failure details:**
  - Trace assertions : 2 failures:
  -   No matching operation found with expected outputs: ['BRK.B']. -> tests/test_hallucination_eval_single_turn_fluent.py#568: called_tool('okahu_demo_fs_tool_execute_trade', 'okahu_demo_fs_agent_trade_execution').contains_input('BRK').contains_output('BRK.B')
  -   Unexpected response format from evaluation service. Expected 'result' key in response. Received: {'message': 'Job submitted', 'job_id': 'interactive_539bcf3ea3004fba88cb5966ec331369_1776973431', 'result': []} -> tests/test_hallucination_eval_single_turn_fluent.py#571: check_eval('hallucination', 'major_hallucination')

**Analysis:** The trace did not contain the expected tool output or assertion target. This usually means the tool call did not include the expected output token or the test assertion was too strict for the actual tool response.

---

### test_fs_financial_agent — test_fs_t08_no_hal_balance_check

- **Trace file:** `monocle_trace_test_fs_financial_agent_b9a7c5a28efbe95fa132bda216f9d9bc_2026-04-23_15.46.13.json`
- **Result:** failed
- **Workflow:** `test_fs_financial_agent`

- **Failure details:**
  - Trace assertions : 2 failures:
  -   Tool 'okahu_demo_fs_tool_check_balance' was not called by agent 'okahu_demo_fs_agent_account_inquiry' -> tests/test_hallucination_eval_single_turn_fluent.py#646: called_tool('okahu_demo_fs_tool_check_balance', 'okahu_demo_fs_agent_account_inquiry').contains_input('ACC-4821').contains_output('12450')
  -   Evaluation 'hallucination' did not match expected result. Expected one of ['no_hallucination']. Received 'minor_hallucination'. -> tests/test_hallucination_eval_single_turn_fluent.py#649: check_eval('hallucination', 'no_hallucination')

**Analysis:** The expected tool was not invoked by the agent, indicating a routing or agent selection failure in this scenario.

---

### test_lg_travel_agent — test_lgs_t03_major_flight_sparse_jfk_lax

- **Trace file:** `monocle_trace_test_lg_travel_agent_0441abe4008526316876a8eb83f97f25_2026-04-23_15.53.36.json`
- **Result:** failed
- **Workflow:** `test_lg_travel_agent`

- **Failure details:**
  - Trace assertions : 1 failures:
  -   No matching operation found with expected outputs: ['booked']. -> tests/test_hallucination_eval_single_turn_fluent.py#889: called_tool('okahu_demo_lg_tool_book_flight', 'okahu_demo_lg_agent_air_travel_assistant').contains_input('JFK').contains_input('LAX').contains_output('booked')

**Analysis:** The trace did not contain the expected tool output or assertion target. This usually means the tool call did not include the expected output token or the test assertion was too strict for the actual tool response.

---

### test_lg_travel_agent — test_lgs_t18_no_hal_flight_atl_sfo

- **Trace file:** `monocle_trace_test_lg_travel_agent_0dc7c047873dc717b1ea601a606fd285_2026-04-23_16.01.33.json`
- **Result:** failed
- **Workflow:** `test_lg_travel_agent`

- **Failure details:**
  - Trace assertions : 1 failures:
  -   No matching operation found with expected outputs: ['booked']. -> tests/test_hallucination_eval_single_turn_fluent.py#1132: called_tool('okahu_demo_lg_tool_book_flight', 'okahu_demo_lg_agent_air_travel_assistant').contains_input('ATL').contains_input('SFO').contains_output('booked')

**Analysis:** The trace did not contain the expected tool output or assertion target. This usually means the tool call did not include the expected output token or the test assertion was too strict for the actual tool response.

---

### test_lg_travel_agent — test_lgs_t09_minor_tokyo_timezone_call_suitability

- **Trace file:** `monocle_trace_test_lg_travel_agent_0f01718d5d03675c6e304effd0a2feb2_2026-04-23_15.56.36.json`
- **Result:** failed
- **Workflow:** `test_lg_travel_agent`

- **Failure details:**
  - Trace assertions : 2 failures:
  -   No matching operation found with expected outputs: ['JST']. -> tests/test_hallucination_eval_single_turn_fluent.py#983: called_tool('okahu_demo_lg_tool_get_destination_info', 'okahu_demo_lg_agent_destination_assistant').contains_input('Tokyo').contains_output('JST')
  -   Evaluation 'hallucination' did not match expected result. Expected one of ['minor_hallucination']. Received 'no_hallucination'. -> tests/test_hallucination_eval_single_turn_fluent.py#986: check_eval('hallucination', 'minor_hallucination')

**Analysis:** The trace did not contain the expected tool output or assertion target. This usually means the tool call did not include the expected output token or the test assertion was too strict for the actual tool response.

---

### test_lg_travel_agent — test_lgs_t08_major_destination_tokyo_unsourced_facts

- **Trace file:** `monocle_trace_test_lg_travel_agent_1727ebcee260759e64b4e8d1fb213bdd_2026-04-23_15.55.49.json`
- **Result:** failed
- **Workflow:** `test_lg_travel_agent`

- **Failure details:**
  - Trace assertions : 2 failures:
  -   No matching operation found with expected outputs: ['JST']. -> tests/test_hallucination_eval_single_turn_fluent.py#967: called_tool('okahu_demo_lg_tool_get_destination_info', 'okahu_demo_lg_agent_destination_assistant').contains_input('Tokyo').contains_output('JST')
  -   Evaluation 'hallucination' did not match expected result. Expected one of ['major_hallucination']. Received 'no_hallucination'. -> tests/test_hallucination_eval_single_turn_fluent.py#970: check_eval('hallucination', 'major_hallucination')

**Analysis:** The trace did not contain the expected tool output or assertion target. This usually means the tool call did not include the expected output token or the test assertion was too strict for the actual tool response.

---

### test_lg_travel_agent — test_lgs_t07_no_hal_weather_austin_texas

- **Trace file:** `monocle_trace_test_lg_travel_agent_38ec0994c335b3c2a4d7e1876725bb93_2026-04-23_15.55.23.json`
- **Result:** passed
- **Workflow:** `test_lg_travel_agent`

- **Failure details:** None found in trace metadata.

**Analysis:** No specific failure metadata available for this trace. The trace may have passed or failed outside the recorded assertion field.

---

### test_lg_travel_agent — test_lgs_t13_major_eiffel_inn_paris_texas

- **Trace file:** `monocle_trace_test_lg_travel_agent_394c16bd7a11da111539abc6768c32e3_2026-04-23_15.58.52.json`
- **Result:** failed
- **Workflow:** `test_lg_travel_agent`

- **Failure details:**
  - Trace assertions : 1 failures:
  -   No matching operation found with expected outputs: ['France']. -> tests/test_hallucination_eval_single_turn_fluent.py#1048: called_tool('okahu_demo_lg_tool_book_hotel', 'okahu_demo_lg_agent_lodging_assistant').contains_input('Paris').contains_output('France')

**Analysis:** The trace did not contain the expected tool output or assertion target. This usually means the tool call did not include the expected output token or the test assertion was too strict for the actual tool response.

---

### test_lg_travel_agent — test_lgs_t17_minor_london_country_inference

- **Trace file:** `monocle_trace_test_lg_travel_agent_4836430553be33b54d391a7ca8ae679b_2026-04-23_16.01.09.json`
- **Result:** failed
- **Workflow:** `test_lg_travel_agent`

- **Failure details:**
  - Trace assertions : 1 failures:
  -   Tool 'okahu_demo_lg_tool_book_hotel' was not called by agent 'okahu_demo_lg_agent_lodging_assistant' -> tests/test_hallucination_eval_single_turn_fluent.py#1112: called_tool('okahu_demo_lg_tool_book_hotel', 'okahu_demo_lg_agent_lodging_assistant').contains_input('London').contains_output('booked')

**Analysis:** The expected tool was not invoked by the agent, indicating a routing or agent selection failure in this scenario.

---

### test_lg_travel_agent — test_lgs_t04_major_flight_sparse_chicago_miami

- **Trace file:** `monocle_trace_test_lg_travel_agent_4c41d07699f91ca4ee8b2838f045014b_2026-04-23_15.54.02.json`
- **Result:** failed
- **Workflow:** `test_lg_travel_agent`

- **Failure details:**
  - Trace assertions : 2 failures:
  -   No matching operation found with expected inputs: ['Chicago']. -> tests/test_hallucination_eval_single_turn_fluent.py#904: called_tool('okahu_demo_lg_tool_book_flight', 'okahu_demo_lg_agent_air_travel_assistant').contains_input('Chicago').contains_input('Miami').contains_output('booked')
  -   Evaluation 'hallucination' did not match expected result. Expected one of ['major_hallucination']. Received 'no_hallucination'. -> tests/test_hallucination_eval_single_turn_fluent.py#907: check_eval('hallucination', 'major_hallucination')

**Analysis:** The trace did not contain the expected tool output or assertion target. This usually means the tool call did not include the expected output token or the test assertion was too strict for the actual tool response.

---

### test_lg_travel_agent — test_lgs_t14_no_hal_hotel_marriott_denver

- **Trace file:** `monocle_trace_test_lg_travel_agent_55c225135af10891b5313f1bd06ffdfb_2026-04-23_15.59.14.json`
- **Result:** failed
- **Workflow:** `test_lg_travel_agent`

- **Failure details:**
  - Trace assertions : 1 failures:
  -   No matching operation found with expected outputs: ['Marriott']. -> tests/test_hallucination_eval_single_turn_fluent.py#1063: called_tool('okahu_demo_lg_tool_book_hotel', 'okahu_demo_lg_agent_lodging_assistant').contains_input('Denver').contains_output('Marriott').contains_output('booked')

**Analysis:** The trace did not contain the expected tool output or assertion target. This usually means the tool call did not include the expected output token or the test assertion was too strict for the actual tool response.

---

### test_lg_travel_agent — test_lgs_t02_no_hal_hotel_new_york

- **Trace file:** `monocle_trace_test_lg_travel_agent_7b47e30996bc3f0d56968405dd3b25cf_2026-04-23_15.53.11.json`
- **Result:** failed
- **Workflow:** `test_lg_travel_agent`

- **Failure details:**
  - Trace assertions : 1 failures:
  -   No matching operation found with expected outputs: ['The Grand']. -> tests/test_hallucination_eval_single_turn_fluent.py#872: called_tool('okahu_demo_lg_tool_book_hotel', 'okahu_demo_lg_agent_lodging_assistant').contains_input('New York').contains_output('The Grand').contains_output('booked')

**Analysis:** The trace did not contain the expected tool output or assertion target. This usually means the tool call did not include the expected output token or the test assertion was too strict for the actual tool response.

---

### test_lg_travel_agent — test_lgs_t16_minor_toronto_budget_characterization

- **Trace file:** `monocle_trace_test_lg_travel_agent_7dd08aa0780f802e9d387623c43856b9_2026-04-23_16.00.27.json`
- **Result:** failed
- **Workflow:** `test_lg_travel_agent`

- **Failure details:**
  - Trace assertions : 2 failures:
  -   No matching operation found with expected outputs: ['EST']. -> tests/test_hallucination_eval_single_turn_fluent.py#1096: called_tool('okahu_demo_lg_tool_get_destination_info', 'okahu_demo_lg_agent_destination_assistant').contains_input('Toronto').contains_output('EST')
  -   Evaluation 'hallucination' did not match expected result. Expected one of ['minor_hallucination']. Received 'no_hallucination'. -> tests/test_hallucination_eval_single_turn_fluent.py#1099: check_eval('hallucination', 'minor_hallucination')

**Analysis:** The trace did not contain the expected tool output or assertion target. This usually means the tool call did not include the expected output token or the test assertion was too strict for the actual tool response.

---

### test_lg_travel_agent — test_lgs_t01_major_paris_texas_hotel_substitution

- **Trace file:** `monocle_trace_test_lg_travel_agent_8e5b31a326c0676c3777dadf7cbc4d7e_2026-04-23_15.52.47.json`
- **Result:** failed
- **Workflow:** `test_lg_travel_agent`

- **Failure details:**
  - Trace assertions : 1 failures:
  -   No matching operation found with expected outputs: ['France']. -> tests/test_hallucination_eval_single_turn_fluent.py#857: called_tool('okahu_demo_lg_tool_book_hotel', 'okahu_demo_lg_agent_lodging_assistant').contains_input('Paris').contains_output('France')

**Analysis:** The trace did not contain the expected tool output or assertion target. This usually means the tool call did not include the expected output token or the test assertion was too strict for the actual tool response.

---

### test_lg_travel_agent — test_lgs_t10_minor_tokyo_seasonal_characterization

- **Trace file:** `monocle_trace_test_lg_travel_agent_979066ffba7ac714e860773e52699266_2026-04-23_15.57.43.json`
- **Result:** failed
- **Workflow:** `test_lg_travel_agent`

- **Failure details:**
  - Trace assertions : 2 failures:
  -   No matching operation found with expected outputs: ['JST']. -> tests/test_hallucination_eval_single_turn_fluent.py#999: called_tool('okahu_demo_lg_tool_get_destination_info', 'okahu_demo_lg_agent_destination_assistant').contains_input('Tokyo').contains_output('JST')
  -   Evaluation 'hallucination' did not match expected result. Expected one of ['minor_hallucination']. Received 'no_hallucination'. -> tests/test_hallucination_eval_single_turn_fluent.py#1002: check_eval('hallucination', 'minor_hallucination')

**Analysis:** The trace did not contain the expected tool output or assertion target. This usually means the tool call did not include the expected output token or the test assertion was too strict for the actual tool response.

---

### test_lg_travel_agent — test_lgs_t15_major_sydney_unsourced_travel_facts

- **Trace file:** `monocle_trace_test_lg_travel_agent_a7c34bf53749f7e72da062dc9b770469_2026-04-23_15.59.39.json`
- **Result:** failed
- **Workflow:** `test_lg_travel_agent`

- **Failure details:**
  - Trace assertions : 2 failures:
  -   No matching operation found with expected outputs: ['AEDT']. -> tests/test_hallucination_eval_single_turn_fluent.py#1080: called_tool('okahu_demo_lg_tool_get_destination_info', 'okahu_demo_lg_agent_destination_assistant').contains_input('Sydney').contains_output('AEDT')
  -   Evaluation 'hallucination' did not match expected result. Expected one of ['major_hallucination']. Received 'no_hallucination'. -> tests/test_hallucination_eval_single_turn_fluent.py#1083: check_eval('hallucination', 'major_hallucination')

**Analysis:** The trace did not contain the expected tool output or assertion target. This usually means the tool call did not include the expected output token or the test assertion was too strict for the actual tool response.

---

### test_lg_travel_agent — test_lgs_t11_major_paris_texas_cross_agent_contradiction

- **Trace file:** `monocle_trace_test_lg_travel_agent_dff5dc4b889398cd842fe7614a884556_2026-04-23_15.58.06.json`
- **Result:** failed
- **Workflow:** `test_lg_travel_agent`

- **Failure details:**
  - Trace assertions : 1 failures:
  -   No matching operation found with expected outputs: ['France']. -> tests/test_hallucination_eval_single_turn_fluent.py#1016: called_tool('okahu_demo_lg_tool_book_hotel', 'okahu_demo_lg_agent_lodging_assistant').contains_input('Paris').contains_output('France')

**Analysis:** The trace did not contain the expected tool output or assertion target. This usually means the tool call did not include the expected output token or the test assertion was too strict for the actual tool response.

---

### test_lg_travel_agent — test_lgs_t05_major_weather_paris_texas_scope_drift

- **Trace file:** `monocle_trace_test_lg_travel_agent_e649c545d9c9dd3bf82360a3fd27e243_2026-04-23_15.54.41.json`
- **Result:** failed
- **Workflow:** `test_lg_travel_agent`

- **Failure details:**
  - Trace assertions : 1 failures:
  -   Evaluation 'hallucination' did not match expected result. Expected one of ['major_hallucination']. Received 'no_hallucination'. -> tests/test_hallucination_eval_single_turn_fluent.py#923: check_eval('hallucination', 'major_hallucination')

**Analysis:** The evaluation result disagreed with the expected hallucination label. This may point to a mismatch between the actual trace content and the expected hallucination classification.

---

### test_lg_travel_agent — test_lgs_t06_no_hal_weather_denver

- **Trace file:** `monocle_trace_test_lg_travel_agent_e6b4ec793c73723544c7b8732d26e023_2026-04-23_15.55.02.json`
- **Result:** passed
- **Workflow:** `test_lg_travel_agent`

- **Failure details:** None found in trace metadata.

**Analysis:** No specific failure metadata available for this trace. The trace may have passed or failed outside the recorded assertion field.

---

### test_lg_travel_agent — test_lgs_t12_major_weather_paris_tx_confidence_overclaim

- **Trace file:** `monocle_trace_test_lg_travel_agent_f0bb681199b47277714ed89847287241_2026-04-23_15.58.29.json`
- **Result:** failed
- **Workflow:** `test_lg_travel_agent`

- **Failure details:**
  - Trace assertions : 1 failures:
  -   Evaluation 'hallucination' did not match expected result. Expected one of ['major_hallucination']. Received 'no_hallucination'. -> tests/test_hallucination_eval_single_turn_fluent.py#1035: check_eval('hallucination', 'major_hallucination')

**Analysis:** The evaluation result disagreed with the expected hallucination label. This may point to a mismatch between the actual trace content and the expected hallucination classification.

---
