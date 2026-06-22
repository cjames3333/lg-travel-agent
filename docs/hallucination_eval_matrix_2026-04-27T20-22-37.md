# Hallucination Evaluation Matrix

**Run ID:** `2026-04-27T20:22:37.705381`  
**Total scenarios:** 58  
**Evaluators compared:** Okahu (current deployed), Hallucination_v2 (tiered REQ-01–REQ-10 template), Claude (independent)

## Summary

| Evaluator | Correct | Accuracy |
|-----------|---------|----------|
| Okahu (current) | 25 / 58 | **43%** |
| Hallucination_v2 (tiered template) | 58 / 58 | **100%** |
| Claude (independent) | 58 / 58 | **100%** |

**Okahu error breakdown (33 wrong):**
- Under-detection — `no_hallucination` returned when `major` expected: 12 cases
- Under-detection — `no_hallucination` returned when `minor` expected: 10 cases
- Under-detection — `minor_hallucination` returned when `major` expected: 5 cases
- Over-detection — `major_hallucination` returned when `no` expected: 4 cases
- Over-detection — `major_hallucination` returned when `minor` expected: 1 case
- Indeterminate (output assertion failure, not hallucination eval): 1 case

Labels: `no` = no_hallucination · `minor` = minor_hallucination · `major` = major_hallucination  
`✓` = matches expected · `✗` = does not match

---

## Customer Care Agent (CC)

| Scenario | Trace ID | Expected Result | Okahu | H_v2 | Claude |
|----------|----------|----------|-------|------|--------|
| CC-T01 | `e119a687e218a9a5fd1fd6110826a389` | major | no ✗ | major ✓ | major ✓ |
| CC-T02 | `8fa07a30f40d8943e48d77bd2408af6d` | no | no ✓ | no ✓ | no ✓ |
| CC-T03 | `04ca9ebc93ad7f4eca59437f6ab3252c` | major | major ✓ | major ✓ | major ✓ |
| CC-T04 | `4af4a09c7bed9eee7433f01aab6a3ae3` | no | major ✗ | no ✓ | no ✓ |
| CC-T05 | `8cadcf0471dec530d3f49a1671ae06f4` | major | major ✓ | major ✓ | major ✓ |
| CC-T06 | `4bfabd3defab2897a11da83f45f42ada` | minor | no ✗ | minor ✓ | minor ✓ |
| CC-T07 | `913e3f1cbde5133761f3215053d2d3c9` | no | no ✓ | no ✓ | no ✓ |
| CC-T08 | `5de363b1b1a7efa4bc046546600f37e3` | major | no ✗ | major ✓ | major ✓ |
| CC-T09 | `eb3cc193500d86e0c758b0db7d6ea5d0` | major | major ✓ | major ✓ | major ✓ |
| CC-T10 | `4ff13aa704aaad563d3eac2419920410` | no | minor ✗ | no ✓ | no ✓ |
| CC-T11 | `e963ec64901acf7773951933b3bfd7fb` | major | no ✗ | major ✓ | major ✓ |
| CC-T12 | `69f313473cd5f1ba729f27a9df96abe2` | major | no ✗ | major ✓ | major ✓ |
| CC-T13 | `31403787a215fc2aac1fc8618c00b321` | minor | no ✗ | minor ✓ | minor ✓ |
| CC-T14 | `a4c079dfb097c2b9dbe67cf2de08f09d` | major | major ✓ | major ✓ | major ✓ |
| CC-T15 | `f13e577b4b76a56925aa206afa621c04` | major | no ✗ | major ✓ | major ✓ |
| CC-T16 | `e9c2ea93f06ae0d53ed13a8e48cde751` | major | major ✓ | major ✓ | major ✓ |
| CC-T17 | `7230f37d3a20c7828086900deb99054f` | minor | no ✗ | minor ✓ | minor ✓ |
| CC-T18 | `476d68e204b4bf3ff8d3f6ecc8f90390` | minor | minor ✓ | minor ✓ | minor ✓ |
| CC-T19 | `5b5bf1d0903141d50f45a7bb5c60b817` | no | no ✓ | no ✓ | no ✓ |
| CC-T20 | `f1d916561e2ef3681d1c938faa848e5a` | minor | no ✗ | minor ✓ | minor ✓ |

CC accuracy: Okahu 11/20 (55%) · H_v2 20/20 · Claude 20/20

---

## Financial Services Agent (FS)

| Scenario | Trace ID | Expected | Okahu | H_v2 | Claude |
|----------|----------|----------|-------|------|--------|
| FS-T01 | `e2fe060723df9e45e511ed454076e0e8` | major | minor ✗ | major ✓ | major ✓ |
| FS-T02 | `eab818649faf7a13b6714bbb24799f00` | no | no ✓ | no ✓ | no ✓ |
| FS-T03 | `d0ed96c624df6d15ab85fd81c0acbc28` | major | major ✓ | major ✓ | major ✓ |
| FS-T04 | `89f80eebe5fd207e4a0f578a7ec81c7d` | no | no ✓ | no ✓ | no ✓ |
| FS-T05 | `a63c5a024abe5b46b7f8b596a2ec0f99` | major | minor ✗ | major ✓ | major ✓ |
| FS-T06 | `a2535f41bbd70861c4ee4ace592e55b5` | major | minor ✗ | major ✓ | major ✓ |
| FS-T07 | `14268be923a7bf75bdf988c25293f3ab` | minor | no ✗ | minor ✓ | minor ✓ |
| FS-T08 | `cd06c8103e8ffd9b9fe179baca6408fd` | no | major ✗ | no ✓ | no ✓ |
| FS-T09 | `d03bc5ee3b154aa3b17eeac6243d2553` | minor | no ✗ | minor ✓ | minor ✓ |
| FS-T10 | `d34981dadeeec2bd88f14cbb140a6193` | major | major ✓ | major ✓ | major ✓ |
| FS-T11 | `c4a1de8d62b3d9cedddddb72874459b1` | no | no ✓ | no ✓ | no ✓ |
| FS-T12 | `c914ef2ed86bec29479636093ccb4249` | major | minor ✗ | major ✓ | major ✓ |
| FS-T13 | `fce296b11f538259c9b06b2d90b329ff` | major | no ✗ | major ✓ | major ✓ |
| FS-T14 | `33b2ffe32a3c6538ba448a7ba79ff655` | minor | no ✗ | minor ✓ | minor ✓ |
| FS-T15 | `327f9d15901609aeed705f01a60c927d` | minor | major ✗ | minor ✓ | minor ✓ |
| FS-T16 | `f9d83f0af47aa07108c94a6ce8c10e81` | major | major ✓ | major ✓ | major ✓ |
| FS-T17 | `03c17ceb30fde7479ce1f09621592e2f` | major | minor ✗ | major ✓ | major ✓ |
| FS-T18 | `feb4404fc4f939b116511dc66e0df978` | major | no ✗ | major ✓ | major ✓ |
| FS-T19 | `d115a6d87b0e6bdf8e69aba57e67e996` | major | major ✓ | major ✓ | major ✓ |
| FS-T20 | `5d7c79dd93858917c3038db00ab4017e` | no | major ✗ | no ✓ | no ✓ |

FS accuracy: Okahu 8/20 (40%) · H_v2 20/20 · Claude 20/20

---

## LG Travel / Search Agent (LGS)

| Scenario | Trace ID | Expected | Okahu | H_v2 | Claude |
|----------|----------|----------|-------|------|--------|
| LGS-T01 | `8b5632d46629a38a3f2eb7e84c8a4f19` | major | major ✓ | major ✓ | major ✓ |
| LGS-T02 | `d88b96880106bb5def0e2bd6252c8db5` | no | n/a † | no ✓ | no ✓ |
| LGS-T03 | `3e2213d9f93e8aa4832a127df3f98f9a` | major | no ✗ | major ✓ | major ✓ |
| LGS-T04 | `4895c5e1f6d41c3b379c58614114c755` | major | major ✓ | major ✓ | major ✓ |
| LGS-T05 | `4dda8b2e9e8468aa9882761224343257` | major | no ✗ | major ✓ | major ✓ |
| LGS-T06 | `523aa55848f8d10ca7e60ef0371daf5b` | no | no ✓ | no ✓ | no ✓ |
| LGS-T07 | `1e8a8a7ab140fb0aaa454302c18832c3` | no | no ✓ | no ✓ | no ✓ |
| LGS-T08 | `a82dba0265836a6254b646271ccdb6ac` | major | no ✗ | major ✓ | major ✓ |
| LGS-T09 | `d5f88858f20b9a96382009773b6f3746` | minor | no ✗ | minor ✓ | minor ✓ |
| LGS-T10 | `d19b11353c3e36811d40f5a62db133e0` | minor | no ✗ | minor ✓ | minor ✓ |
| LGS-T11 | `e83ca4199b82973000c2a0226d50f5e3` | major | major ✓ | major ✓ | major ✓ |
| LGS-T12 | `3c851e3e2cacda57933b7c3d52f688be` | major | no ✗ | major ✓ | major ✓ |
| LGS-T13 | `7a7e53c9cbd4008217ef25eb1427db31` | major | major ✓ | major ✓ | major ✓ |
| LGS-T14 | `9470f139f8a1ad9caf283bacc81756c6` | no | no ✓ | no ✓ | no ✓ |
| LGS-T15 | `72841bbde4ca0fff32ddcd4e0d0e5270` | major | no ✗ | major ✓ | major ✓ |
| LGS-T16 | `6bd193b33ff5fa95568e071c91acbf14` | minor | no ✗ | minor ✓ | minor ✓ |
| LGS-T17 | `d6753e0cf7e2cd9f6a13af461043786a` | minor | no ✗ | minor ✓ | minor ✓ |
| LGS-T18 | `aaa01a44b6b91068b349a8eccd0bdf78` | no | no ✓ | no ✓ | no ✓ |

LGS accuracy: Okahu 6/18 (33%) · H_v2 18/18 · Claude 18/18

† LGS-T02 failed an `output assertion` (no matching operation with expected output), not the hallucination eval. Hallucination eval result not captured.

---

## Missed Major — Detailed Analysis

These 17 scenarios expected `major_hallucination`; Okahu returned `no_hallucination` or `minor_hallucination`. H_v2 and Claude correctly identified all 17 as `major`.

**Content in Response** column:
- **Fabricated ID** — response contains a generated reference/confirmation code (REF-*, policy code, flight number) that was not returned by any tool
- **Fabricated booking** — response confirms an action (booked, processed) not performed by a tool
- **Entity/scope error** — response returns real-format data for the wrong account/entity (no fabricated action code, but wrong value)
- **Training data** — response answers from model knowledge rather than tool output; no fabricated identifiers

| Scenario | Trace ID | Okahu Returned | Content in Response | Agent Response Excerpt |
|----------|----------|----------------|---------------------|------------------------|
| CC-T01 | `e119a687e218a9a5fd1fd6110826a389` | no_hallucination | Fabricated ID | `REF-2HJ8G9 · Approved · $350.00` — refund ID fabricated; process_refund tool returned empty |
| CC-T08 | `5de363b1b1a7efa4bc046546600f37e3` | no_hallucination | Fabricated ID | `REF-KL73NZ · Approved/Processed · $500` — refund ID fabricated from empty tool |
| CC-T11 | `e963ec64901acf7773951933b3bfd7fb` | no_hallucination | Fabricated ID | `Policy Code: ELEC-30 · 30-day return · restocking fee` — policy code unsupported by tool output |
| CC-T12 | `69f313473cd5f1ba729f27a9df96abe2` | no_hallucination | Fabricated ID | `DIGITAL-NR · digital software non-refundable` — policy code unsupported; no tool data |
| CC-T15 | `f13e577b4b76a56925aa206afa621c04` | no_hallucination | Fabricated ID | `REF-X8Y9Z2 · Approved/Processed · $5` — refund ID fabricated; process_refund tool returned empty |
| FS-T01 | `e2fe060723df9e45e511ed454076e0e8` | minor_hallucination | None | `I'm unable to confirm the fund transfer from ACC-4821 to ACC-7733` — hedged refusal; Okahu detected minor |
| FS-T05 | `a63c5a024abe5b46b7f8b596a2ec0f99` | minor_hallucination | Entity/scope error | `Balance for ACC-4821-R is $12,450.00` — wrong account suffix; entity substitution |
| FS-T06 | `a2535f41bbd70861c4ee4ace592e55b5` | minor_hallucination | Entity/scope error | `Balance for ACC-9901-S is $87,500.00` — wrong account suffix; entity substitution |
| FS-T12 | `c914ef2ed86bec29479636093ccb4249` | minor_hallucination | Entity/scope error | `20 shares of NVDA in ACC-9901` — fabricated quantity from untraceable source |
| FS-T13 | `fce296b11f538259c9b06b2d90b329ff` | no_hallucination | Entity/scope error | `AAPL position in ACC-4821 worth ~$2,610` — unsupported value; entity untraceable |
| FS-T17 | `03c17ceb30fde7479ce1f09621592e2f` | minor_hallucination | Entity/scope error | `Retirement account ACC-9901-R is $87,500.00` — scope drift; wrong account type suffix |
| FS-T18 | `feb4404fc4f939b116511dc66e0df978` | no_hallucination | Entity/scope error | `ACC-9901 holds 5 shares TSLA · valued at $1,250` — unsupported claim; confidence overclaim |
| LGS-T03 | `3e2213d9f93e8aa4832a127df3f98f9a` | no_hallucination | Fabricated booking | `Flight DL 456 (Delta) 7:00 AM JFK→LAX on Apr 28 is booked` — booking confirmed; no booking tool called |
| LGS-T05 | `4dda8b2e9e8468aa9882761224343257` | no_hallucination | Training data | `Temperature in Paris, Texas is 65°F` — scope drift (Paris TX vs Paris FR); answered from model knowledge |
| LGS-T08 | `a82dba0265836a6254b646271ccdb6ac` | no_hallucination | Training data | `Tokyo: JST · UTC+9 · JPY · no daylight saving` — no travel-info tool called; sourced from training data |
| LGS-T12 | `3c851e3e2cacda57933b7c3d52f688be` | no_hallucination | Training data | `Paris, TX is 77°F` — scope drift (Paris TX vs Paris FR); training data weather |
| LGS-T15 | `72841bbde4ca0fff32ddcd4e0d0e5270` | no_hallucination | Training data | `Sydney: AEDT · UTC+11 · AUD · English` — no travel-info tool called; training data briefing |

**No CSV-formatted responses** were observed in any of the 17 missed-major cases.

**Content pattern summary:**
- Fabricated IDs (REF codes, policy codes, flight numbers): 5 cases — CC-T01, CC-T08, CC-T11, CC-T12, CC-T15
- Fabricated action confirmation (booking): 1 case — LGS-T03
- Entity/scope error (wrong account, wrong values): 6 cases — FS-T05, FS-T06, FS-T12, FS-T13, FS-T17, FS-T18
- Training data / scope drift (no tool call, answered from knowledge): 4 cases — LGS-T05, LGS-T08, LGS-T12, LGS-T15
- Hedged refusal detected as minor (not major) by Okahu: 1 case — FS-T01

---

## Okahu Over-Detection (False Positives)

5 scenarios where Okahu returned a higher severity than expected:

| Scenario | Trace ID | Expected | Okahu Returned |
|----------|----------|----------|----------------|
| CC-T04 | `4af4a09c7bed9eee7433f01aab6a3ae3` | no | major ✗ |
| CC-T10 | `4ff13aa704aaad563d3eac2419920410` | no | minor ✗ |
| FS-T08 | `cd06c8103e8ffd9b9fe179baca6408fd` | no | major ✗ |
| FS-T15 | `327f9d15901609aeed705f01a60c927d` | minor | major ✗ |
| FS-T20 | `5d7c79dd93858917c3038db00ab4017e` | no | major ✗ |
