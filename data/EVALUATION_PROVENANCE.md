# Fixed-evaluation case provenance

This ledger gives stable primary links and evidence-scope notes for the six
development-facing cases in the fixed split. The machine-readable
`artifacts/results/case_provenance.json` covers **all 24 cases**, resolving every
decomposition to its exact corpus root record and recorded source. Regenerate it with
`python tools/build_case_provenance.py`.

The table below records
what each public source supports and what the authors added when converting the source into
an ordered benchmark reference.
The source supports the case mechanism; it does **not** independently certify the exact
SPARTA mapping, step boundary, order, or check wording chosen by the authors.

The local SPARTA corpus was audited against the live Aerospace SPARTA pages on
2026-07-27 after a mapping recheck on 2026-07-26. SPARTA is a living resource,
so these dates identify the locally audited snapshot rather than an official SPARTA
release number. The public framework and STIX 2.1 download are described in the
[official SPARTA user guide](https://sparta.aerospace.org/resources/user-guide).

| Case | Classification | Primary public evidence | Evidence scope | Author-constructed benchmark content |
|---|---|---|---|---|
| `Turla-satellite-C2` | Documented incident analysis | Kaspersky GReAT, [Satellite Turla: APT Command and Control in the Sky](https://securelist.com/satellite-turla-apt-command-and-control-in-the-sky/72081/) (2015) | Abuse of unencrypted one-way satellite Internet downlinks, subscriber IPs, and passive reception within the footprint | Three-step boundary, SPARTA identifiers, ordering, and deterministic checks |
| `GNSS-spoofing` | Controlled research demonstration | Bhatti and Humphreys, [Hostile Control of Ships via False GPS Signals: Demonstration and Detection](https://doi.org/10.1002/navi.183), *NAVIGATION* 64(1), 51--66 (2017) | Counterfeit civil GNSS transmission and controlled carry-off of a receiver solution | Satellite-security framing, two-step abstraction, SPARTA identifiers, and checks |
| `Pavur-SATCOM-eavesdrop` | Controlled research measurement | James Pavur, [Whispers Among the Stars](https://i.blackhat.com/USA-20/Wednesday/us-20-Pavur-Whispers-Among-The-Stars-Perpetrating-And-Preventing-Satellite-Eavesdropping-Attacks.pdf), Black Hat USA (2020) | Consumer-equipment capture and reconstruction of unencrypted satellite broadband traffic | Three-step boundary, SPARTA identifiers, ordering, and checks |
| `Space-Odyssey-unauth-TC` | Authorized research on real satellites; evaluation is bench-only | Willbold et al., [Space Odyssey: An Experimental Software Security Analysis of Satellites](https://doi.org/10.1109/SP46215.2023.00131), IEEE S&P (2023) | Experimental evidence of missing telecommand authentication and security consequences | Emulated objective, four-step abstraction, SPARTA identifiers, causal order, and checks |
| `TC-replay-no-SDLS` | Canonical emulated pattern | CCSDS, [Space Data Link Security Protocol, CCSDS 355.0-B-2](https://public.ccsds.org/Pubs/355x0b2.pdf) (2022); Aerospace SPARTA `EX-0001` | Security protocol context and replay/anti-replay threat model | Vulnerable no-SDLS counterfactual, four-step reference, mappings, and checks |
| `JTAG-debug-firmware` | Canonical development-bench pattern | Aerospace SPARTA entries used by the reference and the framework's countermeasure to disable or remove test/debug ports; see the [SPARTA user guide](https://sparta.aerospace.org/resources/user-guide) | Enabled debug-interface risk and applicable SPARTA vocabulary | Entire bench scenario, three-step order, mappings, and checks; this is not represented as a historical incident |

## Validation status

- Dataset construction and identifier/page-title consistency are machine-audited.
- All 24 decomposition roots resolve uniquely to a nonempty source record in the complete
  case-provenance manifest.
- Case mechanisms are traceable to the sources above.
- The reference decompositions were collectively reviewed within the author team against
  the recorded sources and SPARTA. They operationalize the benchmark and are not represented
  as independently certified ground truth.
- `artifacts/results/reference_evidence_coverage.md` reports 24/24 source resolution,
  complete step fields, exact candidate inclusion, and 34/34 live identifier/title checks.
- The blank packet in `artifacts/expert_audit/` supports optional future community review;
  it is not an acceptance gate for the present paper.
