# Authored-reference evidence coverage

This is a construct-transparency audit, not independent semantic certification.
References are author-defined benchmark operationalizations reviewed against the
recorded public sources, SPARTA identifiers, and explicit construction rules.

## Exhaustive checks

- Cases with a resolved, non-empty recorded source: 24/24
- References with an Action and Check field on every step: 24/24
- References whose technique identifiers all occur in the supplied candidate set: 24/24
- Unique reference technique identifiers checked against live SPARTA pages: 34/34

## Per-case coverage

| Case | Split | Steps | Unique techniques | Source | Complete fields | All techniques supplied |
|---|---:|---:|---:|---:|---:|---:|
| Starlink-UT-fault-injection | train | 4 | 4 | yes | yes | yes |
| Santamarta-SATCOM-terminals | train | 4 | 4 | yes | yes | yes |
| Turla-satellite-C2 | test | 3 | 3 | yes | yes | yes |
| GNSS-spoofing | test | 2 | 2 | yes | yes | yes |
| Pavur-SATCOM-eavesdrop | test | 3 | 3 | yes | yes | yes |
| Landsat-7/Terra-EOS-AM-1 | train | 3 | 3 | yes | yes | yes |
| generic-fw-memsafety | train | 4 | 3 | yes | yes | yes |
| generic-cmd-injection | train | 4 | 3 | yes | yes | yes |
| generic-protocol-integrity | train | 3 | 3 | yes | yes | yes |
| generic-supply-chain | train | 3 | 3 | yes | yes | yes |
| ROSAT | train | 3 | 3 | yes | yes | yes |
| jamming | train | 3 | 3 | yes | yes | yes |
| BEESAT-1 | train | 4 | 4 | yes | yes | yes |
| Space-Odyssey-unauth-TC | test | 4 | 4 | yes | yes | yes |
| OPS-SAT-authorized-takeover | train | 4 | 4 | yes | yes | yes |
| GMR-satphone-cipher | train | 3 | 3 | yes | yes | yes |
| Iridium-plaintext-eavesdrop | train | 3 | 3 | yes | yes | yes |
| TC-replay-no-SDLS | test | 4 | 4 | yes | yes | yes |
| JTAG-debug-firmware | test | 3 | 3 | yes | yes | yes |
| UART-console-check | train | 4 | 4 | yes | yes | yes |
| SDR-link-bringup-check | train | 4 | 4 | yes | yes | yes |
| Onboard-bus-probe-check | train | 3 | 3 | yes | yes | yes |
| Safe-mode-abuse | train | 3 | 3 | yes | yes | yes |
| ViaSat-KA-SAT | train | 5 | 5 | yes | yes | yes |

Source records support case provenance and mechanisms. Exact step boundaries,
mappings, order, and check wording remain authored task definitions.
