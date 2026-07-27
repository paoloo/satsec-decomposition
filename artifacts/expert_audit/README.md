# Optional community-review packet

Status: **optional extension, not undertaken**. The paper treats its references as
author-defined benchmark operationalizations and does not require this packet to pass.
It is supplied so later work can compare author-team judgments with a qualified non-author.

`items.jsonl` contains 78 items: all 24 author references (`REF-*`) and 54 blinded
generated outputs (`OUT-*`) for the six fixed cases, covering one preregistered
representative decoding seed (seed 0), three model sizes, and adapter/schema/two-shot conditions. The output labels
do not reveal model size or condition. The adjudication key is intentionally written
outside this directory to `../results/expert_audit_key.json`; do not give it to auditors.

If undertaken, the frozen comparison threshold covers the 24 `REF-*` items. The 54
`OUT-*` items are a separately reported semantic-output study. Blank templates are the
expected released state and do not indicate missing evidence for the present paper.

Audit protocol:

1. Complete `reference_responses.csv`: exactly one `internal-author` and one
   `independent-non-author` row per reference. Use stable blinded IDs. Record expertise,
   years of relevant practice, conflicts, review date, exact material reviewed, and whether
   the reviewer is independent of the authors. A non-author reviewer qualifies through
   documented professional or research experience in security plus direct familiarity with
   at least one of satellite, embedded, or communications security and with SPARTA-style
   technique mapping; record that basis rather than relying on a title alone.
2. Rate each dimension from 1 (incorrect/unusable) to 5 (correct/strong). Judge mappings
   against SPARTA, actions against the stated authorized development-time objective,
   order against causal prerequisites, and checks for deterministic executability.
3. Mark any safety, factual, mapping, or causal defect that invalidates the plan as a
   critical error. Mark overall acceptability independently of formatting quality.
4. Freeze each first-pass record before discussion or condition disclosure. Keep the signed
   form privately, put its stable record ID and SHA-256 in every applicable row, and set the
   attestation field only after signing. This anonymous packet publishes hashes, not names.
5. Record requested corrections without changing the first-pass rows. Put later disagreement
   resolutions in `adjudications.csv` and source changes in `correction_ledger.csv`.
6. The optional study's frozen threshold is: both roles on every reference, no critical-error rating, no
   unresolved correction, and median >=4 on every dimension. The analyzer reports agreement
   only for ratings of the same item and fails closed on missing role or provenance fields.

Only after real reviewers volunteer to perform this follow-up, run
`python tools/analyze_expert_audit.py`. The untouched template intentionally produces an
`incomplete` optional-study report and a nonzero exit status.

The packet deliberately does not convert blank ratings into a result. Analysis must occur
only after real, completed responses are received.
