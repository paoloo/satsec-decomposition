# Release-risk and intended-use statement

This artifact is released for reproducible research and authorized, development-time
security testing. It is not a planner for operations against live spacecraft.

## Included capability

The corpus contains public-source and canonical security behaviors, SPARTA identifiers,
author-written action summaries, and deterministic bench checks. Model outputs select and
order candidates already supplied in the prompt. The artifact does not retrieve targets,
discover live assets, establish radio links, execute commands, validate credentials, or
automate any physical or network action.

## Risk controls

- All evaluation objectives are passive research cases, authorized research abstractions,
  canonical standards patterns, or emulated development-bench scenarios.
- Prompts explicitly restrict use to emulated or consented targets.
- Raw outputs are research measurements and may be wrong, unsafe, or hallucinated. They
  must not be treated as operational instructions.
- The oracle candidate-set experiment does not show that the adapter can find relevant
  techniques in an open environment.
- Exact identifier scores do not validate action semantics, causal feasibility, or safe
  execution. Independent expert validation is still pending.
- No exploit code, credentials, target coordinates, transmit parameters, or live-system
  automation is included.

## Residual dual-use risk

Public descriptions of eavesdropping, replay, unauthenticated commanding, GNSS spoofing,
and debug-interface abuse are inherently dual use. Packaging them into structured plans
may reduce the effort needed to organize already-public information. The countervailing
research value is a transparent benchmark for testing whether small local models can
select standards-grounded defensive test steps and expose their failure modes. We limit
claims, retain the authorization boundary in prompts and documentation, publish failed
outputs and uncertainty rather than a deployment recipe, and require human review before
any use in a real test program.

Users are responsible for authorization, spectrum rules, export controls, safety review,
and responsible disclosure applicable to their jurisdiction and target environment.

