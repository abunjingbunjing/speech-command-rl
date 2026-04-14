# Ethics Statement

## Risks and mitigations
1. **Accent/dialect bias** — model may perform worse on non-native English.
   Mitigation: Report per-class accuracy; acknowledge limitation in disclaimers.

2. **Privacy** — audio data collection risks.
   Mitigation: Only public consented data (Google Speech Commands) used.
   No user audio is collected or stored.

3. **Accessibility** — may fail for users with speech impairments.
   Mitigation: Display confidence scores; provide text fallback.

4. **Misuse** — could be used for voice surveillance.
   Mitigation: Clearly scope to command recognition only

## Fairness checks
- Per-class accuracy reported to detect systematic failures.
- No PII in dataset.

## Intended use
Research and education only. Not for clinical or safety-critical deployment.
