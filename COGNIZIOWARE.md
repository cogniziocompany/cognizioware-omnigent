# Cognizioware fork of omnigent

- `main`: exact mirror of upstream `omnigent-ai/omnigent`, fast-forwarded daily by the local routine `agents/orchestrator/scripts/omnigent-fork-sync.sh` on VM211 (claude-code-teams; no GitHub-hosted Actions).
- `cognizioware-base` (default branch): the base our deployed server is built from.
  - Upstream pin: `d98b5a7df35eaa8c6bc27b1f4c87458406b15dcc`.
  - Patch 1: GET elicitation response returns `requestedSchema` (byte-identical to the file in `cognizioware-omnigent-server:qualified` on VM310, checked 2026-10-24).
  - V1.5 work (CAP check on elicitation resolve, task q-33327dce814945c3) lands here by PR.
- Moving the pin to a newer upstream commit is a reviewed PR from `main`, never automatic.
