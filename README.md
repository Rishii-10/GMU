# Beyond Triage / Rural Health Triage POC

All project documentation is in [`docs/`](docs/README.md).

| Document | What it covers |
|---|---|
| [`docs/README.md`](docs/README.md) | Project overview, architecture diagrams, implementation status |
| [`docs/RULE_ENGINE_DESIGN_FINAL.md`](docs/RULE_ENGINE_DESIGN_FINAL.md) | Rule engine: CP gate, AHP scorer, adaptive follow-up, validation results |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | System-wide architecture and repository structure |
| [`docs/AGENT1_README.md`](docs/AGENT1_README.md) | Agent 1 LLM extraction layer deep-dive |

## Quick start

```
cd triage-poc
./.venv/bin/python -m pytest tests/ -v
```
