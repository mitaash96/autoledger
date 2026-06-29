# opencode sandbox

Per-run working directories for the sandboxed `opencode` codegen backend
(`src/autogen/codegen/opencode_runner.py`) are created here as
`<bank>_<instrument>_<timestamp>/`.

Each run dir is confined via `bwrap`: opencode can read/write only inside it and
cannot reach the repo, `data/`, or `.env`. It contains **only anonymized assets** —
`fixtures/anon_tables.json`, `target_schema.json`, the in-sandbox test harness
`test_logic.py`, `AGENTS.md` (the task spec), `opencode.json`, and the generated
`pipeline.py`.

These directories are git-ignored (this README is the only tracked file). You can
inspect and edit the assets here to see exactly what opencode is working with.
