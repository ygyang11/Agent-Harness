# Agent Harness Examples

## Prerequisites

1. Install project dependencies from the repository root:
   `pip install -e ".[dev]"`
2. Ensure a valid `config.yaml` exists at the repository root.
3. For trace file verification, configure:
   - `tracing.enabled: true`
   - `tracing.exporter: json_file`
   - `tracing.export_path: ./traces`
4. `plan_and_execute.py` and `deep_research.py` use the built-in `web_search` tool,
   which reads search settings from the loaded global `config.yaml`.

## Run examples independently

- `python examples/react_agent.py`
- `python examples/plan_and_execute.py`
- `python examples/multi_agent_pipeline.py`
- `python examples/agent_team.py`
- `python examples/deep_research.py`

| File | Description |
|------|-------------|
| `react_agent.py` | ReActAgent with `@tool` decorated functions — the fundamental agent pattern |
| `plan_and_execute.py` | PlanAgent for multi-step task decomposition/execution with built-in `web_search` |
| `multi_agent_pipeline.py` | Pipeline sequential chaining + DAG parallel orchestration |
| `agent_team.py` | AgentTeam with supervisor, debate, and round-robin collaboration modes |
| `deep_research.py` | Multi-agent deep research workflow — planner + parallel DAG researchers + team review + final synthesis |
