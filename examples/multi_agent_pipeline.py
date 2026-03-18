"""Pipeline sequential chaining + DAG parallel orchestration.

Demonstrates two orchestration patterns in one file:
  Part 1 — Pipeline: sequential agent chain with input transforms.
  Part 2 — DAG: parallel execution with dependency resolution.
"""

import asyncio
from pathlib import Path

from agent_harness import ConversationalAgent, HarnessConfig
from agent_harness.orchestration import (
    Pipeline,
    PipelineStep,
    DAGOrchestrator,
    DAGNode,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent


async def run_pipeline(config: HarnessConfig) -> None:
    """Part 1: Sequential pipeline — researcher feeds into writer."""
    print("=" * 60)
    print("PART 1: Pipeline (Sequential)")
    print("=" * 60)

    researcher = ConversationalAgent(
        name="researcher",
        system_prompt=(
            "You are a research assistant. Given a topic, provide 3-4 key facts "
            "and recent developments. Be concise and factual."
        ),
        config=config,
    )

    writer = ConversationalAgent(
        name="writer",
        system_prompt=(
            "You are a skilled writer. Given research notes, craft a polished "
            "two-paragraph summary suitable for a newsletter. Keep it engaging."
        ),
        config=config,
    )

    pipeline = Pipeline(
        steps=[
            PipelineStep(agent=researcher, name="research"),
            PipelineStep(
                agent=writer,
                name="writing",
                transform=lambda research: (
                    f"Write a newsletter paragraph based on these research notes:\n\n"
                    f"{research}"
                ),
            ),
        ],
    )

    result = await pipeline.run("The impact of large language models on software engineering")

    print(f"\nFinal output:\n{result.output}\n")
    print(f"Steps completed: {list(result.step_results.keys())}")
    if result.skipped_steps:
        print(f"Skipped: {result.skipped_steps}")


async def run_dag(config: HarnessConfig) -> None:
    """Part 2: DAG orchestration — parallel branches with merge."""
    print("\n" + "=" * 60)
    print("PART 2: DAG (Parallel Orchestration)")
    print("=" * 60)

    technical = ConversationalAgent(
        name="technical_analyst",
        system_prompt=(
            "You are a technical analyst. Provide a brief technical assessment "
            "of the given topic. Focus on capabilities and limitations. 2-3 sentences."
        ),
        config=config,
    )

    market = ConversationalAgent(
        name="market_analyst",
        system_prompt=(
            "You are a market analyst. Provide a brief market/business assessment "
            "of the given topic. Focus on adoption and economics. 2-3 sentences."
        ),
        config=config,
    )

    social = ConversationalAgent(
        name="social_analyst",
        system_prompt=(
            "You are a social impact analyst. Assess the societal implications "
            "of the given topic. Focus on jobs and education. 2-3 sentences."
        ),
        config=config,
    )

    synthesizer = ConversationalAgent(
        name="synthesizer",
        system_prompt=(
            "You synthesize multiple analyst perspectives into a cohesive "
            "executive summary. Be concise — one paragraph max."
        ),
        config=config,
    )

    def merge_analyses(results: dict) -> str:
        parts = []
        for node_id in ["technical", "market", "social"]:
            if node_id in results:
                parts.append(f"[{node_id.upper()}]: {results[node_id].output}")
        return "Synthesize these analyst reports:\n\n" + "\n\n".join(parts)

    dag = DAGOrchestrator(
        nodes=[
            DAGNode(id="technical", agent=technical),
            DAGNode(id="market", agent=market),
            DAGNode(id="social", agent=social),
            DAGNode(
                id="synthesis",
                agent=synthesizer,
                dependencies=["technical", "market", "social"],
                input_transform=merge_analyses,
            ),
        ],
        config=config,
    )

    result = await dag.run("Autonomous vehicles in urban transportation")

    print(f"\nExecution order: {result.execution_order}")
    for node_id, agent_result in result.outputs.items():
        label = node_id.upper()
        print(f"\n[{label}]: {agent_result.output[:200]}...")

    print(f"\n--- Final Synthesis ---\n{result.outputs['synthesis'].output}")


async def main() -> None:
    config = HarnessConfig.load(PROJECT_ROOT / "config.yaml")

    await run_pipeline(config)
    await run_dag(config)


if __name__ == "__main__":
    asyncio.run(main())
