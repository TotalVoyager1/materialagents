from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from orchestrator.agents import AgentSuite
from orchestrator.documents import DocumentIndex


@dataclass
class WorkflowState:
    task: str
    template: str
    results: dict[str, str] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    artifacts_ready: bool = False
    search_query: str = ""


class Orchestrator:
    def __init__(self, agents: AgentSuite):
        self.agents = agents

    def run_step(
        self,
        state: WorkflowState,
        step: str,
        index: DocumentIndex,
        source_files: dict[str, str],
    ) -> str:
        task = state.task.strip()
        if not task:
            raise ValueError("Введите описание задачи.")
        required_previous = {
            "research": "plan",
            "draft": "research",
            "critic": "draft",
            "tests": "critic",
            "code_review": "tests",
            "summary": "code_review",
        }
        prerequisite = required_previous.get(step)
        if prerequisite and prerequisite not in state.results:
            raise ValueError(
                f"Сначала выполните предыдущий этап «{prerequisite}»."
            )
        input_keys = sorted(state.results.keys())
        if step == "plan":
            output = self.agents.planner(task, state.template)
        elif step == "research":
            query = state.search_query.strip() or task
            matches = index.search(query, limit=6)
            excerpts = "\n\n".join(
                f"[{chunk.filename}; релевантность {score:.2f}]\n{chunk.text}"
                for chunk, score in matches
            ) or (
                f"Для запроса «{query}» подходящих фрагментов локальных документов "
                "не найдено."
            )
            output = self.agents.researcher(task, excerpts, state.results.get("plan", ""))
        elif step == "draft":
            output = self.agents.writer(
                task, state.template, state.results.get("plan", ""), state.results.get("research", "")
            )
        elif step == "critic":
            output = self.agents.critic(task, state.results.get("draft", ""))
        elif step == "tests":
            output = self.agents.tester(
                task, state.results.get("plan", ""), state.results.get("draft", "")
            )
        elif step == "code_review":
            code = "\n\n".join(
                f"### {name}\n{text}" for name, text in source_files.items()
                if name.lower().endswith((".py", ".js", ".ts", ".java", ".cs", ".cpp", ".c", ".go", ".rs"))
            )
            output = self.agents.code_reviewer(task, code)
        elif step == "summary":
            output = self.agents.summarizer(
                task,
                state.results.get("draft", ""),
                state.results.get("critic", ""),
                state.results.get("tests", ""),
            )
        else:
            raise ValueError(f"Неизвестный этап: {step}")
        state.results[step] = output
        state.events.append(
            {
                "step": step,
                "agent": {
                    "plan": "Planner",
                    "research": "Researcher",
                    "draft": "Writer",
                    "critic": "Critic",
                    "tests": "Tester",
                    "code_review": "CodeReviewer",
                    "summary": "Summarizer",
                }[step],
                "status": "completed",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "input_keys": input_keys,
                "output_chars": len(output),
            }
        )
        return output
