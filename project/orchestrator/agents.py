from collections.abc import Callable

from orchestrator.openrouter import OpenRouterClient


class AgentSuite:
    def __init__(
        self,
        client: OpenRouterClient,
        on_delta: Callable[[str], None] | None = None,
    ):
        self.client = client
        self.on_delta = on_delta

    def _ask(
        self, name: str, role: str, user_task: str, instruction: str, context: str
    ) -> str:
        return self.client.generate(
            system_instruction=(
                f"Ты агент {name} в мультиагентной системе помощи студентам. "
                f"{role} Работай на русском языке, будь точным и не выдумывай факты."
            ),
            prompt=(
                f"Задача пользователя:\n{user_task}\n\n"
                f"Текущее задание:\n{instruction}\n\n"
                f"Контекст предыдущих этапов:\n{context}"
            ),
            on_delta=self.on_delta,
        )

    def planner(self, task: str, template: str) -> str:
        return self._ask(
            "Planner",
            "Составляй выполнимый план проекта с этапами, результатами и критериями завершения.",
            task,
            f"Составь пошаговый план. Шаблон отчёта: {template}.",
            "",
        )

    def researcher(self, task: str, retrieved: str, plan: str) -> str:
        return self._ask(
            "Researcher",
            "Отвечай только с опорой на переданные выдержки локальных документов. "
            "Указывай название исходного файла; если данных не хватает, прямо сообщай об этом.",
            task,
            f"Найди полезные сведения по теме. Выдержки из локальных документов:\n{retrieved}",
            f"План:\n{plan}",
        )

    def writer(self, task: str, template: str, plan: str, research: str) -> str:
        return self._ask(
            "Writer",
            "Создавай структурированный черновик учебного отчёта в Markdown, "
            "разделяй подтверждённые сведения и требующие проверки утверждения.",
            task,
            f"Подготовь черновик для шаблона «{template}».",
            f"План:\n{plan}\n\nИсследование:\n{research}",
        )

    def critic(self, task: str, draft: str) -> str:
        return self._ask(
            "Critic",
            "Проверяй результат по чек-листу: соответствие задаче, полнота, логика, "
            "источники, ясность, непроверенные утверждения. Для каждого пункта укажи статус.",
            task,
            "Проведи проверку черновика и предложи конкретные исправления.",
            f"Черновик:\n{draft}",
        )

    def tester(self, task: str, plan: str, draft: str) -> str:
        return self._ask(
            "Tester",
            "Генерируй проверяемые тест-кейсы с идентификатором, предусловиями, шагами и ожидаемым результатом.",
            task,
            "Сформируй тест-кейсы для проверки результата проекта.",
            f"План:\n{plan}\n\nЧерновик:\n{draft}",
        )

    def code_reviewer(self, task: str, code: str) -> str:
        return self._ask(
            "CodeReviewer",
            "Проверяй код на корректность, обработку ошибок, безопасность, читаемость и тестируемость. "
            "Не утверждай, что запускал код.",
            task,
            "Проведи статическое ревью предоставленного кода. Если кода нет, явно укажи это.",
            f"Фрагменты кода из прикреплённых файлов:\n{code or 'Кодовые файлы не предоставлены.'}",
        )

    def summarizer(self, task: str, draft: str, critic: str, tests: str) -> str:
        return self._ask(
            "Summarizer",
            "Составляй краткое итоговое резюме и перечисляй ограничения и оставшиеся действия.",
            task,
            "Подготовь резюме результата проекта.",
            f"Черновик:\n{draft}\n\nПроверка:\n{critic}\n\nТест-кейсы:\n{tests}",
        )
