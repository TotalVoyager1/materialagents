import hashlib
from datetime import datetime, timezone
from io import BytesIO

import pandas as pd
import httpx
import streamlit as st
from dotenv import load_dotenv

from orchestrator.agents import AgentSuite
from orchestrator.analytics import analyze_csv
from orchestrator.artifacts import build_docx, build_json, build_markdown, build_pdf
from orchestrator.documents import DocumentIndex, SUPPORTED_EXTENSIONS, extract_text
from orchestrator.openrouter import OpenRouterClient
from orchestrator.workflow import Orchestrator, WorkflowState

load_dotenv()
st.set_page_config(page_title="AI", page_icon="Z", layout="wide")

STEPS = [
    ("plan", "1. План", "Построить план"),
    ("research", "2. Поиск", "Сформировать обзор с ИИ"),
    ("draft", "3. Черновик", "Собрать черновик"),
    ("critic", "4. Контроль качества", "Проверить черновик"),
    ("tests", "5. Тест-кейсы", "Сгенерировать тест-кейсы"),
    ("code_review", "6. Проверка кода", "Проверить код"),
    ("summary", "7. Резюме", "Составить резюме"),
]
STEP_PREREQUISITES = {
    "research": "plan",
    "draft": "research",
    "critic": "draft",
    "tests": "critic",
    "code_review": "tests",
    "summary": "code_review",
}
TEMPLATES = {
    "Курсовой проект": "Введение, цель и задачи, обзор, методология, результаты, выводы, источники.",
    "Лабораторная работа": "Цель, оборудование, ход работы, результаты, анализ, вывод.",
    "Исследовательский отчёт": "Аннотация, проблема, методы, результаты, обсуждение, ограничения, выводы.",
    "Свободная структура": "Подбери структуру под задачу и объясни выбор.",
}


def initialize_state() -> None:
    if "workflow" not in st.session_state:
        st.session_state.workflow = None
    if "index" not in st.session_state:
        st.session_state.index = DocumentIndex()
    if "source_files" not in st.session_state:
        st.session_state.source_files = {}
    if "upload_signature" not in st.session_state:
        st.session_state.upload_signature = ()


def main() -> None:
    initialize_state()
    st.title("Мультиагентный ИИ-оркестратор")
    st.caption("Последовательная работа агентов через OpenRouter с поиском по локальным файлам и редактированием промежуточных результатов.")

    with st.sidebar:
        st.header("Настройка проекта")
        task = st.text_area("Задача", height=130, placeholder="Опишите проект или вопрос")
        template_name = st.selectbox("Шаблон отчёта", list(TEMPLATES))
        uploads = st.file_uploader(
            "Локальные документы и файлы кода",
            type=[extension.lstrip(".") for extension in SUPPORTED_EXTENSIONS]
            + ["js", "ts", "java", "cs", "cpp", "c", "go", "rs"],
            accept_multiple_files=True,
        )
        signature = tuple(
            (item.name, hashlib.sha256(item.getvalue()).hexdigest())
            for item in (uploads or [])
        )
        if signature != st.session_state.upload_signature:
            index = DocumentIndex()
            source_files: dict[str, str] = {}
            for item in uploads or []:
                try:
                    text = extract_text(item.name, item.getvalue())
                    source_files[item.name] = text
                    if item.name.lower().endswith(tuple(SUPPORTED_EXTENSIONS)):
                        index.add(item.name, text)
                except (ValueError, UnicodeError) as error:
                    st.error(f"{item.name}: {error}")
            st.session_state.index = index
            st.session_state.source_files = source_files
            st.session_state.upload_signature = signature
        st.caption(f"Индексировано фрагментов: {len(st.session_state.index.chunks)}")
        st.caption(
            "Загружено документов: "
            + (", ".join(st.session_state.source_files) or "нет")
        )

        if st.button("Начать новый процесс", type="primary", use_container_width=True):
            if not task.strip():
                st.error("Сначала введите задачу.")
            else:
                st.session_state.workflow = WorkflowState(task, TEMPLATES[template_name])
                st.rerun()

    state: WorkflowState | None = st.session_state.workflow
    if state is None:
        st.info("Задайте задачу и нажмите «Начать новый процесс». Затем запускайте этапы по порядку.")
        st.markdown("**Цепочка:** Planner → Researcher → Writer → Critic → Tester → CodeReviewer → Summarizer → Formatter")
    else:
        st.subheader("Ход выполнения")
        client_error = None
        client = None
        try:
            client = OpenRouterClient()
        except RuntimeError as error:
            client_error = str(error)
        if client_error:
            st.warning(client_error)

        tabs = st.tabs([label for _, label, _ in STEPS] + ["Аналитика", "Артефакты", "Логи"])
        for tab, (step, label, button_label) in zip(tabs, STEPS):
            with tab:
                st.subheader(label)
                if step == "research":
                    st.caption(
                        "Сначала выполните локальный поиск по тексту загруженных "
                        "файлов. Запрос можно вводить отдельно от основной задачи."
                    )
                    with st.form("local_document_search"):
                        search_query = st.text_input(
                            "Что искать в документах?",
                            value=state.search_query or state.task,
                            key="document_search_query",
                        )
                        search_submitted = st.form_submit_button(
                            "Найти фрагменты", disabled=not st.session_state.index.chunks
                        )
                    if not st.session_state.index.chunks:
                        st.info(
                            "Сначала загрузите поддерживаемый текстовый документ "
                            "(TXT, MD, PDF с выделяемым текстом, DOCX, CSV или код)."
                        )
                    elif search_submitted:
                        state.search_query = search_query.strip()
                        if not state.search_query:
                            st.warning("Введите слова для поиска.")
                        else:
                            matches = st.session_state.index.search(
                                state.search_query, limit=6
                            )
                            st.session_state.document_search_matches = [
                                (chunk.filename, chunk.text, score)
                                for chunk, score in matches
                            ]
                    if "document_search_matches" in st.session_state:
                        matches = st.session_state.document_search_matches
                        if matches:
                            st.success(f"Найдено фрагментов: {len(matches)}")
                            for number, (filename, text, score) in enumerate(
                                matches, start=1
                            ):
                                with st.expander(
                                    f"{number}. {filename} — релевантность {score:.0%}"
                                ):
                                    st.write(text)
                        elif state.search_query:
                            st.warning(
                                f"По запросу «{state.search_query}» совпадений "
                                "в загруженных документах не найдено. Попробуйте "
                                "синоним или более короткую формулировку."
                            )
                if step in state.results:
                    edited = st.text_area(
                        "Промежуточный результат (можно редактировать)",
                        value=state.results[step],
                        height=280,
                        key=f"edit_{step}",
                    )
                    state.results[step] = edited
                prerequisite = STEP_PREREQUISITES.get(step)
                waiting = prerequisite is not None and prerequisite not in state.results
                enabled = client is not None and bool(state.task.strip()) and not waiting
                if waiting:
                    st.caption(f"Сначала завершите этап: {prerequisite}.")
                if st.button(button_label, key=f"run_{step}", disabled=not enabled):
                    try:
                        response_area = st.empty()
                        streamed_parts: list[str] = []

                        def render_delta(delta: str) -> None:
                            streamed_parts.append(delta)
                            response_area.markdown("".join(streamed_parts))

                        orchestrator = Orchestrator(
                            AgentSuite(client, on_delta=render_delta)
                        )
                        orchestrator.run_step(
                            state, step, st.session_state.index, st.session_state.source_files
                        )
                        st.rerun()
                    except Exception as error:
                        state.events.append(
                            {
                                "step": step,
                                "agent": label,
                                "status": "error",
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                "error": str(error),
                            }
                        )
                        if isinstance(error, httpx.TransportError) and (
                            "UNEXPECTED_EOF_WHILE_READING" in str(error)
                        ):
                            st.error(
                                f"OpenRouter разорвал защищённое сетевое соединение на этапе "
                                f"«{label}». Автоматические повторы не помогли. Проверьте "
                                "интернет, VPN/прокси или сетевой экран и повторите этап."
                            )
                        else:
                            st.error(f"Этап «{label}» завершился с ошибкой: {error}")

        analytics_tab = tabs[len(STEPS)]
        with analytics_tab:
            st.subheader("Локальный анализ таблицы")
            csv_files = [
                (item.name, item.getvalue())
                for item in (uploads or [])
                if item.name.lower().endswith(".csv")
            ]
            if not csv_files:
                st.info("Загрузите CSV, чтобы увидеть профиль данных, кластеры и визуализацию.")
            else:
                selected = st.selectbox("CSV-файл", [name for name, _ in csv_files])
                cluster_count = st.slider("Количество кластеров", min_value=2, max_value=8, value=3)
                selected_data = next(data for name, data in csv_files if name == selected)
                csv_frame = pd.read_csv(BytesIO(selected_data))
                numeric_columns = csv_frame.select_dtypes(include="number").columns.tolist()
                target_options = ["Не выполнять"] + list(csv_frame.columns)
                target = st.selectbox("Целевой столбец для классификации", target_options)
                forecast_options = ["Не выполнять"] + numeric_columns
                forecast_column = st.selectbox("Числовой столбец для прогноза", forecast_options)
                if st.button("Выполнить анализ", key="analyze_csv"):
                    try:
                        result = analyze_csv(
                            selected_data,
                            cluster_count,
                            None if target == "Не выполнять" else target,
                            None if forecast_column == "Не выполнять" else forecast_column,
                        )
                        st.session_state.analytics_result = result
                        state.events.append(
                            {
                                "step": "analytics",
                                "agent": "AnalyticsAgent",
                                "status": "completed",
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                "input_keys": [selected],
                                "output_chars": len(result.summary),
                            }
                        )
                    except Exception as error:
                        st.error(f"Анализ не выполнен: {error}")
                result = st.session_state.get("analytics_result")
                if result:
                    st.write(result.summary)
                    st.dataframe(result.profile, use_container_width=True)
                    if result.clusters is not None:
                        st.dataframe(result.clusters, use_container_width=True)
                        st.pyplot(result.chart)
                    else:
                        st.info("Для кластеризации нужны минимум два числовых столбца и две строки.")
                    if result.classification:
                        st.write(result.classification)
                    if result.forecast is not None:
                        st.write("Линейный прогноз следующих пяти значений по индексу строк:")
                        st.dataframe(result.forecast, use_container_width=True)

        artifact_tab = tabs[len(STEPS) + 1]
        with artifact_tab:
            st.subheader("Экспорт результата")
            if st.button("Сформировать итоговые артефакты"):
                markdown = build_markdown(state.results, state.task)
                state.events.append(
                    {
                        "step": "format",
                        "agent": "Formatter",
                        "status": "completed",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "input_keys": sorted(state.results.keys()),
                        "output_chars": len(markdown),
                    }
                )
                state.artifacts_ready = True
            if state.artifacts_ready:
                markdown = build_markdown(state.results, state.task)
                st.download_button("Скачать Markdown", markdown, "project-report.md", "text/markdown")
                st.download_button(
                    "Скачать JSON с логами",
                    build_json(state.results, state.task, state.events),
                    "project-artifacts.json",
                    "application/json",
                )
                st.download_button(
                    "Скачать DOCX", build_docx(markdown), "project-report.docx",
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )
                st.download_button("Скачать PDF", build_pdf(markdown), "project-report.pdf", "application/pdf")
                st.markdown(markdown)
            else:
                st.info("Запустите Formatter, чтобы подготовить и зарегистрировать итоговые файлы.")

        log_tab = tabs[len(STEPS) + 2]
        with log_tab:
            st.subheader("Обмен данными между агентами")
            if state.events:
                st.dataframe(state.events, use_container_width=True)
                st.json(state.events)
            else:
                st.info("После запуска этапов здесь появятся статусы, отметки времени и объёмы выходных данных.")


if __name__ == "__main__":
    main()
