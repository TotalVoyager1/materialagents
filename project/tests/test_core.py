import json
import unittest
from unittest.mock import Mock

import httpx
import pandas as pd

from orchestrator.agents import AgentSuite
from orchestrator.analytics import analyze_csv
from orchestrator.artifacts import build_docx, build_json, build_markdown, build_pdf
from orchestrator.documents import DocumentIndex, extract_text
from orchestrator.openrouter import OpenRouterClient
from orchestrator.workflow import Orchestrator, WorkflowState


class DocumentIndexTests(unittest.TestCase):
    def test_extract_text_and_retrieve_relevant_chunk(self):
        text = extract_text("notes.txt", b"Neural networks learn useful patterns from data.")
        index = DocumentIndex(chunk_size=64, overlap=8)
        index.add("notes.txt", text)

        matches = index.search("neural networks")

        self.assertEqual(matches[0][0].filename, "notes.txt")
        self.assertGreater(matches[0][1], 0)

    def test_search_handles_russian_inflections_and_finds_best_document(self):
        index = DocumentIndex()
        index.add(
            "rice.txt",
            "Рисовая культура выращивается на полях. Для приготовления риса "
            "промойте зерна, залейте водой и варите до готовности.",
        )
        index.add(
            "orchestrator.txt",
            "Мультиагентная система распределяет задачи между программными агентами.",
        )

        matches = index.search("тесты по рису")

        self.assertTrue(matches)
        self.assertEqual(matches[0][0].filename, "rice.txt")
        self.assertEqual([chunk.filename for chunk, _ in matches], ["rice.txt"])

    def test_search_ignores_weak_character_only_matches(self):
        index = DocumentIndex()
        index.add(
            "orchestrator.txt",
            "Мультиагентная система распределяет задачи между программными агентами.",
        )

        self.assertEqual(index.search("совершенно неизвестный термин"), [])

    def test_empty_or_blank_search_returns_no_matches(self):
        index = DocumentIndex()
        index.add("notes.txt", "Текст документа о проекте.")

        self.assertEqual(index.search("  "), [])
        self.assertEqual(DocumentIndex().search("проект"), [])

    def test_rejects_unsupported_file(self):
        with self.assertRaises(ValueError):
            extract_text("archive.zip", b"")


class ArtifactTests(unittest.TestCase):
    def test_exports_include_results_and_logs(self):
        results = {"plan": "Step one", "draft": "A report"}
        events = [{"agent": "Planner", "status": "completed"}]
        markdown = build_markdown(results, "Example")

        self.assertIn("# Итог проекта", markdown)
        self.assertIn("Step one", markdown)
        payload = json.loads(build_json(results, "Example", events))
        self.assertEqual(payload["events"], events)
        self.assertTrue(build_docx(markdown).startswith(b"PK"))
        self.assertTrue(build_pdf(markdown).startswith(b"%PDF"))


class OrchestrationTests(unittest.TestCase):
    def test_later_step_requires_previous_agent(self):
        agents = Mock()
        orchestrator = Orchestrator(agents)
        state = WorkflowState(task="Prepare a report", template="Basic")

        with self.assertRaisesRegex(ValueError, "Сначала выполните"):
            orchestrator.run_step(state, "research", DocumentIndex(), {})
        agents.researcher.assert_not_called()

    def test_agent_prompt_contains_user_task(self):
        client = Mock()
        client.generate.return_value = "Plan"

        AgentSuite(client).planner("Study renewable energy", "Course report")

        self.assertIn("Study renewable energy", client.generate.call_args.kwargs["prompt"])


class OpenRouterClientTests(unittest.TestCase):
    def test_defaults_to_verified_openrouter_model(self):
        client = OpenRouterClient(api_key="test-key")

        self.assertEqual(client.model, "google/gemma-4-26b-a4b-it")

    def test_sends_chat_completion_request(self):
        client = OpenRouterClient(api_key="test-key")
        response = Mock()
        response.iter_lines.return_value = [
            'data: {"model":"google/gemma-4-26b-a4b-it","choices":[{"delta":{"content":"Содержательный пошаговый план "}}]}',
            'data: {"model":"google/gemma-4-26b-a4b-it","choices":[{"delta":{"content":"проекта готов."}}]}',
            "data: [DONE]",
        ]
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)

        with unittest.mock.patch(
            "orchestrator.openrouter.httpx.stream", return_value=response
        ) as request:
            deltas = []
            result = client.generate("Инструкция", "Составь план", deltas.append)

        self.assertEqual(result, "Содержательный пошаговый план проекта готов.")
        self.assertEqual("".join(deltas), result)
        self.assertEqual(client.last_model, "google/gemma-4-26b-a4b-it")
        self.assertEqual(
            request.call_args.kwargs["json"]["model"],
            "google/gemma-4-26b-a4b-it",
        )
        self.assertEqual(
            request.call_args.kwargs["headers"]["Authorization"],
            "Bearer test-key",
        )
        self.assertEqual(request.call_args.args[0], "POST")
        self.assertTrue(request.call_args.kwargs["json"]["stream"])
        self.assertEqual(
            request.call_args.kwargs["json"]["reasoning"],
            {"enabled": True, "exclude": True},
        )
        self.assertEqual(
            request.call_args.kwargs["json"]["messages"],
            [
                {"role": "system", "content": "Инструкция"},
                {"role": "user", "content": "Составь план"},
            ],
        )
        self.assertEqual(request.call_args.kwargs["json"]["max_tokens"], 4096)

    def test_retries_unexpected_tls_eof_once(self):
        client = OpenRouterClient(api_key="test-key")
        response = Mock()
        response.iter_lines.return_value = [
            'data: {"choices":[{"delta":{"content":"Содержательный пошаговый план проекта готов."}}]}',
            "data: [DONE]",
        ]
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        with unittest.mock.patch(
            "orchestrator.openrouter.httpx.stream", return_value=response
        ) as request, unittest.mock.patch("orchestrator.openrouter.time.sleep"):
            result = client.generate("Системная инструкция", "Задача")

        self.assertEqual(result, "Содержательный пошаговый план проекта готов.")
        self.assertEqual(request.call_count, 1)

    def test_retries_tls_eof_before_any_stream_content(self):
        client = OpenRouterClient(api_key="test-key")
        request_error = httpx.ConnectError(
            "[SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred in violation of protocol"
        )
        response = Mock()
        response.iter_lines.return_value = [
            'data: {"choices":[{"delta":{"content":"Содержательный пошаговый план проекта готов."}}]}',
            "data: [DONE]",
        ]
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        with unittest.mock.patch(
            "orchestrator.openrouter.httpx.stream",
            side_effect=[request_error, response],
        ) as request, unittest.mock.patch("orchestrator.openrouter.time.sleep"):
            result = client.generate("Системная инструкция", "Задача")

        self.assertEqual(result, "Содержательный пошаговый план проекта готов.")
        self.assertEqual(request.call_count, 2)

    def test_retries_safety_classifier_output_without_streaming_it(self):
        client = OpenRouterClient(api_key="test-key")
        bad_response = Mock()
        bad_response.iter_lines.return_value = [
            'data: {"model":"google/gemma-4-26b-a4b-it","choices":[{"delta":{"content":"User Safety: safe"}}]}',
            "data: [DONE]",
        ]
        good_response = Mock()
        good_response.iter_lines.return_value = [
            'data: {"model":"google/gemma-4-26b-a4b-it","choices":[{"delta":{"content":"Пошаговый содержательный план проекта выполнен."}}]}',
            "data: [DONE]",
        ]
        for response in (bad_response, bad_response, good_response):
            response.__enter__ = Mock(return_value=response)
            response.__exit__ = Mock(return_value=False)
        deltas = []
        with unittest.mock.patch(
            "orchestrator.openrouter.httpx.stream",
            side_effect=[bad_response, bad_response, good_response],
        ) as request:
            result = client.generate("Инструкция", "Составь план", deltas.append)

        self.assertEqual(result, "Пошаговый содержательный план проекта выполнен.")
        self.assertEqual(request.call_count, 3)
        self.assertEqual("".join(deltas), result)

    def test_rejects_safety_classifier_after_retries(self):
        client = OpenRouterClient(api_key="test-key")
        response = Mock()
        response.iter_lines.return_value = [
            'data: {"choices":[{"delta":{"content":"User Safety: safe"}}]}',
            "data: [DONE]",
        ]
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        with unittest.mock.patch(
            "orchestrator.openrouter.httpx.stream", return_value=response
        ) as request:
            with self.assertRaisesRegex(RuntimeError, "содержательный ответ"):
                client.generate("Инструкция", "Составь план")

        self.assertEqual(request.call_count, 3)

    def test_does_not_retry_other_network_errors(self):
        client = OpenRouterClient(api_key="test-key")
        with unittest.mock.patch(
            "orchestrator.openrouter.httpx.stream",
            side_effect=httpx.ConnectError("network unavailable"),
        ) as request:
            with self.assertRaises(httpx.ConnectError):
                client.generate("Системная инструкция", "Задача")

        self.assertEqual(request.call_count, 1)

    def test_reports_openrouter_http_errors_without_hiding_message(self):
        client = OpenRouterClient(api_key="test-key")
        request = httpx.Request(
            "POST", "https://openrouter.ai/api/v1/chat/completions"
        )
        api_response = httpx.Response(
            404,
            json={"error": {"message": "Model not found"}},
            request=request,
        )
        status_error = httpx.HTTPStatusError(
            "Not found", request=request, response=api_response
        )
        response = Mock()
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        response.raise_for_status.side_effect = status_error
        with unittest.mock.patch(
            "orchestrator.openrouter.httpx.stream",
            return_value=response,
        ):
            with self.assertRaisesRegex(RuntimeError, "HTTP 404.*Model not found"):
                client.generate("Системная инструкция", "Задача")

    def test_rejects_missing_api_key(self):
        with unittest.mock.patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "OPENROUTER_API_KEY"):
                OpenRouterClient()


class AnalyticsTests(unittest.TestCase):
    def test_csv_profile_classification_clustering_and_forecast(self):
        frame = pd.DataFrame(
            {
                "feature": list(range(20)),
                "second_feature": list(range(20, 40)),
                "label": ["A"] * 10 + ["B"] * 10,
            }
        )
        data = frame.to_csv(index=False).encode("utf-8")

        result = analyze_csv(data, clusters=2, target_column="label", forecast_column="feature")

        self.assertIsNotNone(result.clusters)
        self.assertIsNotNone(result.chart)
        self.assertIn("accuracy=", result.classification or "")
        self.assertEqual(len(result.forecast), 5)


if __name__ == "__main__":
    unittest.main()
