"""
Tests for the chat logic in chat_ui.py.

Ollama is not called for real: requests.post is replaced with a fake
("mocked") version so each error case can be tested on any machine.
"""

import requests

import chat_ui
from asta_prompt import SYSTEM_PROMPT


class FakeResponse:
    """Stands in for a requests.Response object."""

    def __init__(self, status_code, json_data=None, text=""):
        self.status_code = status_code
        self.ok = status_code < 400
        self._json_data = json_data
        self.text = text

    def json(self):
        if self._json_data is None:
            raise ValueError("no JSON")
        return self._json_data


def fake_post_returning(response):
    def fake_post(url, json, timeout):
        return response
    return fake_post


def fake_post_raising(error):
    def fake_post(url, json, timeout):
        raise error
    return fake_post


# ----- building the messages -----

def test_first_message_has_system_prompt_and_question():
    messages = chat_ui.build_messages("What is a pulsar?", [])
    assert messages == [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "What is a pulsar?"},
    ]


def test_history_in_messages_format_is_included():
    history = [
        {"role": "user", "content": "What is a pulsar?"},
        {"role": "assistant", "content": "A rotating neutron star."},
    ]
    messages = chat_ui.build_messages("How fast do they spin?", history)
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]


def test_history_with_gradio_content_parts_is_included():
    # Gradio 6 passes content as a list of parts rather than a string.
    history = [
        {"role": "user", "content": [{"type": "text", "text": "What is a pulsar?"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "A rotating neutron star."}]},
    ]
    messages = chat_ui.build_messages("How fast do they spin?", history)
    assert messages[1] == {"role": "user", "content": "What is a pulsar?"}
    assert messages[2] == {"role": "assistant", "content": "A rotating neutron star."}


def test_non_text_parts_are_ignored():
    history = [{"role": "user", "content": [{"type": "file", "file": {"path": "x.png"}}]}]
    messages = chat_ui.build_messages("Hello", history)
    assert messages == chat_ui.build_messages("Hello", [])


def test_error_replies_are_not_sent_back_to_the_model():
    history = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": chat_ui.ERROR_PREFIX + "Could not connect"},
    ]
    messages = chat_ui.build_messages("Hello again", history)
    assert all(not m["content"].startswith(chat_ui.ERROR_PREFIX) for m in messages)


def test_long_history_is_trimmed():
    history = [{"role": "user", "content": f"question {i}"} for i in range(50)]
    messages = chat_ui.build_messages("latest", history)
    # system prompt + trimmed history + new question
    assert len(messages) == 1 + chat_ui.MAX_HISTORY_MESSAGES + 1
    assert messages[-1]["content"] == "latest"


# ----- talking to Ollama -----

def test_successful_reply_is_returned(monkeypatch):
    reply = {"message": {"role": "assistant", "content": "A star is a ball of hot gas."}}
    monkeypatch.setattr(requests, "post", fake_post_returning(FakeResponse(200, reply)))
    assert chat_ui.ask_asta("What is a star?", []) == "A star is a ball of hot gas."


def test_ollama_not_running(monkeypatch):
    monkeypatch.setattr(requests, "post", fake_post_raising(requests.exceptions.ConnectionError()))
    answer = chat_ui.ask_asta("What is a star?", [])
    assert answer.startswith(chat_ui.ERROR_PREFIX)
    assert "ollama serve" in answer


def test_ollama_too_slow(monkeypatch):
    monkeypatch.setattr(requests, "post", fake_post_raising(requests.exceptions.ReadTimeout()))
    answer = chat_ui.ask_asta("What is a star?", [])
    assert "too long" in answer


def test_model_missing(monkeypatch):
    response = FakeResponse(404, {"error": "model 'asta' not found"})
    monkeypatch.setattr(requests, "post", fake_post_returning(response))
    answer = chat_ui.ask_asta("What is a star?", [])
    assert "ollama create asta -f Modelfile" in answer


def test_other_server_error(monkeypatch):
    response = FakeResponse(500, text="internal error")
    monkeypatch.setattr(requests, "post", fake_post_returning(response))
    answer = chat_ui.ask_asta("What is a star?", [])
    assert answer.startswith(chat_ui.ERROR_PREFIX)
    assert "500" in answer


def test_unexpected_reply_format(monkeypatch):
    monkeypatch.setattr(requests, "post", fake_post_returning(FakeResponse(200, {"wrong": "shape"})))
    answer = chat_ui.ask_asta("What is a star?", [])
    assert "unexpected format" in answer


def test_empty_message_is_not_sent(monkeypatch):
    monkeypatch.setattr(requests, "post", fake_post_raising(AssertionError("should not be called")))
    assert "Please type a question" in chat_ui.ask_asta("   ", [])
