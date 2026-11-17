"""
chat_ui.py - Gradio chat window for Asta, served by a local Ollama model.

    ollama serve           # if Ollama is not already running
    python chat_ui.py

Each request sends the Asta system prompt, the recent conversation and the
new question to Ollama's /api/chat endpoint, so Asta can follow up on
earlier messages.
"""

import os

import requests

from asta_prompt import build_chat

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
# "asta" by default; set the ASTA_MODEL environment variable to try another
# Ollama model without editing the code.
MODEL_NAME = os.environ.get("ASTA_MODEL", "asta")

# (connect timeout, read timeout) in seconds: fail fast if Ollama is not
# running, but give a slow CPU plenty of time to generate an answer.
TIMEOUT_SECONDS = (5, 120)

# Only the most recent messages are sent. Older ones would eventually
# overflow the model's context window, and Asta was trained on single
# questions, so long histories add little.
MAX_HISTORY_MESSAGES = 10

# Error replies start with this, so they can be left out of the history
# sent back to the model.
ERROR_PREFIX = "[Asta error] "


def content_to_text(content):
    """
    Get the text out of one Gradio message's content.

    Gradio 6 stores content as a list of parts, e.g.
    [{"type": "text", "text": "What is a pulsar?"}], but a plain string is
    also accepted. Non-text parts such as files are ignored.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = [part["text"] for part in content
                 if isinstance(part, dict) and part.get("type") == "text"]
        return "\n".join(texts)
    return ""


def history_to_messages(history):
    """Convert Gradio's chat history (a list of role/content dicts) into Ollama chat messages."""
    messages = []
    for item in history:
        role = item.get("role")
        text = content_to_text(item.get("content"))
        # Skip empty messages and our own error messages.
        if role not in ("user", "assistant") or not text or text.startswith(ERROR_PREFIX):
            continue
        messages.append({"role": role, "content": text})
    return messages


def build_messages(message, history):
    """System prompt + recent history + the new question."""
    recent = history_to_messages(history)[-MAX_HISTORY_MESSAGES:]
    return build_chat(recent + [{"role": "user", "content": message}])


def ask_asta(message, history):
    """Send the conversation to Ollama and return Asta's reply (or a friendly error)."""
    if not message.strip():
        return "Please type a question about astronomy."

    payload = {
        "model": MODEL_NAME,
        "messages": build_messages(message, history),
        "stream": False,
    }

    try:
        response = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=TIMEOUT_SECONDS)
    except requests.exceptions.ReadTimeout:
        return ERROR_PREFIX + "Asta took too long to answer. Please try again."
    except requests.exceptions.ConnectionError:
        return (ERROR_PREFIX + f"Could not connect to Ollama at {OLLAMA_CHAT_URL}. "
                "Is it running? Start it with `ollama serve`.")
    except requests.exceptions.RequestException as error:
        return ERROR_PREFIX + f"The request to Ollama failed: {error}"

    if response.status_code == 404:
        return (ERROR_PREFIX + f"The Ollama model '{MODEL_NAME}' was not found. "
                "Create it with `ollama create asta -f Modelfile`.")
    if not response.ok:
        return ERROR_PREFIX + f"Ollama returned an error ({response.status_code}): {response.text}"

    try:
        return response.json()["message"]["content"]
    except (ValueError, KeyError, TypeError):
        return ERROR_PREFIX + "Ollama sent a reply in an unexpected format."


def main():
    # Imported here rather than at the top so the chat logic above can be
    # tested without Gradio installed.
    import gradio as gr

    demo = gr.ChatInterface(
        fn=ask_asta,
        title="Asta",
        description="An astronomy chatbot: Qwen2.5-1.5B-Instruct fine-tuned with LoRA, running locally with Ollama.",
        examples=["What is a black hole?", "Why do stars twinkle?", "How old is the universe?"],
    )
    demo.launch()


if __name__ == "__main__":
    main()
