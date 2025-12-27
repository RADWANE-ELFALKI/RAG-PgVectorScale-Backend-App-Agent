from .prompt import SYSTEM_PROMPT
import os
from dotenv import load_dotenv
from .embeddings import get_similar_embeddings

load_dotenv()

# Use a mock fallback when no valid OpenAI key is configured
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
USE_MOCK_OPENAI = not OPENAI_API_KEY or OPENAI_API_KEY == "DUMMY"

if not USE_MOCK_OPENAI:
    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY)
    model = "gpt-4o-mini"
else:
    client = None
    model = None


messages = [{"role": "system", "content": SYSTEM_PROMPT}]


def add_message(message, role="user"):
    messages.append({"role": role, "content": f"User question: {message}"})
    similar_embeddings = get_similar_embeddings(message, limit=3)
    # append retrieved context (may be empty)
    messages.append(
        {
            "role": "assistant",
            "content": f"retrieved context: {'\n'.join(similar_embeddings)}",
        }
    )
    return messages


class _MockChoice:
    def __init__(self, text):
        self.message = type("M", (), {"content": text})


class _MockResponse:
    def __init__(self, text):
        self.choices = [type("C", (), {"message": type("M", (), {"content": text})})]


def conversation_agent(query):
    add_message(query)
    if USE_MOCK_OPENAI:
        # Simple mocked reply that uses the latest context and query
        ctx = messages[-1]["content"]
        reply = f"[mock reply] I received your question: '{query}'. Context: {ctx[:200]}"
        add_message(reply, role="assistant")
        return reply

    response = client.chat.completions.create(model=model, messages=messages)
    add_message(response.choices[0].message.content, role="assistant")
    return response.choices[0].message.content
