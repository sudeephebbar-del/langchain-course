from __future__ import annotations

import argparse
import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from tavily import TavilyClient


def build_agent(*, model: str, temperature: float):
    tavily = TavilyClient()

    @tool
    def search(query: str) -> str:
        """Search the web using Tavily."""
        return tavily.search(query=query)

    llm = ChatOllama(temperature=temperature, model=model)
    return create_agent(model=llm, tools=[search])


def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Terminal chatbot using LangChain + Ollama.")
    parser.add_argument("--model", default="llama3.1:latest", help="Ollama model name")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    args = parser.parse_args()

    print(f"LangSmith tracing: {os.getenv('LANGCHAIN_TRACING_V2')}")
    print(f"LangSmith project: {os.getenv('LANGCHAIN_PROJECT')}")
    print("Type a message and press Enter. Type 'exit' or 'quit' to stop.\n")

    agent = build_agent(model=args.model, temperature=args.temperature)

    while True:
        try:
            user_text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_text:
            continue
        if user_text.lower() in {"exit", "quit"}:
            break

        result = agent.invoke({"messages": HumanMessage(content=user_text)})

        if isinstance(result, dict):
            messages = result.get("messages")
            if messages:
                print(messages[-1].content)
            else:
                print(result)
        else:
            print(result)

        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
