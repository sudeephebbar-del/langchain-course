from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable

MAX_ITERATIONS = 10
#MODEL = "llama3.1:405b"
MODEL = "qwen3:1.7b"
#MODEL = "gpt-5"
OLLAMA_HOST = None  # e.g. "http://localhost:11434"
# Prevent "stuck forever" behavior on slow models / stalled server calls.
OLLAMA_TIMEOUT_S = 120.0
# Cap generation so small models don't ramble indefinitely.
OLLAMA_NUM_PREDICT = 512

# --- Tools (LangChain @tool decorator) ---


@tool
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog."""
    print(f"    >> Executing get_product_price(product='{product}')")
    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}
    return prices.get(product, 0)


@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return the final price.
    Available tiers: bronze, silver, gold."""
    print(f"    >> Executing apply_discount(price={price}, discount_tier='{discount_tier}')")
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)


# --- Agent Loop (supports multiple tool calls per iteration) ---


@traceable(name="LangChain Agent Loop (Multi-tool)")
def run_agent(question: str):
    tools = [get_product_price, apply_discount]
    tools_dict = {t.name: t for t in tools}

    ollama_client_kw = {"timeout": OLLAMA_TIMEOUT_S}
    llm_init_kw = dict(
        temperature=0,
        num_predict=OLLAMA_NUM_PREDICT,
        reasoning=False,
        sync_client_kwargs=ollama_client_kw,
        async_client_kwargs=ollama_client_kw,
    )
    if OLLAMA_HOST:
        llm_init_kw["base_url"] = OLLAMA_HOST

    llm = init_chat_model(f"ollama:{MODEL}", **llm_init_kw)
    # llm = init_chat_model(f"openai:{MODEL}", temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    print(f"Question: {question}")
    print("=" * 60)

    messages = [
        SystemMessage(
            content=(
                "You are a helpful shopping assistant. "
                "You have access to a product catalog tool "
                "and a discount tool.\n\n"
                "STRICT RULES — you must follow these exactly:\n"
                "1. NEVER guess or assume any product price. "
                "You MUST call get_product_price first to get the real price.\n"
                "2. Only call apply_discount AFTER you have received "
                "a price from get_product_price. Pass the exact price "
                "returned by get_product_price — do NOT pass a made-up number.\n"
                "3. NEVER calculate discounts yourself using math. "
                "Always use the apply_discount tool.\n"
                "4. If the user does not specify a discount tier, "
                "ask them which tier to use — do NOT assume one."
            )
        ),
        HumanMessage(content=question),
    ]

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iteration} ---")
        print(
            f"  [LLM] Calling Ollama model='{MODEL}' "
            f"(timeout={OLLAMA_TIMEOUT_S}s, num_predict={OLLAMA_NUM_PREDICT})"
        )

        ai_message = llm_with_tools.invoke(messages)
        tool_calls = ai_message.tool_calls or []

        # If no tool calls, this is the final answer
        if not tool_calls:
            print(f"\nFinal Answer: {ai_message.content}")
            return ai_message.content

        # IMPORTANT: Append the AI message once, then respond to EACH tool call.
        messages.append(ai_message)

        for idx, tool_call in enumerate(tool_calls, start=1):
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("args", {})
            tool_call_id = tool_call.get("id")

            print(f"  [Tool Selected {idx}/{len(tool_calls)}] {tool_name} with args: {tool_args}")

            tool_to_use = tools_dict.get(tool_name)
            if tool_to_use is None:
                raise ValueError(f"Tool '{tool_name}' not found")

            observation = tool_to_use.invoke(tool_args)
            print(f"  [Tool Result {idx}/{len(tool_calls)}] {observation}")

            messages.append(ToolMessage(content=str(observation), tool_call_id=tool_call_id))

    print("ERROR: Max iterations reached without a final answer")
    return None


if __name__ == "__main__":
    print("Hello LangChain Agent (.bind_tools, multi-tool)!")
    print()
    run_agent("What is the price of a laptop after applying a gold discount ans silver discount?")
