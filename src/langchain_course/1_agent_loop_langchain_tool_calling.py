from dotenv import load_dotenv
load_dotenv()


from langchain.chat_models import init_chat_model

# @tool:
# Converts a normal Python function into a LangChain tool.
# This gives the function metadata such as:
# - tool name
# - description
# - input schema
# - .invoke()
from langchain.tools import tool

# LangChain message types:
#
# HumanMessage  -> message from the user
# SystemMessage -> instructions/rules for the LLM
# ToolMessage   -> result returned from a tool back to the LLM
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

# traceable lets LangSmith trace this function,
# so we can inspect the agent loop in LangSmith.
from langsmith import traceable


# Safety limit.
# The agent is allowed to loop at most 10 times.
# This prevents an infinite loop if the LLM keeps requesting tools.
MAX_ITERATIONS = 10
MODEL = "qwen3:4b"
# ============================================================
# TOOLS
# ============================================================
# @tool converts this normal Python function into a LangChain tool.
#
# The LLM will see:
# - tool name: get_product_price
# - description from the docstring
# - required argument: product: str
@tool
def get_product_price(product: str) -> float:
    """
    Look up the price of a product in the catalog.

    This docstring is important because the LLM reads the
    tool description to understand WHEN and WHY it should use it.
    """

    # This is only for us to see in the terminal
    # when the tool is actually executed.
    print(f"    >> Executing get_product_price(product='{product}')")

    # A simple fake product database/catalog.
    prices = {
        "laptop": 1299.99,
        "headphones": 149.95,
        "keyboard": 89.50,
    }

    # Search the dictionary using the product name.
    #
    # Example:
    # prices.get("laptop", 0)
    # -> 1299.99
    #
    # If the product does not exist, return 0.
    return prices.get(product, 0)


# Second tool.
#
# The LLM can use this tool after it gets the real price.
@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """
    Apply a discount tier to a price and return the final price.

    Available tiers:
    bronze, silver, gold
    """

    # Show us in the terminal that this tool is executing.
    print(
        f"    >> Executing apply_discount("
        f"price={price}, discount_tier='{discount_tier}')"
    )

    # Mapping between discount tier and percentage.
    discount_percentages = {
        "bronze": 5,
        "silver": 12,
        "gold": 23,
    }

    # Get the discount percentage.
    #
    # Example:
    # discount_tier = "gold"
    # discount = 23
    #
    # If the tier does not exist, use 0.
    discount = discount_percentages.get(discount_tier, 0)

    # Calculate the final price.
    #
    # Example:
    # price = 1299.99
    # discount = 23
    #
    # price * (1 - 23 / 100)
    #
    # round(..., 2) keeps only 2 decimal places.
    return round(price * (1 - discount / 100), 2)


# ============================================================
# AGENT LOOP
# ============================================================


# This tells LangSmith:
# trace everything that happens inside this function.
#
# In LangSmith the trace will be called:
# "LangChain Agent Loop"
@traceable(name="LangChain Agent Loop")
def run_agent(question: str):

    # Create a list containing all tools available to the LLM.
    #
    # The agent is only allowed to use these tools.
    tools = [
        get_product_price,
        apply_discount,
    ]

    # Create a dictionary mapping:
    #
    # tool name -> actual tool object
    #
    # Conceptually:
    #
    # {
    #     "get_product_price": get_product_price,
    #     "apply_discount": apply_discount
    # }
    #
    # Why?
    #
    # The LLM returns the selected tool name as TEXT:
    #
    # "get_product_price"
    #
    # But Python needs the real tool object so it can execute:
    #
    # get_product_price.invoke(...)
    tools_dict = {
        t.name: t
        for t in tools
    }


    # Create our local LLM.
    #
    # "ollama:qwen3:1.7b"
    #
    # tells LangChain:
    #
    # provider = Ollama
    # model    = qwen3:1.7b
    #
    # temperature=0 makes the model more predictable.
    llm = init_chat_model(
        f"ollama:{MODEL}",
        temperature=0,
    )


    # Give the LLM knowledge about the available tools.
    #
    # Before bind_tools:
    #
    # LLM
    #
    # After bind_tools:
    #
    # LLM
    #  ├── knows get_product_price exists
    #  └── knows apply_discount exists
    #
    # This does NOT execute the tools.
    #
    # It only tells the model:
    # "These tools are available if you need them."
    llm_with_tools = llm.bind_tools(tools)


    # Print the user's question.
    print(f"Question: {question}")
    print("=" * 60)


    # messages represents the conversation history.
    #
    # Initially we have:
    #
    # SystemMessage
    #      ↓
    # HumanMessage
    messages = [

        # SystemMessage gives the LLM rules/instructions.
        SystemMessage(
            content=(
                "You are a helpful shopping assistant. "
                "You have access to a product catalog tool "
                "and a discount tool.\n\n"

                "STRICT RULES — you must follow these exactly:\n"

                # Rule 1:
                # The model must not invent prices.
                "1. NEVER guess or assume any product price. "
                "You MUST call get_product_price first to get the real price.\n"

                # Rule 2:
                # The model cannot apply a discount before getting the real price.
                "2. Only call apply_discount AFTER you have received "
                "a price from get_product_price. Pass the exact price "
                "returned by get_product_price — do NOT pass a made-up number.\n"

                # Rule 3:
                # Force the LLM to use the tool instead of doing math itself.
                "3. NEVER calculate discounts yourself using math. "
                "Always use the apply_discount tool.\n"

                # Rule 4:
                # If the user doesn't specify bronze/silver/gold,
                # the model should ask instead of guessing.
                "4. If the user does not specify a discount tier, "
                "ask them which tier to use — do NOT assume one."
            )
        ),

        # HumanMessage represents the actual user question.
        HumanMessage(
            content=question
        ),
    ]


    # ========================================================
    # THE AGENT LOOP
    # ========================================================
    #
    # This is the important part.
    #
    # The loop follows roughly:
    #
    # LLM
    #  ↓
    # Does it request a tool?
    #  ├── YES -> execute tool
    #  │           ↓
    #  │       send result back
    #  │           ↓
    #  │        loop again
    #  │
    #  └── NO -> final answer
    #
    for iteration in range(1, MAX_ITERATIONS + 1):

        print(f"\n--- Iteration {iteration} ---")


        # Send the full conversation history to the LLM.
        #
        # On the first iteration:
        #
        # [
        #   SystemMessage(...),
        #   HumanMessage(...)
        # ]
        #
        # Later iterations will also contain:
        #
        # AIMessage
        # ToolMessage
        #
        # .invoke() means:
        # RUN the LLM one time using these messages.
        ai_message = llm_with_tools.invoke(messages)


        # Extract the list of tool requests made by the LLM.
        #
        # Example:
        #
        # ai_message.tool_calls
        #
        # might contain:
        #
        # [
        #   {
        #       "name": "get_product_price",
        #       "args": {
        #           "product": "laptop"
        #       },
        #       "id": "call_123"
        #   }
        # ]
        #
        # If the LLM does not want a tool:
        #
        # []
        tool_calls = ai_message.tool_calls


        # ----------------------------------------------------
        # NO TOOL CALL = FINAL ANSWER
        # ----------------------------------------------------

        # If the list is empty:
        #
        # tool_calls = []
        #
        # then the LLM is finished and has produced
        # its final answer.
        if not tool_calls:

            print(
                f"\nFinal Answer: {ai_message.content}"
            )

            # Return the final generated text
            # and stop the function.
            return ai_message.content


        # ----------------------------------------------------
        # TOOL CALL EXISTS
        # ----------------------------------------------------

        # The model may theoretically request multiple tools.
        #
        # Here we intentionally process only the FIRST one.
        #
        # Example:
        #
        # tool_calls = [
        #   {... first tool ...},
        #   {... second tool ...}
        # ]
        #
        # tool_calls[0]
        # means:
        # take the first tool request.
        tool_call = tool_calls[0]


        # Extract the name of the tool.
        #
        # Example:
        #
        # "get_product_price"
        tool_name = tool_call.get("name")


        # Extract the arguments the LLM wants
        # to send to the tool.
        #
        # Example:
        #
        # {
        #     "product": "laptop"
        # }
        #
        # If there are no arguments,
        # return an empty dictionary {}.
        tool_args = tool_call.get(
            "args",
            {}
        )


        # Every tool call has an ID.
        #
        # Example:
        #
        # "call_123"
        #
        # We need this later so LangChain knows
        # which ToolMessage belongs to which tool request.
        tool_call_id = tool_call.get("id")


        print(
            f"  [Tool Selected] "
            f"{tool_name} "
            f"with args: {tool_args}"
        )


        # ----------------------------------------------------
        # FIND THE ACTUAL TOOL
        # ----------------------------------------------------

        # tool_name is just TEXT.
        #
        # Example:
        #
        # tool_name =
        # "get_product_price"
        #
        # We cannot execute a string.
        #
        # So we use tools_dict to convert:
        #
        # "get_product_price"
        #          ↓
        # actual get_product_price tool object
        #
        tool_to_use = tools_dict.get(tool_name)


        # Safety check:
        # If the model requested a tool that doesn't exist,
        # stop with an error.
        if tool_to_use is None:
            raise ValueError(
                f"Tool '{tool_name}' not found"
            )


        # ----------------------------------------------------
        # EXECUTE THE TOOL
        # ----------------------------------------------------

        # Execute the actual tool with the arguments
        # selected by the LLM.
        #
        # Example:
        #
        # tool_to_use =
        # get_product_price
        #
        # tool_args =
        # {"product": "laptop"}
        #
        # This effectively becomes:
        #
        # get_product_price.invoke(
        #     {"product": "laptop"}
        # )
        #
        # Result:
        #
        # 1299.99
        #
        # "observation" is the result returned by the tool.
        observation = tool_to_use.invoke(
            tool_args
        )


        print(
            f"  [Tool Result] {observation}"
        )


        # ----------------------------------------------------
        # ADD WHAT HAPPENED TO CONVERSATION HISTORY
        # ----------------------------------------------------

        # First save the AI message that requested the tool.
        #
        # Example:
        #
        # AI:
        # "I want to call get_product_price"
        messages.append(ai_message)


        # Then create a ToolMessage containing
        # the result of the tool.
        #
        # Example:
        #
        # ToolMessage(
        #     content="1299.99",
        #     tool_call_id="call_123"
        # )
        #
        # This tells the LLM:
        #
        # "The tool you requested returned 1299.99."
        messages.append(
            ToolMessage(
                content=str(observation),

                # Connect this result to the exact
                # tool call made by the LLM.
                tool_call_id=tool_call_id,
            )
        )


        # Then the loop starts again.
        #
        # On the next iteration the LLM sees:
        #
        # SystemMessage
        # HumanMessage
        # AIMessage(tool request)
        # ToolMessage(tool result)
        #
        # and decides what to do next.


    # ========================================================
    # MAX ITERATIONS REACHED
    # ========================================================

    # If we reach here, the LLM never produced a final answer
    # within the allowed 10 iterations.
    print(
        "ERROR: Max iterations reached "
        "without a final answer"
    )

    return None


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

# This condition is true when this file is executed directly.
#
# Example:
#
# python agent.py
#
if __name__ == "__main__":

    print(
        "Hello LangChain Agent (.bind_tools)!"
    )

    print()


    # Start the agent with this user question.
    #
    # Expected reasoning flow:
    #
    # User asks:
    # laptop + gold discount
    #
    #         ↓
    #
    # LLM calls:
    # get_product_price("laptop")
    #
    #         ↓
    #
    # Tool returns:
    # 1299.99
    #
    #         ↓
    #
    # LLM calls:
    # apply_discount(
    #     price=1299.99,
    #     discount_tier="gold"
    # )
    #
    #         ↓
    #
    # Tool returns discounted price
    #
    #         ↓
    #
    # LLM gives final answer
    #
    result = run_agent(
        "What is the price of a laptop "
        "after applying a gold discount?"
    )