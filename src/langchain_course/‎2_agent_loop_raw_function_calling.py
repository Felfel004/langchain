# Import load_dotenv so we can load environment variables
# from a .env file, such as LANGSMITH_API_KEY.
from dotenv import load_dotenv

# Actually load the variables from .env
load_dotenv()


# IMPORTANT DIFFERENCE FROM THE PREVIOUS CODE:
#
# Previous code:
#   used LangChain to communicate with Ollama:
#
#       from langchain.chat_models import init_chat_model
#
# Current code:
#   talks directly to Ollama using the Ollama Python package.
#
# So the flow is now:
#
# Python -> Ollama SDK -> Qwen
#
# instead of:
#
# Python -> LangChain -> Ollama -> Qwen
import ollama


# traceable is used ONLY for LangSmith tracing.
#
# It does NOT convert a Python function into a LangChain tool.
#
# It allows us to see the function execution inside LangSmith.
from langsmith import traceable


# Maximum number of times the agent is allowed
# to go through the LLM -> Tool -> LLM loop.
#
# This prevents infinite loops.
MAX_ITERATIONS = 10


# Local model running through Ollama.
MODEL = "qwen3:4b"



# ============================================================
# TOOLS
# ============================================================


# DIFFERENCE FROM PREVIOUS CODE:
#
# Previous code:
#
#   @tool
#   def get_product_price(...):
#
# @tool converted the Python function into a LangChain Tool object.
#
# Current code:
#
#   @traceable(run_type="tool")
#
# This keeps the function as a normal Python function.
# It only tells LangSmith:
#
# "Trace this function and display it as a tool run."
#
@traceable(run_type="tool")
def get_product_price(product: str) -> float:
    """
    Look up the price of a product in the catalog.
    """

    # Print this so we can see when the real Python
    # function is being executed.
    print(
        f"    >> Executing "
        f"get_product_price(product='{product}')"
    )

    # Fake product database.
    prices = {
        "laptop": 1299.99,
        "headphones": 149.95,
        "keyboard": 89.50,
    }

    # Example:
    #
    # product = "laptop"
    #
    # prices.get("laptop", 0)
    # -> 1299.99
    #
    # If product does not exist:
    # -> return 0
    return prices.get(product, 0)



# Same idea here.
#
# This is a NORMAL Python function.
#
# @traceable only lets LangSmith monitor it.
@traceable(run_type="tool")
def apply_discount(
    price: float,
    discount_tier: str
) -> float:
    """
    Apply a discount tier to a price
    and return the final price.

    Available tiers:
    bronze, silver, gold.
    """

    print(
        f"    >> Executing apply_discount("
        f"price={price}, "
        f"discount_tier='{discount_tier}')"
    )

    discount_percentages = {
        "bronze": 5,
        "silver": 12,
        "gold": 23,
    }

    # Example:
    #
    # discount_tier = "gold"
    #
    # discount = 23
    discount = discount_percentages.get(
        discount_tier,
        0
    )

    # Calculate final price.
    #
    # Example:
    #
    # 1299.99 * (1 - 23 / 100)
    #
    # round(..., 2)
    # keeps two decimal places.
    return round(
        price * (1 - discount / 100),
        2
    )



# ============================================================
# TOOL SCHEMAS
# ============================================================


# THIS IS ONE OF THE BIGGEST DIFFERENCES
# BETWEEN THIS CODE AND THE PREVIOUS LANGCHAIN CODE.
#
#
# PREVIOUS CODE:
#
# @tool
# def get_product_price(product: str) -> float:
#     """Look up the price..."""
#
#
# LangChain automatically generated information like:
#
# - tool name
# - tool description
# - parameter names
# - parameter types
# - required parameters
#
#
# CURRENT CODE:
#
# We are NOT using LangChain's @tool.
#
# Therefore we manually describe the tools
# using JSON schemas.
#
#
# The LLM needs this information so it knows:
#
# 1. What tools exist
# 2. What each tool does
# 3. What arguments each tool accepts
#
tools_for_llm = [

    # --------------------------------------------------------
    # TOOL SCHEMA 1
    # get_product_price
    # --------------------------------------------------------

    {
        # Tell Ollama that this tool represents a function.
        "type": "function",

        "function": {

            # This is the name the LLM will return
            # if it decides to use this tool.
            "name": "get_product_price",

            # Helps the LLM decide WHEN to use the tool.
            "description":
                "Look up the price of a product "
                "in the catalog.",


            # Describe the arguments the function accepts.
            "parameters": {

                # Function arguments are represented
                # as a dictionary/object.
                "type": "object",

                "properties": {

                    # The function has one parameter:
                    #
                    # product
                    "product": {

                        # product must be a string.
                        "type": "string",

                        # Extra explanation for the LLM.
                        "description":
                            "The product name, e.g. "
                            "'laptop', 'headphones', "
                            "'keyboard'",
                    },
                },

                # The LLM MUST provide product.
                "required": ["product"],
            },
        },
    },


    # --------------------------------------------------------
    # TOOL SCHEMA 2
    # apply_discount
    # --------------------------------------------------------

    {
        "type": "function",

        "function": {

            # Tool name visible to the LLM.
            "name": "apply_discount",

            # Helps the LLM understand
            # when this tool should be used.
            "description":
                "Apply a discount tier to a price "
                "and return the final price. "
                "Available tiers: bronze, silver, gold.",


            "parameters": {

                "type": "object",

                "properties": {

                    # First argument:
                    #
                    # price: float
                    "price": {

                        # JSON uses "number"
                        # for values like floats.
                        "type": "number",

                        "description":
                            "The original price",
                    },


                    # Second argument:
                    #
                    # discount_tier: str
                    "discount_tier": {

                        "type": "string",

                        "description":
                            "The discount tier: "
                            "'bronze', 'silver', or 'gold'",
                    },
                },


                # Both values are required.
                "required": [
                    "price",
                    "discount_tier",
                ],
            },
        },
    },
]


# Conceptually, the schemas above tell the LLM:
#
#
# AVAILABLE TOOL 1
#
# get_product_price(
#     product: string
# )
#
#
# AVAILABLE TOOL 2
#
# apply_discount(
#     price: number,
#     discount_tier: string
# )
#
#
# Notice:
#
# These schemas DO NOT execute the functions.
#
# They only DESCRIBE the functions to the LLM.



# ============================================================
# OPTIONAL OLLAMA FEATURE
# ============================================================


# Ollama can also generate the schemas automatically
# if we pass Python functions directly:
#
# tools_for_llm = [
#     get_product_price,
#     apply_discount
# ]
#
# But then the docstrings need to be written
# in a suitable format.
#
#
# This example intentionally uses manual JSON schemas
# because the goal is to show what LangChain's @tool
# decorator was hiding from us.



# ============================================================
# WRAPPER AROUND OLLAMA CHAT
# ============================================================


# DIFFERENCE FROM PREVIOUS CODE:
#
# Previous code:
#
#   llm_with_tools.invoke(messages)
#
# LangChain automatically traced many parts of the LLM call.
#
#
# Current code:
#
#   ollama.chat(...)
#
# Because we bypass LangChain,
# we manually add @traceable so LangSmith can see this call.
#
@traceable(
    name="Ollama Chat",
    run_type="llm"
)
def ollama_chat_traced(messages):

    # Send the request DIRECTLY to Ollama.
    #
    # We provide:
    #
    # 1. Which model to use
    # 2. Which tools exist
    # 3. The conversation history
    #
    return ollama.chat(

        # Example:
        # qwen3:1.7b
        model=MODEL,

        # Tell Qwen which tools are available.
        tools=tools_for_llm,

        # Send the whole conversation.
        messages=messages,
    )



# ============================================================
# AGENT LOOP
# ============================================================


# Trace the entire agent loop in LangSmith.
@traceable(name="Ollama Agent Loop")
def run_agent(question: str):


    # --------------------------------------------------------
    # TOOL LOOKUP DICTIONARY
    # --------------------------------------------------------

    # The LLM returns the tool NAME as text.
    #
    # Example:
    #
    # "get_product_price"
    #
    # But Python needs the ACTUAL Python function.
    #
    # Therefore we create:
    #
    # tool name -> actual function
    #
    tools_dict = {

        "get_product_price":
            get_product_price,

        "apply_discount":
            apply_discount,
    }


    # Conceptually:
    #
    # {
    #   "get_product_price":
    #       <Python function get_product_price>,
    #
    #   "apply_discount":
    #       <Python function apply_discount>
    # }



    print(f"Question: {question}")
    print("=" * 60)



    # --------------------------------------------------------
    # CONVERSATION HISTORY
    # --------------------------------------------------------

    # BIG DIFFERENCE FROM PREVIOUS CODE:
    #
    # Previous LangChain version:
    #
    # messages = [
    #     SystemMessage(...),
    #     HumanMessage(...)
    # ]
    #
    #
    # Current raw Ollama version:
    #
    # messages are plain dictionaries.
    #
    #
    # Instead of:
    #
    # SystemMessage(...)
    #
    # we use:
    #
    # {
    #     "role": "system",
    #     "content": "..."
    # }
    #
    messages = [

        # SYSTEM MESSAGE
        #
        # Gives behavior/rules to the LLM.
        {
            "role": "system",

            "content": (

                "You are a helpful shopping assistant. "

                "You have access to a product catalog tool "
                "and a discount tool.\n\n"

                "STRICT RULES — "
                "you must follow these exactly:\n"


                # Rule 1
                #
                # Force the LLM to get the real price
                # from the tool.
                "1. NEVER guess or assume any product price. "
                "You MUST call get_product_price first "
                "to get the real price.\n"


                # Rule 2
                #
                # Prevent applying a discount
                # before getting the real product price.
                "2. Only call apply_discount AFTER "
                "you have received a price "
                "from get_product_price. "
                "Pass the exact price returned "
                "by get_product_price — "
                "do NOT pass a made-up number.\n"


                # Rule 3
                #
                # Do not allow the LLM to calculate
                # the discount itself.
                "3. NEVER calculate discounts yourself "
                "using math. "
                "Always use the apply_discount tool.\n"


                # Rule 4
                #
                # Do not let the LLM assume
                # bronze/silver/gold.
                "4. If the user does not specify "
                "a discount tier, "
                "ask them which tier to use — "
                "do NOT assume one."
            ),
        },


        # USER MESSAGE
        #
        # Previous LangChain version:
        #
        # HumanMessage(content=question)
        #
        # Current Ollama version:
        #
        {
            "role": "user",
            "content": question,
        },
    ]



    # ========================================================
    # MANUAL AGENT LOOP
    # ========================================================

    # Same fundamental idea as the previous code:
    #
    #
    # LLM
    #  ↓
    # tool call?
    #  ↓
    # execute tool
    #  ↓
    # observation
    #  ↓
    # give observation back to LLM
    #  ↓
    # LLM again
    #
    #
    # Continue until the model returns
    # a normal final answer.
    #
    for iteration in range(
        1,
        MAX_ITERATIONS + 1
    ):

        print(
            f"\n--- Iteration {iteration} ---"
        )



        # ----------------------------------------------------
        # CALL THE LLM
        # ----------------------------------------------------

        # DIFFERENCE FROM PREVIOUS CODE:
        #
        #
        # PREVIOUS:
        #
        # ai_message =
        #     llm_with_tools.invoke(messages)
        #
        #
        # CURRENT:
        #
        # We directly call Ollama.
        #
        response = ollama_chat_traced(
            messages=messages
        )


        # ollama.chat() returns a Response object.
        #
        # The assistant message is inside:
        #
        # response.message
        #
        ai_message = response.message



        # ----------------------------------------------------
        # GET TOOL REQUESTS
        # ----------------------------------------------------

        # Ask:
        #
        # Did the model request any tools?
        #
        # Example:
        #
        # [
        #     ToolCall(
        #         function.name =
        #             "get_product_price"
        #
        #         function.arguments =
        #             {"product": "laptop"}
        #     )
        # ]
        #
        tool_calls = ai_message.tool_calls



        # ----------------------------------------------------
        # NO TOOL CALL = FINAL ANSWER
        # ----------------------------------------------------

        # If:
        #
        # tool_calls = []
        #
        # then the LLM does not want another tool.
        #
        # Therefore its content is treated
        # as the final answer.
        #
        if not tool_calls:

            print(
                f"\nFinal Answer: "
                f"{ai_message.content}"
            )

            return ai_message.content



        # ----------------------------------------------------
        # TAKE FIRST TOOL CALL
        # ----------------------------------------------------

        # The model could theoretically request:
        #
        # tool_calls[0]
        # tool_calls[1]
        # ...
        #
        # This program intentionally processes
        # ONE tool at a time.
        #
        tool_call = tool_calls[0]



        # ----------------------------------------------------
        # GET TOOL NAME
        # ----------------------------------------------------

        # DIFFERENCE FROM LANGCHAIN:
        #
        #
        # LangChain version:
        #
        # tool_name =
        #     tool_call.get("name")
        #
        #
        # Ollama version:
        #
        # tool_call
        #     ↓
        # function
        #     ↓
        # name
        #
        tool_name = (
            tool_call.function.name
        )



        # ----------------------------------------------------
        # GET TOOL ARGUMENTS
        # ----------------------------------------------------

        # Example:
        #
        # {
        #     "product": "laptop"
        # }
        #
        #
        # Or:
        #
        # {
        #     "price": 1299.99,
        #     "discount_tier": "gold"
        # }
        #
        tool_args = (
            tool_call.function.arguments
        )



        print(
            f"  [Tool Selected] "
            f"{tool_name} "
            f"with args: {tool_args}"
        )



        # ----------------------------------------------------
        # CONVERT TOOL NAME -> PYTHON FUNCTION
        # ----------------------------------------------------

        # Example:
        #
        # tool_name =
        # "get_product_price"
        #
        #
        # tools_dict.get(tool_name)
        #
        # becomes:
        #
        # tools_dict.get(
        #     "get_product_price"
        # )
        #
        #
        # which returns:
        #
        # the real get_product_price function.
        #
        tool_to_use = tools_dict.get(
            tool_name
        )



        # Safety check.
        #
        # If Qwen requests a function
        # we do not have:
        #
        # stop the program.
        #
        if tool_to_use is None:

            raise ValueError(
                f"Tool '{tool_name}' "
                f"not found"
            )



        # ----------------------------------------------------
        # EXECUTE THE PYTHON FUNCTION
        # ----------------------------------------------------

        # ANOTHER BIG DIFFERENCE:
        #
        #
        # PREVIOUS LANGCHAIN CODE:
        #
        # observation =
        #     tool_to_use.invoke(tool_args)
        #
        #
        # Because @tool turned it into
        # a LangChain Tool object.
        #
        #
        # CURRENT CODE:
        #
        # observation =
        #     tool_to_use(**tool_args)
        #
        #
        # Because tool_to_use is now
        # a NORMAL Python function.
        #
        observation = tool_to_use(
            **tool_args
        )


        # What does ** mean?
        #
        # If:
        #
        # tool_args =
        # {
        #     "product": "laptop"
        # }
        #
        #
        # then:
        #
        # tool_to_use(**tool_args)
        #
        # becomes:
        #
        # get_product_price(
        #     product="laptop"
        # )
        #
        #
        # If:
        #
        # tool_args =
        # {
        #     "price": 1299.99,
        #     "discount_tier": "gold"
        # }
        #
        #
        # it becomes:
        #
        # apply_discount(
        #     price=1299.99,
        #     discount_tier="gold"
        # )



        # observation stores
        # the result returned by the tool.
        #
        # Example:
        #
        # observation = 1299.99
        #
        print(
            f"  [Tool Result] "
            f"{observation}"
        )



        # ----------------------------------------------------
        # ADD AI TOOL REQUEST TO HISTORY
        # ----------------------------------------------------

        # Save what the AI just requested.
        #
        # This is important because next time
        # the model must see:
        #
        # "I requested get_product_price."
        #
        messages.append(ai_message)



        # ----------------------------------------------------
        # ADD TOOL RESULT TO HISTORY
        # ----------------------------------------------------

        # DIFFERENCE FROM PREVIOUS CODE:
        #
        #
        # PREVIOUS:
        #
        # ToolMessage(
        #     content=str(observation),
        #     tool_call_id=...
        # )
        #
        #
        # CURRENT:
        #
        # We manually create the raw tool message:
        #
        messages.append(
            {
                "role": "tool",

                # Example:
                #
                # 1299.99
                #
                # converted to:
                #
                # "1299.99"
                #
                "content":
                    str(observation),
            }
        )



        # After this, messages may look conceptually like:
        #
        #
        # SYSTEM
        # "Use the tools..."
        #
        # USER
        # "Laptop with gold discount?"
        #
        # ASSISTANT
        # [tool call:
        #  get_product_price]
        #
        # TOOL
        # "1299.99"
        #
        #
        # Then the loop goes back up
        # and sends everything to Qwen again.
        #
        #
        # Qwen can now reason:
        #
        # "The laptop costs 1299.99.
        #  Now I need to call apply_discount."
        #



    # ========================================================
    # MAXIMUM ITERATIONS REACHED
    # ========================================================

    # If Qwen keeps requesting tools
    # and never produces a final response,
    # stop after MAX_ITERATIONS.
    #
    print(
        "ERROR: Max iterations reached "
        "without a final answer"
    )

    return None



# ============================================================
# PROGRAM ENTRY POINT
# ============================================================


# Run this only when the file
# is executed directly.
if __name__ == "__main__":

    # NOTE:
    #
    # This message says "LangChain Agent",
    # but this version is actually using
    # the Ollama SDK directly.
    #
    # A more accurate message could be:
    #
    # "Hello Raw Ollama Tool-Calling Agent!"
    #
    print(
        "Hello LangChain Agent "
        "(.bind_tools)!"
    )

    print()


    # Start the agent.
    #
    # Expected flow:
    #
    # User
    #   ↓
    # Qwen
    #   ↓
    # get_product_price
    #   ↓
    # 1299.99
    #   ↓
    # Qwen again
    #   ↓
    # apply_discount
    #   ↓
    # discounted price
    #   ↓
    # Qwen again
    #   ↓
    # final answer
    #
    result = run_agent(
        "What is the price of a laptop "
        "after applying a gold discount?"
    )