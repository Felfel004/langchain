from typing import List

from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch


class Source(BaseModel):
    """Schema for a source used by the agent"""

    url: str = Field(
        description="The URL of the source"
    )


class AgentResponse(BaseModel):
    """Schema for agent response with answer and sources"""

    answer: str = Field(
        description="The agent's answer to the query"
    )

    sources: List[Source] = Field(
        default_factory=list,
        description="List of sources used to generate the answer"
    )


# Local model running through Ollama
llm = ChatOllama(
    model="qwen3:4b",
    temperature=0
)


# Ready-made Tavily search tool
tools = [
    TavilySearch()
]


# Create the agent
#
# ToolStrategy is used because the agent already uses tools
# and we also want a structured AgentResponse.
agent = create_agent(
    model=llm,
    tools=tools,
    response_format=ToolStrategy(AgentResponse)
)


def main():

    print("Hello from langchain-course!")

    result = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "Search the internet for the 3 most-viewed matches "
                        "in the 2022 FIFA World Cup. "
                        "For each match, provide the teams, tournament stage, "
                        "viewership figures, and supporting sources."
                    )
                )
            ]
        }
    )

    print(result)


if __name__ == "__main__":
    main()