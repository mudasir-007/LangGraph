# from typing import TypedDict, List
# from langchain_core.messages import HumanMessage
# from langchain_openai import ChatOpenAI
# from langgraph.graph import StateGraph, START, END
# from dotenv import load_dotenv # used to store secret stuff like API keys or configuration values

# load_dotenv()

# class AgentState(TypedDict):
#     messages: List[HumanMessage]

# llm = ChatOpenAI(model="gpt-4o")

# def process(state: AgentState) -> AgentState:
#     response = llm.invoke(state["messages"])
#     print(f"\nAI: {response.content}")
#     return state

# graph = StateGraph(AgentState)
# graph.add_node("process", process)
# graph.add_edge(START, "process")
# graph.add_edge("process", END) 
# agent = graph.compile()

# user_input = input("Enter: ")
# while user_input != "exit":
#     agent.invoke({"messages": [HumanMessage(content=user_input)]})
#     user_input = input("Enter: ")













from typing import TypedDict, List

from langchain_core.messages import HumanMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

load_dotenv()


class AgentState(TypedDict):
    messages: List[HumanMessage]


llm = ChatHuggingFace(
    llm=HuggingFaceEndpoint(
        repo_id="google/gemma-2-2b-it",
        task="text-generation",
        max_new_tokens=512,
    )
)


def process(state: AgentState) -> AgentState:
    response = llm.invoke(state["messages"])

    print(f"\nAI: {response.content}")

    return state


graph = StateGraph(AgentState)

graph.add_node("process", process)

graph.add_edge(START, "process")
graph.add_edge("process", END)

agent = graph.compile()

user_input = input("Enter: ")

while user_input != "exit":
    agent.invoke({
        "messages": [HumanMessage(content=user_input)]
    })

    user_input = input("Enter: ")