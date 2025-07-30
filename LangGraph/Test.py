from typing import Annotated, Sequence, TypedDict
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import tool

# 1. Define the Graph State
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], lambda x, y: x + y]
    next_node: str

# 2. Initialize Ollama LLM (Gemma:2b)
llm = ChatOllama(model="llama3:latest")

# 3. Define the Agents (Nodes)

class GeneralChatAgent:
    def __init__(self, llm):
        self.llm = llm
        # UPDATED PROMPT FOR BETTER MEMORY RECALL
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a friendly general-purpose chatbot. You have access to the full conversation history. Your goal is to continue the conversation naturally, answer questions, and respond to queries about previous interactions. When asked about previous questions or topics, refer to the provided 'Conversation History' section directly and accurately."),
            ("human", "---Conversation History---\n{history}\n\n---Current User Input---\nUser: {question}\n\n---Your Response---")
        ])

    def invoke(self, state: AgentState):
        current_question = state["messages"][-1].content
        
        history_messages = state["messages"][:-1] 
        history_str = "\n".join([f"{m.type}: {m.content}" for m in history_messages])

        response = self.llm.invoke(self.prompt.format_messages(history=history_str, question=current_question))
        
        return {"messages": [AIMessage(content=response.content)], "next_node": ""}

class MathAgent:
    def __init__(self, llm):
        self.llm = llm
        # UPDATED PROMPT FOR BETTER MEMORY RECALL
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful math assistant. You have access to the full conversation history. Solve mathematical problems or clarify math concepts. If it's not a math problem, say you can't help. When asked about previous math questions or answers, refer to the 'Conversation History' section directly."),
            ("human", "---Conversation History---\n{history}\n\n---Current User Input---\nUser: {question}\n\n---Your Response---")
        ])
        
    def invoke(self, state: AgentState):
        current_question = state["messages"][-1].content
        
        history_messages = state["messages"][:-1]
        history_str = "\n".join([f"{m.type}: {m.content}" for m in history_messages])

        try:
            if any(op in current_question for op in ['+', '-', '*', '/', 'calculate', 'what is']) and \
               all(c.isdigit() or c in ' .+-*/()' for c in current_question.replace('calculate', '').replace('what is', '').strip()):
                expression = current_question.lower().replace("calculate", "").replace("what is", "").strip()
                result = eval(expression)
                return {"messages": [AIMessage(content=f"The result is: {result}")], "next_node": ""}
            else:
                response = self.llm.invoke(self.prompt.format_messages(history=history_str, question=current_question))
                return {"messages": [AIMessage(content=response.content)], "next_node": ""}
        except Exception as e:
            print(f"Math Agent Error: {e}")
            return {"messages": [AIMessage(content="I'm sorry, I couldn't solve that math problem. Please provide a clear arithmetic expression or concept.")], "next_node": ""}


class Supervisor:
    def __init__(self, llm):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a supervisor that directs user queries to the appropriate agent. 
            Your goal is to decide whether the user's request is related to 'math' or 'general' conversation.
            Respond with 'math' if the query is about calculation, mathematical concepts, or equations.
            Respond with 'general' for all other types of queries, including greetings, casual chat, remembering history, or non-math facts.
            Only respond with 'math' or 'general' as a single word, without punctuation or extra text."""),
            ("human", "{question}")
        ])

    def route_agent(self, state: AgentState):
        question = state["messages"][-1].content
        response = self.llm.invoke(self.prompt.format_messages(question=question))
        
        raw_decision = response.content.strip().lower()
        print(f"--- RAW LLM ROUTER DECISION: '{response.content}' ---")
        print(f"--- PROCESSED ROUTER DECISION: '{raw_decision}' ---")

        if "math" in raw_decision:
            final_decision = "math"
        else:
            final_decision = "general"

        print(f"--- FINAL ROUTING TO: '{final_decision}' ---")
        return {"next_node": final_decision}

# Initialize our agents and supervisor
general_chat_agent = GeneralChatAgent(llm)
math_agent = MathAgent(llm)
supervisor = Supervisor(llm)

# 4. Build the LangGraph
workflow = StateGraph(AgentState)

workflow.add_node("general_chat_agent", general_chat_agent.invoke)
workflow.add_node("math_agent", math_agent.invoke)
workflow.add_node("supervisor", supervisor.route_agent)

workflow.set_entry_point("supervisor")

workflow.add_conditional_edges(
    "supervisor",
    lambda state: state.get("next_node", "general"),
    {
        "math": "math_agent",
        "general": "general_chat_agent",
    }
)

workflow.add_edge("general_chat_agent", END)
workflow.add_edge("math_agent", END)

app = workflow.compile()

# 5. Run the Chatbot with conversational memory
print("Multi-Agent Chatbot (Type 'exit' to quit)")

current_state = {"messages": [], "next_node": ""} 

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    current_state["messages"].append(HumanMessage(content=user_input))
    current_state["next_node"] = "" 

    final_state_of_turn = None # Initialize to None for safety

    for s in app.stream(current_state, stream_mode="values"):
        final_state_of_turn = s # This will always be the last state yielded

    if final_state_of_turn: # Check if a state was actually yielded
        current_state = final_state_of_turn
        
        if "messages" in current_state:
            latest_message = current_state["messages"][-1]
            if isinstance(latest_message, AIMessage):
                print(f"Bot: {latest_message.content}")