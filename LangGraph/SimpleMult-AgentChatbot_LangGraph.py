# pip install -r requirements.txt

print("###########################################################################")
print("Loading Packages.............")
print("###########################################################################\n")

from typing import Annotated, Sequence, TypedDict
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import tool

print("|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")
print("Completed Loading of all the required Python Packages")
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n")


print("###########################################################################")
print("1. Define the Graph State.............")
print("###########################################################################\n")

# 1. Define the Graph State
# This represents the overall state of our chatbot's conversation
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], lambda x, y: x + y]
    next_node: str

print("|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")
print("Completed refining Graph State")
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n")


print("###########################################################################")
print("2. Initialize Ollama LLM (Gemma:2b).............")
print("###########################################################################\n")

# 2. Initialize Ollama LLM (Gemma:2b)
# Make sure Ollama is running and gemma:2b is pulled
llm = ChatOllama(model="gemma:2b")

print("|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")
print("Completed Initializig Ollam LLM (Gemma:2b)")
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n")


print("###########################################################################")
print("3. Define the Agents (Nodes).............")
print("###########################################################################\n")

# 3. Define the Agents (Nodes)
# General Chat Agent
class GeneralChatAgent:
    def __init__(self, llm):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a friendly general-purpose chatbot. Your goal is to continue the conversation naturally, answer questions, and remember previous interactions. Respond concisely and politely."),
            # IMPORTANT: The prompt now includes the entire message history
            ("human", "Previous conversation: {history}\n\nCurrent question: {question}")
        ])

    def invoke(self, state: AgentState):
        # Extract the current question (last human message)
        current_question = state["messages"][-1].content
        
        # Build the conversation history excluding the current question
        # We start from index 0 up to the second-to-last message (i.e., excluding the latest HumanMessage)
        history_messages = state["messages"][:-1] 
        history_str = "\n".join([f"{m.type}: {m.content}" for m in history_messages])

        # Invoke the LLM with the full history
        response = self.llm.invoke(self.prompt.format_messages(history=history_str, question=current_question))
        
        return {"messages": [AIMessage(content=response.content)], "next_node": ""}

class MathAgent:
    def __init__(self, llm):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful math assistant. Solve mathematical problems or clarify math concepts. If it's not a math problem, say you can't help. You can also remember previous math questions if the user refers to them."),
            # For math, we might still want the history to understand context if they say "what about the previous answer?"
            ("human", "Previous conversation: {history}\n\nCurrent math query: {question}")
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
            ("human", "{question}") # The supervisor still only needs the current question for routing
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

print("|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")
print("Completed Defining the Agents (Nodes)")
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n")


print("###########################################################################")
print("4. Build the LangGraph.............")
print("###########################################################################\n")
# 4. Build the LangGraph
workflow = StateGraph(AgentState)

# Add nodes for each agent
workflow.add_node("general_chat_agent", general_chat_agent.invoke)
workflow.add_node("math_agent", math_agent.invoke)
workflow.add_node("supervisor", supervisor.route_agent)

# Set the entry point to the supervisor
workflow.set_entry_point("supervisor")

# Add edges based on the supervisor's decision
workflow.add_conditional_edges(
    "supervisor",
    lambda state: state.get("next_node", "general"), # Reads 'next_node' from state
    {
        "math": "math_agent",   # Must match the forced 'final_decision' from supervisor
        "general": "general_chat_agent", # Must match the forced 'final_decision' from supervisor
    }
)

# Agents transition to END after responding
workflow.add_edge("general_chat_agent", END)
workflow.add_edge("math_agent", END)
# Compile the graph
app = workflow.compile()

print("|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")
print("Completed Building the LangGraph")
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n")


print("###########################################################################")
print("5. Run the Chatbot.............")
print("###########################################################################\n")
# 5. Run the Chatbot
print("Multi-Agent Chatbot (Type 'exit' to quit)")

# Initialize the state before the loop
# Start with an empty list of messages and an empty next_node
current_state = {"messages": [], "next_node": ""} 

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    # Add the new human message to the current state
    current_state["messages"].append(HumanMessage(content=user_input))
    # Reset next_node for the new turn
    current_state["next_node"] = "" 

    # Run the graph with the accumulated state as input
    # The 'stream_mode="values"' is crucial as it yields the full state at each step
    for s in app.stream(current_state, stream_mode="values"):
        # The 's' here is the state *after* a node has run.
        # We only want to process the *final* state after the graph execution for this turn.
        # LangGraph's stream typically yields multiple states as it progresses.
        # The last state yielded will be the final state of the graph for the turn.
        pass # We use 'pass' here to simply let the loop complete and get the final 's'

    # After the loop finishes (meaning the graph reached END), 's' will hold the final state
    final_state_of_turn = s 
    
    # Update current_state with the final state of the turn
    current_state = final_state_of_turn

    # Extract and print the latest AI response from the updated state
    if current_state and "messages" in current_state:
        latest_message = current_state["messages"][-1]
        if isinstance(latest_message, AIMessage):
            print(f"Bot: {latest_message.content}")