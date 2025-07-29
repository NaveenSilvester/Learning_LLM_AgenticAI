import os
from langchain.agents import initialize_agent, AgentType, Tool
from langchain_community.llms import Ollama
from langchain_community.tools import SerperDevTool
from langchain.chains.conversation.memory import ConversationBufferMemory

# --- Configuration (Ensure these are set in your environment or uncomment and set here for testing) ---
# os.environ["SERPER_API_KEY"] = "YOUR_SERPER_API_KEY" # Replace with your actual Serper API Key

# Check if SERPER_API_KEY is set
if not os.getenv("SERPER_API_KEY"):
    raise ValueError("SERPER_API_KEY environment variable not set. Please set it before running the script.")

# --- LLM and Tools Initialization ---
# 1. Initialize the LLM with Ollama
# Replace 'llama2' with the name of the model you pulled (e.g., 'mistral', 'gemma')
# Ensure Ollama server is running and the chosen model is pulled.
try:
    llm = Ollama(model="llama2", temperature=0) # You can change 'llama2' to 'mistral', 'gemma', etc.
    # A quick test to see if the LLM is accessible
    _ = llm.invoke("Hello world")
    print(f"Successfully connected to Ollama model: llama2")
except Exception as e:
    print(f"Error connecting to Ollama or model not found. Make sure Ollama is running and 'llama2' (or your chosen model) is pulled.")
    print(f"Error details: {e}")
    exit() # Exit if Ollama setup isn't correct

# 2. Define the Tools for the Agent
# SerperDevTool provides web search capabilities.
search_tool = SerperDevTool()
tools = [
    Tool(
        name=search_tool.name,  # The name the LLM will use to refer to this tool
        func=search_tool.run,   # The actual function to call for the tool's operation
        description="useful for when you need to answer questions about current events or factual information by searching the web."
    )
]

# 3. Initialize Conversation Memory
# This will store the history of the conversation, allowing the LLM to maintain context.
memory = ConversationBufferMemory(memory_key="chat_history")

# --- Agent Initialization ---
# 4. Initialize the Agent
# AgentType.CONVERSATIONAL_REACT_DESCRIPTION is suitable for multi-turn conversations
# and uses the ReAct framework for reasoning and action.
agent_chain = initialize_agent(
    tools,           # The list of tools the agent can use
    llm,             # The Large Language Model that drives the agent's decisions
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,    # Set to True to see the agent's internal thought process (very helpful!)
    memory=memory,   # The conversation memory for context
    handle_parsing_errors=True # Helps to recover if the LLM generates malformed output
)

# --- Conversation Loop ---
# 5. Run the Chatbot
print("Smart Research Chatbot (powered by Ollama): Hello! I can answer your questions by searching the web. Type 'exit' to quit.")

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Smart Research Chatbot: Goodbye!")
        break

    try:
        # The agent_chain.run() method orchestrates the LLM's reasoning, tool usage, and response generation.
        response = agent_chain.run(input=user_input)
        print(f"Smart Research Chatbot: {response}")
    except Exception as e:
        print(f"Smart Research Chatbot: An error occurred: {e}")
        print("Smart Research Chatbot: I might have trouble understanding that. Please try rephrasing or ask a different question.")