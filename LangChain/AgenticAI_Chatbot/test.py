import os
from langchain.agents import initialize_agent, AgentType, Tool
from langchain_community.chat_models import ChatOllama # Use ChatOllama for better agent compatibility
#from langchain_community.tools import SerperDevTool
#from langchain_community.tools.google_serper import SerperDevTool
from langchain.chains.conversation.memory import ConversationBufferMemory

import os
from langchain.agents import initialize_agent, AgentType, Tool
from langchain_community.chat_models import ChatOllama
from langchain.chains.conversation.memory import ConversationBufferMemory

os.environ["SERPER_API_KEY"] = "250c25fae94c058a794dac29637f79940a35b5c0"

import os
from langchain.agents import initialize_agent, AgentType, Tool
from langchain_community.chat_models import ChatOllama
from langchain_community.tools.google_serper import SerperDevTool # Reverting to the direct import
from langchain.chains.conversation.memory import ConversationBufferMemory

# --- Configuration ---
if not os.getenv("SERPER_API_KEY"):
    raise ValueError("SERPER_API_KEY environment variable not set. "
                     "Please set it (e.g., 'export SERPER_API_KEY=\"your_key\"') before running the script.")

# --- LLM and Tools Initialization ---
try:
    llm = ChatOllama(model="gemma:2b", temperature=0)
    _ = llm.invoke("Hello")
    print("Ollama Chat model 'gemma:2b' initialized successfully.")
except Exception as e:
    print(f"Error initializing Ollama Chat LLM: {e}")
    print("Please ensure Ollama is running and the specified model ('gemma:2b') is pulled.")
    print("You might need to adjust the 'model' parameter if you pulled a different one (e.g., 'llama2:chat').")
    exit()

# 2. Define the Tools for the Agent
# Directly initialize SerperDevTool. This requires google-search-results to be installed.
try:
    serper_tool = SerperDevTool()
    tools = [
        Tool(
            name=serper_tool.name, # This will typically be "Serper_Dev_Tool"
            func=serper_tool.run,
            description="useful for when you need to answer questions about current events, factual information, or perform web searches. Input should be a concise search query string."
        )
    ]
    print("SerperDevTool initialized successfully.")
except Exception as e:
    print(f"Error initializing SerperDevTool: {e}")
    print("Please ensure 'google-search-results' is installed: pip install google-search-results")
    print("And that SERPER_API_KEY is correctly set.")
    exit()


# 3. Initialize Conversation Memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# --- Agent Initialization ---
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    memory=memory,
    handle_parsing_errors=True
)

# --- Conversation Loop ---
print("--- Gemma:2b Powered Chatbot with Web Search ---")
print("Hello! Ask me anything, and I'll try to find the answer by searching the web.")
print("Type 'exit' to quit.")
print("---------------------------------------------")

while True:
    user_query = input("You: ")
    if user_query.lower() == 'exit':
        print("Chatbot: Goodbye!")
        break
    try:
        response = agent.run(user_query)
        print(f"Chatbot: {response}")
    except Exception as e:
        print(f"Chatbot Error: {e}")
        print("Chatbot: Sorry, I couldn't process that request. Can you rephrase?")