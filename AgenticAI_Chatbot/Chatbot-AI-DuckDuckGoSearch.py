########################################################################
#### This chatbot takes the user input and searches using the Tool, 
#### DuckDuckGoSearchRu, LLM - Gemma:2b
########################################################################


from langchain.chat_models import ChatOllama
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain.tools import DuckDuckGoSearchRun
from langchain.schema import SystemMessage

# 1. Set up the LLM via Ollama
llm = ChatOllama(model="gemma:2b")

# 2. Create the web search tool
search_tool = DuckDuckGoSearchRun()

# 3. Wrap the tool with LangChain's Tool interface
tools = [
    Tool(
        name="WebSearch",
        func=search_tool.run,
        description="Useful for answering questions about current events or retrieving recent information from the web."
    )
]

# 4. Optional: Customize system instructions for behavior
system_message = SystemMessage(content="You are a helpful assistant. Use tools when needed.")

# 5. Initialize the agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,  # Uses function-calling logic to invoke tools
    verbose=True
)

# 6. Run the chatbot
def chatbot():
    print("🤖 Chatbot with Web Search is ready. Type 'exit' to quit.\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        response = agent.run(user_input)
        print(f"AI: {response}\n")

if __name__ == "__main__":
    chatbot()















