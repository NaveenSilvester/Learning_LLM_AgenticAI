##########################################################################################
#### This Chatbot has two tools DuckDuckGoSearch and PubMed to fetch results using
#### LLM Ollama - gemma:2b
##########################################################################################

#from langchain.chat_models import ChatOllama
#from langchain_community.chat_models import ChatOllama
#from langchain.ollama import ChatOllama
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain.schema import SystemMessage
from langchain_community.tools import DuckDuckGoSearchRun
#from langchain.tools import DuckDuckGoSearchRun
from Bio import Entrez

from langchain_ollama import ChatOllama
from langchain_community.tools import DuckDuckGoSearchRun







# 1. Configure Gemma:2b via Ollama
llm = ChatOllama(model="gemma:2b")

# 2. DuckDuckGo Search Tool
duckduckgo_tool = Tool(
    name="DuckDuckGoSearch",
    func=DuckDuckGoSearchRun().run,
    description="Search the web for current information or general topics."
)

# 3. PubMed Search Tool
def pubmed_search(query: str, max_results: int = 1) -> str:
    Entrez.email = "your_email@example.com"  # Required by NCBI
    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
    record = Entrez.read(handle)
    ids = record["IdList"]

    if not ids:
        return "No PubMed results found."

    summaries = []
    for pubmed_id in ids:
        summary_handle = Entrez.efetch(db="pubmed", id=pubmed_id, rettype="medline", retmode="text")
        summary = summary_handle.read()
        summaries.append(f"PubMed ID {pubmed_id}:\n{summary[:500]}...\n")  # Trimmed for brevity
    return "\n".join(summaries)

pubmed_tool = Tool(
    name="PubMedSearch",
    func=pubmed_search,
    description="Search scientific medical literature on PubMed."
)

# 4. Initialize the Agent
#tools = [duckduckgo_tool, pubmed_tool]
tools = [pubmed_tool]

agent = initialize_agent(
    tools=tools,
    llm=llm,
#    agent=AgentType.OPENAI_FUNCTIONS,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False
)

# 5. Chatbot Interaction Loop
def chatbot():
    print("Gemma Chatbot Ready!\nType 'exit' to quit.\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        response = agent.invoke(user_input)
        print(f"AI: {response}\n")

if __name__ == "__main__":
    chatbot()
