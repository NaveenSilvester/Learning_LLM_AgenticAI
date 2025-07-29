from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chains.conversational_retrieval import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, Tool
from langchain.llms import HuggingFaceHub
from duckduckgo_search import DDGS
from serpapi import GoogleSearch

# STEP 1: Load and Split Documents
def load_documents(file_paths):
    docs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    for path in file_paths:
        if path.endswith(".pdf"):
            loader = PyPDFLoader(path)
        elif path.endswith(".txt"):
            loader = TextLoader(path)
        else:
            continue
        docs.extend(splitter.split_documents(loader.load()))
    return docs

# STEP 2: Create the Vector Store
def create_vectorstore(documents):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma.from_documents(documents, embedding=embeddings)

# STEP 3: Setup Gemma LLM (via Hugging Face)
def get_llm():
    return HuggingFaceHub(repo_id="google/gemma-2b", model_kwargs={"temperature": 0.5, "max_new_tokens": 512})

# STEP 4: Set up Conversational Memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# STEP 5: DuckDuckGo Search Tool
def ddg_search(query: str) -> str:
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=3))
        return "\n".join([r["body"] for r in results])

# STEP 6: Google Search Tool
def google_search(query: str) -> str:
    params = {"q": query, "api_key": "YOUR_SERPAPI_KEY"}
    search = GoogleSearch(params)
    results = search.get_dict().get("organic_results", [])
    return "\n".join([r.get("snippet", "") for r in results[:3]])

# STEP 7: Combine All in a RAG + Search Agent
def create_agent(file_paths):
    docs = load_documents(file_paths)
    vs = create_vectorstore(docs)
    retriever = vs.as_retriever()
    llm = get_llm()

    # Conversational QA Chain
    qa_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)

    # External Search Tools
    tools = [
        Tool(name="DuckDuckGo", func=ddg_search, description="Search the web using DuckDuckGo."),
        Tool(name="GoogleSearch", func=google_search, description="Search the web using Google.")
    ]

    # Wrap agent
    agent = initialize_agent(tools + [Tool(name="RAG QA", func=lambda q: qa_chain.run(q), description="Answer using RAG pipeline.")],
                             llm, agent_type="zero-shot-react-description", verbose=True, memory=memory)

    return agent

# 🗣️ Run Example
if __name__ == "__main__":
    file_paths = ["yourfile.pdf", "yourfile.txt"]
    chatbot = create_agent(file_paths)
    while True:
        query = input("You: ")
        response = chatbot.run(query)
        print("Bot:", response)
