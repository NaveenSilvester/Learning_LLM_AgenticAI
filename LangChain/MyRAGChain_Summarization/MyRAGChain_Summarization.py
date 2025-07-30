##########################################################################################################################
#### This Python code implements a sophisticated RAG (Retrieval-Augmented Generation) chatbot using the LangChain framework. 
#### It's designed to answer questions based on a corpus of local documents (PDFs, TXT files) and can fall back to web 
#### search if it can't find an answer in the provided documents. It also incorporates chat history summarization to handle 
#### longer conversations effectively.
##########################################################################################################################


import os
from typing import List, Literal, TypedDict 
from langchain_community.document_loaders import PyPDFLoader # To load PDF documents
from langchain.text_splitter import RecursiveCharacterTextSplitter # To split loaded documents into smaller, manageable chunks
from langchain_community.embeddings import OllamaEmbeddings # To create numerical representation (embeddings) of text using models served by Ollama
from langchain_community.vectorstores import Chroma #  A vector store to efficiently store and retrieve document embeddings
from langchain_community.llms import Ollama # For LangChain 0.1.x, often enough. If issues, use langchain_ollama.OllamaLLM
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate # For Defining Structured prompts for LLMs
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda # Core components of LangChain Expression Language (LCEL) for building flexible and composable chains.
from langchain_core.runnables.history import RunnableWithMessageHistory # Manages convestational history for a chain
from langchain_core.output_parsers import StrOutputParser # Parses LLM output into a simple string.
from langchain_core.messages import HumanMessage, AIMessage # Represents messaages in a chat conversation
from langchain.memory import ChatMessageHistory # An in-memory store for chat messages.
from langchain.retrievers import ContextualCompressionRetriever # Improves retrieval by compressing retrieved documents
from langchain.retrievers.document_compressors import LLMChainExtractor # Uses an LLM to extract relevant information from documents for compression
from langchain.retrievers.multi_query import MultiQueryRetriever # Generates multiple variations of a user's query to retrieve a widers range of relevant documents
from langchain_community.tools import TavilySearchResults # For web search fallback (requires API key if used heavily)
from langchain_core.documents import Document # Represents a document in LangChain

print ("Loading Libraries ....................")

# --- Configuration ---
DATA_PATH = "data/"
CHROMA_PERSIST_DIRECTORY = "./chroma_db"
EMBEDDING_MODEL_NAME = "nomic-embed-text"  # Ensure this model is pulled in Ollama (ollama pull nomic-embed-text)
LLM_MODEL_NAME_MAIN = "llama3:latest"             # Ensure this model is pulled in Ollama (ollama pull llama3)
LLM_MODEL_NAME_SUMMARIZER = "phi3:mini"    # Ensure this model is pulled in Ollama (ollama pull phi3:mini)

print ("Loading Configurations ....................")
# Optional: Set Tavily API key for web search fallback
# os.environ["TAVILY_API_KEY"] = "YOUR_TAVILY_API_KEY" # Get from https://tavily.com/

print ("Ingestion Phase ....................")
# --- Ingestion Layer Functions ---

def load_and_chunk_documents(data_dir: str) -> List[Document]:
    """Loads documents from the specified directory and splits them into chunks."""
    documents = []
    if not os.path.exists(data_dir):
        print(f"Warning: Data directory '{data_dir}' does not exist. No documents loaded.")
        return []

    print(f"Loading documents from {data_dir}...")
    for filename in os.listdir(data_dir):
        file_path = os.path.join(data_dir, filename)
        loader = None
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif filename.endswith(".txt"):
            loader = TextLoader(file_path, encoding='utf-8') # Specify encoding
        # Add more loaders for .csv, .html etc., as needed
        
        if loader:
            try:
                docs = loader.load()
                for doc in docs:
                    doc.metadata["source_file"] = filename
                    doc.metadata["source_path"] = file_path
                documents.extend(docs)
                print(f"Loaded {filename}")
            except Exception as e:
                print(f"ERROR: Failed to load {file_path}: {type(e).__name__} - {e}")
        else:
            print(f"Skipping unsupported file type: {filename}")
    
    if not documents:
        print("No documents were loaded. Please check the 'data/' directory and file types.")
        return []

    print(f"Splitting {len(documents)} loaded documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")
    return chunks

def get_vectorstore(chunks: List[Document], embeddings_model_name: str) -> Chroma:
    """Initializes or loads a Chroma vectorstore."""
    embeddings = OllamaEmbeddings(model=embeddings_model_name)

    if not os.path.exists(CHROMA_PERSIST_DIRECTORY) or not os.listdir(CHROMA_PERSIST_DIRECTORY):
        print(f"Creating new Chroma vectorstore at {CHROMA_PERSIST_DIRECTORY}...")
        # from_documents handles initial persistence
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=CHROMA_PERSIST_DIRECTORY,
            collection_name="rag_knowledge"
        )
        print("Chroma vectorstore created and automatically persisted.")
    else:
        print(f"Loading existing Chroma vectorstore from {CHROMA_PERSIST_DIRECTORY}...")
        vectorstore = Chroma(
            persist_directory=CHROMA_PERSIST_DIRECTORY,
            embedding_function=embeddings,
            collection_name="rag_knowledge"
        )
        print("Chroma vectorstore loaded.")
    
    print(f"Vectorstore contains {vectorstore._collection.count()} documents.")
    return vectorstore

print ("LLM & Prompt Definitions ....................")
# --- LLM & Prompt Definitions ---

def get_main_llm():
    """Returns the main LLM for generation."""
    return Ollama(model=LLM_MODEL_NAME_MAIN, temperature=0.1) # Low temp for factual answers

def get_summarization_llm():
    """Returns a smaller LLM for summarization tasks."""
    return Ollama(model=LLM_MODEL_NAME_SUMMARIZER, temperature=0.3)

def get_qa_prompt():
    """Returns the prompt template for Q&A."""
    template = """You are an AI assistant for answering questions about documents.
    Use ONLY the following retrieved context to answer the question.
    If you don't know the answer based on the context, just say that you don't have enough information to answer that question.
    Do not try to make up an answer or use external knowledge.
    Provide as much detail as possible from the context.

    Context:
    {context}

    Question: {question}
    """
    return ChatPromptTemplate.from_template(template)

def get_chat_history_summarizer_prompt():
    """Returns the prompt for summarizing chat history."""
    return PromptTemplate.from_template(
        """Given the following conversation history and a new question,
        summarize the conversation history concisely to help answer the new question.
        If there's no relevant history, return "No relevant history".

        Conversation History:
        {chat_history}

        New Question: {question}

        Summarized History:"""
    )

print ("Retrieval Layer Phase ....................")
# --- Retrieval Layer Functions ---

def format_docs(docs: List[Document]) -> str:
    """Formats retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

def get_retriever(vectorstore: Chroma, llm_for_retrieval: Ollama):
    """Configures and returns a multi-query retriever with contextual compression."""
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 7}) # Retrieve more for compression/multi-query

    # Multi-Query Retriever to improve recall
    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever, llm=llm_for_retrieval
    )

    # Contextual Compression to filter and focus results
    compressor = LLMChainExtractor.from_llm(llm_for_retrieval)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=multi_query_retriever # Apply compression on multi-query results
    )
    return compression_retriever

print ("Fallback Layer Phase ....................")
# --- Fallback Layer (Conceptual) ---
def web_search_fallback(query: str):
    """Performs a web search using TavilySearchResults."""
    print("---FALLBACK: Performing web search---")
    try:
        if not os.getenv("TAVILY_API_KEY"):
            print("TAVILY_API_KEY not set. Cannot perform web search. Skipping fallback.")
            return "Web search skipped: TAVILY_API_KEY is not set."

        tool = TavilySearchResults(max_results=3) # Adjust max_results as needed
        search_results = tool.invoke({"query": query})
        # Format results for the LLM
        formatted_results = "\n".join([f"Source: {res['url']}\nContent: {res['content']}" for res in search_results])
        return f"Web Search Results:\n{formatted_results}"
    except Exception as e:
        print(f"Error during web search fallback: {e}")
        return "Error performing web search."

# --- Orchestration with LangChain Expression Language (LCEL) ---

# In-memory session store for chat history (replace with Redis/DB for production)
store = {}
def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def create_rag_chain(vectorstore: Chroma, main_llm: Ollama, summarizer_llm: Ollama):
    """Creates the full RAG conversational chain with summarization and fallback."""
    retriever = get_retriever(vectorstore, summarizer_llm) # Use summarizer LLM for retriever operations
    qa_prompt = get_qa_prompt()
    chat_history_summarizer_prompt = get_chat_history_summarizer_prompt()

    # Define a runnable to decide if history needs summarization
    # This is a simple heuristic; more complex logic could be in LangGraph
    def needs_history_summarization(chat_history: List[AIMessage], max_history_length=500):
        # A simple token count check or number of messages
        total_len = sum(len(msg.content) for msg in chat_history)
        return total_len > max_history_length

    # Chain to summarize chat history
    summarize_history_chain = (
        {"chat_history": RunnablePassthrough(), "question": RunnablePassthrough()}
        | chat_history_summarizer_prompt
        | summarizer_llm
        | StrOutputParser()
    )

    # This chain decides whether to summarize history or use it raw
    def process_history(inputs: dict) -> str:
        chat_history = inputs.get("chat_history", [])
        question = inputs.get("question", "")
        
        # Format history for summarization or direct use
        formatted_history = "\n".join([f"{msg.type}: {msg.content}" for msg in chat_history])

        if needs_history_summarization(chat_history):
            print("---Summarizing chat history---")
            return summarize_history_chain.invoke({"chat_history": formatted_history, "question": question})
        else:
            print("---Using full chat history---")
            return formatted_history


    # Core RAG logic that takes context and question
    rag_chain_content = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | qa_prompt
        | main_llm
        | StrOutputParser()
    )

    # Define the full conversational chain with a simple fallback check (can be expanded with LangGraph)
    def invoke_with_fallback(inputs: dict) -> str:
        try:
            # Attempt to get an answer from RAG
            answer = rag_chain_content.invoke(inputs)
            # Simple check for "don't have enough information" as a signal for fallback
            if "i don't have enough information" in answer.lower() or "not in the provided context" in answer.lower():
                print("---RAG response indicates insufficient info, attempting web search---")
                web_results = web_search_fallback(inputs["question"])
                if web_results and "skipped" not in web_results:
                    # Re-run LLM with web results as additional context
                    fallback_context = inputs["context"] + "\n\n" + web_results
                    print("---Re-generating with web search context---")
                    return (
                        {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
                        | ChatPromptTemplate.from_template(f"Based on the following context and web results, answer the question:\n\n{{context}}\n\nQuestion: {{question}}")
                        | main_llm
                        | StrOutputParser()
                    ).invoke({"context": fallback_context, "question": inputs["question"]})
                else:
                    return answer # Return original "don't know" if web search also fails/skipped
            return answer
        except Exception as e:
            print(f"Error during RAG chain invocation: {e}. Falling back to general response.")
            # General fallback if there's any unexpected error in RAG chain
            return "I apologize, but I encountered an issue while trying to answer your question. Please try rephrasing."


    # Combine all parts:
    # This chain implicitly uses history as a part of the "question" input, 
    # if `process_history` prepends the summary to the question.
    # For robust history handling, especially for LLMs that expect messages,
    # consider adapting `qa_prompt` to use `MessagesPlaceholder` for history.
    final_rag_chain = RunnableParallel(
        question=RunnablePassthrough(),
        chat_history=RunnableLambda(lambda x: x.get("chat_history", [])) # Pass raw history for session management
    ) | {
        "context": retriever | format_docs, # Context from documents
        "question": RunnablePassthrough.assign(
            original_question=lambda x: x["question"], # Keep original question
            # Conditionally summarize history and append to question if needed
            processed_history_and_question=lambda x: (
                process_history({"chat_history": x["chat_history"], "question": x["question"]}) + 
                "\n" + x["question"] # Append original question
            ) if needs_history_summarization(x["chat_history"]) else x["question"] # Use original if not summarizing
        ) | (lambda x: x["processed_history_and_question"] if x.get("processed_history_and_question") else x["question"])
    } | invoke_with_fallback # This step gets context and question, then handles logic
    

    # Add `RunnableWithMessageHistory` to manage session state
    conversational_rag_chain_with_history = RunnableWithMessageHistory(
        final_rag_chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history", # This maps to where `RunnableWithMessageHistory` expects to pass history
    )

    return conversational_rag_chain_with_history


# --- Main Execution Logic ---

if __name__ == "__main__":
    print("--- Starting RAG Chatbot Setup ---")

    # 1. Ingestion: Load, chunk, and set up vectorstore
    chunks = load_and_chunk_documents(DATA_PATH)
    
    if not chunks:
        print("Exiting: No documents loaded or chunks created. Please check your 'data/' directory.")
        exit() # Exit if no documents to process

    vectorstore = get_vectorstore(chunks, EMBEDDING_MODEL_NAME)

    # 2. Initialize LLMs
    main_llm_instance = get_main_llm()
    summarizer_llm_instance = get_summarization_llm()

    # 3. Create the RAG chain
    rag_chain = create_rag_chain(vectorstore, main_llm_instance, summarizer_llm_instance)

    print("\n--- RAG Chatbot Ready ---")
    print("Type 'exit' or 'quit' to end the chat.")

    session_id = "user_session_123" # A fixed session ID for a simple CLI chat

    # 4. Start the interactive chat
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Chatbot: Goodbye!")
            break

        try:
            # Invoke the conversational RAG chain
            # The `chat_history` is automatically managed by RunnableWithMessageHistory
            response = rag_chain.invoke(
                {"question": user_input},
                config={"configurable": {"session_id": session_id}}
            )
            print(f"Chatbot: {response}")
        except Exception as e:
            print(f"Chatbot Error: An unexpected error occurred: {e}")
            print("Please try again or restart the chatbot.")