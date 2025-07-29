##################################################################################################################
### This Script is an Example of RAG Pipeline involving the following steps
### 1. Ingesting multiple files (TXT, PDF ) from a folder. The script navigates to every sub-folder and ingests documents
###    The parent folder where all the documents are located is named my_data_directory
### 2. Chunking the Dataset using RecursiveCharacterTextSplitter
### 3. Embedding using FastEmbedEmbeddings
### 4. Create/Load a vectorstore. The vector store used is ChromaDB. The DB is made persistent
### 5. Creating a Retriever
### 6. Using LLM - Ollama(model="tinyllama")
### 7. Creating a Prompt template 
### 8. Building a RAG Pipeline
### 9. Invoking/Query the RAG Pipeline
##################################################################################################################

import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

print ("########################################################################################################")
print ("STEP-1: Document Loading")
print ("########################################################################################################\n")
def load_documents_from_folders(base_path):
    all_documents = []
    # Using os.path.abspath to ensure we're clear about the starting point
    print(f"Starting document loading from: {os.path.abspath(base_path)}")

    # Check if the base_path actually exists
    if not os.path.exists(base_path):
        print(f"Error: Base path does not exist: {os.path.abspath(base_path)}")
        return [] # Return empty list if base path is not found

    for root, _, files in os.walk(base_path):
        for file in files:
            file_path = os.path.join(root, file)
            loader = None
            print(f"Attempting to process file: {file_path}")

            if file.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            elif file.endswith(".txt"):
                # Always try UTF-8 first, as it's most common
                # You can also use autodetect_encoding=True, but ensure 'chardet' is installed
                loader = TextLoader(file_path, encoding="utf-8")
                # Alternatively, for unknown encoding:
                # loader = TextLoader(file_path, autodetect_encoding=True)
            elif file.endswith(".csv"):
                loader = CSVLoader(file_path)
            # Add more loaders for other file types as needed

            if loader:
                try:
                    docs = loader.load()
                    for doc in docs:
                        doc.metadata["source_folder"] = root
                        doc.metadata["source_file"] = file_path
                    all_documents.extend(docs)
                    print(f"SUCCESS: Loaded {file_path}")
                except FileNotFoundError:
                    print(f"ERROR: File not found at {file_path}. Please check the path and file existence.")
                except PermissionError:
                    print(f"ERROR: Permission denied for {file_path}. Check file/folder permissions.")
                except UnicodeDecodeError as e:
                    print(f"ERROR: Encoding issue with {file_path}: {e}. Try different encoding (e.g., latin-1) or autodetect_encoding=True.")
                except Exception as e:
                    # Catch any other unexpected errors
                    print(f"UNKNOWN ERROR: Failed to load {file_path} due to: {type(e).__name__} - {e}")
            else:
                print(f"INFO: Skipping unsupported file type: {file_path}")

    return all_documents

# Example usage:
documents = load_documents_from_folders("./my_data_directory")
print(len(documents))
#print(type(documents))
print ("#############################################################\n")
print (">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
print ("STEP-1: Completed - Document Loading")
print (">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n\n")
#print(documents[0])


print ("########################################################################################################")
print ("STEP-2: Chunking")
print ("########################################################################################################\n")
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Assuming 'documents' is a list of LangChain Document objects from ingestion
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,  # Use len for character count, or a tokenizer for token count
    add_start_index=True,
)
chunks = text_splitter.split_documents(documents)
print(len(chunks))
#print(chunks[len(chunks)-1])
print (">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
print ("STEP-2: Completed - Document Chunking")
print (">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n\n")


print ("########################################################################################################")
print ("STEP-3: Embedding Model")
print ("########################################################################################################\\nn")
print ("########################################################################################################")
print ("STEP-3a: Initialize Embedding Model")
print ("########################################################################################################\n")
# 1. Initialize Embedding Model
#embeddings = OpenAIEmbeddings(model="text-embedding-ada-002") # Or FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
from langchain_community.embeddings import FastEmbedEmbeddings
embeddings = FastEmbedEmbeddings()

print ("########################################################################################################")
print ("STEP-3b: Create/Load Vector Store")
print ("########################################################################################################\n")
# 2. Create/Load Vector Store
# For a new collection:
#vectorstore = FAISS.from_documents(chunk_documents,FastEmbedEmbeddings())
#vectorstore = Chroma.from_documents(chunks, embeddings, collection_name="my_rag_collection")
# For an existing collection:
# vectorstore = Chroma(embedding_function=embeddings, collection_name="my_rag_collection")

# Define a directory to store your Chroma database files
PERSIST_DIRECTORY = './chroma_db'
# Ensure the directory exists
if not os.path.exists(PERSIST_DIRECTORY):
    os.makedirs(PERSIST_DIRECTORY)

# --- Ingestion (Creating/Updating the Vectorstore) ---
# If you are creating the vectorstore for the first time or adding new documents:
# Assume 'chunks' is your list of LangChain Document objects

# This line will create the Chroma DB in 'PERSIST_DIRECTORY' if it doesn't exist,
# or load it if it does, and then add the 'chunks'.
# It automatically persists the data.
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY,
        collection_name="my_rag_collection" # Recommended to specify a collection name
    )
    print(f"Chroma DB created/updated and persisted to {PERSIST_DIRECTORY}")


# --- Querying (Loading an existing Vectorstore) ---
# When your application restarts, you don't need to re-index.
# Just load the existing vectorstore:

# Re-initialize the embeddings with the same model used for creation
# It's crucial that the embedding model matches the one used during persistence!
#loaded_vectorstore = Chroma(
vectorstore = Chroma(    
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=embeddings,
    collection_name="my_rag_collection"
)
print(f"Chroma DB loaded from {PERSIST_DIRECTORY}")


print (">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
print ("STEP-3: Completed - Embedding Model")
print (">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n\n")



print ("########################################################################################################")
print ("STEP-4: Get a REtriever from Vector Store")
print ("########################################################################################################\n")
# Get a retriever from the vector store
retriever = vectorstore.as_retriever(search_kwargs={"k": 5}) # Retrieve top 5 similar chunks


#query="The encoder is composed of a stack of N = 6 identical layers"
query="Naveen Silvester AND Dell"

retrieved_result=vectorstore.similarity_search(query)
print("Here is the Result from RAG Query:\n")
print(retrieved_result[0].page_content)

print (">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
print ("STEP-4: Completed - Creating an Retriever")
print (">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n\n")



print ("########################################################################################################")
print ("STEP-5: LLM Orchestration")
print ("########################################################################################################\n")

print ("########################################################################################################")
print ("STEP-5a: LLM Orchestration")
print ("########################################################################################################\n")
#from langchain_openai import ChatOpenAI # Or from langchain_community.llms import Ollama
#llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1) # or Ollama(model="llama3")
from langchain_community.llms import Ollama
#llm=Ollama(model="tinyllama")
llm=Ollama(model="llama3:latest")


print ("########################################################################################################")
print ("STEP-5b: Define Prompt Template")
print ("########################################################################################################\n")
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# MessagesPlaceholder is not used in this specific template, but good to have if you plan to add chat history later.
template = """You are an AI assistant for answering questions about documents.
Use the following retrieved context to answer the question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use as much detail as possible from the context.

{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)


print ("########################################################################################################")
print ("STEP-5c: Build the RAG Chain (using LCEL): This is where the retriever and LLM are combined")
print ("########################################################################################################\n")

from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
# Helper function to format retrieved documents for the prompt
# 1. RunnableParallel executes 'context' and 'question' branches concurrently.
#    - 'context' branch: Takes the input question, passes it to the 'retriever',
#      then 'format_docs' formats the retrieved documents into a single string.
#    - 'question' branch: Simply passes the original input question through.
# 2. The combined output (a dict with 'context' and 'question' keys) is passed to the 'prompt'.
# 3. The prompt template is rendered with the context and question.
# 4. The rendered prompt is sent to the 'llm'.
# 5. The LLM's raw output is passed to 'StrOutputParser' to get a clean string answer.


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
rag_chain = (
    RunnableParallel(
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
    )
    | prompt
    | llm
    | StrOutputParser()
)
print (">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
print ("STEP-5: Completed - LLM Orchestration")
print (">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n\n")

print ("########################################################################################################")
print ("STEP-6: Invoke the Chain")
print ("########################################################################################################\n")
#query = "What are the main features of LangChain?"
#query = "Who is Naveen Silvester, where is he located?"
#query = "What is the largest living reptile?"
#query = "Can you provide me some information on Crocodylus porosus"
query = "What is Acinonyx jubatus?"
response = rag_chain.invoke(query)
print (f"Query: {query}")
retrieved_result=vectorstore.similarity_search(query)
print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n")
print(len(retrieved_result))
print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n")
#print (f"Here is the document retrieved from Vector Store {retrieved_result}")
#print(retrieved_result[0].page_content)
print("Here is the response:\n")
print(response)


