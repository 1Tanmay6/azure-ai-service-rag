# Basic Packages
import os
import time
from pprint import pprint
import streamlit as st
from dotenv import find_dotenv, load_dotenv

#? Neccesary langchain imports
from langchain_groq.chat_models import ChatGroq
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.embeddings import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain

#? Azure specific imports
from langchain_community.vectorstores.azuresearch import AzureSearch
# from langchain_community.retrievers import AzureAISearchRetriever #! Didn't worked

#! CONFIGURATION AND INITIALIZATION
load_dotenv(find_dotenv())

#* Vector store Creds
vector_store_address = os.environ['AZURE_SEARCH_ENDPOINT']
vector_store_password = os.environ['AZURE_SEARCH_ADMIN_KEY']
vecor_store_index_name = os.environ['AZURE_AI_SEARCH_INDEX_NAME']
service_name = os.environ['AZURE_AI_SEARCH_SERVICE_NAME']
api_key = os.environ['AZURE_AI_SEARCH_API_KEY']

#* Groq API key
groq_api_key = os.environ['GROQ_API_KEY']

#* Embeddings
embeddings = OllamaEmbeddings(model='gemma:7b')

# #* Vector Store
vector_store = AzureSearch(
    azure_search_endpoint=vector_store_address,
    azure_search_key=vector_store_password,
    index_name=vecor_store_index_name,
    embedding_function=embeddings.embed_query
)


#* Loader and chunker
loader = PyPDFDirectoryLoader('/home/tanmaypatil/Documents/Vanquisher_Tech/azure-ai-service-rag/docs/')
docs = loader.load()
chunker = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunked_docs = chunker.split_documents(docs)

#* Adds documents to the azure vector store [Not neccesary when already done]
# print(vector_store.add_documents(documents=chunked_docs))

#! Embedding Lengths (azure vector dimensions limit=3072):
#! 1. Llama2: 4096
#! 2. Llama3: 4096
#! 3. phi:3B: 2560
#! 4. Gemma: 3072

#* Definig the prompt, for the specific use case
prompt = ChatPromptTemplate.from_template(
"""
You are an expert in research, and you can easily formulate answers from the limited data given in context.
Please provide the most accurate response and tell it in a nicely formatted and elaborated manner.
<context>
{context}
</context>

QUESTIONS: {input}
"""
)

# #* LLM
llm = ChatGroq(groq_api_key=groq_api_key, model='gemma-7b-it')

# #* Chains: Stuff Document Chain and Retriever Chain
document_chain = create_stuff_documents_chain(llm, prompt)
retriever = vector_store.as_retriever()
retriever_chain = create_retrieval_chain(retriever, document_chain)

# Streamlit UI
st.title("Research Assistant")
st.write("Ask a question based on the provided context.")

# Input from user
user_input = st.text_input("Your Question:", "What is SVD?")

if user_input:
    with st.spinner('Fetching response...'):
        start = time.time()
        response = retriever_chain.invoke({'input': user_input})
        end = time.time()
        
        st.write(f"**Response Time:** {end - start:.2f} seconds")
        
        st.write("### Answer:")
        st.write(response['answer'])
        
        st.write("### Relevant Context:")
        for i, doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write('_____________________________')

# #* Retriever 
# retriever = AzureAISearchRetriever(
#     content_key="content", top_k=1, index_name="langchain-vector-demo"
# )

# retriever.invoke("does the hybrid recommedation system outperforms traditional methods of recommending?")
