import os
from dotenv import find_dotenv, load_dotenv
from langchain_groq.chat_models import ChatGroq
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.embeddings import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.vectorstores.azuresearch import AzureSearch


class VectorStore():
    def __init__(self, shouldLoad, path) -> None:
        load_dotenv(find_dotenv())
        self._vector_store_address = os.environ['AZURE_SEARCH_ENDPOINT']
        self._vector_store_password = os.environ['AZURE_SEARCH_ADMIN_KEY']
        self._vecor_store_index_name = os.environ['AZURE_AI_SEARCH_INDEX_NAME']
        self.shouldLoad = shouldLoad
        self.path = path
        self.__loader_and_chunker__()

    def __Embedder__(self) :
        embeddings = OllamaEmbeddings(model='gemma:7b')
        return embeddings
    
    def __loader_and_chunker__(self) -> None:
        if self.shouldLoad:
            loader = PyPDFDirectoryLoader(self.path)
            docs = loader.load()
            chunker = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunked_docs = chunker.split_documents(docs)

            print(self.access_vector().add_documents(documents=chunked_docs))
    
    def access_vector(self) -> AzureSearch:

        vector_store = AzureSearch(
        azure_search_endpoint=self._vector_store_address,
        azure_search_key=self._vector_store_password,
        index_name=self._vecor_store_index_name,
        embedding_function=self.__Embedder__().embed_query
        )

        return vector_store

# if __name__ == '__main__':
    # vector_store = VectorStore(False, '/home/tanmaypatil/Documents/Vanquisher_Tech/azure-ai-service-rag/docs/')
    # print(vector_store)
