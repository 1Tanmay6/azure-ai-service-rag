#! Package imports
import os
import time
from pprint import pprint
from dotenv import find_dotenv, load_dotenv
from langchain_groq.chat_models import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
#! File imports
from vector_store import VectorStore

class GroqLLMInteraction():
    
    def __init__(self, model, shouldLoad, path) -> None:
        load_dotenv(find_dotenv())
        self._groq_api_key = os.environ['GROQ_API_KEY']
        self.shouldLoad = shouldLoad
        self.path = path
        self.model = model
    
    def __llm__(self) -> ChatGroq:
        llm = ChatGroq(groq_api_key=self._groq_api_key, model=self.model)
        return llm
    
    def __prompt__(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_template(
        """
        You are an expert in research, and you can easily fetch information from the limited data given in context.
        Answer the following question only on the given context and nothing else.
        Please provide the most accurate response and tell it in a nicely formatted and elaborated manner.
        <context>
        {context}
        </context>

        QUESTIONS: {input}
        """
        )
    
    def __define_chains__(self) -> create_retrieval_chain:
        vector_store = VectorStore(self.shouldLoad, self.path).access_vector()
        document_chain = create_stuff_documents_chain(self.__llm__(), self.__prompt__())
        retriever = vector_store.as_retriever()
        retriever_chain = create_retrieval_chain(retriever, document_chain)
        return retriever_chain
    
    def ask_llm(self, user_input) -> dict:
        if user_input:
            start = time.time()
            response = self.__define_chains__().invoke({'input': user_input})
            end = time.time()
            return {
                'response_time': end-start,
                'answer': response['answer'],
                'context': [{'page_content': doc.page_content, 'metadata': doc.metadata}  for doc in response['context']]
            }
        return {'error': UserWarning}

# 'gemma-7b-it'

if __name__ == '__main__':
    llm = GroqLLMInteraction('gemma-7b-it', False, '/home/tanmaypatil/Documents/Vanquisher_Tech/azure-ai-service-rag/docs/')
    pprint(llm.ask_llm('why is SVD used?'))