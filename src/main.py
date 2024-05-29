from flask import Flask, request
import json
from vector_store import VectorStore
from llm_interaction import GroqLLMInteraction

app = Flask(__name__)

@app.route('/', methods=['GET'])
def welcome():
    return 'Welcome to the LLM Interaction API!'

# Route for retrieving the vector store
@app.route('/vector-store', methods=['POST'])
def get_vector_store():
    data = request.json
    shouldLoad = data['shouldLoad']
    path = data['path']
    # Logic to retrieve the vector store
    vector_store = VectorStore(shouldLoad, path).access_vector()
    print(vector_store.fields)
    return json.dumps('Azure AI service, Vector Database')

# Route for querying the LLM
@app.route('/query-llm', methods=['POST'])
def query_llm():
    data = request.json
    model = data['model']
    shouldLoad = data['shouldLoad']
    path = data['path']
    query = data['query']
    # Logic to query the LLM
    llm_interact = GroqLLMInteraction(model, shouldLoad, path)
    
    return json.dumps(llm_interact.ask_llm(query))

if __name__ == '__main__':
    app.run(debug=True)