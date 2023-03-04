import os

# NOTE: for local testing only, do NOT deploy with your key hardcoded
os.environ['OPENAI_API_KEY'] = "your_key_here"

from flask import Flask, request
from llama_index import SimpleDirectoryReader, GPTSimpleVectorIndex

app = Flask(__name__)
index_name = "./index.json"
documents_folder = "./documents"

index = None
def initialize_index():
    global index
    if os.path.exists(index_name):
        index = GPTSimpleVectorIndex.load_from_disk(index_name)
    else:
        documents = SimpleDirectoryReader(documents_folder).load_data()
        index = GPTSimpleVectorIndex(documents)
        index.save_to_disk(index_name)


@app.route("/query", methods=["GET"])
def query_index():
    global index
    query_text = request.args.get("text", None)
    if query_text is None:
        return "No text found, please include a ?text=blah parameter in the URL", 200
    
    response = index.query(query_text)
    return str(response), 200


@app.route("/")
def home():
    return "Hello, World! Welcome to the llama_index docker image!"


if __name__ == "__main__":
    initialize_index()
    app.run(host="0.0.0.0", port=5601)

