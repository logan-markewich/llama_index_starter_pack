import os
from multiprocessing.managers import BaseManager
from flask import Flask, request
from werkzeug.utils import secure_filename

app = Flask(__name__)

# initialize manager connection
# NOTE: you might want to handle the password in a less hardcoded way
manager = BaseManager(('', 5602), b'password')
manager.register('query_index')
manager.register('insert_into_index')
manager.connect()


@app.route("/query", methods=["GET"])
def query_index():
    global manager
    query_text = request.args.get("text", None)
    if query_text is None:
        return "No text found, please include a ?text=blah parameter in the URL", 400
    
    response = manager.query_index(query_text)
    return str(response), 200


@app.route("/upload", methods=["POST"])
def upload_file():
    if 'file' not in request.files:
        return "Please send a POST request with a file", 400

    uploaded_file = request.files["file"]
    filename = secure_filename(uploaded_file.filename)
    if request.form.get("filename_as_doc_id", None) is not None:
        manager.insert_into_index(uploaded_file.read().decode(), doc_id=filename)
    else:
        manager.insert_into_index(uploaded_file.read().decode())

    return "File inserted!", 200


@app.route("/")
def home():
    return "Hello, World! Welcome to the llama_index docker image!"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5601)
