import os
from multiprocessing.managers import BaseManager
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
from werkzeug.utils import secure_filename
from typing import Any, Dict, List, Tuple
import threading    
import requests
import time
import requests
import json
import datetime
#from PseudoBaseManager import PseudoBaseManager as manager
class Config(object):
    DEBUG=True
    JSON_AS_ASCII=False
    
app = Flask(__name__)
app.config.from_object(Config)
CORS(app)

# initialize manager connection
# NOTE: you might want to handle the password in a less hardcoded way
manager = BaseManager(('', 5602), b'password')
manager.register('query_index')
manager.register('chat_index')
manager.register('insert_doc_index')
manager.register('insert_chunk_index')
manager.register('delete_from_index')
manager.register('get_documents_list')
manager.register('test')
manager.connect()

def worker(company, data: Dict):
    global manager
    name = data['name']
    age = data['age']
    
    response = manager.test(name, age)._getvalue()
    print(response)
 
       
@app.route("/test", methods=["GET"])
def test():
    time.sleep(10)
    return "ok", 200

def download_file(url, filename, save_dir):
    print(f"download url = {url}")
    # 发送 GET 请求并获取响应对象
    response = requests.get(url)

    # 拼接文件保存路径
    save_path = os.path.join(save_dir, filename)

    # 保存文件
    with open(save_path, "wb") as f:
        f.write(response.content)

    print("File downloaded and saved to:", save_path)

    # 返回已经下载的文件的完整路径
    return save_path

def query_worker(key, question, userData):
    print(key)
    print(question);
    url = "http://test-api.qi.work/tpd/api/aiWechatMessage/message/callback"

    global manager
    response = manager.query_index(key, question)._getvalue()
    response_json = {
        "answer": str(response)
    }
    response_json['userData'] = userData
    json_data = json.dumps(response_json)
    now = datetime.datetime.now()
    print(f"post = {now}:{json_data}")
    headers = {"Content-Type": "application/json;charset=utf-8"}
    response = requests.post(url, data=json_data, headers=headers)
    print(response)
        
@app.route("/query", methods=["POST"])
=======
@app.route("/query", methods=["GET"])
>>>>>>> 39422b1a8dd7c0c93e3860e97fd6161d1dfda5c3
def query_index():
    """_summary_

    Returns:
    {
        "errorCode": 0, 非0失败
        "errorMsg": "操作成功",
        "answer":”1，保养；2，检查”,
        "sessionId":1,
        "robotId":1
    }
    """
    data = request.json
    user_data = data['userData']
    key = data['key']
    if key is None:
        return "No key found, please include a ?key=blah parameter in the URL", 400

    question = data['question']
    if question is None:
        return "No question found, please include a ?question=blah parameter in the URL", 400
    
    thread = threading.Thread(target=query_worker, args=(key, question, user_data, ))
    thread.start()

    response_json = {
        "errorCode": 0,
        "errorMsg": ""
    }
    return make_response(jsonify(response_json)), 200

def chat_worker(key, question, history, userData):
    global manager
    url = "http://test-api.qi.work/tpd/api/aiWechatMessage/message/callback"

    response = manager.chat_index(key, question, history)._getvalue()
    response_json = {
        "answer": str(response)
    }
    response_json['userData'] = userData
    json_data = json.dumps(response_json)
    now = datetime.datetime.now()
    print(f"post = {now}:{json_data}")
    headers = {"Content-Type": "application/json;charset=utf-8"}
    response = requests.post(url, data=json_data, headers=headers)
    print(response)
    
@app.route("/chat", methods=["POST"])
def chat_index_ex():
    """_summary_

    Returns:
    {
        "errorCode": 0, 非0失败
        "errorMsg": "操作成功",
        "answer": "1，保养；2，检查",
        "userData": {}
    }
    """
    data = request.json
    print(data)
    user_data = data['userData']
    key = data['key']
    if key is None:
        return "No key found, please include a ?key=blah parameter in the URL", 400

    question = data['question']
    if question is None:
        return "No question found, please include a ?question=blah parameter in the URL", 400
    
    messages = data['messages']
    print(messages)
    if isinstance(messages, list):
        history = []
        for message in messages:
            if 'user' in message and 'assistant' in message:
                history.append((message["user"],message["assistant"]))
            
    chathistory = history if len(history) else None
    
    print(key)
    print(history)
    print(question)
    
    thread = threading.Thread(target=chat_worker, args=(key, question, history, user_data, ))
    thread.start()

    response_json = {
        "errorCode": 0,
        "errorMsg": ""
    }
    return make_response(jsonify(response_json)), 200
    
@app.route("/deleteFile", methods=["GET"])
def delete_index():
    global manager
    doc_id = request.args.get("doc_id", None)
    if doc_id is None:
        return "No doc_id found, please include a ?doc_id=blah parameter in the URL", 400

    manager.delete_from_index(doc_id)
    return "File deleted!", 200

def upload_chunk_worker(companyId, chunks: List):
    import requests
    import json
    global manager
    url = "http://test-api.qi.work/tpd/api/knowledgeRobotTrain/callback"
    for chunk in chunks:
        chatgptKey = chunk['chatgptKey']
        doc_id = chunk['doc_id'] if 'doc_id' in chunk else None
        doc = chunk['chunk']
        
        fileinfo = manager.insert_chunk_index(chatgptKey, doc, doc_id)._getvalue()
        print(f"doc_id = {fileinfo['doc_id']}")
        chunk['doc_id'] = fileinfo['doc_id']
        chunk['doc_type'] = 1
        chunk['companyId'] = companyId
        chunk['trainType'] = 3
        json_data = json.dumps(chunk)
        print(f"post = {json_data}")
        headers = {"Content-Type": "application/json;charset=utf-8"}
        response = requests.post(url, data=json_data, headers = headers)
        print(response)


@app.route("/uploadChunk", methods=["POST"])
def upload_chunk():
    global manager
    print(f"uploadChunk = {request.json}")
    companyId = request.json['companyId']
    if companyId is None:
        return "No companyId found, please include a ?companyId=str parameter in the URL", 400
    print(f"recv company_id = {companyId}") 
    chunks = request.json['chunk']
    if chunks is None:
        return "No chunk found, please include a ?chunk=List parameter in the URL", 400
    print(f"recv_chunks = {chunks}")
        
    thread = threading.Thread(target=upload_chunk_worker, args=(companyId, chunks,))
    thread.start()
    
    response_json = {
        "errorCode":0,
        "errorMsg":"操作成功",
        "data":"null"
        }
    return make_response(jsonify(response_json)), 200


def upload_file_worker(companyId, files: List):
    import requests
    import json
    global manager
    callback_url = "http://test-api.qi.work/tpd/api/knowledgeRobotTrain/callback"
    save_dir = "/root/documents"
    for file in files:
        key = file['chatgptKey']
        infos = file['uploadFile']
        doc_id = None
        for info in infos:
            url = info['url']
            name = info['name']
            file_path = download_file(url, name, save_dir)
            fileinfo = manager.insert_doc_index(key, file_path, doc_id=doc_id)._getvalue()
            #cleanup temp file
            if file_path is not None and os.path.exists(file_path):
               os.remove(file_path)

    # {
    #     "knowledgeId":1,
    #     "companyId":701,
    #     "trainType":1,
    #     "robotId":1,
    #     "doc_id":"docId",
    # 	  "doc_type": 1 (1文本，2文件)
    # }
            res = {}
            res['doc_type'] = 2
            res['doc_id'] = fileinfo['doc_id']
            res['companyId'] = companyId
            res['robotId'] = file['robotId']
            res['trainType'] = 3
            res['knowledgeId'] = file['knowledgeId']
            json_data = json.dumps(res)
            print(f"post = {json_data}")
            headers = {"Content-Type": "application/json;charset=utf-8"}
            response = requests.post(callback_url, data=json_data, headers = headers)
            print(response)

    
@app.route("/uploadFile", methods=["POST"])
def upload_file():
    global manager
    
    print(f"uploadFile = {request.json}")
    
    data = json.loads(request.data)
    
    companyId = data.get('companyId')
    if companyId is None:
        return "No companyId found, please include a ?companyId=str parameter in the URL", 400
    print(f"recv company_id = {companyId}") 

    files = data.get('uploadFile')
    if files is None:
        return "No files found, please include a ?files=List parameter in the URL", 400
    print(f"files = {files}")
    
    thread = threading.Thread(target=upload_file_worker, args=(companyId, files,))
    thread.start()

    response_json = {
        "errorCode":0,
        "errorMsg":"操作成功",
        "data":"null"
        }
    return make_response(jsonify(response_json)), 200


@app.route("/getDocuments", methods=["GET"])
def get_documents():
    key = request.args.get("key", None)
    if key is None:
        return "No key found, please include a ?key=blah parameter in the URL", 400
    document_list = manager.get_documents_list(key)._getvalue()

    return make_response(jsonify(document_list)), 200
    
    
@app.route("/")
def home():
    return "Hello, World! Welcome to the llama_index docker image!"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
