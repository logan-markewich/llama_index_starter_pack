import os
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from werkzeug.utils import secure_filename
from typing import Any, Dict, List, Tuple
from llama_index.llms.base import ChatMessage
import threading    
import requests
import time
import requests
import json
import datetime
import index_server
    
app = FastAPI()
openai_api_key = "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"
use_global_api = True
# initialize manager connection
# NOTE: you might want to handle the password in a less hardcoded way

# def worker(company, data: Dict):
#     name = data['name']
#     age = data['age']
    
#     response = index_server.test(name, age)
#     print(response)
 
       
@app.get("/test")
def test():
    print("test")
    time.sleep(10)
    return "ok", 200

def chat_history(messages : List) -> List[ChatMessage]:
    print(messages)
    
    history = []
    
    if isinstance(messages, list):
        for message in messages:
            for key, value in message.items():
                history.append(ChatMessage(content=value, role=key)) 
            
    return history

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

    response = index_server.query_index(key, question)
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
        
@app.post("/query")
def query_index(data: dict):
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
    print(data)
    user_data = data['userData']
    key = data['key']
    if key is None:
        return "No key found, please include a ?key=blah parameter in the URL", 400

    question = data['question']
    if question is None:
        return "No question found, please include a ?question=blah parameter in the URL", 400

    if use_global_api:
        key = openai_api_key
                    
    thread = threading.Thread(target=query_worker, args=(key, question, user_data, ))
    thread.start()

    response_json = {
        "errorCode": 0,
        "errorMsg": ""
    }
    return JSONResponse(content=response_json, status_code=200)

@app.post("/querySync")
def query_index_sync(data: dict):
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
    user_data = data['userData']
    key = data['key']
    if key is None:
        return "No key found, please include a ?key=blah parameter in the URL", 400

    question = data['question']
    if question is None:
        return "No question found, please include a ?question=blah parameter in the URL", 400

    if use_global_api:
        key = openai_api_key
    
    print(f"key============{key}")
    response = index_server.query_index(key, question)
    response_json = {
        "answer": str(response)
    }
    response_json['userData'] = user_data
    headers = {"Content-Type": "application/json;charset=utf-8"}
    return JSONResponse(content=response_json, status_code=200, headers=headers)

def chat_worker(key, question, history, userData):
    url = "http://test-api.qi.work/tpd/api/aiWechatMessage/message/callback"

    response = index_server.chat_index(key, question, history)
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
    
@app.post("/chat")
def chat_index_ex(data: dict):
    """_summary_

    Returns:
    {
        "errorCode": 0, 非0失败
        "errorMsg": "操作成功",
        "answer": "1，保养；2，检查",
        "userData": {}
    }
    """
    print(data)
    user_data = data['userData']
    key = data['key']
    if key is None:
        return "No key found, please include a ?key=blah parameter in the URL", 400

    question = data['question']
    if question is None:
        return "No question found, please include a ?question=blah parameter in the URL", 400
    
    messages = data['messages']            
    chathistory = chat_history(messages)

    if use_global_api:
        key = openai_api_key
    
    print(key)
    print(question)
    
    thread = threading.Thread(target=chat_worker, args=(key, question, chathistory, user_data, ))
    thread.start()

    response_json = {
        "errorCode": 0,
        "errorMsg": ""
    }
    return JSONResponse(content=response_json, status_code=200)

@app.post("/chatSync")
def chat_index_sync(data: dict):
    """_summary_

    Returns:
    {
        "errorCode": 0, 非0失败
        "errorMsg": "操作成功",
        "answer": "1，保养；2，检查",
        "userData": {}
    }
    """
    print(data)
    user_data = data['userData']
    key = data['key']
    if key is None:
        return "No key found, please include a ?key=blah parameter in the URL", 400

    question = data['question']
    if question is None:
        return "No question found, please include a ?question=blah parameter in the URL", 400
    
    messages = data['messages']
    chathistory = chat_history(messages)

    if use_global_api:
        data['key'] = openai_api_key
    
    print(key)
    print(question)
    
    response = index_server.chat_index(key, question, chathistory)
    response_json = {
        "answer": str(response)
    }
    response_json['userData'] = user_data
    json_data = json.dumps(response_json)
    now = datetime.datetime.now()
    print(f"post = {now}:{json_data}")
    headers = {"Content-Type": "application/json;charset=utf-8"}
    return JSONResponse(content=response_json, status_code=200)

 
@app.get("/deleteFile/{doc_id}")
def delete_index(doc_id: str):
    if doc_id is None:
        return "No doc_id found, please include a ?doc_id=blah parameter in the URL", 400

    index_server.delete_from_index(doc_id)
    return "File deleted!", 200

def upload_chunk_worker(companyId, chunks: List):
    import requests
    import json
    url = "http://test-api.qi.work/tpd/api/knowledgeRobotTrain/callback"
    for chunk in chunks:
        chatgptKey = openai_api_key if use_global_api else chunk['chatgptKey']
        doc_id = chunk['doc_id'] if 'doc_id' in chunk else None
        doc = chunk['chunk']
        if doc_id is not None:
            index_server.delete_from_index(chatgptKey,doc_id)        
        
        fileinfo = index_server.insert_chunk_index(chatgptKey, doc, doc_id)
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


@app.post("/uploadChunk")
def upload_chunk(data: dict):
    companyId = data['companyId']
    if companyId is None:
        return "No companyId found, please include a ?companyId=str parameter in the URL", 400
    print(f"recv company_id = {companyId}") 
    chunks = data['chunk']
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
    return JSONResponse(content=response_json, status_code=200)


def upload_file_worker(companyId, files: List):
    import requests
    import json
    callback_url = "http://test-api.qi.work/tpd/api/knowledgeRobotTrain/callback"
    save_dir = "/root/documents"
    for file in files:
        key = file['chatgptKey']
        
        if use_global_api:
            key = openai_api_key

        infos = file['uploadFile']
        doc_id = file['doc_id'] if 'doc_id' in file else None 
        if doc_id is not None:
            index_server.delete_from_index(key,doc_id)
        for info in infos:
            url = info['url']
            name = info['name']
            file_path = download_file(url, name, save_dir)
            fileinfo = index_server.insert_doc_index(key, file_path, doc_id=doc_id)
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

    
@app.post("/uploadFile")
def upload_file(data: dict):    
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
    return JSONResponse(content=response_json, status_code=200)


@app.get("/getDocuments/{key}")
def get_documents(key: str):
    if key is None:
        return "No key found, please include a ?key=blah parameter in the URL", 400
    
    if use_global_api:
        key = openai_api_key

    document_list = index_server.get_documents_list(key)

    return JSONResponse(content=document_list, status_code=200)

    
@app.get("/")
def home():
    return "Hello, World! Welcome to the llama_index docker image!"


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5001)
