import os
import glob
import shutil
import subprocess
import torch
import json
import faulthandler

faulthandler.enable()


from fastapi import FastAPI, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from websocket.socketManager import WebSocketManager

from pydantic import BaseModel

# langchain
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult
from langchain.vectorstores import Chroma

from prompt_template_utils import get_prompt_template
from load_models import load_model

from constants import CHROMA_SETTINGS, EMBEDDING_MODEL_NAME, PERSIST_DIRECTORY, MODEL_ID, MODEL_BASENAME, PATH_NAME_SOURCE_DIRECTORY, SHOW_SOURCES, CONTEXT_WINDOW_SIZE, MAX_NEW_TOKENS

class Predict(BaseModel):
    prompt: str

class Delete(BaseModel):
    filename: str

# if torch.backends.mps.is_available():
#     DEVICE_TYPE = "mps"
# elif torch.cuda.is_available():
#     DEVICE_TYPE = "cuda"
# else:
#     DEVICE_TYPE = "cpu"

DEVICE_TYPE = "cuda"

EMBEDDINGS = HuggingFaceInstructEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={"device": DEVICE_TYPE})
DB = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=EMBEDDINGS, client_settings=CHROMA_SETTINGS)
RETRIEVER = DB.as_retriever()

LLM = load_model(device_type=DEVICE_TYPE, model_id=MODEL_ID, model_basename=MODEL_BASENAME, stream=True)

prompt, memory = get_prompt_template(promptTemplate_type="llama", history=False)

QA = RetrievalQA.from_chain_type(
    llm=LLM,
    chain_type="stuff",
    retriever=RETRIEVER,
    return_source_documents=SHOW_SOURCES,
    chain_type_kwargs={
        "prompt": prompt,
    },
)

def sendPromptChain(QA, user_prompt):
    res = QA(user_prompt)

    answer, docs = res["result"], res["source_documents"]

    prompt_response_dict = {
        "Prompt": user_prompt,
        "Answer": answer,
    }

    prompt_response_dict["Sources"] = []
    for document in docs:
        prompt_response_dict["Sources"].append(
            (os.path.basename(str(document.metadata["source"])), str(document.page_content))
        )

    return prompt_response_dict;

# socket_manager = WebSocketManager()

app = FastAPI(title="homepage-app")
api_app = FastAPI(title="api app")

app.mount("/api", api_app, name="api")
app.mount("/", StaticFiles(directory="static",html = True), name="static")

@api_app.get("/training")
def run_ingest_route():
    global DB
    global RETRIEVER
    global QA

    try:
        if os.path.exists(PERSIST_DIRECTORY):
            try:
                shutil.rmtree(PERSIST_DIRECTORY)
            except OSError as e:
                raise HTTPException(status_code=500, detail=f"Error: {e.filename} - {e.strerror}.")
        else:
            raise HTTPException(status_code=500, detail="The directory does not exist")

        run_langest_commands = ["python", "ingest.py"]

        # if DEVICE_TYPE == "cpu":
        #     run_langest_commands.append("--device_type")
        #     run_langest_commands.append(DEVICE_TYPE)

        result = subprocess.run(run_langest_commands, capture_output=True)

        if result.returncode != 0:
            raise HTTPException(status_code=400, detail="Script execution failed: {}")

        # load the vectorstore
        DB = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=EMBEDDINGS,
            client_settings=CHROMA_SETTINGS,
        )

        RETRIEVER = DB.as_retriever()

        QA = RetrievalQA.from_chain_type(
            llm=LLM,
            chain_type="stuff",
            retriever=RETRIEVER,
            return_source_documents=SHOW_SOURCES,
            chain_type_kwargs={
                "prompt": prompt
            },
        )

        return {"response": "The training was successfully completed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error occurred: {str(e)}")

@api_app.get("/api/files")
def get_files():
    upload_dir = os.path.join(os.getcwd(), PATH_NAME_SOURCE_DIRECTORY)
    files = glob.glob(os.path.join(upload_dir, '*'))

    return {"directory": upload_dir, "files": files}

@api_app.delete("/api/delete_document")
def delete_source_route(data: Delete):
    filename = data.filename
    path_source_documents = os.path.join(os.getcwd(), PATH_NAME_SOURCE_DIRECTORY)
    file_to_delete = f"{path_source_documents}/{filename}"

    if os.path.exists(file_to_delete):
        try:
            os.remove(file_to_delete)
            print(f"{file_to_delete} has been deleted.")

            return {"message": f"{file_to_delete} has been deleted."}
        except OSError as e:
            raise HTTPException(status_code=400, detail=print(f"error: {e}."))
    else:
         raise HTTPException(status_code=400, detail=print(f"The file {file_to_delete} does not exist."))

@api_app.post('/predict')
def predict(data: Predict):
    global QA
    try:
        user_prompt = data.prompt
        if user_prompt:
            prompt_response_dict = sendPromptChain(QA, user_prompt)

            return {"response": prompt_response_dict}
        else:
            raise HTTPException(status_code=400, detail="Prompt Incorrect")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error occurred: {str(e)}")

@api_app.post("/save_document/")
async def create_upload_file(file: UploadFile):
    # Get the file size (in bytes)
    file.file.seek(0, 2)
    file_size = file.file.tell()

    # move the cursor back to the beginning
    await file.seek(0)

    if file_size > 10 * 1024 * 1024:
        # more than 10 MB
        raise HTTPException(status_code=400, detail="File too large")

    content_type = file.content_type

    if content_type not in [
        "text/plain",
        "text/markdown",
        "text/x-markdown",
        "text/csv",
        "application/msword",
        "application/pdf",
        "application/vnd.ms-excel",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "text/x-python",
        "application/x-python-code"]:
        raise HTTPException(status_code=400, detail="Invalid file type")

    upload_dir = os.path.join(os.getcwd(), PATH_NAME_SOURCE_DIRECTORY)
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)

    dest = os.path.join(upload_dir, file.filename)

    with open(dest, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {"filename": file.filename}

# @api_app.websocket("/ws/{user_id}")
# async def websocket_endpoint_student(websocket: WebSocket, user_id: str):
#     global QA

#     message = {
#         "message": f"Student {user_id} connected"
#     }

#     await socket_manager.add_user_to_room(user_id, websocket)
#     await socket_manager.broadcast_to_room(user_id, json.dumps(message))

#     try:
#         while True:
#             data = await websocket.receive_text()

#             prompt_response_dict = sendPromptChain(QA, data)

#             await socket_manager.broadcast_to_room(user_id, json.dumps(prompt_response_dict))

#     except WebSocketDisconnect:
#         await socket_manager.remove_user_from_room(user_id, websocket)

#         message = {
#             "message": f"Student {user_id} disconnected"
#         }

#         await socket_manager.broadcast_to_room(user_id, json.dumps(message))
#     except RuntimeError as error:
#         print(error)

# @api_app.websocket("/ws/{room_id}/{user_id}")
# async def websocket_endpoint_room(websocket: WebSocket, room_id: str, user_id: str):
#     global QA

#     message = {
#         "message": f"Student {user_id} connected to the classroom"
#     }

#     await socket_manager.add_user_to_room(room_id, websocket)
#     await socket_manager.broadcast_to_room(room_id, json.dumps(message))

#     try:
#         while True:
#             data = await websocket.receive_text()

#             prompt_response_dict = sendPromptChain(QA, data)

#             await socket_manager.broadcast_to_room(room_id, json.dumps(prompt_response_dict))

#     except WebSocketDisconnect:
#         await socket_manager.remove_user_from_room(room_id, websocket)

#         message = {
#             "message": f"Student {user_id} disconnected from room - {room_id}"
#         }

#         await socket_manager.broadcast_to_room(room_id, json.dumps(message))
#     except RuntimeError as error:
#         print(error)

