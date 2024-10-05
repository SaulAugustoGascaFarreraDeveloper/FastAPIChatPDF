from dotenv import load_dotenv
from pydantic import BaseModel
from fastapi import FastAPI,File,UploadFile
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAI,OpenAIEmbeddings
from langchain_ollama import OllamaLLM,OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
import tempfile
import os
import shutil

load_dotenv()


app = FastAPI()


llm_ollama = OllamaLLM(model="llama3.2")
embed_ollama = OllamaEmbeddings(model="nomic-embed-text")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


class PDFFilesModel(BaseModel):
    files: UploadFile



vector_store = None

doc_pages = []


text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500,chunk_overlap=200)

@app.get('/')
async def root():
    return  {"message" : "Chat PDF Api Works :)"}


@app.post('/api/vectorize')
async def vectorize_docs(items: List[UploadFile] = File(...)):
    global vector_store
    files = [file for file in items if file.filename.endswith(".pdf")]

    if  len(files) == 0:
        return {f"Your file(s) is not a pdf file ,please upload a valid pdf file !!"}
    
    docs = []

    #process every pdf file
    for file in files:
        #save the file temporality
        temp_dir = tempfile.mkdtemp()
        file_path = os.path.join(temp_dir,file.filename)


        #save the file into temp route
        with open(file_path,"wb") as f:
            shutil.copyfileobj(file.file,f)

    
    loader = PyPDFLoader(file_path)

    docs = loader.load()

    doc_pages.extend(docs)

    for page in doc_pages:
        chunks = text_splitter.split_text(page.page_content)

    return {"message chunks": chunks}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app,host="0.0.0.0",port=8000)
