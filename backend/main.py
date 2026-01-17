import os
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from langchain_community.llms import HuggingFaceHub
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="IA de Apoio Operacional e Assistencial")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Servir arquivos estáticos do frontend
if os.path.exists("./frontend"):
    app.mount("/static", StaticFiles(directory="./frontend"), name="static")

@app.get("/")
async def read_index():
    return FileResponse("./frontend/index.html")

# Configurações da Base de Conhecimento
PERSIST_DIRECTORY = "./data/chroma_db"
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)

SYSTEM_PROMPT = """Você é uma IA de apoio operacional e assistencial para equipes de saúde.
Sua função é fornecer informações precisas baseadas nos manuais e protocolos cadastrados.

REGRAS OBRIGATÓRIAS:
1. Responda APENAS com base no conhecimento fornecido no contexto.
2. NUNCA invente informações ou use conhecimentos externos.
3. Se a resposta não estiver na base de conhecimento, responda EXATAMENTE: "Não sei responder. Procure sua liderança direta."
4. Responda de forma clara, objetiva e padronizada, seguindo o tom de um manual de conduta.
5. Priorize sempre a informação mais recente disponível no contexto."""

class ChatMessage(BaseModel):
    message: str
    user_type: str

@app.post("/chat")
async def chat(payload: ChatMessage):
    query = payload.message
    
    # Busca na base de conhecimento
    docs = vectorstore.similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in docs])
    
    if not context:
        return {"response": "Não sei responder. Procure sua liderança direta."}

    # Usando Hugging Face Hub (Gratuito)
    llm = HuggingFaceHub(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        model_kwargs={"temperature": 0.1, "max_new_tokens": 512},
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
    )
    
    prompt = f"<|system|>\n{SYSTEM_PROMPT}</s>\n<|user|>\nContexto:\n{context}\n\nPergunta: {query}</s>\n<|assistant|>\n"
    
    try:
        response = llm.invoke(prompt)
        # Limpar a resposta para pegar apenas o que a IA gerou após o prompt
        content = response.split("<|assistant|>\n")[-1].strip()
        
        if not content or "não sei" in content.lower() or "desculpe" in content.lower():
             return {"response": "Não sei responder. Procure sua liderança direta."}
             
        return {"response": content}
    except Exception as e:
        return {"response": "Erro ao processar sua pergunta. Por favor, tente novamente mais tarde."}

@app.post("/upload-text")
async def upload_text(text: str = Form(...), user_type: str = Form(...)):
    if user_type != "master":
        raise HTTPException(status_code=403, detail="Apenas usuários Master podem atualizar o conhecimento.")
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    documents = [Document(page_content=chunk) for chunk in chunks]
    
    vectorstore.add_documents(documents)
    vectorstore.persist()
    return {"message": "Conhecimento atualizado com sucesso!"}

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...), user_type: str = Form(...)):
    if user_type != "master":
        raise HTTPException(status_code=403, detail="Apenas usuários Master podem atualizar o conhecimento.")
    
    file_path = f"./{file.filename}"
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
    
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = text_splitter.split_documents(documents)
    
    vectorstore.add_documents(split_docs)
    vectorstore.persist()
    os.remove(file_path)
    
    return {"message": f"PDF {file.filename} processado e adicionado ao conhecimento!"}
