import os
import requests
from fastapi import FastAPI, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
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

# Base de conhecimento simples em memória (para ser leve no Render Free)
# Em um cenário real com muitos dados, usaríamos um banco externo, 
# mas para garantir que o site suba no Render Free agora, vamos usar esta abordagem.
knowledge_base = []

# Servir arquivos estáticos do frontend
if os.path.exists("./frontend"):
    app.mount("/static", StaticFiles(directory="./frontend"), name="static")

@app.get("/")
async def read_index():
    return FileResponse("./frontend/index.html")

SYSTEM_PROMPT = """Você é uma IA de apoio operacional e assistencial para equipes de saúde.
Responda APENAS com base no contexto fornecido.
Se não souber, responda EXATAMENTE: "Não sei responder. Procure sua liderança direta." """

class ChatMessage(BaseModel):
    message: str
    user_type: str

@app.post("/chat")
async def chat(payload: ChatMessage):
    query = payload.message
    
    # Busca simples por palavra-chave no conhecimento cadastrado
    context = "\n".join([item for item in knowledge_base if any(word.lower() in item.lower() for word in query.split())])
    
    if not context:
        # Se não achou nada específico, tenta pegar os últimos 3 cadastros como contexto geral
        context = "\n".join(knowledge_base[-3:])
    
    if not context:
        return {"response": "Não sei responder. Procure sua liderança direta."}

    # Chamada direta para API da Hugging Face (Leve e Gratuita)
    api_url = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"
    headers = {"Authorization": f"Bearer {os.getenv('HUGGINGFACEHUB_API_TOKEN')}"}
    
    prompt = f"<|system|>\n{SYSTEM_PROMPT}\nContexto:\n{context}</s>\n<|user|>\n{query}</s>\n<|assistant|>\n"
    
    try:
        response = requests.post(api_url, headers=headers, json={"inputs": prompt, "parameters": {"max_new_tokens": 250, "temperature": 0.1}})
        result = response.json()
        
        if isinstance(result, list) and len(result) > 0:
            content = result[0].get("generated_text", "").split("<|assistant|>\n")[-1].strip()
        else:
            content = "Não sei responder. Procure sua liderança direta."
            
        if not content or "não sei" in content.lower():
             return {"response": "Não sei responder. Procure sua liderança direta."}
             
        return {"response": content}
    except:
        return {"response": "Não sei responder. Procure sua liderança direta."}

@app.post("/upload-text")
async def upload_text(text: str = Form(...), user_type: str = Form(...)):
    if user_type != "master":
        raise HTTPException(status_code=403, detail="Apenas usuários Master podem atualizar o conhecimento.")
    
    knowledge_base.append(text)
    return {"message": "Conhecimento atualizado com sucesso!"}

@app.post("/upload-pdf")
async def upload_pdf(user_type: str = Form(...)):
    return {"message": "Para manter o sistema leve no plano gratuito, use o envio de texto direto no chat Master."}
