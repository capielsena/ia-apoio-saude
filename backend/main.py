import os
import requests
import json
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

# Caminho para salvar o conhecimento de forma persistente
DATA_FILE = "/opt/render/project/src/data/knowledge.json"
if not os.path.exists(os.path.dirname(DATA_FILE)):
    os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)

def load_knowledge():
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return []
    return []

def save_knowledge(data):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# Carrega o conhecimento inicial
knowledge_base = load_knowledge()

# Servir arquivos estáticos do frontend
if os.path.exists("./frontend"):
    app.mount("/static", StaticFiles(directory="./frontend"), name="static")

@app.get("/")
async def read_index():
    return FileResponse("./frontend/index.html")

SYSTEM_PROMPT = """Você é uma IA de apoio operacional e assistencial para equipes de saúde.
Sua função é responder perguntas baseando-se EXCLUSIVAMENTE no contexto fornecido abaixo.
Se a informação não estiver no contexto, responda exatamente: "Não sei responder. Procure sua liderança direta."
Responda de forma curta, clara e objetiva."""

class ChatMessage(BaseModel):
    message: str
    user_type: str

@app.post("/chat")
async def chat(payload: ChatMessage):
    query = payload.message
    current_kb = load_knowledge()
    
    # Busca de contexto: pega tudo o que foi cadastrado para garantir que a IA tenha acesso
    # Como são textos curtos de manuais, podemos enviar tudo como contexto para a IA decidir
    context = "\n---\n".join(current_kb)
    
    if not context:
        return {"response": "Não sei responder. Procure sua liderança direta."}

    # Chamada para API da Hugging Face
    api_url = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"
    headers = {"Authorization": f"Bearer {os.getenv('HUGGINGFACEHUB_API_TOKEN')}"}
    
    # Prompt estruturado para forçar o uso do contexto
    full_prompt = f"<|system|>\n{SYSTEM_PROMPT}\n\nCONTEXTO CADASTRADO:\n{context}</s>\n<|user|>\n{query}</s>\n<|assistant|>\n"
    
    try:
        response = requests.post(api_url, headers=headers, json={
            "inputs": full_prompt, 
            "parameters": {"max_new_tokens": 500, "temperature": 0.1, "return_full_text": False}
        }, timeout=15)
        
        result = response.json()
        
        if isinstance(result, list) and len(result) > 0:
            content = result[0].get("generated_text", "").strip()
        elif isinstance(result, dict) and "generated_text" in result:
            content = result["generated_text"].strip()
        else:
            content = "Não sei responder. Procure sua liderança direta."
            
        # Validação de segurança: se a IA começar a inventar ou divagar
        if not content or len(content) < 5:
             return {"response": "Não sei responder. Procure sua liderança direta."}
             
        return {"response": content}
    except Exception as e:
        print(f"Erro na API: {e}")
        return {"response": "Não sei responder. Procure sua liderança direta."}

@app.post("/upload-text")
async def upload_text(text: str = Form(...), user_type: str = Form(...)):
    if user_type != "master":
        raise HTTPException(status_code=403, detail="Apenas usuários Master podem atualizar o conhecimento.")
    
    current_kb = load_knowledge()
    current_kb.append(text)
    save_knowledge(current_kb)
    return {"message": "Conhecimento atualizado com sucesso!"}

@app.post("/upload-pdf")
async def upload_pdf(user_type: str = Form(...)):
    return {"message": "Use o envio de texto direto para maior precisão no plano gratuito."}
