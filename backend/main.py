import os
import uvicorn
from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import io
try:
    import pypdf
except ImportError:
    pypdf = None
from supabase import create_client, Client

app = FastAPI(title="IA de Apoio Operacional e Assistencial")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configurações do Supabase
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Configuração da IA (Hugging Face)
HF_API_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"
HF_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN")

class ChatRequest(BaseModel):
    message: str
    user_type: str

def get_all_knowledge():
    try:
        response = supabase.table("knowledge").select("content").execute()
        return "\n---\n".join([item['content'] for item in response.data])
    except Exception as e:
        print(f"Erro ao buscar conhecimento: {e}")
        return ""

@app.get("/")
async def read_index():
    return FileResponse("./frontend/index.html")

@app.post("/chat")
async def chat(req: ChatRequest):
    knowledge = get_all_knowledge()
    
    system_prompt = f"""Você é uma assistente virtual de apoio operacional e assistencial para uma equipe de saúde.
Sua personalidade é profissional, educada e prestativa.

DIRETRIZES DE RESPOSTA:
1. SAUDAÇÕES: Se o usuário disser "Oi", "Olá", "Bom dia" ou fizer interações sociais básicas, responda de forma cordial e se coloque à disposição para ajudar com dúvidas sobre os protocolos.
2. DÚVIDAS TÉCNICAS: Para perguntas sobre exames, horários, locais ou procedimentos, use APENAS o contexto fornecido abaixo.
3. REGRA DE OURO: Se a pergunta for sobre um procedimento ou regra que NÃO consta no contexto abaixo, você deve responder EXATAMENTE: "Não sei responder. Procure sua liderança direta."
4. Não invente informações. Seja objetiva.

CONTEXTO CADASTRADO:
{knowledge}"""

    full_prompt = f"<|system|>\n{system_prompt}</s>\n<|user|>\n{req.message}</s>\n<|assistant|>\n"

    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {
        "inputs": full_prompt,
        "parameters": {"max_new_tokens": 500, "temperature": 0.1, "return_full_text": False}
    }
    
    try:
        response = requests.post(HF_API_URL, headers=headers, json=payload)
        result = response.json()
        
        if isinstance(result, list) and len(result) > 0:
            answer = result[0].get("generated_text", "").strip()
        elif isinstance(result, dict) and "generated_text" in result:
            answer = result["generated_text"].strip()
        else:
            answer = "Não sei responder. Procure sua liderança direta."

        return {"response": answer}
    except Exception as e:
        return {"response": "Não sei responder. Procure sua liderança direta."}

@app.post("/upload-text")
async def upload_text(text: str = Form(...), user_type: str = Form(...)):
    if user_type != "master":
        raise HTTPException(status_code=403, detail="Acesso negado.")
    try:
        supabase.table("knowledge").insert({"content": text}).execute()
        return {"message": "Conhecimento atualizado com sucesso!"}
    except Exception as e:
        return {"message": f"Erro ao salvar: {str(e)}"}

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...), user_type: str = Form(...)):
    if user_type != "master":
        raise HTTPException(status_code=403, detail="Acesso negado.")
    try:
        content = await file.read()
        pdf_reader = pypdf.PdfReader(io.BytesIO(content))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        
        if text.strip():
            supabase.table("knowledge").insert({"content": text}).execute()
            return {"message": f"PDF '{file.filename}' processado com sucesso!"}
        return {"message": "PDF sem texto legível."}
    except Exception as e:
        return {"message": f"Erro ao ler PDF: {str(e)}"}

# Servir arquivos estáticos
if os.path.exists("./frontend"):
    app.mount("/static", StaticFiles(directory="./frontend"), name="static")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
