from fastapi import FastAPI,Request
from fastapi.concurrency import ContextManager
from fastapi.responses import JSONResponse
from fastapi.requests import Request
from typing import List
import os 
from transformers import AutoTokenizer, AutoModel
import torch
from semantic_router.encoders import HuggingFaceEncoder
import pinecone
from pinecone import Pinecone
import uvicorn
import time
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

#initalize Pinecorn index
pinecone_api =os.environ["pinecorn"]
pc = Pinecone(api_key = pinecone_api)
index_name = "vectormultipdf"

#connect to the index
index = pc.Index(index_name)
time.sleep(1)

# Load Hugging Face model and tokenizer
model_name = "dwzhu/e5-base-4k"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Initialize Groq client
groq_api = os.environ["groq_key"]
groq_client = Groq(api_key= groq_api)

async def get_docs(query : str , top_k : int) -> List[str]:
    # Encode the query using the Hugging Face model and tokenizer
    inputs = tokenizer(query, return_tensors="pt",max_length=512, truncation= True)
    
    with torch.no_grad():
        query_embedding = model(**inputs).last_hidden_state.mean(dim = 1).cpu().numpy()
        
    #search Pinecorn index
    results = index.query(vector= query_embedding.tolist(),top_k=top_k, include_metadata=True)
    
    #extract document texts
    #docs = [match["matadata"].get('content','')for match in results.get("matches",[])]
    docs = [match["metadata"].get('content','') for match in results.get("matches",[])]
    return docs

async def genrate_response(query :str , docs: List[str]) -> str:
    system_message = (
        "You are a helpful assistant that answers questions about AI using the "
        "context provided below.\n\n"
        "CONTEXT:\n"
        "\n---\n".join(docs)
    )
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": query}
    ]
    
    # generate response
    chat_response = groq_client.chat.completions.create(
        model="llama3-70b-8192",
        messages=messages
    )
    return chat_response.choices[0].message.content

@app.get("/query")
async def query_endpoint(query: str, top_k: int = 5):
    docs = await get_docs(query,top_k)
    response = await genrate_response(query,docs)
    return JSONResponse(content={"response":response},media_type="application/json")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", workers=4)
