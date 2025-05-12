import httpx
import uvicorn

from fastapi import FastAPI
from FlagEmbedding import BGEM3FlagModel
from pydantic import BaseModel

app = FastAPI()
model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

class EmbeddingsRequest(BaseModel):
    queries: list[str]

class RerankRequest(BaseModel):
    query: str
    candidates: list[str]

def embedding(
    sentences: list[str],
) -> tuple[list[list[float]], list[list[int]], list[list[float]]]:
    output = model.encode(sentences, return_dense=True, return_sparse=True, return_colbert_vecs=False)
    dense_embedding = output["dense_vecs"].tolist()
    sparse_indices = [list(map(int, list(el.keys()))) for el in output["lexical_weights"]]
    sparse_values = [list(map(float, list(el.values()))) for el in output["lexical_weights"]]
    return dense_embedding, sparse_indices, sparse_values

@app.post("/fetch_embeddings")
async def fetch_embeddings(request: EmbeddingsRequest):
    dense_embeddings, sparse_indices, sparse_values = embedding(request.queries)
    embeddings = [
        {"sparse_val": sparse_val, "sparse_ind": sparse_ind, "dense": dense}
        for dense, sparse_ind, sparse_val in zip(dense_embeddings, sparse_indices, sparse_values)
    ]

    return {"success": True, "model_length": len(model.tokenizer), "data": embeddings}

class BGEInteractor:
    def __init__(self, url):
        self.url = url

    def fetch_embeddings(self, queries):
        body = {"queries": queries}
        with httpx.Client(timeout=10000) as client:
            response = client.post(f"{self.url}/fetch_embeddings", json=body)
            response = response.json()
            return response["model_length"], response["data"]

    async def afetch_embeddings(self, queries):
        body = {"queries": queries}
        async with httpx.AsyncClient(timeout=10000) as client:
            response = await client.post(f"{self.url}/fetch_embeddings", json=body)
            response = response.json()
            return response["model_length"], response["data"]

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8004)
