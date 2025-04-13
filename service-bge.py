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



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8004)
