import sys
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from FlagEmbedding import FlagReranker

app = FastAPI()
reranker = FlagReranker('BAAI/bge-reranker-large', use_fp16=True) 

class RerankRequest(BaseModel):
    query: str
    candidates: list[str]
    result_elems_num: int

@app.post('/rerank_retrievals')
def rerank_retrievals(request: RerankRequest):
    if len(request.candidates) < request.result_elems_num:
        print("Number of candidates is less than a requested result size!\nAborting request", file=sys.stderr)
        return None
    arg = [(request.query, doc) for doc in request.candidates]
    scores = reranker.compute_score(arg)
    sorted_candidates_with_score = sorted([(score, candidate) for score, candidate in zip(scores, request.candidates)], reverse=True)
    return {'top_candidates': [c[1] for c in sorted_candidates_with_score[:request.result_elems_num]]}

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8007)