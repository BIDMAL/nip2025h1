import httpx

class RerankerInteractor:
    def __init__(self, url):
        self.url = url

    def rerank_contexts(self, prompt, context_candidates, k=3):
        '''Requests BGEReranker to get scores for every candidate
        and returns topk from them'''
        args = {'query': prompt, 'candidates': context_candidates, 'result_elems_num': k}
        with httpx.Client(timeout=10000) as client:
            response = client.post(f'{self.url}/rerank_retrievals', json=args)
            response = response.json()
            return response['top_candidates']
