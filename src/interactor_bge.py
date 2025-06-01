import httpx

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