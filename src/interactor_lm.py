import httpx

class LMInteractor:
    def __init__(self, url):
        self.url = url
        self.history = []

    def get_history(self):
        ''' Returns prompts histore and optional LM response to them'''
        return self.history

    def generate(self, prompt, context) -> tuple[list[str], list[str]]:
        '''Request model to generate answer to prompts\n
        Returns thinking_content and content'''
        args = {'prompt': prompt, 'context': context}
        self.history.append(args)
        with httpx.Client(timeout=10000) as client:
            response = client.post(f'{self.url}/generate', json=args)
            response = response.json()
            return response['thinking'], response['response']
