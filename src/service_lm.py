from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import uvicorn

model_name = "Qwen/Qwen3-1.7B"
app = FastAPI()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

model = model.to(device)

class GenerateArgs(BaseModel):
    prompt: str
    context: str

@app.post('/get_history')
def get_history(return_responses=False):
    ''' Returns prompts histore and optional LM response to them'''

@app.post('/generate')
def generate(args: GenerateArgs):
    '''Request model to generate answer to prompts\n
    Returns thinking_content and content'''
    msgs = [
        {'role': 'system', 'content': args.context},
        {'role': 'user', 'content': args.prompt}
    ]

    text = tokenizer.apply_chat_template(
        msgs,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device) 
    
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=32768
    )

    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
    try:
        thinking_id = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        thinking_id = 0

    thinking = tokenizer.decode(output_ids[:thinking_id], skip_special_tokens=True).strip("\n") 
    response = tokenizer.decode(output_ids[thinking_id:], skip_special_tokens=True).strip("\n") 

    return {'thinking': thinking, 'response': response} 

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8005)
