import markdown
import yaml 
import psycopg

from markupsafe import Markup
from datetime import datetime
from flask import Flask, request, render_template
from src import db_query_process_tools as qtools
from src import interactor_lm as lm
from src import interactor_bge as bge
from src import interactor_reranker as rerank

def load_config(config_file):
    config = None
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

CONTEXT_CHUNKS_NUM = 3

app = Flask(__name__)
table_name = 'clear_docs_v0'
config = load_config('config.yaml')
connection = psycopg.connect(**config['db_params'])
cursor = connection.cursor()
bge_interacrtor = bge.BGEInteractor(url='http://0.0.0.0:8004')
lm_interactor = lm.LMInteractor(url='http://0.0.0.0:8005')
reranker_interactor = rerank.RerankerInteractor('http://0.0.0.0:8007')
chat_history = []

@app.template_filter('markdown')
def markdown_filter(text):
    import markdown2
    return Markup(markdown2.markdown(text))

@app.route('/', methods=['POST', 'GET'])
def chat():
    if request.method == 'POST':
        prompt = request.form['user_query']
        timestamp = datetime.now().strftime('%H:%M:%S')
        
        chat_history.append({
            'role': 'user',
            'content': prompt,
            'timestamp': timestamp,
            'context': None
        })

        nearest_elems = qtools.get_topk_elems(
            prompt=prompt, 
            cursor=cursor, 
            connection=connection, 
            table_name=table_name, 
            bge_interactor=bge_interacrtor,
            k=20
        )

        contexts = [el[3] for el in nearest_elems]
        top_contexts = reranker_interactor.rerank_contexts(prompt, contexts, CONTEXT_CHUNKS_NUM)
        
        combined_context = ''
        for i, s in enumerate(top_contexts):
            combined_context += f'source{i}: ' + s + '\n'

        _, response = lm_interactor.generate(prompt, combined_context)

        chat_history.append({
            'role': 'bot',
            'content': Markup(markdown.markdown(response)),
            'timestamp': datetime.now().strftime('%H:%M:%S'),
            'context': combined_context
        })
    
    return render_template('web_chat.html', chat_history=chat_history)

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=8006, debug=True)
    except Exception as e:
        print(e)
        connection.rollback()
        raise e
    finally:
        cursor.close()
        connection.close()