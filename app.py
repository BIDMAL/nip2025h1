import markdown
import yaml 
import psycopg

from markupsafe import Markup
from datetime import datetime
from flask import Flask, request, render_template
from src import db_query_process_tools as qtools
from src import interactor_lm as lm
from src import intercator_bge as bge

def load_config(config_file):
    config = None
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

app = Flask(__name__)
table_name = 'clear_docs_v0'
config = load_config('config.yaml')
connection = psycopg.connect(**config['db_params'])
cursor = connection.cursor()
bge_interacrtor = bge.BGEInteractor(url='http://0.0.0.0:8004')
lm_interactor = lm.LMInteractor(url='http://0.0.0.0:8005')
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

        nearest_elem = qtools.get_topk_elems(
            prompt=prompt, 
            cursor=cursor, 
            connection=connection, 
            table_name=table_name, 
            bge_interactor=bge_interacrtor,
            k=1
        )
        
        context = nearest_elem[0][3] if nearest_elem else ''
        _, response = lm_interactor.generate(prompt, context)

        chat_history.append({
            'role': 'bot',
            'content': Markup(markdown.markdown(response)),
            'timestamp': datetime.now().strftime('%H:%M:%S'),
            'context': context
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