from turtle import bye
import psycopg
import yaml
from src import db_query_process_tools as qtools
from src import interactor_lm as lm


def load_config(config_file):
    config = None
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config

table_name = 'clear_docs_v0'
config = load_config('config.yaml')
connection = psycopg.connect(**config["db_params"])
cursor = connection.cursor() # object to communicate with databse
bge_interacrtor = bye.BGEInteractor(url='http://0.0.0.0:8004')
lm_interactor = lm.LMInteractor(url='http://0.0.0.0:8005')

if __name__ == "__main__":
    prompt = 'How to use PosgreSQL'
    nearest_elems = qtools.get_topk_elems(
        prompt,
        table_name,
        cursor, 
        connection, 
        bge_interacrtor, 
        k=5
    )

    def test_rag():
        print('rag')
        print()
        print()
        for elem in nearest_elems:
            print('id: ', elem[0])
            print('title: ', elem[1])
            print('uri: ', elem[2])
            print()
            print('-----------text------------')
            print(elem[3][:200])
            print('---------------------------')
            print(len(elem[3]))
            print('---------------------------')
            print()
        print()
        print()
        print()
        print()
            

    def test_lm():
        context = nearest_elems[0][3]
        thinking, response = lm_interactor.generate(prompt, context)
        
        print('generation')
        print()
        print()
        print('-----------thinking-----------')
        print(thinking[:600] + '...cut off')
        print('-----------answer------------')
        print(response[:600] + '...cut off')
        print()
        print()
        print()
        print()

    test_rag()
    test_lm()



