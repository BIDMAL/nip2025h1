import psycopg
import yaml
import csv
import pandas as pd

from tqdm import tqdm
from src import db_query_process_tools as qtools
from src import interactor_lm as lm
from src import intercator_bge as bge

def load_config(config_file):
    config = None
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config

table_name = 'clear_docs_v0'
config = load_config('config.yaml')
connection = psycopg.connect(**config["db_params"])
cursor = connection.cursor() # object to communicate with databse
bge_interacrtor = bge.BGEInteractor(url='http://0.0.0.0:8004')
lm_interactor = lm.LMInteractor(url='http://0.0.0.0:8005')

def prepare_benchmark_data():
    prompt = "Напиши вопросительный промт, на который бы отвечал следующий текст. Ответь только одним предложением. \n"
    cursor.execute(
            f"""SELECT uri, text
                FROM {table_name}"""
        )
    db_data = cursor.fetchall()

    csv_data = []
    with open('./data/data_dict.csv', mode='w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['uri', 'text', 'prompt1', 'prompt2'])
        writer.writeheader()
        for _, elem in tqdm(zip(range(100), db_data)):
            _, response1 = lm_interactor.generate(prompt + elem[1], '')
            _, response2 = lm_interactor.generate(prompt + elem[1], '')
            csv_data = {'uri': elem[0], 'text': elem[1], 'prompt1': response1, 'prompt2': response2}
            if csv_data is not None:
                writer.writerow(csv_data)

def measure_accuracy():
    '''Measures accuracy for how RAG retrieves similar text for a prompt'''
    df = pd.read_csv("./data/data_dict.csv")

    total_amount = len(df)
    correct_predicts_count = 0.
    for elem in tqdm(df.values):
        uri, _, p1, p2 = elem
        nearest_elems1 = qtools.get_topk_elems(
            prompt=p1,
            bge_interactor=bge_interacrtor,
            connection=connection,
            cursor=cursor,
            table_name=table_name,
            k=3
        )

        nearest_elems2 = qtools.get_topk_elems(
            prompt=p2,
            bge_interactor=bge_interacrtor,
            connection=connection,
            cursor=cursor,
            table_name=table_name,
            k=3
        )

        correct_predicts_count += 0.5 if any(uri == el[1] for el in nearest_elems1) else 0.0
        correct_predicts_count += 0.5 if any(uri == el[1] for el in nearest_elems2) else 0.0
        
    return correct_predicts_count / total_amount

if __name__ == "__main__":
    try:
        # prepare_benchmark_data()
        print('Accuracy: ' + str(measure_accuracy()))
    except Exception as e:
        print(e)
        connection.rollback()
        raise e
    finally:
        cursor.close()
        connection.close()


