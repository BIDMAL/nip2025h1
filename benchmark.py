import psycopg
from tqdm import tqdm
import yaml
import csv

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

if __name__ == "__main__":
    config = load_config('config.yaml')
    prompt = "Напиши вопросительный промт, на который бы отвечал следующий текст. Ответь только одним предложением. \n"

    try:
        cursor.execute(
                f"""SELECT uri, text
                    FROM {table_name}"""
            )
        db_data = cursor.fetchall()
    except Exception as e:
        print(e)
        connection.rollback()
        raise e
    finally:
        cursor.close()
        connection.close()

    csv_data = []
    with open('./data/data_dict.csv', mode='w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['uri', 'text', 'prompt1', 'prompt2'])
        writer.writeheader()
        for i, elem in tqdm(enumerate(db_data)):
            if i % 10 == 0:
                _, response1 = lm_interactor.generate(prompt + elem[1], '')
                _, response2 = lm_interactor.generate(prompt + elem[1], '')
                csv_data = {'uri': elem[0], 'text': elem[1], 'prompt1': response1, 'prompt2': response2}
                print('--------------------')
                print('Generated prompt: ')
                print('.........................')
                print(response1)
                print('.........................')
                print('text')
                print('.........................')
                print(elem[1][:200])
                print('--------------------')
                if csv_data is not None:
                    writer.writerow(csv_data)