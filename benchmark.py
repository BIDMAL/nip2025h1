import psycopg
import yaml
import csv
import pandas as pd

from tqdm import tqdm
from src import db_query_process_tools as qtools
from src import interactor_lm as lm
from src import interactor_bge as bge
from src import interactor_reranker as rerank

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
reranker_interactor = rerank.RerankerInteractor('http://0.0.0.0:8007')

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

def measure_accuracy(topk_num_without_reranker=5, topk_num_with_reranker=5):
    '''Measures accuracies for how RAG retrieves similar text for a prompt with and without reranker'''
    df = pd.read_csv("./data/data_dict.csv")

    total_amount = len(df)
    correct_predicts_count = [0.] * (topk_num_with_reranker + 1)
    for elem in tqdm(df.values):
        _, text, p1, p2 = elem
        
        # Getting 20 nearest to prompts text chunks before reanker
        nearest_elems1 = qtools.get_topk_elems(
            prompt=p1,
            bge_interactor=bge_interacrtor,
            connection=connection,
            cursor=cursor,
            table_name=table_name,
            k=10
        )

        nearest_elems2 = qtools.get_topk_elems(
            prompt=p2,
            bge_interactor=bge_interacrtor,
            connection=connection,
            cursor=cursor,
            table_name=table_name,
            k=10
        )

        # Evaluationg accuracy for use with no reranker
        correct_predicts_count[0] += 0.5 if any(text == el[3] for el in nearest_elems1[:topk_num_without_reranker]) else 0.0
        correct_predicts_count[0] += 0.5 if any(text == el[3] for el in nearest_elems2[:topk_num_without_reranker]) else 0.0

        # Evaluationg accuracy for use with reranker
        contexts1 = [el[3] for el in nearest_elems1]
        top_contexts1 = reranker_interactor.rerank_contexts(p1, contexts1, topk_num_with_reranker)

        contexts2 = [el[3] for el in nearest_elems2]
        top_contexts2 = reranker_interactor.rerank_contexts(p2, contexts2, topk_num_with_reranker)

        for i in range(topk_num_with_reranker):
            correct_predicts_count[i + 1] += 0.5 if any(text == el for el in top_contexts1[:i + 1]) else 0.0
            correct_predicts_count[i + 1] += 0.5 if any(text == el for el in top_contexts2[:i + 1]) else 0.0
        
    with open('./benchmark_results.txt', 'w', encoding='utf-8') as f:
        f.write(f'Accuracy without reranker on top{topk_num_without_reranker}: {correct_predicts_count[0] / total_amount:}\n')
        print(f'Accuracy without reranker on top{topk_num_without_reranker}: ' + str(correct_predicts_count[0] / total_amount))
        for i in range(1, topk_num_with_reranker + 1):
            print(f'Accuracy with reranker on top{i}: ' + str(correct_predicts_count[i] / total_amount))
            f.write(f'Accuracy with reranker on top{i}: {correct_predicts_count[i] / total_amount:}\n')

if __name__ == "__main__":
    try:
        # prepare_benchmark_data()
        measure_accuracy()
    except Exception as e:
        print(e)
        connection.rollback()
        raise e
    finally:
        cursor.close()
        connection.close()


