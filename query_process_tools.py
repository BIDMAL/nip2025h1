import psycopg
import yaml

from service_bge import BGEInteractor

def load_config(config_file):
    config = None
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config

config = load_config('config.yaml')
connection = psycopg.connect(**config["db_params"])
cursor = connection.cursor() # object to communicate with databse
bge_interacrtor = BGEInteractor(url='http://0.0.0.0:8004')

def get_topk_elems(query, table_name, k=20):
    '''Get db objects from table which are 20 nearest in embedding space entities to query\n
    Return object = (doc_id, uri, title, text)'''
    
    topk_elems = []
    _, query_embs = bge_interacrtor.fetch_embeddings([query])
    emb_str = '[' + ','.join(map(str, query_embs[0]['dense'])) + ']'

    try:
        cursor.execute(
                f"""SELECT doc_id, uri, title, text
                    FROM "{table_name}" 
                    ORDER BY dense <=> %s::vector
                    LIMIT %s""",
            (emb_str, k)
            )
        topk_elems = cursor.fetchall()
    except Exception as e:
        print(e)
        connection.rollback()
        raise e
    finally:
        cursor.close()
        connection.close()

    return topk_elems

