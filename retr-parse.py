import ast
import ebooklib
import httpx
import numpy as np
import psycopg
import re
import yaml

from bs4 import BeautifulSoup
from copy import copy
from datetime import datetime
from ebooklib import epub
from langchain_text_splitters import RecursiveCharacterTextSplitter
from multiprocessing import Pool
from sklearn.cluster import KMeans
from tqdm import tqdm

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

def load_config(config_file):
    config = None

    with open(config_file, "r") as file:
        config = yaml.safe_load(file)

    return config

def prepare_tables(config):
    tables = []
    connection = psycopg.connect(**config["db_params"])
    cursor = connection.cursor()

    sql = "CREATE EXTENSION IF NOT EXISTS vectors;"
    cursor.execute(sql)
    connection.commit()

    # "clear_docs" table
    cd_table = f"clear_docs{config['version']}"
    cursor.execute(f"""DROP TABLE IF EXISTS "{cd_table}" CASCADE;""")
    sql = f"""
        CREATE TABLE IF NOT EXISTS "{cd_table}" (
            "doc_id" SERIAL NOT NULL,
            "uri" TEXT NOT NULL,
            "title" TEXT NULL DEFAULT NULL,
            "text" TEXT NULL DEFAULT NULL,
            "dense" vector(1024) NULL DEFAULT NULL,
            PRIMARY KEY ("uri")
        );
        """
    cursor.execute(sql)
    connection.commit()
    tables.append(cd_table)

    # add vector index
    cursor.execute(f"DROP INDEX IF EXISTS hnsw_index_{cd_table}")
    cursor.execute(f"""
                CREATE INDEX hnsw_index_{cd_table} ON {cd_table} 
                USING vectors (dense vector_cos_ops) 
                WITH (options = \"[indexing.hnsw]\");
            """)
    connection.commit()

    return tables

def clear_text(text):
    clean_text = re.sub(r"\n{3,}", "\n\n", text) # "\n\n\n..."" -> "\n\n"
    clean_text = re.sub("(?<!\n)\n(?!\n)", " ", clean_text) # \n -> " "
    clean_text = re.sub(" +", " ", clean_text) # " ..." -> " "
    clean_text = clean_text.replace("\xa0", " ")
    return clean_text.strip()

base_url = 'https://postgrespro.ru/docs/postgrespro/17/'
bge_interacrtor = BGEInteractor(url='http://0.0.0.0:8004')

def rec_sects_processing(cur_sect, deep=1, prefix=''):
    clear_docs = []
    for sect in cur_sect.find_all('div', class_=f'sect{deep}'):
        id = sect.find('a')['id']

        clear_docs += rec_sects_processing(sect, deep + 1, prefix + ('#' if deep > 1 else '') + (id.upper() if deep > 1 else id))

        title = sect.find(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']).get_text()
        text = sect.get_text()
        text = clear_text(text)

        _, emb = bge_interacrtor.fetch_embeddings([text])
        clear_docs.append({
                    'doc_id': None,
                    'uri': base_url + prefix + ('#' if deep > 1 else '') + (id.upper() if deep > 1 else id),
                    'title': title,
                    'text': text,
                    'dense': emb[0]['dense']
                })
    
    return clear_docs

def get_clear_docs(config):
    clear_docs = []

    file = epub.read_epub('./data/17.4-ru.epub')
    
    # text_splitter = RecursiveCharacterTextSplitter(
    #     chunk_size=config.get('chunk_size', 1000),
    #     chunk_overlap=config.get('chunk_overlap', 200),
    #     length_function=len
    # )
    
    items = list(file.get_items())
    for item in tqdm(items, desc='Process data and embedding texts'):
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            soup = BeautifulSoup(item.get_content(), 'html.parser')
            clear_docs += rec_sects_processing(soup)
    
    return clear_docs

def fill_clear_docs(docs, doc_table, config):
    connection = psycopg.connect(**config["db_params"])
    cursor = connection.cursor()
    
    try:
        for i, doc in tqdm(enumerate(docs), desc="Inserting documents"):
            dense_str = '[' + ','.join(map(str, doc['dense'])) + ']'
            args = (i, doc['uri'], doc['title'], doc['text'], dense_str)
            cursor.execute(
                f"""INSERT INTO "{doc_table}" 
                    (doc_id, uri, title, text, dense) 
                    VALUES (%s, %s, %s, %s, %s)""",
                args
            )

            connection.commit()
    except Exception as e:
        connection.rollback()
        raise e
    finally:
        cursor.close()
        connection.close()

if __name__ == "__main__":
    config = load_config('config.yaml')
    tables = prepare_tables(config)
    clear_docs = get_clear_docs(config)
    # import pickle
    # with open('clear_docs.pkl', 'wb') as f:
    #     pickle.dump(clear_docs, f)
    # with open('clear_docs.pkl', 'rb') as f:
    #     clear_docs = pickle.load(f)
    fill_clear_docs(clear_docs, tables[0], config)