import ast
import ebooklib
import httpx
import numpy as np
import psycopg
import re
import yaml

from bs4 import BeautifulSoup
from ebooklib import epub
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

def rec_sects_processing(src_sect, dist_sects_data, deep=1, prefix=''): 
    ''' Recursive function to retrieve data from section. '''
    
    for sect in src_sect.find_all('div', class_=f'sect{deep}'):
        id = sect.find('a')['id']

        rec_sects_processing(sect, deep + 1, prefix + ('#' if deep > 1 else '') + (id.upper() if deep > 1 else id))

        title = sect.find(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']).get_text()
        text = sect.get_text()
        text = clear_text(text)

        dist_sects_data.append(dict(
            title = title,
            text = text,
            id = id,
            uri = base_url + prefix + ('#' if deep > 1 else '') + (id.upper() if deep > 1 else id)
        )) 

def get_new_corteges(config):
    ''' Gets corteges for database. '''
    file = epub.read_epub('./data/17.4-ru.epub')
   
    db_corteges = [] # array of data extracted from sections + embeddindgs for section's text 
    sects_data = [] # contains [text, title, uri, etc] for each section
    
    # text_splitter = RecursiveCharacterTextSplitter(
    #     chunk_size=config.get('chunk_size', 1000),
    #     chunk_overlap=config.get('chunk_overlap', 200),
    #     length_function=len
    # )
    
    items = list(file.get_items())
    for item in tqdm(items, desc='Process data and embedding texts'):
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            soup = BeautifulSoup(item.get_content(), 'html.parser')
            rec_sects_processing(soup, sects_data)

    if len(sects_data) > 0:
        _, embs = bge_interacrtor.fetch_embeddings([sect_data['text'] for sect_data in sects_data])
        for emb, sect_data in zip(embs, sects_data):
            db_corteges.append({
                'doc_id': None,
                'uri': sect_data['uri'],
                'title': sect_data['title'],
                'text': sect_data['text'],
                'dense': emb['dense']
            }) 

    return db_corteges

def fill_db(db_corteges, table, config):
    connection = psycopg.connect(**config["db_params"])
    cursor = connection.cursor() # object to communicate with databse
    
    try:
        for i, doc in tqdm(enumerate(db_corteges), desc="Inserting documents"):
            dense_str = '[' + ','.join(map(str, doc['dense'])) + ']'
            args = (i, doc['uri'], doc['title'], doc['text'], dense_str)
            cursor.execute(
                f"""INSERT INTO "{table}" 
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
    new_corteges = get_new_corteges(config)
    fill_db(new_corteges, tables[0], config)