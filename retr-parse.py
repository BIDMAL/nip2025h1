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
            "doc_id" INTEGER NOT NULL,
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

def get_clear_docs(config):
    clear_docs = []
    # TODO: parse document
    return clear_docs

def fill_clear_docs(docs, doc_table, config):
    # TODO: fill parsed chunks
    pass

if __name__ == "__main__":
    config = load_config('config.yaml')
    tables = prepare_tables(config)
    clear_docs = get_clear_docs(config)
    fill_clear_docs(clear_docs, tables[0], config)