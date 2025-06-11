from src import interactor_bge as bge

def get_topk_elems(prompt, table_name, cursor, connection, bge_interactor, k=20):
    '''Get db objects from table which are 20 nearest in embedding space entities to query\n
    Return object = (doc_id, uri, title, text)'''
    
    topk_elems = []
    _, query_embs = bge_interactor.fetch_embeddings([prompt])
    emb_str = '[' + ','.join(map(str, query_embs[0]['dense'])) + ']'

    cursor.execute(
            f'''SELECT doc_id, uri, title, text
                FROM {table_name} 
                ORDER BY dense <=> %s::vector
                LIMIT %s''',
        (emb_str, k)
        )
    topk_elems = cursor.fetchall()

    return topk_elems

