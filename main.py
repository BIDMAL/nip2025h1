import query_process_tools as qtools

table_name = 'clear_docs_v0'

if __name__ == "__main__":
    query = 'How to use PostgreSQL'
    nearest_elems = qtools.get_topk_elems(query, table_name, k=5)
    for elem in nearest_elems:
        print('id: ', elem[0])
        print('title: ', elem[1])
        print('uri: ', elem[2])
        print()
        print('-----------text------------')
        print(elem[3][:200])
        print('---------------------------')
        print()