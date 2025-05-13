from src import query_process_tools as qtools
from src import interactor_lm as lm

table_name = 'clear_docs_v0'
lm_interactor = lm.LMInteractor(url='http://0.0.0.0:8005')

if __name__ == "__main__":
    query = 'How to use PostgreSQL'
    nearest_elems = qtools.get_topk_elems(query, table_name, k=1)
    print('rag')
    print()
    for elem in nearest_elems:
        print('id: ', elem[0])
        print('title: ', elem[1])
        print('uri: ', elem[2])
        print()
        print('-----------text------------')
        print(elem[3][:200])
        print('---------------------------')
        print()

    thinkings, responses = lm_interactor.generate(query, nearest_elems[0][3])
    
    print('generation')
    print()
    print('-----------thinking-----------')
    print(thinkings)
    print('-----------answer------------')
    print(responses)
    print()


