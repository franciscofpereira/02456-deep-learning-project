from src.data_utils.document import Document
import hydra
from src.data_utils.corpus import Corpus
from src.retrievers import tf_idf
from src.retrievers.tf_idf import TFIDF
from src.data_utils.query import Query
from src.retrievers.utils import retrieveQueryAndGetRelevancies
from src.evaluation.metrics import Evaluate

@hydra.main(config_path='config', config_name='fiqa_experiment.yaml', version_base=None)
def main(cfg):
    ### Load a dataset
    corpus = Corpus(dataset_name=cfg.dataset.name, base_dir=f'{cfg.base_path}/retrieval_data')
  
    def make_queries(split):
      return [Query(q, id=qid, relevant_document_ids=corpus.relevant_ids[qid])
              for q, qid in zip(split['question'], split['query_id'])]

    train_queries = make_queries(corpus.train)
    val_queries = make_queries(corpus.val)
    test_queries = make_queries(corpus.test)
    
    ### Initialize TF-IDF class
    tf_idf = TFIDF(documents=corpus.all_documents)

    row = corpus.train[0]
    print('Question:', row['question'])
    print('Query ID:', row['query_id'])
    print('Document IDs:', row['document_id'])
    print('First document:', row['text'][0])

    # print number of queries
    #print(f'Number of train queries: {len(train_queries)}')
    #print(f'Number of val queries: {len(val_queries)}')
    #print(f'Number of test queries: {len(test_queries)}')
    #print(f'Number of documents in corpus: {sum(len(ids) for ids in corpus.all_documents["document_id"])}')
    #print(f'Number of relevant documents in corpus: {len(corpus.relevant_ids)}')
    #print(f'Number of terms in corpus vocabulary: {len(tf_idf.corpus_vocabulary)}')
     
    ### Retrieve top-k relevant queries
    tfidf_train_relevancies = retrieveQueryAndGetRelevancies(tf_idf, train_queries, k=cfg.top_k)
    tfidf_val_relevancies = retrieveQueryAndGetRelevancies(tf_idf, val_queries, k=cfg.top_k)
    tfidf_test_relevancies = retrieveQueryAndGetRelevancies(tf_idf, test_queries, k=cfg.top_k)

    ### Evaluate
    train_scores = Evaluate(tfidf_train_relevancies, train_queries)
    val_scores = Evaluate(tfidf_val_relevancies, val_queries)
    test_scores = Evaluate(tfidf_test_relevancies, test_queries)

    print(f'Train results: {train_scores}')
    print(f'Val results: {val_scores}')
    print(f'Test results: {test_scores}')

if __name__ == '__main__':
    main()