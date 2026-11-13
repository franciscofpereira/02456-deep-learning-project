from src.data_utils.document import Document
import hydra
from src.data_utils.corpus import Corpus
from src.retrievers import tf_idf
from src.retrievers import dpr
from src.retrievers.tf_idf import TFIDF
from src.retrievers.bm25 import BM25
from src.retrievers.dpr import DPR
from src.data_utils.query import Query
from src.retrievers.utils import retrieveQueryAndGetRelevancies
from src.evaluation.metrics import Evaluate
from src.training.trainer import Trainer
import numpy as np

import warnings

warnings.filterwarnings(
    "ignore",
    message=".*nested tensors is in prototype stage.*",
    category=UserWarning,
)   

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
    """ tf_idf = TFIDF(documents=corpus.all_documents)
    
    ### Retrieve top-k relevant queries
    tfidf_train_relevancies = retrieveQueryAndGetRelevancies(tf_idf, train_queries, k=cfg.top_k)
    tfidf_val_relevancies = retrieveQueryAndGetRelevancies(tf_idf, val_queries, k=cfg.top_k)
    tfidf_test_relevancies = retrieveQueryAndGetRelevancies(tf_idf, test_queries, k=cfg.top_k)

    ### Evaluate
    train_scores = Evaluate(tfidf_train_relevancies, train_queries)
    val_scores = Evaluate(tfidf_val_relevancies, val_queries)
    test_scores = Evaluate(tfidf_test_relevancies, test_queries)

    print(f'TF-IDF Train results: {train_scores}')
    print(f'TF-IDF Val results: {val_scores}')
    print(f'TF-IDF Test results: {test_scores}') """    

    # Initialize BM25 class
    """ bm25 = BM25(documents=corpus.all_documents)
        
    ### Retrieve top-k relevant queries
    bm25_train_relevancies = retrieveQueryAndGetRelevancies(bm25, train_queries, k=cfg.top_k)
    bm25_val_relevancies = retrieveQueryAndGetRelevancies(bm25, val_queries, k=cfg.top_k)
    bm25_test_relevancies = retrieveQueryAndGetRelevancies(bm25, test_queries, k=cfg.top_k)

    ### Evaluate
    train_scores = Evaluate(bm25_train_relevancies, train_queries)
    val_scores = Evaluate(bm25_val_relevancies, val_queries)
    test_scores = Evaluate(bm25_test_relevancies, test_queries)

    print(f'BM25 Train results: {train_scores}')
    print(f'BM25 Val results: {val_scores}')
    print(f'BM25 Test results: {test_scores}') """

    
    #first document should be a list of dictionaries 
    documents = corpus.all_documents

    first_query = train_queries[0]

    dpr = DPR(cfg, documents=documents, index_path=None)

    print("Embedding documents...")
    document_embeddings = dpr.BuildDocumentEmbeddings(batch_size=cfg.batch_size)

    print("Embedding query...")
    query_embeddings = dpr.EncodeQuery([first_query.text], batch_size=cfg.batch_size)

    top_10 = dpr.Lookup([first_query.text], 10)

    top_10_docs = [top_10[0][i] for i in range(len(top_10[0]))]

    print(f'There are {first_query.getNumberOfRelevantDocuments()} relevant documents for the first query.')
    print(f'Relevant document IDs for the first query: {first_query.relevant_document_ids}')

    print(f'Top 10 document IDs: {[doc._id for doc in top_10_docs]}')

    """ print(f'First query: {first_query.text}')
    print(f'First document: {documents["text"]}')

    print(f'Query embeddings: {query_embeddings}')
    print(f'Query embeddings shape: {query_embeddings.shape}')
    print(f'Document embeddings shape: {document_embeddings.shape}')
    print(f'Document embeddings: {document_embeddings}') """


    
    """ print(f'DPR Train results: {dpr.CalculateScores(train_queries)}')
    print(f'DPR Val results: {dpr.CalculateScores(val_queries)}')
    print(f'DPR Test results: {dpr.CalculateScores(test_queries)}') """
    
if __name__ == '__main__':
    main()