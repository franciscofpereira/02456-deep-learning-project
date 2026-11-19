from src.data_utils.document import Document
import hydra
from src.data_utils.corpus import Corpus
from src.retrievers import tf_idf
from src.retrievers.tf_idf import TFIDF
from src.retrievers.bm25 import BM25
from src.data_utils.query import Query
from src.retrievers.dpr import DPR
from src.retrievers.utils import retrieveQueryAndGetRelevancies
from src.evaluation.metrics import Evaluate
from src.training.trainer import Trainer
import os
import torch

@hydra.main(config_path='config', config_name='fiqa_experiment.yaml', version_base=None)
def main(cfg):
    ### Load a dataset
    corpus = Corpus(dataset_name=cfg.dataset.name, base_dir=f'{cfg.base_path}/retrieval_data')

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device(cfg.device)
  
    def make_queries(split):
      return [Query(q, id=qid, relevant_document_ids=corpus.relevant_ids[qid])
              for q, qid in zip(split['question'], split['query_id'])]

    train_queries = make_queries(corpus.train)
    val_queries = make_queries(corpus.val)
    test_queries = make_queries(corpus.test)
    
    ### Initialize TF-IDF class
    tf_idf = TFIDF(documents=corpus.all_documents)
    
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
    print(f'TF-IDF Test results: {test_scores}')    

    # Initialize BM25 class
    bm25 = BM25(documents=corpus.all_documents)
        
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
    print(f'BM25 Test results: {test_scores}')  

    dpr = DPR(cfg, documents=corpus.all_documents, index_path=None)

    # Skip training automatically if weights are already saved on disk
    if not (
        os.path.exists(cfg.model.query_encoder_weights_path)
        and os.path.exists(cfg.model.document_encoder_weights_path)
    ):

        bm25_retriever = BM25(documents=corpus.all_documents, index_path=None)
        hard_negatives_texts = bm25_retriever.mine_hard_negatives(train_queries, top_k=30, skip_top=5)

        trainer = Trainer(device)

        trainer.train(
                    query_encoder=dpr.query_encoder, 
                    document_encoder=dpr.document_encoder,
                    corpus=corpus, 
                    hard_negatives=hard_negatives_texts,
                    epochs=cfg.epochs, 
                    batch_size=cfg.batch_size, 
                    learning_rate=cfg.learning_rate, 
                    max_seq_length=cfg.max_seq_length, 
                    query_encoder_weights_path=cfg.model.query_encoder_weights_path, 
                    document_encoder_weights_path=cfg.model.document_encoder_weights_path,
                    tokenizer=dpr.tokenizer
                )

        document_embeddings = dpr.BuildDocumentEmbeddings(batch_size=cfg.batch_size, save_path=cfg.document_embeddings_path)

        dpr_train_relevancies = retrieveQueryAndGetRelevancies(dpr, train_queries, k=cfg.top_k)
        dpr_val_relevancies = retrieveQueryAndGetRelevancies(dpr, val_queries, k=cfg.top_k)
        dpr_test_relevancies = retrieveQueryAndGetRelevancies(dpr, test_queries, k=cfg.top_k)

        train_scores = Evaluate(dpr_train_relevancies, train_queries)
        val_scores = Evaluate(dpr_val_relevancies, val_queries)
        test_scores = Evaluate(dpr_test_relevancies, test_queries)

        print(f"DPR Train results: {train_scores}")
        print(f"DPR Val results: {val_scores}")
        print(f"DPR Test results: {test_scores}")
    
if __name__ == '__main__':
    main()