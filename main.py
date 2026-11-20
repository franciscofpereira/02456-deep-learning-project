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
from src.evaluation.metrics import Evaluate, save_results
from src.training.trainer import Trainer
import numpy as np
from omegaconf import OmegaConf
from hydra.core.hydra_config import HydraConfig
import torch
import os

import warnings

warnings.filterwarnings(
    "ignore",
    message=".*nested tensors is in prototype stage.*",
    category=UserWarning,
)   

@hydra.main(config_path='config', config_name='fiqa_experiment.yaml', version_base=None)
def main(cfg):

    print(f"Config: \n {OmegaConf.to_yaml(cfg)}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device(cfg.device)

    # Load a dataset
    corpus = Corpus(dataset_name=cfg.dataset.name, base_dir=f'{cfg.base_path}/retrieval_data')

    documents = corpus.all_documents

    def make_queries(split):
      return [Query(q, id=qid, relevant_document_ids=corpus.relevant_ids[qid])
              for q, qid in zip(split['question'], split['query_id'])]

    train_queries = make_queries(corpus.train)
    val_queries = make_queries(corpus.val)
    test_queries = make_queries(corpus.test)

    lengths = [
    len((" ".join(doc["text"]) if isinstance(doc["text"], list) else doc["text"]).split())
    for doc in documents
    ]
    
    print(f"Average document length (words): {np.mean(lengths):.1f}")
    print(f"95th percentile (words): {np.percentile(lengths, 95):.1f}, Max: {np.max(lengths)}")
    print(f"Median document length (words): {np.median(lengths):.1f}, Min: {np.min(lengths)}")

    # Precision@k / Recall@k / MRR@k / nDCG@k for every k in eval_ks
    eval_ks = list(cfg.get("eval_ks", [cfg.top_k]))
    retrieve_k = max(max(eval_ks), cfg.top_k)

    def evaluate_and_report(method_name, retriever):
        """Retrieve top-k for every split, compute the metrics, print and save them."""
        results = {}
        for split, queries in [("train", train_queries), ("val", val_queries), ("test", test_queries)]:
            relevancies = retrieveQueryAndGetRelevancies(retriever, queries, k=retrieve_k)
            results[split] = Evaluate(relevancies, queries, ks=eval_ks)
            print(f"{method_name} {split} results: " + ", ".join(f"{m}={v:.4f}" for m, v in results[split].items()))

        # Name of the run in the comparison table: e.g. "bm25", "dpr_hand_encoder" (override with run_name=...)
        run_name = cfg.get("run_name", None)
        if not run_name:
            run_name = cfg.retriever.method
            if cfg.retriever.method == "dpr":
                run_name += "_" + HydraConfig.get().runtime.choices.get("model", "model")
        path = os.path.join(cfg.get("results_dir", "outputs/results"), f"{run_name}.json")
        save_results(results, path)
        print(f"Saved results to {path}")
        return results

    ### Initialize TF-IDF class

    if cfg.retriever.method == "tf_idf":
        tf_idf = TFIDF(documents=corpus.all_documents)
        evaluate_and_report("TF-IDF", tf_idf)

    # Initialize BM25 class

    elif cfg.retriever.method == "bm25":
        bm25 = BM25(documents=corpus.all_documents)
        evaluate_and_report("BM25", bm25)

    # Initialize DPR class
    
    elif cfg.retriever.method == "dpr":

        dpr = DPR(cfg, documents=documents, index_path=None)

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

        evaluate_and_report("DPR", dpr)


    """ print(f'First query: {first_query.text}')
    print(f'First document: {documents["text"]}')

      #top_10 = dpr.Lookup([first_query.text], 10)
    
        """ 
    
    """
    top_10_docs = [top_10[0][i] for i in range(len(top_10[0]))]
    
    print(f'There are {first_query.getNumberOfRelevantDocuments()} relevant documents for the first query.')
    print(f'Relevant document IDs for the first query: {first_query.relevant_document_ids}')

    print(f'Top 10 document IDs: {[doc._id for doc in top_10_docs]}') """


if __name__ == '__main__':
    main()