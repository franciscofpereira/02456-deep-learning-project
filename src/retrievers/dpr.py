from src.data_utils.document import Document
from src.retrievers.base_retriever import Retriever
from src.retrievers.models.encoder import TransformerEncoder
import torch


class DPR(Retriever):
    def __init__(self, cfg, documents: list[dict] = None, index_path: str = None) -> None:
        #Modified from tf_idf.py: allows for use of Retriever functions so they don't need to be rewritten here
        super(DPR, self).__init__(documents, index_path) 
        self.cfg = cfg
        self.__load_models()

        
        self.doc_embeddings = self.BuildDocumentEmbeddings(...)
    
    def __load_models(self):
        # n_heads, n_layers, emb_in, hidden_dim, feed_forward_dim
        
        # Encode queries and documents and store them
        self.query_encoder = TransformerEncoder(self.cfg.model.n_layers, self.cfg.model.n_heads, self.cfg.model.emb_dim, self.cfg.model.hidden_dim, self.cfg.model.feed_forward_dim)
        self.document_encoder = TransformerEncoder(self.cfg.model.n_layers, self.cfg.model.n_heads, self.cfg.model.emb_dim, self.cfg.model.hidden_dim, self.cfg.model.feed_forward_dim)
        
        query_weights_path = self.cfg.model.get('query_encoder_weights_path', None)
        if self.cfg.model.get('query_encoder_weights_path', None):
            #Instead of loading weight path, loading weights themselves
            query_weights = torch.load(query_weights_path)
            self.query_encoder.load_state_dict(query_weights)

        document_weights_path = self.cfg.model.get('document_encoder_weights_path', None)
        if self.cfg.model.get('document_encoder_weights_path', None):
            #Instead of loading weight path, loading weights themselves
            document_weights = torch.load(document_weights_path)
            self.document_encoder.load_state_dict(document_weights)
    
    # Overhead of process : Tokenize queries/documents -> Encode queries/documents -> Build embeddings -> Calculate scores

    # Tokenize the queries and documents: Turn strings into tensors the encoder can read
    def Tokenize(self, texts: list[str]):
        raise NotImplementedError()

    # Encode Passages
    def Encode(self, texts: list[str]):
        raise NotImplementedError()

    # Encode Queries
    def EncodeQuery(self, queries: list[str]):
        raise NotImplementedError()

    # Build embeddings
    def BuildDocumentEmbeddings(self):
        raise NotImplementedError()


    def CalculateScores(self, queries: list[str]):
        #code foundation copied from tf_idf.py, needs to be modified for DPR
        """Calculate scores for a query
        
        Args:
            queries (list[str]): The queries to calculate scores for
        
        Returns:
            sim(q, d) = E_Q(q) (encoded queries) · E_D(d) (encoded documents)
            Score for relevance 

            Ref: https://www.geeksforgeeks.org/nlp/dense-passage-retrieval-dpr/
        
        """
        encoded_queries = self.EncodeQuery(queries)
        scores = encoded_queries @ self.doc_embeddings.T
        
        return scores.cpu().numpy()

    
    # Optional FAISS index (Mentioned in week 6 notebook)

        

