from src.data_utils.document import Document
from src.retrievers.base_retriever import Retriever
from src.retrievers.models.encoder import TransformerEncoder
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
import os


class DPR(Retriever):
    def __init__(self, cfg, documents: list[dict] = None, index_path: str = None) -> None:
        #Modified from tf_idf.py: allows for use of Retriever functions so they don't need to be rewritten here
        super(DPR, self).__init__(documents, index_path) 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", local_files_only=True)
        self.__load_models()
        
        # token embedding table
        self.token_embedding = nn.Embedding(self.tokenizer.vocab_size, self.cfg.model.emb_dim)

        # positional embedding table
        self.pos_embedding = nn.Embedding(self.cfg.max_seq_length, self.cfg.model.emb_dim)
    
        #self.doc_embeddings = self.BuildDocumentEmbeddings(...)
    
    def __load_models(self):
        # n_heads, n_layers, emb_in, hidden_dim, feed_forward_dim

        # Encode queries and documents and store them
        self.query_encoder = TransformerEncoder(self.cfg.model.n_layers, self.cfg.model.n_heads, self.cfg.model.emb_dim, self.cfg.model.hidden_dim, self.cfg.model.feed_forward_dim, self.cfg.model.pooling_type)
        self.document_encoder = TransformerEncoder(self.cfg.model.n_layers, self.cfg.model.n_heads, self.cfg.model.emb_dim, self.cfg.model.hidden_dim, self.cfg.model.feed_forward_dim, self.cfg.model.pooling_type)
        
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

        return self.tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=self.cfg.model.get("max_seq_len", 256),
        return_tensors="pt",
    )
        

    # Encode Passages
    def Encode(self, texts: list[str], batch_size: int):

        self.document_encoder.eval()

        all_embeddings = []

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]

                tokenized_batch = self.Tokenize(batch_texts)
                input_ids = tokenized_batch["input_ids"].to(self.device)
                attention_mask = tokenized_batch["attention_mask"].to(self.device)

                # 1. Forward pass through token and positional embeddings -> (batch_size, max_seq_length, emb_dim)
                token_embeddings = self.token_embedding(input_ids)
                pos_embeddings = self.pos_embedding(torch.arange(input_ids.size(1)).to(self.device))
                x = token_embeddings + pos_embeddings
                
                # 2. Forward pass through document encoder -> (batch_size, emb_dim)
                batch_embeddings = self.document_encoder(x, attention_mask)

                # 3. Store on CPU to keep GPU/RAM usage low
                all_embeddings.append(batch_embeddings.cpu())

        # 4. Combine all batches -> (len(texts), emb_dim)
        return torch.cat(all_embeddings, dim=0)

        
    # Encode Queries
    def EncodeQuery(self, queries: list[str],  batch_size: int) -> torch.Tensor:
        if not queries:
            return torch.empty((0, self.cfg.model.emb_dim))

        self.query_encoder.eval()
        all_embeddings = []

        with torch.no_grad():
            for i in range(0, len(queries), batch_size):
                batch_queries = queries[i : i + batch_size]

                tokenized_batch = self.Tokenize(batch_queries)
                input_ids = tokenized_batch["input_ids"].to(self.device)
                attention_mask = tokenized_batch["attention_mask"].to(self.device)

                # 1. Token + positional embeddings -> (batch_size, seq_len, emb_dim)
                token_embeddings = self.token_embedding(input_ids)
                pos_embeddings = self.pos_embedding(torch.arange(input_ids.size(1), device=self.device))
                x = token_embeddings + pos_embeddings

                # 2. Forward pass through query encoder -> (batch_size, emb_dim)
                batch_embeddings = self.query_encoder(x, attention_mask)

                # 3. Store on CPU to keep GPU/RAM usage low
                all_embeddings.append(batch_embeddings.cpu())

        # 4. Combine all batches -> (len(queries), emb_dim)
        return torch.cat(all_embeddings, dim=0)
    
    # Build embeddings
    def BuildDocumentEmbeddings(self, batch_size: int, save_path: str = "retrieval_data/doc_embeddings.pt"):

        if os.path.exists(save_path):
            self.doc_embeddings = torch.load(save_path, map_location="cpu", weights_only=True)
            return self.doc_embeddings

        texts = [doc.text for doc in self.index.GetDocuments()]
        self.doc_embeddings = self.Encode(texts, batch_size)

        # Save document embeddings to file
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(self.doc_embeddings, save_path)

        return self.doc_embeddings


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
        encoded_queries = self.EncodeQuery(queries, batch_size=self.cfg.batch_size)
        scores = encoded_queries @ self.doc_embeddings.T
        
        return scores.cpu().numpy()

    
    # Optional FAISS index (Mentioned in week 6 notebook)

        

