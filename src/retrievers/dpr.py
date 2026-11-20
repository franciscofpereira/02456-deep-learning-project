from src.data_utils.document import Document
from src.retrievers.base_retriever import Retriever
from src.retrievers.models.encoder import TransformerEncoder, PretrainedEncoder, HandTransformerEncoder
from transformers import AutoTokenizer
import torch
import torch.nn as nn
import os


class DPR(Retriever):
    def __init__(self, cfg, documents: list[dict] = None, index_path: str = None) -> None:
        #Modified from tf_idf.py: allows for use of Retriever functions so they don't need to be rewritten here
        super(DPR, self).__init__(documents, index_path) 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cfg = cfg
        tokenizer_name = getattr(self.cfg.model, "pretrained_model_name", None) or "bert-base-uncased"
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=True)
        self.__load_models()

        self.doc_embeddings = None
        if index_path is not None and os.path.exists(index_path):
            self.doc_embeddings = torch.load(index_path, map_location="cpu", weights_only=True)
        
    def __load_models(self):
        # n_heads, n_layers, emb_in, hidden_dim, feed_forward_dim
        pretrained_name = getattr(self.cfg.model, "pretrained_model_name", None)

        if pretrained_name is not None:
            
            self.query_encoder = PretrainedEncoder(
                model_name=pretrained_name,
                pooling_type=self.cfg.model.pooling_type,
                emb_dim=self.cfg.model.emb_dim,
                max_seq_length=self.cfg.max_seq_length
            )

        else:
            # encoder_type: "hand" -> HandTransformerEncoder, anything else -> TransformerEncoder (default)
            encoder_type = self.cfg.model.get("encoder_type", "transformer")
            encoder_class = HandTransformerEncoder if encoder_type == "hand" else TransformerEncoder

            # Encode queries and documents and store them and move them to device
            self.query_encoder = encoder_class(
                self.cfg.model.n_layers,
                self.cfg.model.n_heads,
                self.cfg.model.emb_dim,
                self.cfg.model.hidden_dim, 
                self.cfg.model.feed_forward_dim, 
                self.cfg.model.pooling_type, 
                self.tokenizer.vocab_size, 
                self.cfg.max_seq_length
                )

        self.document_encoder = self.query_encoder

        """ self.document_encoder = TransformerEncoder(
            self.cfg.model.n_layers,
            self.cfg.model.n_heads,
            self.cfg.model.emb_dim,
            self.cfg.model.hidden_dim,
            self.cfg.model.feed_forward_dim,
            self.cfg.model.pooling_type,
            self.tokenizer.vocab_size,
            self.cfg.max_seq_length
            token_embedding=self.query_encoder.token_embedding,
            pos_embedding=self.query_encoder.pos_embedding
            ) """

        self.query_encoder.eval()
        self.document_encoder.eval()

        self.query_encoder.to(self.device)
        self.document_encoder.to(self.device)
 
        query_weights_path = self.cfg.model.get('query_encoder_weights_path', None)
        if self.cfg.model.get('query_encoder_weights_path', None) and os.path.exists(query_weights_path):
            #Instead of loading weight path, loading weights themselves
            query_weights = torch.load(query_weights_path)
            self.query_encoder.load_state_dict(query_weights)

        document_weights_path = self.cfg.model.get('document_encoder_weights_path', None)
        if self.cfg.model.get('document_encoder_weights_path', None) and os.path.exists(document_weights_path):
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

        use_amp = str(self.device).startswith("cuda")

        all_embeddings = []

        print(f'len texts = {len(texts)}')

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]

                tokenized_batch = self.Tokenize(batch_texts)
                input_ids = tokenized_batch["input_ids"].to(self.device)
                attention_mask = tokenized_batch["attention_mask"].to(self.device)

                with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    batch_embeddings = self.document_encoder(input_ids, attention_mask)

                all_embeddings.append(batch_embeddings.cpu())

        # 4. Combine all batches -> (len(texts), emb_dim)
        return torch.cat(all_embeddings, dim=0)

        
    # Encode Queries
    def EncodeQuery(self, queries: list[str],  batch_size: int) -> torch.Tensor:
        if not queries:
            return torch.empty((0, self.cfg.model.emb_dim))

        self.query_encoder.eval()
        all_embeddings = []

        #print(f"Query Tokens: {self.tokenizer.tokenize(queries[0])}")
        #print(f"Query IDs: {self.tokenizer(queries[0])['input_ids']}")

        with torch.no_grad():
            for i in range(0, len(queries), batch_size):
                batch_queries = queries[i : i + batch_size]

                tokenized_batch = self.Tokenize(batch_queries)
                input_ids = tokenized_batch["input_ids"].to(self.device)
                attention_mask = tokenized_batch["attention_mask"].to(self.device)

                # Forward pass through query encoder
                batch_embeddings = self.query_encoder(input_ids, attention_mask)

                # Store on CPU to keep GPU/RAM usage low
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
        print(f'Encoded queries shape: {encoded_queries.shape}')
        print(f'Document embeddings shape: {self.doc_embeddings.shape}')
        scores = encoded_queries @ self.doc_embeddings.T
        
        return scores.cpu().numpy()

    
    # Optional FAISS index (Mentioned in week 6 notebook)

        

