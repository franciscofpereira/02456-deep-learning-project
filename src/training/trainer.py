import os
import torch
from src.training.loss import DPRLoss
from src.data_utils.corpus import Corpus
from src.retrievers.models.encoder import TransformerEncoder
import random

class Trainer:
    def __init__(self, device: torch.device):
        self.criterion = DPRLoss()
        self.device = device
        
    def train(
            self,
            query_encoder: TransformerEncoder,
            document_encoder: TransformerEncoder,
            corpus: Corpus,
            tokenizer,
            epochs: int = 10,
            batch_size: int = 32,
            learning_rate: float = 1e-4,
            max_seq_length: int = 256,
            query_encoder_weights_path: str = "retrieval_data/query_encoder_weights.pt",
            document_encoder_weights_path: str = "retrieval_data/document_encoder_weights.pt",       
    ):

        params = list( dict.fromkeys(list(query_encoder.parameters()) + list(document_encoder.parameters())) )
        optimizer = torch.optim.AdamW(params, lr=learning_rate, weight_decay=0.01)

        query_encoder.to(self.device).train()
        document_encoder.to(self.device).train()

        max_seq_length = query_encoder.max_seq_length

        for epoch in range(epochs):
  
            total_loss = 0.0
            total_samples = 0

            for batch in corpus.get_batch(batch_size, split='train', shuffle=True):
                
                optimizer.zero_grad()

                # Tokenize queries
                query_tokenized = tokenizer(
                    batch['question'],
                    padding=True, 
                    truncation=True, 
                    max_length=max_seq_length, 
                    return_tensors='pt'
                    ).to(self.device)

                positive_texts = [random.choice(docs) if docs else "" for docs in batch['text']]

                # Tokenize documents
                document_tokenized = tokenizer(
                    positive_texts, 
                    padding=True, 
                    truncation=True, 
                    max_length=max_seq_length, 
                    return_tensors='pt'
                    ).to(self.device)

                # Run forward pass through query
                query_embeddings = query_encoder(
                    query_tokenized['input_ids'], query_tokenized['attention_mask']
                    )

                # Run forward pass through document
                positive_document_embeddings = document_encoder(
                    document_tokenized['input_ids'], document_tokenized['attention_mask']
                    )

                # Compute loss, propagate gradients, and update model parameters
                loss = self.criterion(query_embeddings, positive_document_embeddings)
                loss.backward()
                optimizer.step()

                b_len = query_embeddings.size(0)
                total_loss += loss.item() * b_len
                total_samples += b_len

            print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {total_loss / total_samples:.4f}")
        
    
        os.makedirs(os.path.dirname(query_encoder_weights_path), exist_ok=True)
        os.makedirs(os.path.dirname(document_encoder_weights_path), exist_ok=True)
        torch.save(query_encoder.state_dict(), query_encoder_weights_path)
        torch.save(document_encoder.state_dict(), document_encoder_weights_path)

        query_encoder.eval()
        document_encoder.eval()