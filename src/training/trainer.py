import os
from sched import scheduler
import torch
from src.training.loss import DPRLoss
from src.data_utils.corpus import Corpus
from src.retrievers.models.encoder import TransformerEncoder
from tqdm import tqdm
import random

class Trainer:
    def __init__(self, device: torch.device, cfg=None, temperature = 0.05):
        self.criterion = DPRLoss(temperature=temperature)
        self.device = device
        self.cfg = cfg
        
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

        if hasattr(corpus, "ds") and "train" in corpus.ds:
            train_len = len(corpus.ds["train"])
            total_batches = (train_len + batch_size - 1) // batch_size
        else:
            total_batches = sum(1 for _ in corpus.get_batch(batch_size, split="train", shuffle=False))

        total_steps = total_batches * epochs
        
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=learning_rate,
            total_steps=total_steps,
            pct_start=0.1,
        )

        query_encoder.to(self.device).train()
        document_encoder.to(self.device).train()

        max_seq_length = query_encoder.max_seq_length
        total_batches = (corpus.get_length("train") + batch_size - 1) // batch_size if corpus.get_length("train") else None

        for epoch in range(epochs):
  
            total_loss = 0.0
            total_samples = 0

            pbar = tqdm(
                corpus.get_batch(batch_size, split="train", shuffle=True),
                total=total_batches,
                desc=f"Epoch {epoch:02d}/{epochs:02d}",
                unit="batch",
            )

            for batch in pbar:
                
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
                scheduler.step()

                b_len = query_embeddings.size(0)
                total_loss += loss.item() * b_len
                total_samples += b_len

                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    avg_loss=f"{total_loss / total_samples:.4f}",
                )

        query_encoder.eval()
        document_encoder.eval()  
        
    
        os.makedirs(os.path.dirname(query_encoder_weights_path), exist_ok=True)
        os.makedirs(os.path.dirname(document_encoder_weights_path), exist_ok=True)
        torch.save(query_encoder.state_dict(), query_encoder_weights_path)
        torch.save(document_encoder.state_dict(), document_encoder_weights_path)

        