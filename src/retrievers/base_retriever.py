import os
import pickle
from abc import ABC, abstractmethod
from src.data_utils.dataset import DocumentDataset
import numpy as np

class Retriever(ABC):
    def __init__(self, documents: list[dict] = None, index_path: str = None) -> None:
        if not index_path is None:
            self.index = self.__LoadIndex(index_path)
        else:
            self.index = self.__BuildIndex(documents)
    
    def __BuildIndex(self, documents: list[dict]):
        """
        @param dataset: The dataset for which an index containing embeddings should be built.
        """
        index = DocumentDataset(documents)
        return index
    
    def __LoadIndex(self, index_path: str):
        """
        @param index_path: The path to the pre-computed index.
        """
        file = open(index_path, 'rb')
        index = pickle.load(file)
        file.close()
        return index
    
    def SaveIndex(self, index_path: str):
        """
        @param index_path: The path to save the pre-computed index.
        """
        if not os.path.exists(os.path.dirname(index_path)):
            os.makedirs(os.path.dirname(index_path))
        file = open(index_path, 'wb')
        pickle.dump(self.index, file)
        file.close()

        
    @abstractmethod
    def CalculateScores(self, query: str):
        raise NotImplementedError("Must overwrite")
    
    def Lookup(self, queries: list[str], k: int, chunk_size: int = 4096) -> list:
        """
        @param queries: The input text(s) to which relevant passages should be found.
        @param k: The number of relevant passages to retrieve for each input query.
        @param chunk_size: Queries are scored in chunks, so the (n_queries x n_documents) score
                           matrix never has to fit in memory at once (needed for SQuAD). Same results.
        """
        documents = self.index.GetDocuments()
        k = min(k, len(documents))
        results = []
        for start in range(0, len(queries), chunk_size):
            scores = self.CalculateScores(queries[start:start + chunk_size])
            top_k = np.argpartition(-scores, k - 1, axis=1)[:, :k]
            order = np.argsort(-np.take_along_axis(scores, top_k, axis=1), axis=1, kind='stable')
            top_k = np.take_along_axis(top_k, order, axis=1)
            results.extend([documents[j] for j in row] for row in top_k)
        return results