from src.retrievers.tf_idf import TFIDF
import numpy as np
import math
from collections import Counter
from scipy.sparse import csr_matrix, diags  

class BM25(TFIDF):
    def __init__(self, documents: list[dict] = None, index_path: str = None, k1: float = 1.5, b: float = 0.75) -> None:
        super(BM25, self).__init__(documents, index_path)
        self.k1 = k1
        self.b = b
        self.doc_lengths, self.avg_doc_length = self.GetDocumentLengths()

        # IDF for BM25
        self.idf_bm25 = self.GetInverseDocumentFrequenciesBM25()

        if type(self) == BM25:
            self.bm25_matrix = self.GetDocumentsBM25Vectors()

    def GetDocumentLengths(self):
        # document_lengths = {}
        # average_document_length = 0
        # for document in self.index.GetDocuments():
        #     length = len(self.PreprocessText(document.GetText()))
        #     document_lengths[document] = length
        #     average_document_length += length
        # return document_lengths, average_document_length/len(self.index.GetDocuments())

        # Modified due to problem with index access, also for efficiency
        document_lengths = [len(tokens) for tokens in self.doc_tokens]
        average_document_length = sum(document_lengths) / len(document_lengths)
        return document_lengths, average_document_length
    
    def GetInverseDocumentFrequenciesBM25(self) -> dict[str, float]:
            """
            Returns a dict with terms and their corresponding inverse document frequencies. (Modified for BM25)
            """
            n_documents = len(self.index.GetDocuments())
            return {term: math.log((n_documents +1) / (df + 0.5)) for term, df in self.GetDocumentFrequencies().items()}

    def GetDocumentsBM25Vectors(self):
            """
            Returns a n_documents x n_terms matrix whose element (i, j) is the bm25 score of the j'th term in the i'th document.
            """
            self.term_to_index = {term: idx for idx, term in enumerate(self.corpus_vocabulary)}
    
            n_documents = len(self.doc_tokens)
            n_terms = len(self.corpus_vocabulary)
            rows, cols, vals = [], [], []
    
            # raw term frequency × inverse document frequency
            for doc_idx, tokens in enumerate(self.doc_tokens):
                for term, count in Counter(tokens).items():
                    rows.append(doc_idx)
                    cols.append(self.term_to_index[term])
                    # Modified TF-IDF formula to BM25 formula
                    vals.append(((count * (self.k1 + 1))/(count+self.k1*(1-self.b+self.b*(self.doc_lengths[doc_idx]/self.avg_doc_length)))) * self.idf_bm25[term])
    
            bm25_matrix = csr_matrix((vals, (rows, cols)), shape=(n_documents, n_terms), dtype=np.float32)
            norms = np.sqrt(bm25_matrix.multiply(bm25_matrix).sum(axis=1)).A1
            norms[norms == 0] = 1  # avoid dividing by zero for empty documents
            return (diags(1 / norms) @ bm25_matrix).tocsr()  # divides documents by their own length, normalizing bm25 vectors
           
    
    def CalculateScores(self, queries: list[str]):
        """Calculate scores for a query
        
        Args:
            query (str): The query to calculate scores for
        
        Returns:
            np.array: An array of scores for each document
        """
        query_vector = self.QueryToVector(queries)
        scores = query_vector.dot(self.bm25_matrix.T).toarray()
        return scores