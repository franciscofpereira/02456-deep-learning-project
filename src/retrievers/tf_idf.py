from src.data_utils.document import Document
from src.retrievers.base_retriever import Retriever
import numpy as np
from scipy.sparse import csr_matrix
import re
import math
from collections import Counter

class TFIDF(Retriever):
    def __init__(self, documents: list[dict] = None, index_path: str = None) -> None:
        super(TFIDF, self).__init__(documents, index_path)
        self.doc_tokens = [self.PreprocessText(doc.GetText()) for doc in self.index.GetDocuments()]
        self.corpus_vocabulary = self.GetCorpusVocabulary()
        self.idf = self.GetInverseDocumentFrequencies()

        # Only run if we are in TFIDF class
        if type(self) == TFIDF:
            self.tfidf_matrix = self.GetDocumentsTFIDFVectors()

    def PreprocessText(self, text: str) -> list[str]:
        """
        Remove unwanted charatcters from text and lowercase
        """
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)  # replace non-alphanumerics with space

        # potentially add stopword removal and stemming

        return text.split()  
        
    def GetQueryVocabulary(self, query: str) -> list[str]:
        """
        Returns all unique words in a query
        """
        return list(dict.fromkeys(self.PreprocessText(query)))

    def GetDocumentVocabulary(self, document: Document) -> list[str]:
        """
        Returns all unique words in a document
        """
        return list(dict.fromkeys(self.PreprocessText(document.GetText())))

    def GetCorpusVocabulary(self) -> set[str]:
        """
        Returns all unique words in ALL documents.
        """
        return set().union(*self.doc_tokens)  # union of all document vocabularies

    def GetDocumentFrequencies(self) -> dict[str, int]:
        """
        Returns a dict with terms and counts of how many documents the term occurs in (counted once per document).
        """
        df = Counter()
        for tokens in self.doc_tokens:
            df.update(set(tokens))  
        return dict(df)

    def GetInverseDocumentFrequencies(self) -> dict[str, float]:
        """
        Returns a dict with terms and their corresponding inverse document frequencies.
        """
        n_documents = len(self.index.GetDocuments())
        return {term: math.log(n_documents / df) for term, df in self.GetDocumentFrequencies().items()}
    
    
    def GetDocumentTermCounts(self, document: Document):
        """
        Returns a dict on the form {term: num_occurences} for an input Document.
        """
        return Counter(self.PreprocessText(document.GetText()))

    def GetDocumentsTFIDFVectors(self):
        """
        Returns a n_documents x n_terms matrix whose element (i, j) is the td-idf score of the j'th term in the i'th document.
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
                vals.append(count * self.idf[term])

        return csr_matrix((vals, (rows, cols)), shape=(n_documents, n_terms), dtype=np.float32)
    
    def QueryToVector(self, queries: list[str]):
        """ Converts a query or batch of queries into a sparse vector matching the vocab size with counts of how many times the term occurs.
                
                Args:
                    query (str): The query to convert
                
                Returns:
                    csr_matrix: A sparse vector representation of the query
        """
        rows, cols, vals = [], [], []
        for i, query in enumerate(queries):
            for term, count in Counter(self.PreprocessText(query)).items():
                j = self.term_to_index.get(term)
                if j is not None:
                    rows.append(i)
                    cols.append(j)
                    vals.append(count)
                    
        return csr_matrix((vals, (rows, cols)), shape=(len(queries), len(self.corpus_vocabulary)), dtype=np.float32)

    #@time_func
    def CalculateScores(self, queries: list[str]):
        """Calculate scores for a query
        
        Args:
            query (str): The query to calculate scores for
        
        Returns:
            np.array: An array of scores for each document

            num_queries x num_documents where (i, j) = ∑_i TF-IDF(q_i, D_j; D) 
        
        """
        # Convert the queries to a vector
        query_vectors = self.QueryToVector(queries) # n_queries x vocab size
        scores = query_vectors.dot(self.tfidf_matrix.T).toarray() # n_queries x n_documents
        return scores
    