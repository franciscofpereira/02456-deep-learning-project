from src.data_utils.document import Document


class DocumentDataset:
    def __init__(self, documents: list[dict]) -> None:
        self.documents: list[Document] = self.__BuildDocuments(documents)

    def __BuildDocuments(self, documents):
        docs = []
        for entries in documents:
            texts = entries["text"]
            ids = entries["document_id"]
            if isinstance(texts, str):
                docs.append(Document(text=texts, _id=ids))
            else:
                for text, _id in zip(texts, ids):
                    docs.append(Document(text=text, _id=_id))
        return docs

    def GetDocuments(self) -> list[Document]:
        return self.documents
