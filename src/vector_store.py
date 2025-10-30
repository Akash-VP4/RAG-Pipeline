import os
import chromadb
from typing import List, Any
import numpy as np
from src.embedding import EmbeddingPipeline
import uuid


class ChromaVectorStore:

    def __init__(
        self,
        embedding: EmbeddingPipeline,
        collection_name="store",
        persistent_directory="../data",
    ):

        self.persistnet_directory = persistent_directory
        self.collection_name = collection_name
        self.index = None
        self.collection = None
        self.client = None
        self.embedding = embedding
        self._initialize_store()

    def _initialize_store(self):
        """Ineitialize chromadb client and collection"""

        try:
            os.makedirs(self.persistnet_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persistnet_directory)
            print(f"[DEBUG]: client created: {self.client}")

            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"decription": "document embeddings"},
            )

            print(f"[DEBUG]: collection created: {self.collection}")

        except Exception as e:
            print(f"[ERROR]: Failed to create client/collection: {e}")

    def add_documents(self, document: List[Any], embeddings: np.ndarray):
        """Add document to the DB
        Args:
          document;
          embeddings:
        """

        if len(document) != len(embeddings):
            raise ValueError("Number of document must match number of embeddings")

        ids = []
        metadatas = []
        document_text = []
        embedding_list = []

        for i, (doc, embedding) in enumerate(zip(document, embeddings)):

            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)

            metadata = dict(doc.metadata)
            metadata["doc_indez"] = i
            metadata["content_length"] = len(doc.page_content)
            metadatas.append(metadata)

            document_text.append(doc.page_content)
            embedding_list.append(embedding.tolist())

        print(f"[DEBUG]: Added document_text and embedding_list")

        try:
            print(self.collection)
            self.collection.upsert(
                ids=ids,
                documents=document_text,
                embeddings=embeddings,
                metadatas=metadatas,
            )
            print(f"[DEBUG]: Data added ")

        except Exception as e:
            print(f"[ERROR]: Failed to add data: {e}")

    def search(self, query, n_results=5):
        query_embedding = self.embedding.embed_text([query])[0]

        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
        )

        # print("printing resultssss", type(results), results["documents"])
        # print(results)

        return results
