from typing import List, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
from src.data_loader import load_all_documents


class EmbeddingPipeline:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        """This function chunk the document and convert into embedding"""

        self.chunl_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            self.model = SentenceTransformer(self.model_name)
            print(f"[DEBUG] loaded embdding model : {self.model_name}")

        except Exception as e:
            print(f"Loading errorr! {e}")

    def chunk_document(self, document: List[Any]) -> List[Any]:
        """This will split the document in to chunks
        Args:
          document:lsit of document
        """

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunl_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )

        doc_chunks = text_splitter.split_documents(document)
        print(f"[DEBUG] split{len(document)} into {len(doc_chunks)}")

        return doc_chunks

    def embed_text(self, document_text:List[Any]) -> np.ndarray:
        """This will embedd the document chunk
        Args:
          document:
        """

        embeddings = self.model.encode(document_text, show_progress_bar=True)
        print(f"[DEBUG] Embedded succesfully shape:{embeddings.shape}")

        return embeddings
