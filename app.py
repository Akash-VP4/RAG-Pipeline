from src.data_loader import load_all_documents
from src.embedding import EmbeddingPipeline
from src.vector_store import ChromaVectorStore
from src.search import RAGSearch


# example

if __name__ == "__main__":
    docs = load_all_documents("../data/pdf")

    # initialize embedding
    embedding = EmbeddingPipeline()
    chunks = EmbeddingPipeline().chunk_document(docs)  # chunking documents

    text = [chunk.page_content for chunk in chunks]  # Extracting text from the chunks
    vectors = EmbeddingPipeline().embed_text(text)  # converting to vectors
    print(len(vectors), len(chunks))

    db = ChromaVectorStore(embedding)  # initalize db
    db.add_documents(chunks, vectors)  # Add document to db

    search = RAGSearch(vector_store=db)
    print(search.search("what is leave policy"))  # Retrieving data
