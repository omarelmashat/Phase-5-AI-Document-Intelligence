import chromadb

DEFAULT_TOP_K = 4


class VectorStoreError(Exception):
class DocumentVectorStore:
    def __init__(self, collection_name: str = "document_session"):
        self._client = chromadb.EphemeralClient()
        self._collection_name = collection_name
        self._collection = self._client.get_or_create_collection(name=collection_name)

    def reset(self) -> None:
        try:
            self._client.delete_collection(self._collection_name)
        except Exception:
            pass
        self._collection = self._client.get_or_create_collection(name=self._collection_name)

    def add_chunks(self, chunks: list[dict], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        if len(chunks) != len(embeddings):
            raise VectorStoreError(
                f"Got {len(chunks)} chunks but {len(embeddings)} embeddings — they must match 1:1."
            )
        try:
            self._collection.add(
                ids=[chunk["id"] for chunk in chunks],
                embeddings=embeddings,
                documents=[chunk["text"] for chunk in chunks],
                metadatas=[
                    {"source": chunk.get("source", ""), "chunk_index": chunk.get("chunk_index", 0)}
                    for chunk in chunks
                ],
            )
        except Exception as error:
            raise VectorStoreError(f"Failed to add chunks to the vector store: {error}") from error

    def search(self, query_embedding: list[float], top_k: int = DEFAULT_TOP_K) -> list[dict]:
        count = self._collection.count()
        if count == 0:
            return []
        try:
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, count),
            )
        except Exception as error:
            raise VectorStoreError(f"Vector search failed: {error}") from error

        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        return [
            {"id": ids[i], "text": docs[i], "metadata": metas[i], "distance": dists[i]}
            for i in range(len(ids))
        ]

    def is_empty(self) -> bool:
        return self._collection.count() == 0

    def chunk_count(self) -> int:
        return self._collection.count()
