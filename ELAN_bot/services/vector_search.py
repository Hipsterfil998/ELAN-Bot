"""
Vector search functionality for ELAN-Bot application.
"""

from pathlib import Path
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from config.settings import DEFAULT_ENCODER_MODEL, QDRANT_CLIENT_PATH, COLLECTION_NAME, SEARCH_LIMIT


class VectorSearchService:
    """Service for handling vector-based document search."""

    def __init__(self, encoder_model: str = DEFAULT_ENCODER_MODEL):
        """
        Initialize the vector search service.

        Args:
            encoder_model: The sentence transformer model to use for encoding
        """
        self.encoder_model = encoder_model
        self.encoder = None
        self.client = None

    def _initialize_encoder(self):
        """Initialize the sentence transformer encoder."""
        if self.encoder is not None:
            return
        try:
            self.encoder = SentenceTransformer(self.encoder_model, trust_remote_code=True)
            print(f"Successfully loaded encoder: {self.encoder_model}")
        except Exception as e:
            print(f"Error initializing encoder {self.encoder_model}: {e}. Falling back to all-MiniLM-L6-v2")
            self.encoder = SentenceTransformer("all-MiniLM-L6-v2")

    def _initialize_client(self):
        """Initialize the Qdrant client."""
        if self.client is not None:
            return
        Path(QDRANT_CLIENT_PATH).mkdir(parents=True, exist_ok=True)
        self.client = QdrantClient(path=QDRANT_CLIENT_PATH)

    def get_context(self, query: str) -> str:
        """
        Retrieve relevant context from vector database based on query.

        Args:
            query: The search query string

        Returns:
            str: Combined context from relevant documents
        """
        try:
            self._initialize_encoder()
            self._initialize_client()

            available = [col.name for col in self.client.get_collections().collections]
            if COLLECTION_NAME not in available:
                print(f"Collection '{COLLECTION_NAME}' not found. Available: {available}")
                return "I'm sorry, the knowledge base is not available yet. Please ensure your ELAN documentation is properly loaded in the vector database."

            query_vector = self.encoder.encode(query).tolist()
            hits = self.client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_vector,
                limit=SEARCH_LIMIT,
            ).points

            if not hits:
                return "I'm sorry, I couldn't find relevant information for your query in the knowledge base."

            context_parts = []
            for hit in hits:
                if hasattr(hit, 'payload') and hit.payload:
                    content = (
                        hit.payload.get('content')
                        or hit.payload.get('text')
                        or hit.payload.get('document')
                        or str(hit.payload)
                    )
                    context_parts.append(content)

            if not context_parts:
                return "I'm sorry, the knowledge base structure is not compatible. Please check the data format."

            print(f"Found {len(hits)} relevant documents for query: '{query[:50]}'")
            return "\n".join(context_parts)

        except Exception as e:
            print(f"Error in vector search: {e}")
            return "I'm sorry, it was not possible to find any relevant information."
