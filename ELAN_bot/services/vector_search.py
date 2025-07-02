"""
Vector search functionality for ELAN-Bot application.
"""

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
        if self.encoder is None:
            try:
                self.encoder = SentenceTransformer(
                    self.encoder_model, 
                    trust_remote_code=True
                )
                print(f"Successfully loaded encoder: {self.encoder_model}")
            except Exception as e:
                print(f"Error initializing encoder {self.encoder_model}: {e}")
                # Fallback to a stable model if nomic fails
                print("Falling back to all-MiniLM-L6-v2")
                self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
    
    def _initialize_client(self):
        """Initialize the Qdrant client."""
        if self.client is None:
            try:
                self.client = QdrantClient(path=QDRANT_CLIENT_PATH)
            except Exception as e:
                print(f"Error initializing Qdrant client: {e}")
                # Initialize with in-memory mode as fallback
                self.client = QdrantClient(":memory:")
    
    def get_context(self, query: str) -> str:
        """
        Retrieve relevant context from vector database based on query.
        
        Args:
            query: The search query string
            
        Returns:
            str: Combined context from relevant documents
        """
        try:
            # Initialize components if needed
            self._initialize_encoder()
            self._initialize_client()
            
            # Check if collection exists and get available collections
            try:
                collections = self.client.get_collections()
                collection_names = [col.name for col in collections.collections]
                print(f"Available collections: {collection_names}")
                
                # Try to use the configured collection name first
                target_collection = COLLECTION_NAME
                
                # If configured collection doesn't exist, try common alternatives
                if target_collection not in collection_names:
                    possible_names = ["elan_docs_pages", "ELAN_docs_pages", "collection", "documents"]
                    for name in possible_names:
                        if name in collection_names:
                            target_collection = name
                            print(f"Using collection: {target_collection}")
                            break
                    else:
                        print(f"No suitable collection found. Available: {collection_names}")
                        return "I'm sorry, the knowledge base is not available yet. Please ensure your ELAN documentation is properly loaded in the vector database."
                
            except Exception as e:
                print(f"Error checking collections: {e}")
                return "I'm sorry, there was an issue connecting to the knowledge base."
            
            # Encode the query
            query_vector = self.encoder.encode(query).tolist()
            
            # Execute the search
            hits = self.client.query_points(
                collection_name=target_collection,
                query=query_vector,
                limit=SEARCH_LIMIT,
            ).points
            
            # Extract and combine context content
            if not hits:
                return "I'm sorry, I couldn't find relevant information for your query in the knowledge base."
            
            # Check if hits have the expected payload structure
            context_parts = []
            for hit in hits:
                if hasattr(hit, 'payload') and hit.payload:
                    # Try different possible content keys
                    content = hit.payload.get('content') or hit.payload.get('text') or hit.payload.get('document') or str(hit.payload)
                    context_parts.append(content)
            
            if not context_parts:
                return "I'm sorry, the knowledge base structure is not compatible. Please check the data format."
            
            context = "\n".join(context_parts)
            print(f"Found {len(hits)} relevant documents for query: '{query[:50]}...'")
            
            return context
            
        except Exception as e:
            print(f"Error in vector search: {e}")
            return "I'm sorry, it was not possible to find any relevant information."