from qdrant_client import models, QdrantClient
from sentence_transformers import SentenceTransformer
import pickle

encoder = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)

def read_pickle(file_path):
    """
    Read a pickle file and return the content.
    """
    with open(file_path, 'rb') as file:
        content = pickle.load(file)
    return content

chunks = read_pickle('/home/filippo/Scrivania/ELAN_bot/ELAN_chunks.pkl')


client = QdrantClient(path="/home/filippo/Scrivania/ELAN_bot/qdrant_data")

client.create_collection(
    collection_name="ELAN_docs_pages",
    vectors_config=models.VectorParams(
        size=encoder.get_sentence_embedding_dimension(),  # Vector size is defined by used model
        distance=models.Distance.COSINE,
    ),
)


client.upload_points(
    collection_name="ELAN_docs_pages",
    points=[
        models.PointStruct(
            id=idx, vector=encoder.encode(doc["title"]).tolist(), payload=doc
        )
        for idx, doc in enumerate(chunks)
    ],
)

