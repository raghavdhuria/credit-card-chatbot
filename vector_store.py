import weaviate
from weaviate.classes.config import Property, DataType
from langchain.vectorstores import Weaviate
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
from langchain.schema import Document
from typing import List, Dict, Any
import numpy as np

class BGEEmbeddings(Embeddings):
    def __init__(self, model_name: str = "BAAI/bge-large-en"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text])[0].tolist()

class VectorStore:
    def __init__(self, cluster_url: str, api_key: str):
        self.cluster_url = cluster_url
        self.api_key = api_key
        self.client = None
        self.index_name = "CreditCards"
        self.embeddings = BGEEmbeddings()
        self.vector_store = None

    def connect(self):
        self.client = weaviate.connect_to_weaviate_cloud(
            cluster_url=self.cluster_url,
            auth_credentials=weaviate.auth.AuthApiKey(api_key=self.api_key)
        )
        print("Connected to Weaviate cluster")

    def create_schema(self):
        if self.client is None:
            self.connect()

        if self.client.collections.exists(self.index_name):
            print(f"Collection '{self.index_name}' already exists")
            return

        self.client.collections.create(
            name=self.index_name,
            description="Credit card and debit card information for recommendation system",
            vectorizer_config=weaviate.classes.config.Configure.Vectorizer.none(),
            vector_index_config=weaviate.classes.config.Configure.VectorIndex.hnsw(distance_metric=weaviate.classes.config.VectorDistances.COSINE),
            properties=[
                Property(name="text", data_type=DataType.TEXT),
                Property(name="card_name", data_type=DataType.TEXT),
                Property(name="issuer", data_type=DataType.TEXT),
                Property(name="annual_fee", data_type=DataType.NUMBER),
                Property(name="joining_fee", data_type=DataType.NUMBER),
                Property(name="category", data_type=DataType.TEXT_ARRAY),
                Property(name="rewards_category", data_type=DataType.TEXT_ARRAY),
                Property(name="eligibility", data_type=DataType.TEXT_ARRAY),
                Property(name="description", data_type=DataType.TEXT)
            ]
        )
        print(f"Collection '{self.index_name}' created")

    def reset_collection(self):
        if self.client is None:
            self.connect()
        if self.client.collections.exists(self.index_name):
            self.client.collections.delete(self.index_name)
            print(f"Deleted collection '{self.index_name}'")
        self.create_schema()
        print(f"Reset collection '{self.index_name}'")

    def load_documents(self, documents: List[Dict[str, Any]]):
        if self.client is None:
            self.connect()

        try:
            self.reset_collection()
        except Exception as e:
            print(f"Warning: Could not reset collection: {e}")

        collection = self.client.collections.get(self.index_name)

        with collection.batch.dynamic() as batch:
            for i, doc in enumerate(documents):
                if i % 100 == 0:
                    print(f"Loading document {i+1}/{len(documents)}")

                props = {
                    "text": doc["text"],
                    "card_name": doc["metadata"]["card_name"],
                    "issuer": doc["metadata"]["issuer"],
                    "annual_fee": doc["metadata"]["annual_fee"],
                    "joining_fee": doc["metadata"]["joining_fee"],
                    "category": doc["metadata"]["category"],
                    "rewards_category": doc["metadata"]["rewards_category"],
                    "eligibility": doc["metadata"]["eligibility"],
                    "description": doc["metadata"]["description"],
                }
                vector = self.embeddings.embed_query(doc["text"])
                batch.add_object(properties=props, vector=vector)

        self.vector_store = Weaviate(
            client=self.client,
            index_name=self.index_name,
            text_key="text",
            embedding=self.embeddings
        )
        print(f"Loaded {len(documents)} documents into vector store")

    def filter_search(self, query: str, filters: Dict[str, Any], top_k: int = 5) -> List[Document]:
        if self.client is None:
            self.connect()

        collection = self.client.collections.get(self.index_name)
        weaviate_filter = None

        # Build property filters
        if "annual_fee_max" in filters:
            try:
                fee = float(filters["annual_fee_max"])
                from weaviate import Filter
                f1 = Filter.property("annual_fee").less_than_equal(fee)
                weaviate_filter = f1 if weaviate_filter is None else Filter.and_(weaviate_filter, f1)
            except Exception:
                pass

        if "joining_fee_max" in filters:
            try:
                fee = float(filters["joining_fee_max"])
                from weaviate import Filter
                f2 = Filter.property("joining_fee").less_than_equal(fee)
                weaviate_filter = f2 if weaviate_filter is None else Filter.and_(weaviate_filter, f2)
            except Exception:
                pass

        if "categories" in filters and filters["categories"]:
            try:
                cats = filters["categories"]
                if not isinstance(cats, list):
                    cats = [cats]
                from weaviate import Filter
                f3 = Filter.property("category").contains_any(cats)
                weaviate_filter = f3 if weaviate_filter is None else Filter.and_(weaviate_filter, f3)
            except Exception:
                pass

        query_vector = self.embeddings.embed_query(query)

        if weaviate_filter:
            response = collection.query.near_vector(near_vector=query_vector, filters=weaviate_filter, limit=top_k)
        else:
            response = collection.query.near_vector(near_vector=query_vector, limit=top_k)

        results = []
        for obj in response.objects:
            props = obj.properties
            doc = Document(page_content=props.get("text", ""), metadata={k:v for k,v in props.items() if k != "text"})
            results.append(doc)
        return results
