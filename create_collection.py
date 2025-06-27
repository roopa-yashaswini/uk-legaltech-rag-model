import os
from pymilvus import MilvusClient, CollectionSchema, FieldSchema, DataType

def create_collection(client: MilvusClient):
    try:
        collection_name = os.getenv("COLLECTION_NAME")
        VECTOR_DIM = 3072

        # 1. Define schema fields
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=4096),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=VECTOR_DIM),
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=1024),
        ]

        schema = CollectionSchema(fields=fields, description="Chunks from UK sponsorship license guidelines")

        # 2. Create collection
        try:
            client.create_collection(
                collection_name=collection_name,
                schema=schema
            )
            print(f"📁 Collection '{collection_name}' created.")
        except Exception as e:
            print(f"❌ Failed to create collection '{collection_name}': {e}")
            raise e

        # 3. Create index
        try:
            index_params = client.prepare_index_params()
            index_params.add_index(
                field_name="vector",
                index_type="HNSW",
                metric_type="COSINE",
                params={"M": 16, "efConstruction": 200}
            )

            client.create_index(
                collection_name=collection_name,
                index_params=index_params,
                timeout=None
            )
            print(f"🔍 Index created on collection '{collection_name}'.")
        except Exception as e:
            print(f"❌ Failed to create index: {e}")
            raise e

        # 4. Load collection into memory
        try:
            client.load_collection(collection_name)
            print(f"✅ Collection '{collection_name}' loaded into memory.")
        except Exception as e:
            print(f"❌ Failed to load collection '{collection_name}': {e}")
            raise e

    except Exception as final_error:
        print(f"🔥 Unexpected error during collection setup: {final_error}")
