from pymilvus import MilvusClient
from typing import List, Dict

def zilliz_retriever(milvus_client: MilvusClient, collection_name: str, search_field: str = "vector", output_fields: List[str] = ["text", "source"], top_k: int = 5):
    """
    Returns a retriever object with an `invoke(query_text, query_vector)` method.
    """

    class Retriever:
        def __init__(self, client, collection):
            self.client = client
            self.collection_name = collection

        def invoke(self, query_text: str, query_vector: List[float]) -> List[Dict]:
            try:
                search_results = self.client.search(
                    collection_name=self.collection_name,
                    data=[query_vector],
                    anns_field=search_field,
                    search_params={
                        "metric_type": "COSINE",
                        "params": {"nprobe": 10}
                    },
                    limit=top_k,
                    output_fields=output_fields
                )

                # print(search_results)

                docs = []
                for match in search_results[0]:
                    # match is a dict
                    # print(match.get("entity", {}).get("text", ""))
                    text = match.get("entity", {}).get("text", "")
                    source = match.get("entity", {}).get("source", "")
                    
                    docs.append({
                        "pageContent": text,
                        "metadata": {"source": source}
                    })


                return docs

            except Exception as e:
                print(f"❌ Retrieval failed: {str(e)}")
                return []

    return Retriever(milvus_client, collection_name)
