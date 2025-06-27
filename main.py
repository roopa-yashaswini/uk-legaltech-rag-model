import os
from dotenv import load_dotenv
from pymilvus import MilvusClient
from create_collection import create_collection
from embed_and_upsert import upsert_documents
from rag_model import cover_letter_model
from embed_and_upsert import process_documents
import pathlib

load_dotenv()

ZILLIZ_URI = os.getenv("ZILLIZ_ADDRESS")
ZILLIZ_TOKEN = os.getenv("ZILLIZ_TOKEN")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

client = MilvusClient(uri=ZILLIZ_URI, token=ZILLIZ_TOKEN)

def main():
    if not client.has_collection(COLLECTION_NAME):
        create_collection(client)
        print("Collection created")
    else:
        print("Collection already exists, skipping")

    all_chunks = []
    chunk_id = 1
    embed = False

    docs_dir = pathlib.Path("docs")
    files = list(docs_dir.iterdir())

    if embed:
        folder_path = "./docs"
        print(f"📂 Reading and embedding documents in: {folder_path}")
        
        all_chunks = process_documents(folder_path)

        upsert_result = upsert_documents(client, all_chunks)
        if upsert_result:
            print(f"✅ {len(all_chunks)} chunks embedded and stored in Zilliz.")
        else:
            print("❌ Upsert failed.")

    cover_letter_model(client, "I want to apply for a sponsor licence to hire software engineers. We are a fintech startup in London with 25 employees. I need to justify the business need and structure a formal letter for UKVI.")

if __name__ == "__main__":
    main()
