import os
from openai import OpenAI
from dotenv import load_dotenv
import fitz  # PyMuPDF
from docx import Document

load_dotenv()

MAX_CHARS = 20000
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

def process_documents(folder_path):
    results = []
    file_id = 1

    for file_name in os.listdir(folder_path):
        path = os.path.join(folder_path, file_name)
        ext = os.path.splitext(file_name)[1].lower()

        try:
            if ext == ".pdf":
                print(f"📄 Extracting PDF: {file_name}")
                chunks = extract_chunks_from_pdf(path)
            elif ext == ".docx":
                print(f"📝 Extracting DOCX: {file_name}")
                chunks = extract_chunks_from_docx(path)
            else:
                print(f"⚠️ Skipping unsupported file: {file_name}")
                continue
        except Exception as e:
            print(f"❌ Error extracting from {file_name}: {e}")
            continue

        for chunk in chunks:
            embedding = embed_text(chunk)
            if embedding:
                results.append({
                    "id": file_id,
                    "text": chunk,
                    "embedding": embedding,
                    "source": file_name
                })
                file_id += 1
            else:
                print(f"❌ Embedding failed for chunk from {file_name}")

    print(f"✅ Total valid chunks processed: {len(results)}")
    return results

def extract_chunks_from_pdf(file_path):
    try:
        doc = fitz.open(file_path)
        chunks = []
        for page in doc:
            text = page.get_text()
            while text:
                chunks.append(text[:MAX_CHARS].strip())
                text = text[MAX_CHARS:]
        return chunks
    except Exception as e:
        print(f"❌ PDF read error: {e}")
        return []

def extract_chunks_from_docx(file_path):
    try:
        doc = Document(file_path)
        full_text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        chunks = []
        while full_text:
            chunks.append(full_text[:MAX_CHARS].strip())
            full_text = full_text[MAX_CHARS:]
        return chunks
    except Exception as e:
        print(f"❌ DOCX read error: {e}")
        return []

def embed_text(text):
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=text
        )
        embedding = response.data[0].embedding
        if not embedding or len(embedding) != 3072:
            print("⚠️ Invalid embedding length:", len(embedding))
        return embedding
    except Exception as e:
        print("❌ Error during embedding:", str(e))
        return None

def upsert_documents(client, docs):
    print(f"🆙 Starting upsert to collection: {COLLECTION_NAME}")
    valid_data = []

    for doc in docs:
        vec = doc.get("embedding")
        if vec and isinstance(vec, list) and len(vec) == 3072:
            valid_data.append({
                "id": doc["id"],
                "text": doc["text"],
                "vector": vec,
                "source": doc["source"]
            })
        else:
            print(f"⚠️ Skipping invalid vector for doc ID {doc['id']}")

    print(f"📊 Vectors to insert: {len(valid_data)}")

    if not valid_data:
        print("❌ No valid embeddings to insert.")
        return False

    try:
        insert_result = client.insert(collection_name=COLLECTION_NAME, data=valid_data)
        print("📥 Insert result:", insert_result)

        try:
            client.load_collection(COLLECTION_NAME)
            print("✅ Collection loaded.")
        except Exception as e:
            print(f"❌ Collection load failed: {e}")
        return True
    except Exception as e:
        print(f"❌ Upsert failed: {e}")
        return False
