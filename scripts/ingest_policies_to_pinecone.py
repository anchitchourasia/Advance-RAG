import config  # loads .env and applies proxy first
from pathlib import Path
from config import KNOWLEDGE_DIR, PINECONE_POLICY_NAMESPACE
from retrieval.pinecone_store import get_index, embed_texts

def chunk_text(text, chunk_size=500, overlap=100):
    text = (text or "").strip()
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += max(1, chunk_size - overlap)
    return chunks

def load_policy_files():
    base = Path(KNOWLEDGE_DIR).resolve()
    files = []
    for path in base.glob("*"):
        if path.suffix.lower() in [".md", ".txt"] and path.is_file():
            files.append(path)
    return files

def main():
    index = get_index()
    vectors = []

    files = load_policy_files()
    if not files:
        print("No policy files found.")
        return

    for path in files:
        text = path.read_text(encoding="utf-8").strip()
        chunks = chunk_text(text)

        if not chunks:
            continue

        embeddings = embed_texts(chunks)

        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            vectors.append({
                "id": f"policy#{path.stem}#{i}",
                "values": emb,
                "metadata": {
                    "document_id": path.stem,
                    "chunk_number": i,
                    "source": path.name,
                    "document_type": "policy",
                    "chunk_text": chunk
                }
            })

    if not vectors:
        print("No vectors created. Check if policy files have content.")
        return

    for i in range(0, len(vectors), 100):
        batch = vectors[i:i+100]
        index.upsert(namespace=PINECONE_POLICY_NAMESPACE, vectors=batch)

    print(f"Upserted {len(vectors)} policy chunks to Pinecone.")

if __name__ == "__main__":
    main()
