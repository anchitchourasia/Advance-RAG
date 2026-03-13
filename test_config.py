from config import (
    GOOGLE_API_KEY, MODEL_NAME,
    PINECONE_API_KEY, PINECONE_INDEX_HOST,
    PROXY_CONFIG, LIMIT_ROWS,
    GROQ_API_KEY, GROQ_MODEL_NAME
)

print("Gemini key:    ", bool(GOOGLE_API_KEY))
print("Model:         ", MODEL_NAME)
print("Pinecone key:  ", bool(PINECONE_API_KEY))
print("Pinecone host: ", PINECONE_INDEX_HOST)
print("Proxy:         ", PROXY_CONFIG)
print("LIMIT_ROWS:    ", LIMIT_ROWS)
print("Groq key:      ", bool(GROQ_API_KEY))
print("Groq model:    ", GROQ_MODEL_NAME)
