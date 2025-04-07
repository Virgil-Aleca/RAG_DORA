from docx import Document
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import openai
from deep_translator import GoogleTranslator
from langdetect import detect

openai.api_key = "AICI"

DOCX_PATH = "DORA.docx" 
USER_INPUT = "Ce este Change Management?"
USE_GPT = True  


doc = Document(DOCX_PATH)
chunks = []
section_titles = []
current_section = None

for para in doc.paragraphs:
    text = para.text.strip()
    if not text:
        continue
    if text[0].isdigit() and len(text) < 500:
        current_section = text
        continue
    chunk_text = f"{current_section}\n{text}" if current_section else text
    chunks.append(chunk_text)
    section_titles.append(current_section or "N/A")


print("Generăm embedding-uri...")
model = SentenceTransformer('all-mpnet-base-v2')
embeddings = model.encode(chunks, show_progress_bar=True)
embeddings = np.array(embeddings).astype('float32')


index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)
print("Index FAISS creat cu", index.ntotal, "fragmente.")



translation_prompt = f"Please translate the following question from Romanian to English, keeping the original meaning intact:\n\n'{USER_INPUT}'"

translation_response = openai.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a professional translator, fluent in Romanian and English."},
        {"role": "user", "content": translation_prompt}
    ],
    temperature=0,
    max_tokens=100,
)

QUERY = translation_response.choices[0].message.content.strip()
print("\n=== ÎNTREBARE TRADUSĂ DE GPT ===")
print(QUERY)


query_embedding = model.encode([QUERY])[0].astype('float32')
D, I = index.search(np.array([query_embedding]), k=10)
retrieved_chunks = [chunks[i] for i in I[0]]

print("\n=== FRAGMENTE RELEVANTE ===")
for chunk in retrieved_chunks:
    print("-", chunk[:300].replace("\n", " "), "...\n")


if USE_GPT:
    openai.api_key = "AICI"  # <-- înlocuiește cu cheia ta
    context = "\n\n".join(retrieved_chunks)
    prompt = f"Use the context below from the DORA document to answer the question. Also, mention at the end where was the information extracted from, i want reasoning. I want a clear structure. Please Translate it in romanian\n\nContext:\n{context}\n\nQuestion: {QUERY}\nAnswer:"

    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant with vast DORA experience that answers based on provided documents."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=1000,
    )

    answer_en = response.choices[0].message.content
    print("\n=== RĂSPUNS GPT (EN) ===")
    print(answer_en)
