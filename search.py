"""
Semantic product search — proof of concept
เทียบความหมายของคำค้นกับสินค้าใน catalog.json ด้วย embedding + cosine similarity
"""

import json
from sentence_transformers import SentenceTransformer
import numpy as np

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

model = SentenceTransformer(MODEL_NAME)

with open("catalog.json", encoding="utf-8") as f:
    products = json.load(f)

def to_text(value):
    """รองรับทั้ง string เดี่ยว และ list ของ string"""
    if isinstance(value, list):
        return " ".join(value)
    return value or ""

# รวมชื่อ + คำอธิบาย + หมวดหมู่ + diet tags เป็นข้อความเดียวต่อสินค้า แล้วแปลงเป็นเวกเตอร์ล่วงหน้า
texts = [
    f"{p['name']} {p['description']} {to_text(p.get('brand', ''))} "
    f"{to_text(p.get('category', ''))} {to_text(p.get('diet_tags', []))}"
    for p in products
]

product_vecs = model.encode(texts, normalize_embeddings=True)


def search(query: str, top_k: int = 5):
    q_vec = model.encode([query], normalize_embeddings=True)[0]
    scores = product_vecs @ q_vec  # cosine similarity เพราะ normalize แล้ว
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [(products[i]["name"], float(scores[i])) for i in top_idx]


if __name__ == "__main__":
    test_queries = [
        "ของกินเจไม่ใช่ผัก",
        "เนื้อจากพืชไม่มีคอเลสเตอรอล",
        "อาหารมังสวิรัติกรุบกรอบ",
        "อาหารแช่แข็ง",
        "วีแกน",
        "อาหารเจ",
        "มีทซีโร่",
        "ไก่ทอดจากพืช",
        "เนื้อเทียม"
    ]
    for q in test_queries:
        print(f"\nQuery: {q}")
        for name, score in search(q, top_k=2):
            print(f"  {score:.3f}  {name}")

