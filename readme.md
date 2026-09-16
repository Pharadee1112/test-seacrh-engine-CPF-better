# CPF Semantic Search — Proof of Concept

ทดสอบว่า semantic search (ค้นด้วยความหมาย) แก้ pain point การค้นหาสินค้าใน ALL Online (เว็บของ 7-Eleven) ได้จริงไหม เทียบกับระบบเดิมที่ค้นแบบ keyword ตรงตัวอักษรเท่านั้น

รายละเอียด pain point, การตัดสินใจ, และ progress เต็มๆ ดูที่ `PROGRESS.md`

## วิธีรัน

```
pip install sentence-transformers numpy
python search.py
```

รันครั้งแรกจะโหลดโมเดล `paraphrase-multilingual-MiniLM-L12-v2` (~470MB) ครั้งเดียว ครั้งต่อไปจะเร็วเพราะโมเดล cache ไว้ในเครื่องแล้ว

ถ้าโหลดโมเดลไม่ได้ (connection timeout ไปที่ huggingface.co) ลองตั้ง mirror ก่อนรัน:
```
set HF_ENDPOINT=https://hf-mirror.com
python search.py
```

## ไฟล์

- `catalog.json` — สินค้าจริงที่เก็บมาจาก ALL Online (name, brand, category, description, diet_tags, price)
- `search.py` — search engine หลัก คำนวณ embedding จาก name + description + category + diet_tags
- `PROGRESS.md` — บันทึกความคืบหน้า pain point และการตัดสินใจตลอดโปรเจกต์

## หลักการทำงาน

1. แปลงชื่อ+คำอธิบาย+หมวดหมู่+diet tags ของแต่ละสินค้าเป็นเวกเตอร์ (embedding) ล่วงหน้า
2. แปลงคำค้นของผู้ใช้เป็นเวกเตอร์เดียวกัน
3. เทียบด้วย cosine similarity แล้วคืนสินค้าที่คะแนนใกล้เคียงที่สุด

ต่างจาก keyword search เดิมตรงที่จับ "ความหมาย" ไม่ใช่ "ตัวอักษรตรงเป๊ะ" — พิมพ์ผิด พิมพ์ขาดคำ หรือใช้คำพ้องความหมาย ก็ยังหาเจอได้

## Schema ของสินค้าใน catalog.json

```json
{
  "id": "mz-001",
  "name": "เกี๊ยวซ่าจากพืช มีทซีโร่",
  "brand": "Meat Zero",
  "category": "อาหารแช่แข็ง",
  "description": "เกี๊ยวซ่าทำจากโปรตีนพืช 100% ไม่มีเนื้อสัตว์ ไม่มีคอเลสเตอรอล",
  "diet_tags": ["วีแกน", "เจ", "มังสวิรัติ", "plant-based"],
  "price": 89
}
```

## Known limitation

- Catalog เป็นข้อมูลจำลอง เก็บด้วยมือจากเว็บจริง (ไม่มี public API ของ 7-Eleven/CPF)
- ยังไม่ fine-tune โมเดล ใช้ pretrained model ตรงๆ เหมาะกับ PoC ขนาดนี้
- Pain point เรื่อง brand/category tagging ไม่ครบ เป็นปัญหาข้อมูลของ CPF เอง ไม่ใช่สิ่งที่ semantic search แก้ได้ — ดูรายละเอียดใน PROGRESS.md