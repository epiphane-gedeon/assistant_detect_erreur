from app import app, db
from typing import Union

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

@app.get("/faq")
def get_faq():
    
    cur = db.conn.cursor()

    # Requête SELECT
    cur.execute("SELECT id, question, procede FROM FAQ ORDER BY id;")
    faqs = cur.fetchall()
    
    faqs_list = []
    for faq in faqs:
        faqs_list.append({
            "id": faq[0],
            "question": faq[1],
            "procede": faq[2]
        })
    return faqs_list