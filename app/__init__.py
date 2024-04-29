from typing import Union

from fastapi import FastAPI
from app.handlers.yolo_explorer import YoloHumanExplorer

app = FastAPI()


# uvicorn main:app --reload
@app.get("/")
def read_root():
    user_id = '2'
    yolo_explorer = YoloHumanExplorer()
    yolo_explorer.predict_and_save(user_id,'/vid.mp4')

    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}
