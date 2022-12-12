import os
import services.yolo.src.utils.checker_util as ch
from services.yolo.src.controllers.detection_controller import (
    YOLO_img_to_base64_response as yolo_b64,
)
from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

folder = os.path.abspath(".") + "/weights"


class YOLO(object):
    def __init__(self, image):
        self.image = image

    def get_image(self):
        ch.Weight_checker.start(folder)     
        return { "status": 200, "message": "success", "data": yolo_b64.predict(self.image) }


app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:5000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/image")
async def image(images: UploadFile):
    response = await YOLO(images.read()).get_image()
    print(response)
    return { "status": 200, "message": "success","data": response }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
