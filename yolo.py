import os
import src.utils.checker_util as ch
from src.controllers.detection_controller import YOLO_img_to_base64_response as yolo_b64
from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from flask import Response, abort

folder = os.path.abspath('.') + "/weights"


def get_image(images):
	ch.Weight_checker.start(folder)
	return yolo_b64.predict(images)

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
	print(images)
	# return get_image(images)



if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=5000)
