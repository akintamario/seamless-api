from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
import io
import cv2
import numpy as np
from .image_processor import make_seamless

app = FastAPI()

@app.post("/seamless")
async def make_image_seamless(file: UploadFile = File(...)):
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    processed_image = make_seamless(image)

    _, buffer = cv2.imencode(".png", processed_image)
    return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/png")
