from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from Sticker_validation import StickerValidation
import cv2
import numpy as np
import base64
import tempfile
import os

app = FastAPI(title="Sticker Validation API")

# ✅ Allow browser-based requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost"] to be strict
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Load YOLO model once to save time on each request
validator = StickerValidation()

@app.post("/infer")
async def infer_image(file: UploadFile = File(...)):
    try:
        # ✅ Read the uploaded file
        contents = await file.read()
        np_arr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            return JSONResponse(content={"error": "Invalid image"}, status_code=400)

        # ✅ Run YOLO inference for annotated image
        results = validator.model(img)
        annotated_img = results[0].plot(labels=True, conf=False)

        # ✅ Save to a temp file to run StickerStatus (it expects a path)
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            temp_path = tmp.name
            cv2.imwrite(temp_path, img)

        # ✅ Get message (OK / NOK etc.)
        message = validator.StickerStatus(temp_path)

        # ✅ Cleanup temp file
        os.remove(temp_path)

        # ✅ Convert annotated image to Base64
        _, buffer = cv2.imencode(".jpg", annotated_img)
        base64_img = base64.b64encode(buffer).decode("utf-8")

        return {
            "message": message,
            "image": f"data:image/jpeg;base64,{base64_img}"
        }

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
