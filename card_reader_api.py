from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import cv2
from ID_capturing import detect_card_by_contours
from ID_Reader import extract_selected_mrz_data
import tempfile

app = FastAPI(title="ID Reader API")

@app.post("/extract-id")
async def extract_id(file: UploadFile = File(...)):
    try:
        # Read image bytes
        contents = await file.read()
        npimg = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image format")

        # Detect and crop the ID
        crop, _, _ = detect_card_by_contours(image)
        if crop is None:
            raise HTTPException(status_code=404, detail="ID card not detected")

        # Extract MRZ data
        data = extract_selected_mrz_data(crop)
        if not data:
            raise HTTPException(status_code=422, detail="Unable to extract MRZ data")

        # Return relevant fields
        return JSONResponse({
            "id_number": data.get("national_number"),
            "nationality": data.get("country"),
            "full_name": data.get("full_name")
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
