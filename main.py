# main.py
import fastapi
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import cv2
import numpy as np
import os
import io
from contextlib import asynccontextmanager  # For lifespan

# --- Import the class ---
try:
    from feature_extractor_class import FacialFeatureExtractor
except ImportError as e:
    print(f"Error importing FacialFeatureExtractor class: {e}")
    print(
        "Make sure 'feature_extractor_class.py' is in the same directory and defines the 'FacialFeatureExtractor' class.")
    exit()

# --- Global Variable for the extractor instance ---
feature_extractor_instance: FacialFeatureExtractor | None = None

# --- Configuration for model paths ---
DLIB_FACE_DETECTOR_PATH_API = "mmod_human_face_detector.dat"  # or "" or None for HOG
DLIB_LANDMARK_PREDICTOR_PATH_API = "shape_predictor_68_face_landmarks.dat"


# --- FastAPI Lifespan Event Handler ---
@asynccontextmanager
async def lifespan(current_app: FastAPI):
    global feature_extractor_instance
    print("FastAPI lifespan: Initializing FacialFeatureExtractor...")

    if not os.path.isfile(DLIB_LANDMARK_PREDICTOR_PATH_API):
        print(f"CRITICAL LIFESPAN ERROR: Landmark predictor model not found at '{DLIB_LANDMARK_PREDICTOR_PATH_API}'")
        # Let it yield, but instance will be None or models_loaded=False
    elif DLIB_FACE_DETECTOR_PATH_API and not DLIB_FACE_DETECTOR_PATH_API.strip() == "" and not os.path.isfile(
            DLIB_FACE_DETECTOR_PATH_API):
        print(
            f"WARNING LIFESPAN: MMOD Face detector model specified ('{DLIB_FACE_DETECTOR_PATH_API}') but not found. Extractor will attempt HOG fallback.")

    # Initialize the extractor regardless of file checks above; class __init__ handles it.
    feature_extractor_instance = FacialFeatureExtractor(
        face_detector_path=DLIB_FACE_DETECTOR_PATH_API,
        landmark_predictor_path=DLIB_LANDMARK_PREDICTOR_PATH_API
    )

    if not feature_extractor_instance.models_loaded:
        print(
            "CRITICAL LIFESPAN ERROR: FacialFeatureExtractor models could not be loaded. API may not function correctly.")
    else:
        print("FacialFeatureExtractor initialized successfully via FastAPI lifespan.")

    yield  # Application runs here

    # Code to run on shutdown (if any)
    print("FastAPI lifespan: Shutting down...")
    # feature_extractor_instance = None # Optional: clear instance


# --- FastAPI App Initialization ---
app = FastAPI(title="Facial Feature Extractor API (Class-based)", lifespan=lifespan)


# --- API Endpoint ---
@app.post("/extract-features-class/")
async def create_upload_file_class_based(file: UploadFile = File(...)):
    global feature_extractor_instance

    if feature_extractor_instance is None or not feature_extractor_instance.models_loaded:
        raise HTTPException(status_code=503,
                            detail="Feature extractor not ready or models not loaded. Server may be starting up or encountered an error.")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="No file content received.")

    try:
        nparr = np.frombuffer(contents, np.uint8)
        # Use IMREAD_UNCHANGED to preserve alpha channel if present,
        # the extractor class will handle its conversion.
        img_cv2 = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
        if img_cv2 is None:
            raise HTTPException(status_code=400, detail="Invalid image format or corrupted image.")
    except Exception as e:
        await file.close()  # Ensure file is closed on error
        raise HTTPException(status_code=400, detail=f"Could not decode image: {str(e)}")

    await file.close()  # Close file after successful read

    print(f"Processing image (class-based): {file.filename}, shape: {img_cv2.shape}, dtype: {img_cv2.dtype}")
    features = feature_extractor_instance.extract_all_features(img_cv2)

    if features and "error" in features:
        error_detail = features.get('details', features['error'])
        status_code = 500  # Default internal server error
        if features["error"] == "No faces detected":
            status_code = 422  # Unprocessable Entity
        elif features["error"] == "Models not loaded in extractor" or "dlib runtime" in features["error"]:
            status_code = 503  # Service unavailable (models issue)

        print(f"Feature extraction failed: {features['error']} - {error_detail}")  # Log error
        raise HTTPException(status_code=status_code, detail=f"{features['error']}: {error_detail}")

    if not features:
        print("Feature extraction returned empty or None, failing.")
        raise HTTPException(status_code=500, detail="Feature extraction failed with an unknown error (empty response).")

    return JSONResponse(content=features)


@app.get("/")
async def root():
    global feature_extractor_instance
    status = "Ready and models loaded"
    if feature_extractor_instance is None:
        status = "Not initialized"
    elif not feature_extractor_instance.models_loaded:
        status = "Initialized but models NOT loaded"

    return {"message": f"Facial Feature Extractor API (Class-based) is running. Status: {status}. Use /docs for API."}


# --- To run the app (if this script is executed directly) ---
if __name__ == "__main__":
    # Model paths are defined globally above.
    # The lifespan event handles model loading.
    print("Attempting to run Uvicorn server...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)