from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
import face_recognition
import requests
from PIL import Image
import numpy as np
from io import BytesIO

app = FastAPI()

# Define your CORS settings
origins = [
    "http://127.0.0.1:8000",  # Replace with your development server URL
    "http://192.81.219.48",
]

# Add CORS middleware with the defined settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def find_face_encodings(image_url):
    # Download the image from the URL
    response = requests.get(image_url)
    response.raise_for_status()  # Check for any request errors

    # Open the downloaded image with Pillow (PIL)
    image = Image.open(BytesIO(response.content))

    # Convert the Pillow image to a numpy array for face_recognition
    image_array = np.array(image)

    # Get face encodings from the image
    face_enc = face_recognition.face_encodings(image_array)

    if not face_enc:
        raise ValueError("No faces found in the provided image")

    return face_enc[0]

@app.post("/")
async def hello_world():
    return "Hello, World"

@app.post("/compare")
async def compare_images(
    id_card_photo_url: str = Form(..., description="ID Card Photo URL"),
    recent_camera_photo_url: str = Form(..., description="Recent Camera Photo URL"),
):
    try:
        # Process the images from the provided URLs
        id_card_face_encoding = find_face_encodings(id_card_photo_url)
        recent_camera_face_encoding = find_face_encodings(recent_camera_photo_url)

        is_same = face_recognition.compare_faces([id_card_face_encoding], recent_camera_face_encoding)[0]

        if is_same:
            distance = face_recognition.face_distance([id_card_face_encoding], recent_camera_face_encoding)
            distance = round(distance[0] * 100)
            accuracy = 100 - round(distance)

            response = {"result": "The images are the same", "accuracy": accuracy}
        else:
            response = {"result": "The images are not the same"}

        return JSONResponse(content=jsonable_encoder(response))
    except Exception as e:
        error_message = {"error": str(e)}
        return JSONResponse(content=jsonable_encoder(error_message), status_code=400)
