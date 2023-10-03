from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
import face_recognition
import requests
from PIL import Image
from io import BytesIO

app = FastAPI()

def find_face_encodings(image_url):
    # Download the image from the URL
    response = requests.get(image_url)
    response.raise_for_status()  # Check for any request errors

    # Open the downloaded image with Pillow (PIL)
    image = Image.open(BytesIO(response.content))

    # Convert the Pillow image to a numpy array for face_recognition
    image_array = face_recognition.load_image_file(BytesIO(response.content))

    # Get face encodings from the image
    face_enc = face_recognition.face_encodings(image_array)
    
    if not face_enc:
        raise ValueError("No faces found in the provided image")

    return face_enc[0]

@app.get("/")
async def read_root(
    image_url_1: str = Query(..., description="URL of the first image"),
    image_url_2: str = Query(..., description="URL of the second image"),
):
    try:
        # Download and process the images from the provided URLs
        image_1 = find_face_encodings(image_url_1)
        image_2 = find_face_encodings(image_url_2)

        is_same = face_recognition.compare_faces([image_1], image_2)[0]

        if is_same:
            distance = face_recognition.face_distance([image_1], image_2)
            distance = round(distance[0] * 100)
            accuracy = 100 - round(distance)

            response = {"result": "The images are same", "accuracy": accuracy}
        else:
            response = {"result": "The images are not same"}

        return JSONResponse(content=jsonable_encoder(response))
    except Exception as e:
        error_message = {"error": str(e)}
        return JSONResponse(content=jsonable_encoder(error_message), status_code=400)
