from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
import face_recognition
import requests

app = FastAPI()

def find_face_encodings(image_url):
    # Download the image from the URL
    response = requests.get(image_url)
    response.raise_for_status()  # Check for any request errors

    # Convert the image content to bytes
    image_bytes = response.content

    # Process the image bytes
    image_array = face_recognition.face_encodings(image_bytes)

    if not image_array:
        raise ValueError("No faces found in the provided image")

    return image_array[0]

@app.post("/")
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
