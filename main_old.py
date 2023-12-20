from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
import face_recognition
from PIL import Image
import numpy as np
from io import BytesIO
import boto3
from aws_credentials import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION  # Import your credentials

app = FastAPI()

S3_BUCKET_NAME = 'gwap-development-storage'
S3_FOLDER_NAME = 'verification-images'

def find_face_encodings(image_name):
    # Initialize the S3 client
    s3 = boto3.client('s3', region_name=AWS_DEFAULT_REGION,
                      aws_access_key_id=AWS_ACCESS_KEY_ID,
                      aws_secret_access_key=AWS_SECRET_ACCESS_KEY)

    # Construct the S3 object URL
    s3_object_url = f's3://{S3_BUCKET_NAME}/{S3_FOLDER_NAME}/{image_name}'

    # Download the image from S3
    s3_response = s3.get_object(Bucket=S3_BUCKET_NAME, Key=f'{S3_FOLDER_NAME}/{image_name}')
    image_bytes = s3_response['Body'].read()

    # Open the downloaded image with Pillow (PIL)
    image = Image.open(BytesIO(image_bytes))

    # Convert the Pillow image to a numpy array for face_recognition
    image_array = np.array(image)

    # Get face encodings from the image
    face_enc = face_recognition.face_encodings(image_array)

    if not face_enc:
        raise ValueError("No faces found in the provided image")

    return face_enc[0]

@app.get("/")
async def hello_world():
    return "Hello, World"

@app.post("/compare")
async def compare_images(
    id_card_photo: str = Form(..., description="Id Card Photo.."),
    recent_camera_photo: str = Form(..., description="Recent Camera Photo.."),
):
    try:
        # Process the images from the provided image names
        image_1 = find_face_encodings(id_card_photo)
        image_2 = find_face_encodings(recent_camera_photo)

        is_same = face_recognition.compare_faces([image_1], image_2)[0]

        if is_same:
            distance = face_recognition.face_distance([image_1], image_2)
            distance = round(distance[0] * 100)
            accuracy = 100 - round(distance)

            response = {
                "result": "The images are the same",
                "accuracy": accuracy,
                "id_card_photo": id_card_photo,  
                "recent_camera_photo": recent_camera_photo, 
            }
        else:
            response = {"result": "The images are not the same"}

        return JSONResponse(content=jsonable_encoder(response))
    except Exception as e:
        error_message = {"error": str(e)}
        return JSONResponse(content=jsonable_encoder(error_message), status_code=400)
