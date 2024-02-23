from django.shortcuts import render
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .models import UploadedImage
from .serializers import UploadedImageSerializer
import cv2
from rest_framework.parsers import MultiPartParser
from io import BytesIO
import numpy as np
import base64
import os

@api_view(['POST'])
def detect_face(request):
    base64_image = request.data['image']
    img_data = base64.b64decode(base64_image)    

    # img_bytes = image_file.read()
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Convert the image to grayscale for face detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Load the face cascade classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=8)
    # Create a directory to store cropped images
    cropped_images_dir = 'cropped_images'
    if not os.path.exists(cropped_images_dir):
        os.makedirs(cropped_images_dir)
    # Crop the detected faces and save them
    cropped_images_paths = []
    cropped_images = [];
    for i, (x, y, w, h) in enumerate(faces):        
        x_inc = int(w*0.3)
        y_inc = int(h*0.3)
        sub_face = img[y-y_inc:y+h+y_inc+50, x-x_inc:x+w+x_inc]
        cropped_img = cv2.resize(sub_face,(int(224),int(224))) 
        cropped_img_path = os.path.join(cropped_images_dir, f'cropped_{i}.jpg')
        cv2.imwrite(cropped_img_path, cropped_img)
        cropped_images_paths.append(cropped_img_path)
        # encode
        _, img_encoded = cv2.imencode('.jpg', cropped_img);
        cropped_img_base64 = base64.b64encode(img_encoded).decode('utf-8')        
        cropped_images.append(cropped_img_base64)
    
    return Response({'cropped_images': cropped_images[0]}, status=status.HTTP_200_OK)
