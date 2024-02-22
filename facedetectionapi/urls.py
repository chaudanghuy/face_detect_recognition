# facedetectionapi/urls.py
from django.urls import path
from .views import detect_face

urlpatterns = [
    path('detect-face/', detect_face, name='detect_face'),
]
