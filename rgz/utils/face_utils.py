import cv2
import numpy as np
import torch
from PIL import Image

class FaceProcessor:
    @staticmethod
    def align_face(image, landmarks):
        """Выравнивание лица по глазам"""
        left_eye = landmarks[36:42].mean(axis=0)
        right_eye = landmarks[42:48].mean(axis=0)
        
        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dy, dx))
        
        eyes_center = ((left_eye[0] + right_eye[0]) // 2,
                      (left_eye[1] + right_eye[1]) // 2)
        
        M = cv2.getRotationMatrix2D(eyes_center, angle, 1)
        aligned = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
        
        return aligned

    @staticmethod
    def extract_face_embedding(face_image, model, device):
        """Извлечение эмбеддинга лица"""
        face = cv2.resize(face_image, (160, 160))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = np.transpose(face, (2, 0, 1))
        face = torch.FloatTensor(face).unsqueeze(0).to(device)
        face = (face - 127.5) / 128.0
        
        with torch.no_grad():
            embedding = model(face)
        
        return embedding.cpu().numpy().flatten()

    @staticmethod
    def compare_embeddings(embedding1, embedding2, threshold=0.6):
        """Сравнение эмбеддингов с использованием косинусного сходства"""
        embedding1 = embedding1 / np.linalg.norm(embedding1)
        embedding2 = embedding2 / np.linalg.norm(embedding2)
        
        similarity = np.dot(embedding1, embedding2)
        
        is_same = similarity > threshold
        
        return similarity, is_same