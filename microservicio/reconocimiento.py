import cv2
import os
import numpy as np

class ReconocimientoFacial:
    def __init__(self):
        # Directorio donde se almacenan los rostros
        self.ROSTROS_DIR = 'rostros'

        # Cargar el clasificador de reconocimiento facial
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Cargar el modelo de reconocimiento facial
        self.modelo = self.cargar_modelo()

    def cargar_modelo(self):
        modelo = cv2.face.LBPHFaceRecognizer_create()
        modelo.read('modelos/lbph_modelo.yml')
        return modelo

    def reconocer_rostros(self, imagen):
        # Convertir la imagen a escala de grises
        gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

        # Detectar rostros en la imagen
        rostros_detectados = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        recognized_faces = []

        for (x, y, w, h) in rostros_detectados:
            # Obtener el área de interés (ROI) de la cara
            roi_gray = gray[y:y+h, x:x+w]

            # Realizar la predicción utilizando el modelo
            id_, conf = self.modelo.predict(roi_gray)

            # Si la confianza es menor que 100, es un rostro reconocido
            if conf < 100:
                recognized_faces.append({'id': id_, 'confianza': round(100 - conf, 2)})

        return recognized_faces

# Ejemplo de uso
if __name__ == "__main__":
    # Cargar la imagen de prueba
    imagen_prueba = cv2.imread('imagen_prueba.jpg')

    # Inicializar el reconocimiento facial
    reconocimiento = ReconocimientoFacial()

    # Realizar el reconocimiento de rostros en la imagen de prueba
    rostros_reconocidos = reconocimiento.reconocer_rostros(imagen_prueba)

    print("Rostros reconocidos:", rostros_reconocidos)
