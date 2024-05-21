from flask import Flask, request, jsonify
import os
import numpy as np
import base64
import mysql.connector
import cv2

app = Flask(__name__)

# Configuración de la conexión a la base de datos
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'facial_recognition'
}

# Directorio donde se almacenarán los rostros
ROSTROS_DIR = 'rostros'

# Cargar el clasificador de reconocimiento facial
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Función para cargar el modelo de reconocimiento facial
def cargar_modelo():
    modelo = cv2.face.LBPHFaceRecognizer_create()
    modelo_path = 'modelos/lbph_modelo.yml'

    # Verificar si el archivo existe
    if not os.path.exists(modelo_path):
        raise FileNotFoundError(f"El archivo de modelo {modelo_path} no existe")
    
    # Intentar leer el archivo del modelo manualmente para verificar su contenido
    with open(modelo_path, 'r') as f:
        contenido = f.read()
        if not contenido:
            raise ValueError(f"El archivo de modelo {modelo_path} está vacío o corrupto")
    
    # Intentar cargar el modelo
    try:
        modelo.read(modelo_path)
    except cv2.error as e:
        raise cv2.error(f"Error al leer el archivo del modelo: {modelo_path}\n{e}")
    
    return modelo

def entrenar_modelo():
    rostros = []
    labels = []

    for root, dirs, files in os.walk(ROSTROS_DIR):
        for file in files:
            if file.endswith("jpg") or file.endswith("png"):
                path = os.path.join(root, file)
                try:
                    label = int(file.split("_")[0])  # Extraer la etiqueta del nombre del archivo
                    imagen = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    if imagen is not None:
                        rostros.append(imagen)
                        labels.append(label)
                    else:
                        print(f"Advertencia: No se pudo leer la imagen {path}")
                except ValueError:
                    print(f"Advertencia: No se pudo extraer una etiqueta válida del nombre del archivo {file}")
                except Exception as e:
                    print(f"Error inesperado con el archivo {file}: {e}")

    if len(rostros) > 0 and len(labels) > 0:
        print(f"Entrenando modelo con {len(rostros)} rostros y {len(labels)} etiquetas...")
        modelo = cv2.face.LBPHFaceRecognizer_create()
        modelo.train(rostros, np.array(labels))
        modelo_path = 'modelos/lbph_modelo.yml'
        if not os.path.exists('modelos'):
            os.makedirs('modelos')
        modelo.save(modelo_path)
        print(f"Modelo guardado en {modelo_path}")
    else:
        raise ValueError("No se encontraron rostros o etiquetas para entrenar el modelo")



# Función para guardar información del rostro en la base de datos y en el sistema de archivos
@app.route('/guardar_rostro', methods=['POST'])
def guardar_rostro():
    try:
        data = request.json
        nombre = data.get('nombre')
        imagen_base64 = data.get('imagen')

        if not nombre or not imagen_base64:
            return jsonify({'error': 'Nombre e imagen son campos requeridos'}), 400

        if imagen_base64.startswith('data:image/jpeg;base64,'):
            imagen_base64 = imagen_base64.split('base64,')[1]

        nparr = np.frombuffer(base64.b64decode(imagen_base64), np.uint8)
        imagen = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if imagen is None:
            return jsonify({'error': 'No se pudo decodificar la imagen'}), 400

        gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)

        rostros_detectados = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(rostros_detectados) == 0:
            return jsonify({'error': 'No se detectaron rostros en la imagen'}), 400

        with mysql.connector.connect(**db_config) as db_connection:
            cursor = db_connection.cursor()
            db_connection.start_transaction()

            sql_insert = "INSERT INTO rostros (nombre, imagen_base64) VALUES (%s, %s)"
            cursor.execute(sql_insert, (nombre, imagen_base64))
            rostro_id = cursor.lastrowid

            if not os.path.exists(ROSTROS_DIR):
                os.makedirs(ROSTROS_DIR)

            for i, (x, y, w, h) in enumerate(rostros_detectados):
                rostro = gray[y:y+h, x:x+w]
                rostro_path = os.path.join(ROSTROS_DIR, f'{rostro_id}_{i}.jpg')
                cv2.imwrite(rostro_path, rostro)
                print(f"Rostro guardado en {rostro_path}")

            db_connection.commit()

        entrenar_modelo()

        return jsonify({'message': 'Rostro guardado y modelo entrenado correctamente'}), 200

    except mysql.connector.Error as db_error:
        return jsonify({'error': f'Error en la base de datos: {db_error}'}), 500

    except Exception as e:
        return jsonify({'error': str(e)}), 500
 

def reconocer_rostro_modelo_entrenado(imagen):
    gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    modelo = cargar_modelo()

    rostros_detectados = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    resultados = []

    for (x, y, w, h) in rostros_detectados:
        rostro = gray[y:y+h, x:x+w]
        id_, confianza = modelo.predict(rostro)
        porcentaje_confianza = int((1 - (confianza / 400)) * 100)
        nombre_reconocido = consultar_nombre_por_id(id_)
        print(id_)
        if porcentaje_confianza > 85:
            resultados.append((nombre_reconocido, porcentaje_confianza, id_))
        else:
            resultados.append(("Persona no reconocida", porcentaje_confianza, id_))

    return resultados
def consultar_nombre_por_id(id_):
    # Establecer conexión a la base de datos y realizar la consulta
    try:
        with mysql.connector.connect(**db_config) as db_connection:
            cursor = db_connection.cursor()
            sql_query = "SELECT nombre FROM rostros WHERE id = %s"
            cursor.execute(sql_query, (id_,))
            resultado = cursor.fetchone()
            if resultado:
                return resultado[0]  # Devolver el primer elemento del resultado (nombre)
            else:
                return "Desconocido"  # Si no se encuentra el ID en la base de datos, devolver "Desconocido"
    except mysql.connector.Error as e:
        print(f"Error al consultar el nombre en la base de datos: {e}")
        return "Desconocido"
# Ruta para reconocer rostros
@app.route('/reconocer_rostro', methods=['POST'])
def reconocer_rostro():
    try:
        data = request.json
        imagen_base64 = data.get('imagen')

        # Validar si la imagen base64 comienza con 'data:image/jpeg;base64,'
        if imagen_base64.startswith('data:image/jpeg;base64,'):
            # Si comienza con 'data:image/jpeg;base64,', extraer solo los datos base64
            imagen_base64 = imagen_base64.split('base64,')[1]

        # Convertir la imagen base64 a matriz OpenCV
        nparr = np.frombuffer(base64.b64decode(imagen_base64), np.uint8)
        imagen = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Validar si la imagen se decodificó correctamente
        if imagen is None:
            return jsonify({'error': 'No se pudo decodificar la imagen'}), 400

        # Reconocer rostros utilizando el modelo entrenado
        id_reconocido = reconocer_rostro_modelo_entrenado(imagen)

        return jsonify({'recognized_face_id': id_reconocido}), 200
    except FileNotFoundError as fnf_error:
        return jsonify({'error': str(fnf_error)}), 500
    except cv2.error as cv_error:
        return jsonify({'error': f'Error al leer el archivo del modelo: {cv_error}'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Configuración de CORS
@app.after_request
def allow_origin(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

if __name__ == '__main__':
    app.run(debug=True)
