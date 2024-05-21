# Usar la imagen base oficial de Python
FROM python:3.9

# Establecer el directorio de trabajo en el contenedor
WORKDIR /app

# Copiar el contenido del directorio actual en el contenedor en /app
COPY . .

# Instalar las dependencias del sistema
RUN apt-get update && apt-get install -y libgl1-mesa-glx

# Instalar los paquetes necesarios especificados en requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Hacer el puerto 80 disponible para el mundo exterior
EXPOSE 80

# Definir una variable de entorno
ENV NAME World

# Ejecutar app.py cuando se inicie el contenedor
CMD ["python", "app.py"]
