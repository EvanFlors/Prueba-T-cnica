# Proyecto de Clasificación de Noticias

Este proyecto clasifica titulares de noticias en diferentes categorías usando un modelo de machine learning entrenado con texto y características adicionales.

## Requisitos

Este proyecto está desarrollado y probado con **Python 3.11.12**.  
Se recomienda usar esta versión para evitar problemas de compatibilidad.

## Instalación

### 1. Crear y activar entorno virtual

Se recomienda crear un entorno virtual para aislar las dependencias del proyecto.

```bash
# Crear entorno virtual (puedes cambiar el nombre 'venv' por el que prefieras)
python3 -m venv venv

# PowerShell
.\venv\Scripts\Activate.ps1

# o en CMD
.\venv\Scripts\activate.bat
```

### 2. Instalar dependencias

pip install -r requirements.txt

### 3. Correr pipeline de entrenamiento

python train.py

### 4. Ejecutar API de Flask

flask run

## Estructura del proyecto

📁 src/
├── components/
├── pipeline/
├── utils.py
├── logger.py
├── exception.py
├── train.py

📁 templates/
    └── home.html
    └── predict.html

📁 static/
    └── imagen-1/

📁 artifacts/
    └── model.pkl
    └── bow_vectorizer.pkl
    └── scaler.pkl
    └── label_encoder.pkl

requirements.txt
app.py
README.md