# Clasificador de Paisajes con Redes Neuronales

Este repositorio contiene el proyecto final para la materia **Redes Neuronales Profundas**. El objetivo del proyecto fue desarrollar un ciclo de vida completo de un modelo de machine learning, desde la preparación de los datos y el entrenamiento de la red, hasta su despliegue en una aplicación web interactiva.

La aplicación web final permite a los usuarios subir imágenes de paisajes para que sean clasificadas automáticamente en una de seis categorías, utilizando un modelo de visión por computadora entrenado con PyTorch.

## 📜 Descripción del Proyecto

El problema abordado es la clasificación automática de imágenes de paisajes. La solución implementada es una aplicación web desarrollada con Streamlit que utiliza un modelo **ResNet-18** al que se le aplicó **fine-tuning** para esta tarea específica.

El modelo fue entrenado y evaluado en el dataset público "Intel Image Classification" de Kaggle. Se experimentó con tres arquitecturas diferentes (ResNet-18, VGG-16 y una CNN propia) para seleccionar el modelo con el mejor rendimiento y eficiencia.

## ✨ Características de la Aplicación

* **Clasificación en 6 categorías**: El modelo puede clasificar imágenes en `buildings`, `forest`, `glacier`, `mountain`, `sea` y `street`.
* **Carga Múltiple de Archivos**: Permite subir imágenes individuales (`.jpg`, `.jpeg`, `.png`) o un archivo `.zip` que contenga varias imágenes.
* **Galería de Resultados Visual**: Muestra las imágenes procesadas junto con la etiqueta predicha y el porcentaje de confianza del modelo.
* **Descarga Organizada**: Genera y ofrece para descargar un archivo `.zip` con todas las imágenes originales organizadas en subcarpetas según la categoría predicha.

## 🛠️ Tecnologías Utilizadas

* **Lenguaje**: Python 3.10
* **Deep Learning**: PyTorch, Torchvision
* **Aplicación Web**: Streamlit
* **Análisis y Manipulación de Datos**: Pandas, NumPy, Scikit-learn, Pillow
* **Visualización**: Matplotlib, Seaborn

## 📂 Estructura del Repositorio

El proyecto sigue la estructura definida en las pautas, separando el desarrollo de la producción:

* **`/data`**: Carpeta destinada a contener los datasets de entrenamiento y prueba.
* **`/dev`**: Contiene los Jupyter Notebooks con todo el proceso de experimentación, análisis y entrenamiento de los modelos.
* **`/prod`**: Contiene el código final de la aplicación Streamlit (`app.py`), las funciones de utilidad (`utils.py`), el modelo entrenado (`modelo.pth`) y el archivo de dependencias (`requirements.txt`).

## 🚀 Instalación y Ejecución Local

Para ejecutar la aplicación en tu máquina local, seguí estos pasos:

1.  **Clonar el repositorio**:
    ```bash
    git clone https://github.com/agustinlongarini/clasificador_paisajes.git
    cd clasificador_paisajes
    ```

2.  **Instalar las dependencias**:
    El archivo `requirements.txt` se encuentra en la carpeta `prod`.
    ```bash
    pip install -r prod/requirements.txt
    ```

3.  **Ejecutar la aplicación**:
    Asegurate de estar en la carpeta raíz del proyecto y ejecutá:
    ```bash
    streamlit run prod/app.py
    ```

## 💻 Cómo Usar la Aplicación

1.  Una vez ejecutado el comando anterior, se abrirá una pestaña en tu navegador web.
2.  Arrastrá y soltá imágenes o un archivo `.zip` en el área de carga, o hacé clic en "Browse files" para seleccionarlos.
3.  Hacé clic en el botón **"Clasificar"**.
4.  La aplicación procesará las imágenes y mostrará una galería con los resultados, indicando la clase predicha y la confianza para cada una.
5.  Finalmente, aparecerá un botón para **descargar un archivo `.zip`** con todas tus imágenes ya ordenadas en carpetas por categoría.

## 🤖 Sobre el Modelo Final

* **Arquitectura**: **ResNet-18** con fine-tuning.
* **Precisión Final (en test)**: **93.80%**.
* **Justificación**: Fue seleccionado por ofrecer el mejor equilibrio entre una alta precisión y una eficiencia computacional superior en comparación con otras arquitecturas evaluadas como VGG-16 y una CNN propia.

## 👥 Autores

* Agustín Longarini
* Ignacio Licciardi
