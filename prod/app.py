import io
import os
import tempfile
import zipfile
from pathlib import Path

import streamlit as st
from PIL import Image

import utils


def is_valid_image(data):
    """Verifica si los datos representan una imagen válida."""
    try:
        # Intentar abrir la imagen
        img = Image.open(io.BytesIO(data))
        
        # Verificar el formato
        if img.format not in ['JPEG', 'PNG', 'JPG']:
            return False
            
        # Verificar dimensiones
        if img.size[0] == 0 or img.size[1] == 0:
            return False
            
        # Intentar verificar que se puede cargar completamente
        img_copy = Image.open(io.BytesIO(data))
        img_copy.verify()
        return True
    except Exception:
        return False


@st.cache_resource
def _load_model():
    model_path = Path(__file__).resolve().parent / "modelo.pth"
    return utils.load_model(str(model_path))


def classify_images(files):
    model, device = _load_model()
    results = []
    errors = []
    
    with tempfile.TemporaryDirectory() as tmpdir:
        for cls in utils.CLASSES:
            os.makedirs(os.path.join(tmpdir, cls), exist_ok=True)

        for file in files:
            original_filename = file.name
            try:
                # Procesar imágenes individuales
                if not original_filename.lower().endswith(".zip"):
                    data = file.getvalue()
                    if not is_valid_image(data):
                        errors.append(f"Archivo no es una imagen válida: {original_filename}")
                        continue
                    img = Image.open(io.BytesIO(data)).convert("RGB")
                    tensor = utils.preprocess_image(img)
                    label, confidence = utils.predict(model, device, tensor)
                    
                    save_path = os.path.join(tmpdir, label, original_filename)
                    img.save(save_path)
                    
                    results.append({'name': original_filename, 'label': label, 'confidence': confidence, 'image': img})
                
                # Procesar archivos ZIP
                else:
                    with zipfile.ZipFile(file) as zf:
                        for name in zf.namelist():
                            if name.endswith('/') or name.startswith('.') or '/.DS_Store' in name:
                                continue
                            if name.lower().endswith((".jpg", ".jpeg", ".png")):
                                try:
                                    data = zf.read(name)
                                    if not is_valid_image(data):
                                        errors.append(f"Archivo no es una imagen válida dentro del ZIP: {name}")
                                        continue
                                    img = Image.open(io.BytesIO(data)).convert("RGB")
                                    tensor = utils.preprocess_image(img)
                                    label, confidence = utils.predict(model, device, tensor)
                                    
                                    base_name = os.path.basename(name)
                                    save_path = os.path.join(tmpdir, label, base_name)
                                    img.save(save_path)
                                    
                                    results.append({'name': base_name, 'label': label, 'confidence': confidence, 'image': img})
                                except Exception as e:
                                    errors.append(f"Error procesando {name} en el ZIP: {str(e)}")
            except Exception as e:
                errors.append(f"Error general procesando el archivo {original_filename}: {str(e)}")

        # Comprimir resultados en zip
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zipf:
            for root, _, files_ in os.walk(tmpdir):
                for fname in files_:
                    fpath = os.path.join(root, fname)
                    arcname = os.path.relpath(fpath, tmpdir)
                    zipf.write(fpath, arcname)
        zip_buffer.seek(0)
        
    return results, zip_buffer, errors


def main():
    st.set_page_config(page_title="Clasificador de Paisajes")
    st.title("Clasificador de Paisajes")
    st.write(
        "Sube imágenes sueltas o un archivo .zip. Las imágenes serán clasificadas en las carpetas correspondientes y podrás descargar un zip con el resultado."
    )

    uploaded_files = st.file_uploader(
        "Selecciona archivos", type=["jpg", "jpeg", "png", "zip"], accept_multiple_files=True
    )

    if uploaded_files and st.button("Clasificar"):
        with st.spinner("Procesando..."):
            results, zip_buffer, errors = classify_images(uploaded_files)
        
        if errors:
            st.warning("Se encontraron algunos errores durante el procesamiento:")
            for error in errors:
                st.write(f"⚠️ {error}")
        
        if results:
            st.success(f"Clasificación completada. Se procesaron {len(results)} imágenes exitosamente.")
            st.download_button(
                "Descargar imágenes ordenadas (.zip)", zip_buffer, "imagenes_clasificadas.zip", "application/zip"
            )

            st.header("Galería de Resultados")
            
            # Crear columnas para la galería.
            num_cols = 4
            cols = st.columns(num_cols)
            
            for i, result in enumerate(results):
                col = cols[i % num_cols]
                
                # Extraemos la información del diccionario de resultados
                image_to_show = result['image']
                predicted_label = result['label']
                confidence_score = result['confidence']
                
                col.image(image_to_show, caption=f"{result['name']}", use_container_width=True)
                col.write(f"**Predicción**: {predicted_label}")
                confidence_percent = confidence_score * 100
                col.progress(int(confidence_percent), text=f"Nivel de confianza: {confidence_percent:.2f}%")

        else:
            st.error("No se pudieron procesar imágenes válidas.")


if __name__ == "__main__":
    main()