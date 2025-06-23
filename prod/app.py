import io
import os
import tempfile
import zipfile
from pathlib import Path

import streamlit as st
from PIL import Image, ImageOps
import requests

# Tamaño usado para mostrar las miniaturas en la interfaz
DISPLAY_SIZE = (256, 256)

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


def classify_images(files, urls=None):
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
                    label, confidence, probs = utils.predict(model, device, tensor)

                    save_path = os.path.join(tmpdir, label, original_filename)
                    img.save(save_path)

                    display_img = ImageOps.fit(img, DISPLAY_SIZE)
                    results.append({
                        'name': original_filename,
                        'label': label,
                        'confidence': confidence,
                        'probs': probs,
                        'image': display_img,
                    })
                
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
                                    label, confidence, probs = utils.predict(model, device, tensor)
                                    
                                    base_name = os.path.basename(name)
                                    save_path = os.path.join(tmpdir, label, base_name)
                                    img.save(save_path)

                                    display_img = ImageOps.fit(img, DISPLAY_SIZE)
                                    results.append({
                                        'name': base_name,
                                        'label': label,
                                        'confidence': confidence,
                                        'probs': probs,
                                        'image': display_img,
                                    })
                                except Exception as e:
                                    errors.append(f"Error procesando {name} en el ZIP: {str(e)}")
            except Exception as e:
                errors.append(f"Error general procesando el archivo {original_filename}: {str(e)}")

        # Procesar URLs si se proporcionan
        if urls:
            for url in urls:
                try:
                    resp = requests.get(url, timeout=10)
                    resp.raise_for_status()
                    data = resp.content
                    if not is_valid_image(data):
                        errors.append(f"URL no contiene una imagen válida: {url}")
                        continue
                    img = Image.open(io.BytesIO(data)).convert("RGB")
                    tensor = utils.preprocess_image(img)
                    label, confidence, probs = utils.predict(model, device, tensor)

                    base_name = os.path.basename(url.split("?")[0]) or "imagen_url.jpg"
                    save_path = os.path.join(tmpdir, label, base_name)
                    img.save(save_path)

                    display_img = ImageOps.fit(img, DISPLAY_SIZE)
                    results.append({
                        'name': base_name,
                        'label': label,
                        'confidence': confidence,
                        'probs': probs,
                        'image': display_img,
                    })
                except Exception as e:
                    errors.append(f"Error procesando URL {url}: {str(e)}")

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

    url_input = st.text_area(
        "Enlaces de imágenes (uno por línea)", placeholder="https://example.com/imagen.jpg"
    )
    urls = [u.strip() for u in url_input.splitlines() if u.strip()]

    if (uploaded_files or urls) and st.button("Clasificar"):
        with st.spinner("Procesando..."):
            results, zip_buffer, errors = classify_images(uploaded_files, urls)
        
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
                probs = result.get('probs', {})

                col.image(result['image'], use_container_width=True)
                confidence_percent = confidence_score * 100

                # Mostrar la predicción principal con estilo destacado
                col.markdown(f"Predicción: **{predicted_label.capitalize()}**")
                
                col.markdown("<hr style='margin-top: 0.5rem; margin-bottom: 0.1rem;'>", unsafe_allow_html=True)
                
                # Mostrar la confianza debajo de la línea
                col.markdown(f"**Confianza: {confidence_percent:.1f}%**")
                
                # Barra de progreso visual para la confianza
                col.progress(confidence_percent / 100)
                
                # Espacio adicional antes del expander
                col.markdown("")

                # Expander para ver todas las probabilidades
                with col.expander("Ver detalles"):
                    for cls, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
                        prob_percent = prob * 100
                        # Crear una barra de progreso compacta usando HTML y CSS
                        if cls == predicted_label:
                            st.markdown(
                                f"""
                                <div style="margin-bottom: 8px;">
                                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 2px;">
                                        <span style="font-weight: bold; color: #1f77b4;">{cls}</span>
                                        <span style="font-weight: bold; color: #1f77b4;">{prob_percent:.1f}%</span>
                                    </div>
                                    <div style="background-color: #e0e0e0; border-radius: 10px; height: 8px; overflow: hidden;">
                                        <div style="background-color: #1f77b4; height: 100%; width: {prob_percent}%; border-radius: 10px;"></div>
                                    </div>
                                </div>
                                """, 
                                unsafe_allow_html=True
                            )
                        else:
                            st.markdown(
                                f"""
                                <div style="margin-bottom: 6px;">
                                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 2px;">
                                        <span style="color: #666;">{cls}</span>
                                        <span style="color: #666; font-size: 0.9em;">{prob_percent:.1f}%</span>
                                    </div>
                                    <div style="background-color: #f0f0f0; border-radius: 6px; height: 4px; overflow: hidden;">
                                        <div style="background-color: #ccc; height: 100%; width: {prob_percent}%; border-radius: 6px;"></div>
                                    </div>
                                </div>
                                """, 
                                unsafe_allow_html=True
                            )

        else:
            st.error("No se pudieron procesar imágenes válidas.")


if __name__ == "__main__":
    main()