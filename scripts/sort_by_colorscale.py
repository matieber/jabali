"""
CLASIFICACIÓN DE FRAMES EN COLOR E INFRARROJO (IR)

Este script clasifica las imágenes de un directorio en dos
categorías: fotografías a color e imágenes infrarrojas (IR).

Cada imagen se convierte del espacio BGR, utilizado por OpenCV,
al espacio HSV, que permite analizar por separado el tono,
la saturación y el valor (brillo).

Se calcula la saturación media de todos los píxeles de la
imagen. Este valor representa el nivel promedio de intensidad
cromática de sus colores.

Se compara la saturación media con un umbral establecido.
   - Si supera el umbral, la imagen se clasifica como color.
   - Si es igual o inferior al umbral, se clasifica como IR.
"""

from pathlib import Path
import re
import cv2
import numpy as np
from tqdm import tqdm

def natural_sort_key(path_obj):
    """
    Clave para ordenar nombres con números de forma natural (ej. frame2 antes de frame10)
    para conservar el orden dado por el directorio de donde se extrajeron los frames.
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', path_obj.name)]

def is_color_frame(image_path, saturation_threshold=12.0):
    """
    Retorna True si el frame es a color, False si es Infrarrojo.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return None

    # Conversión de BGR a HSV para analizar la saturación
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mean_saturation = np.mean(hsv[:, :, 1])
    
    return bool(mean_saturation > saturation_threshold)

def process_directory(input_source, out_color="color.out", out_ir="infrared.out", image_range=None, base_dir=None):
    """
    Clasifica las imágenes de un directorio en fotografías a color
    e imágenes infrarrojas (IR), generando archivos .out separados.

    **Parámetros**
        frames_dir (str | Path):
            Ruta del directorio que contiene las imágenes a clasificar.

        out_color (str | Path, opcional):
            Ruta del archivo de salida donde se guardarán los nombres
            de las imágenes clasificadas como fotografías a color.
            Por defecto: "color.out".

        out_ir (str | Path, opcional):
            Ruta del archivo de salida donde se guardarán los nombres
            de las imágenes clasificadas como infrarrojas (IR).
            Por defecto: "infrared.out".
    """
    source_path = Path(input_source)
    color_frames = []
    ir_frames = []

    if source_path.is_file():
        with open(source_path, "r", encoding="utf-8") as f:
            files = [line.strip() for line in f if line.strip()]
        if image_range is not None:
            files = files[image_range[0]:image_range[1]]
        
        target_dir = Path(base_dir) if base_dir else Path(".")
        for file_name in tqdm(files, desc="Clasificando desde .out", unit="frame"):
            file_path = target_dir / file_name
            is_color = is_color_frame(file_path)

            if is_color is True:
                color_frames.append(file_path.name)
            elif is_color is False:
                ir_frames.append(file_path.name)
    else:
        extension = ".jpg"
        files = sorted([f for f in source_path.iterdir() if f.suffix.lower() == extension], key=natural_sort_key)

        if image_range is not None:
            files = files[image_range[0]:image_range[1]]

        for file_path in tqdm(files, desc="Clasificando directorio", unit="frame"):
            is_color = is_color_frame(file_path)

            if is_color is True:
                color_frames.append(file_path.name)
            elif is_color is False:
                ir_frames.append(file_path.name)

    with open(out_color, "w", encoding="utf-8") as f:
        f.write("\n".join(color_frames) + ("\n" if color_frames else ""))

    with open(out_ir, "w", encoding="utf-8") as f:
        f.write("\n".join(ir_frames) + ("\n" if ir_frames else ""))

    print(f"  - Frames Color: {len(color_frames)} -> guardados en '{out_color}'")
    print(f"  - Frames IR:    {len(ir_frames)} -> guardados en '{out_ir}'")

process_directory(
    "data/input/jabali_frames",
    out_color="color.out",
    out_ir="infrared.out",
    image_range=(0, 10300)
)
process_directory("data/input/jabali_frames", out_color= "color.out", out_ir= "infrared.out")