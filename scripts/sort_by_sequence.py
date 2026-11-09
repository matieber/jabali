"""
AGRUPACIÓN DE FRAMES EN SECUENCIAS VISUALES

Este script agrupa imágenes consecutivas en secuencias según su
similitud visual, con el objetivo de identificar fotografías que
pertenecen a una misma escena, tolerando pequeños desplazamientos
de cámara y cambios moderados en la posición de los animales.

El proceso de comparación utiliza tres técnicas:

1. ORB (Oriented FAST and Rotated BRIEF):
   Detecta características visuales locales en ambas imágenes
   y genera descriptores que permiten identificar puntos
   correspondientes, incluso ante pequeños cambios de posición,
   escala o rotación.

2. RANSAC (Random Sample Consensus):
   Utiliza las correspondencias encontradas por ORB para estimar
   una transformación geométrica entre las imágenes mediante
   una homografía. Permite identificar las correspondencias
   geométricamente consistentes y descartar aquellas
   que no se ajustan a la transformación estimada.

3. SSIM (Structural Similarity Index):
   Una vez estimada la transformación, se alinea la segunda
   imagen con la primera. Posteriormente, se calcula la similitud
   estructural entre ambas, considerando únicamente la región
   válida de superposición.

La similitud final combina dos componentes:
    - Proporción de correspondencias consistentes.
    - Similitud estructural de las imágenes alineadas.

El resultado se calcula mediante una combinación ponderada:
    65 % de consistencia geométrica y 35 % de similitud estructural.
"""

from skimage.metrics import structural_similarity
from pathlib import Path
import re
import cv2
import numpy as np
from tqdm import tqdm

def load_grayscale_robust(image_path, base_dir=None):
    """
    Carga de imágenes concatenando la carpeta base si se proporciona.
    """
    try:
        full_path = Path(base_dir) / image_path if base_dir else Path(image_path)
        abs_path = full_path.resolve()

        img_array = np.fromfile(str(abs_path), dtype=np.uint8)
        if img_array.size == 0:
            return None
        return cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
    except Exception:
        return None

def calculate_histogram_similarity(img1_path, img2_path, base_dir=None):
    """
    Compara dos imágenes utilizando:
    - ORB para detectar características locales.
    - RANSAC para encontrar correspondencias geométricas.
    - SSIM para comparar las imágenes una vez alineadas.

    Devuelve un valor de similitud entre 0 y 1.
    """

    img1 = load_grayscale_robust(img1_path, base_dir)
    img2 = load_grayscale_robust(img2_path, base_dir)

    if img1 is None or img2 is None:
        return None

    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    # Detectar características y descriptores ORB
    orb = cv2.ORB_create(nfeatures=1500)

    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        return 0.0

    # Buscar correspondencias entre descriptores
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    matches = matcher.knnMatch(des1, des2, k=2)

    # Filtrar coincidencias mediante Lowe's ratio test
    good_matches = [
        m for pair in matches
        if len(pair) == 2
        for m, n in [pair]
        if m.distance < 0.75 * n.distance
    ]

    if len(good_matches) < 8:
        return 0.0

    # Obtener coordenadas de los puntos correspondientes
    src_pts = np.float32([
        kp2[m.trainIdx].pt for m in good_matches
    ]).reshape(-1, 1, 2)

    dst_pts = np.float32([
        kp1[m.queryIdx].pt for m in good_matches
    ]).reshape(-1, 1, 2)

    # Estimar transformación geométrica mediante RANSAC
    H, mask = cv2.findHomography(
        src_pts,
        dst_pts,
        cv2.RANSAC,
        5.0
    )

    if H is None or mask is None:
        return 0.0

    inliers = int(mask.sum())
    inlier_ratio = inliers / len(good_matches)

    if inliers < 8:
        return 0.0

    # Alinear la segunda imagen con la primera
    h, w = img1.shape

    aligned = cv2.warpPerspective(
        img2,
        H,
        (w, h)
    )

    # Identificar la región válida de superposición
    valid_mask = cv2.warpPerspective(
        np.ones(img2.shape, dtype=np.uint8) * 255,
        H,
        (w, h)
    )

    # Excluir bordes de la región de comparación
    valid_mask = cv2.erode(
        valid_mask,
        np.ones((11, 11), dtype=np.uint8)
    )

    valid_pixels = valid_mask > 0

    if np.mean(valid_pixels) < 0.30:
        return 0.0

    # Comparar estructura después de la alineación
    _, ssim_map = structural_similarity(
        img1,
        aligned,
        data_range=255,
        full=True
    )

    aligned_ssim = float(np.mean(ssim_map[valid_pixels]))
    aligned_ssim = np.clip(aligned_ssim, 0, 1)

    # Combinar consistencia geométrica y similitud estructural
    similarity = (
        0.65 * inlier_ratio +
        0.35 * aligned_ssim
    )

    return float(similarity)


def group_sequences(frames_dir, input_list=None, output_file="ir_sequences.out", threshold=0.95, image_range=None):
    """
    Agrupa frames consecutivos en secuencias según su similitud visual.
    
    **Parámetros**
        frames_dir (str | Path):
            Ruta obligatoria del directorio base donde se encuentran las imágenes.
            Se utiliza para construir la ruta completa de cada frame.

        input_list (str | Path, opcional):
            Ruta de un archivo .out previo que contiene los nombres o rutas
            de los frames a procesar. Si no se proporciona, se escanean
            directamente las imágenes del directorio base.
            Por defecto: None.

        output_file (str | Path, opcional):
            Ruta del archivo de salida donde se guardarán las
            secuencias detectadas.
            Por defecto: "ir_sequences.out".

        threshold (float, opcional):
            Umbral mínimo de similitud para mantener dos frames
            consecutivos dentro de una misma secuencia.
            Si la similitud es inferior a este valor, se inicia
            una nueva secuencia.
            Su valor debe calibrarse según las características
            de las imágenes y el nivel de variación visual
            que se desea tolerar. Por defecto: 0.4.

        image_range (tuple, opcional):
            Rango numérico de imágenes a procesar.
            Por defecto: None.
    """
    base_dir = Path(frames_dir)
    lines = []

    if input_list is not None and Path(input_list).is_file():
        with open(input_list, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
    else:
        extension = ".jpg"
        files = sorted([f.name for f in base_dir.iterdir() if f.suffix.lower() == extension], 
                       key=lambda x: [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', x)])
        lines = files

    if image_range is not None:
        lines = lines[image_range[0]:image_range[1]]

    if not lines:
        print(f"El directorio o lista está vacío o el rango no contiene elementos.")
        return

    sequences = []
    seq_start = lines[0]
    prev_frame = lines[0]
    similarities = []

    for i in tqdm(range(1, len(lines)), desc="Comparando secuencias", unit="par"):
        curr_frame = lines[i]
        
        sim = calculate_histogram_similarity(prev_frame, curr_frame, base_dir=base_dir)
        similarities.append(sim)

        if sim is None:
            print(f"Advertencia: No se pudo cargar '{prev_frame}' o '{curr_frame}' dentro de '{base_dir}'")
            prev_frame = curr_frame
            continue

        if sim < threshold:
            sequences.append(f"{seq_start} - {prev_frame}")
            seq_start = curr_frame
        
        prev_frame = curr_frame

    sequences.append(f"{seq_start} - {prev_frame}")

    with open(output_file, "w", encoding="utf-8") as f:
        for seq in sequences:
            f.write(f"{seq}\n")

    print(f"  - Secuencias detectadas: {len(sequences)}")
    print(f"  - Resultado guardado en: '{output_file}'")

group_sequences(
    frames_dir="data/input/jabali_frames",
    input_list="infrared.out",
    output_file="ir_sequences.out",
    threshold=0.4,
    image_range=None
)
group_sequences(
    frames_dir="data/input/jabali_frames",
    input_list="color.out",
    output_file="color_sequences.out",
    threshold=0.4,
    image_range=None
)