import os
import sys
import shutil
import subprocess
import cv2
from pathlib import Path

# ==============================================================================
# CONFIGURACIÓN 
# ==============================================================================
REMOTE_FOLDER = "onedrive:jabali" # Modificar con el directorio correspondiente
LOCAL_TEMP_DIR = Path("./rclone_local")
FRAMES_DIR = Path("./temp_frames")
RESULTS_DIR = Path("./result") # Salida de resultados

LOCAL_TEMP_DIR.mkdir(exist_ok=True, parents=True)
FRAMES_DIR.mkdir(exist_ok=True, parents=True)
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

# MODO DE EJECUCIÓN:
# True  -> Ejecuta speciesnet (que incluye MegaDetector integrado)
# False -> Ejecuta megadetector.detection.run_md_and_speciesnet 
MEGADETECTOR = True

# Ruta al archivo JSON preexistente generado previamente por MegaDetector
PREEXISTING_MD_JSON = "detection_results.json"

RCLONE_CMD = shutil.which("rclone") or "rclone"
PYTHON_EXEC = sys.executable

# ==============================================================================
# FUNCIONES AUXILIARES
# ==============================================================================
def extract_frames(video_path, output_folder, fps_rate=1):
    """ Extrae fotogramas de video. """
    cap = cv2.VideoCapture(str(video_path.resolve()))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    hop = int(video_fps / fps_rate) if int(video_fps / fps_rate) > 0 else 1

    count = 0
    saved_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % hop == 0:
            frame_filename = output_folder / f"frame_{saved_count:04d}.jpg"
            cv2.imwrite(str(frame_filename.resolve()), frame)
            saved_count += 1
        count += 1
        
    cap.release()
    return saved_count

# ==============================================================================
# PIPELINE
# ==============================================================================
print(f"Sistema Operativo detectado: {os.name} ({sys.platform})")
print("Buscando videos en OneDrive...")

result = subprocess.run(
    [RCLONE_CMD, "lsf", REMOTE_FOLDER, "--include", "*.MP4"], 
    capture_output=True,
    text=True,
    check=True
)

file_list = [f.strip() for f in result.stdout.splitlines() if f.strip()]
print(f"Se encontraron {len(file_list)} videos para procesar.")

for video_name in file_list:
    print(f"\n--- Procesando: {video_name} ---")
    local_video_path = LOCAL_TEMP_DIR / video_name

    # Descarga desde OneDrive
    print("Descargando video...")
    subprocess.run([
        RCLONE_CMD, "copyto", 
        f"{REMOTE_FOLDER}/{video_name}", 
        str(local_video_path.resolve())
    ], check=True)

    # Carpeta temporal de frames
    if FRAMES_DIR.exists():
        shutil.rmtree(FRAMES_DIR)
    FRAMES_DIR.mkdir(exist_ok=True, parents=True)

    print("Extrayendo frames del video...")
    num_frames = extract_frames(local_video_path, FRAMES_DIR, fps_rate=1)
    print(f"Extraídos {num_frames} frames en {FRAMES_DIR}.")

    # Archivo de salida
    json_output_path = RESULTS_DIR / f"salida_{Path(video_name).stem}.json"

    # Comando según la opción seleccionada
    if MEGADETECTOR:
        # OPCIÓN 1: SpeciesNet con MegaDetector
        print("Ejecutando MD + SpeciesNet...")
        cmd = [
            PYTHON_EXEC,
            "-m",
            "speciesnet.scripts.run_model",
            "--folders", str(FRAMES_DIR.resolve()),
            "--predictions_json", str(json_output_path.resolve()),
            "--country", "ARG"
        ]
    else:
        # OPCIÓN 2: SpeciesNet a partir de JSON preexistente de detecciones
        print("Ejecutando peciesNet...")
        cmd = [
            PYTHON_EXEC,
            "-m",
            "megadetector.detection.run_md_and_speciesnet",
            "--detections_file", str(Path(PREEXISTING_MD_JSON).resolve()),
            str(FRAMES_DIR.resolve()),
            str(json_output_path.resolve()),
            "--country", "ARG"
        ]

    # Invocación de la inferencia
    subprocess.run(cmd, check=True)
    
    # Limpieza de temporales
    if local_video_path.exists():
        os.remove(local_video_path)
    if FRAMES_DIR.exists():
        shutil.rmtree(FRAMES_DIR)
    print("Archivos temporales locales eliminados.")

print("\nProcesamiento finalizado.")