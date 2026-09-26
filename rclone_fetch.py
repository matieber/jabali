import os
import sys
import shutil
import subprocess
import argparse
import cv2
from pathlib import Path

# ==============================================================================
# CONFIGURACIÓN
# ==============================================================================

REMOTE_FOLDER = (
    "pps:FotosCamarasTrampas/SL001/20240905/DCIM/100_BTCF"
)

LOCAL_TEMP_DIR = Path("./rclone_local")
FRAMES_DIR = Path("./temp_frames")
RESULTS_DIR = Path("./result")

PREEXISTING_MD_JSON = "detection_results.json"

# Extensiones aceptadas
VIDEO_EXTS = {".mp4", ".avi", ".mov"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

# Ejecutable de rclone
RCLONE_CMD = shutil.which("rclone")

# Python que está ejecutando este script
PYTHON_EXEC = sys.executable

# ==============================================================================
# PARSER DE ARGUMENTOS
# ==============================================================================

parser = argparse.ArgumentParser(
    description=(
        "Pipeline de procesamiento de cámaras trampa "
        "(Rclone + MegaDetector / SpeciesNet)."
    )
)

parser.add_argument(
    "--md",
    action="store_true",
    help=(
        "Ejecuta MegaDetector + SpeciesNet desde cero. "
        "Si no se especifica, utiliza detecciones de MegaDetector existentes."
    ),
)

parser.add_argument(
    "--frames",
    action="store_true",
    help=(
        "Descarga las imágenes de la carpeta remota en lote "
        "y genera un único JSON."
    ),
)

args = parser.parse_args()

MEGADETECTOR = args.md
ONLY_FRAMES = args.frames

# ==============================================================================
# PREPARACIÓN DEL ENTORNO
# ==============================================================================

def prepare_directories():
    """Crea los directorios necesarios para el pipeline."""
    LOCAL_TEMP_DIR.mkdir(exist_ok=True, parents=True)
    FRAMES_DIR.mkdir(exist_ok=True, parents=True)
    RESULTS_DIR.mkdir(exist_ok=True, parents=True)

def check_dependencies():
    """Verifica que las herramientas necesarias estén disponibles."""
    if RCLONE_CMD is None:
        raise RuntimeError(
            "No se encontró 'rclone' en el PATH del sistema."
        )
    md_json = Path(PREEXISTING_MD_JSON)
    if not MEGADETECTOR and not md_json.exists():
        raise FileNotFoundError(
            f"No existe el JSON de MegaDetector: "
            f"{md_json.resolve()}"
        )

# ==============================================================================
# FUNCIONES AUXILIARES
# ==============================================================================

def reset_directory(directory: Path):
    """
    Elimina completamente un directorio y lo vuelve a crear.
    """
    if directory.exists():
        shutil.rmtree(directory)

    directory.mkdir(exist_ok=True, parents=True)

def clean_directory(directory: Path):
    """Elimina un directorio si existe."""
    if directory.exists():
        shutil.rmtree(directory)

def extract_frames(
    video_path: Path,
    output_folder: Path,
    frame_stride: int = 1,
):
    """
    Extrae frames de un video.
    frame_stride = 1:
        procesa todos los frames.
    frame_stride = 5:
        guarda 1 de cada 5 frames.
    """
    cap = cv2.VideoCapture(str(video_path.resolve()))

    if not cap.isOpened():
        raise RuntimeError(
            f"No se pudo abrir el video: {video_path}"
        )

    saved_count = 0
    frame_index = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_index % frame_stride == 0:
            frame_filename = (
                output_folder /
                f"frame_{saved_count:06d}.jpg"
            )
            success = cv2.imwrite(
                str(frame_filename.resolve()),
                frame
            )
            if not success:
                raise RuntimeError(
                    f"No se pudo guardar el frame: "
                    f"{frame_filename}"
                )
            saved_count += 1
        frame_index += 1
    cap.release()
    return saved_count

def run_inference(
    frames_path: Path,
    output_json_path: Path,
    use_md: bool,
):
    """ Ejecuta la inferencia sobre una carpeta de frames. """
    if use_md:
        print("\nEjecutando MegaDetector + SpeciesNet...")
        cmd = [
            PYTHON_EXEC,
            "-m",
            "speciesnet.scripts.run_model",
            "--folders",
            str(frames_path.resolve()),
            "--predictions_json",
            str(output_json_path.resolve()),
            "--country",
            "ARG",
        ]
    else:
        print("\nEjecutando SpeciesNet usando detecciones existentes...")
        md_json = Path(PREEXISTING_MD_JSON)
        cmd = [
            PYTHON_EXEC,
            "-m",
            "megadetector.detection.run_md_and_speciesnet",
            "--detections_file",
            str(md_json.resolve()),
            str(frames_path.resolve()),
            str(output_json_path.resolve()),
            "--country",
            "ARG",
        ]

    print("Comando:")
    print(" ".join(cmd))

    subprocess.run(
        cmd,
        check=True,
    )

# ==============================================================================
# PROCESAMIENTO DEL MODO --FRAMES
# ==============================================================================

def process_images():
    """
    Descarga todas las imágenes seleccionadas del directorio remoto, 
    ejecuta la inferencia y genera un único JSON.
    """
    print("\n" + "=" * 70)
    print("MODO: PROCESAMIENTO DE IMÁGENES")
    print("=" * 70)
    print("\nLimpiando carpeta temporal de frames...")
    reset_directory(FRAMES_DIR)
    print("Descargando imágenes desde el remoto...")

    subprocess.run(
        [
            RCLONE_CMD,
            "copy",
            REMOTE_FOLDER,
            str(FRAMES_DIR.resolve()),
            "--include",
            "*.{jpg,jpeg,png}",
            "--ignore-case",
        ],
        check=True,
    )
    downloaded_images = [
        file
        for file in FRAMES_DIR.iterdir()
        if file.is_file()
        and file.suffix.lower() in IMAGE_EXTS
    ]
    if not downloaded_images:
        print(
            "\nNo se encontraron imágenes "
            "en el directorio remoto."
        )
        clean_directory(FRAMES_DIR)
        return

    print(
        f"\nSe descargaron "
        f"{len(downloaded_images)} imágenes."
    )
    print("Ejecutando inferencia...")
    folder_name = Path(REMOTE_FOLDER).name

    json_output_path = (
        RESULTS_DIR /
        f"salida_{folder_name}.json"
    )
    run_inference(
        FRAMES_DIR,
        json_output_path,
        MEGADETECTOR,
    )
    print(
        f"\nResultados guardados en:\n"
        f"{json_output_path.resolve()}"
    )
    clean_directory(FRAMES_DIR)

# ==============================================================================
# PROCESAMIENTO DEL MODO VIDEOS
# ==============================================================================

def find_remote_videos():
    """ Busca videos en el directorio remoto. """
    print("\nBuscando videos en el remoto...")

    result = subprocess.run(
        [
            RCLONE_CMD,
            "lsf",
            REMOTE_FOLDER,
            "-R",
            "--include",
            "*.{mp4,avi,mov}",
            "--ignore-case",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    file_list = [
        file.strip()
        for file in result.stdout.splitlines()
        if file.strip()
        and not file.endswith("/")
    ]
    return [
        file
        for file in file_list
        if Path(file).suffix.lower() in VIDEO_EXTS
    ]

def process_videos():
    """
    Busca videos en el remoto, descarga uno por uno,
    extrae sus frames, ejecuta la inferencia y elimina
    los archivos temporales.
    """

    print("\n" + "=" * 70)
    print("MODO: PROCESAMIENTO DE VIDEOS")
    print("=" * 70)

    file_list = find_remote_videos()
    print(
        f"\nSe encontraron "
        f"{len(file_list)} videos para procesar."
    )
    if not file_list:
        print("No se encontraron videos.")
        return
    for video_number, remote_file in enumerate(
        file_list,
        start=1,
    ):
        file_path = Path(remote_file)

        safe_name = (
            str(file_path.with_suffix(""))
            .replace("/", "_")
            .replace("\\", "_")
        )
        json_output_path = (
            RESULTS_DIR /
            f"salida_{safe_name}.json"
        )
        print("\n" + "-" * 70)
        print(
            f"Video {video_number}/{len(file_list)}"
        )
        print(f"Procesando: {remote_file}")
        print("-" * 70)
        reset_directory(FRAMES_DIR)
        local_video_path = (
            LOCAL_TEMP_DIR /
            file_path.name
        )
        try:
            # DESCARGA
            print("\nDescargando video...")
            subprocess.run(
                [
                    RCLONE_CMD,
                    "copyto",
                    f"{REMOTE_FOLDER}/{remote_file}",
                    str(local_video_path.resolve()),
                ],
                check=True,
            )
            # EXTRACCIÓN DE FRAMES
            print("Extrayendo frames...")
            num_frames = extract_frames(
                local_video_path,
                FRAMES_DIR,
                frame_stride=1,
            )
            print(
                f"Extraídos {num_frames} frames."
            )
            # INFERENCIA
            run_inference(
                FRAMES_DIR,
                json_output_path,
                MEGADETECTOR,
            )
            print(
                f"Resultados guardados en:\n"
                f"{json_output_path.resolve()}"
            )
        finally:
            # LIMPIEZA
            if local_video_path.exists():
                local_video_path.unlink()
            clean_directory(FRAMES_DIR)
            print("\nArchivos temporales eliminados.")

# ==============================================================================
# PIPELINE PRINCIPAL
# ==============================================================================

def main():

    print("=" * 70)
    print("PIPELINE DE CÁMARAS TRAMPA")
    print("=" * 70)

    print(f"\nSistema operativo: {os.name} ({sys.platform})")
    print(f"Python: {PYTHON_EXEC}")
    print(f"Rclone: {RCLONE_CMD}")
    print(f"MegaDetector desde cero: {MEGADETECTOR}")
    print(f"Procesar imágenes: {ONLY_FRAMES}")

    prepare_directories()
    check_dependencies()

    if ONLY_FRAMES:
        process_images()
    else:
        process_videos()

    print("\n" + "=" * 70)
    print("PROCESAMIENTO FINALIZADO")
    print("=" * 70)

# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    main()