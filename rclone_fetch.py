import sys
import shutil
import subprocess
import argparse
import csv
import cv2
import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Full

from PIL import Image
import numpy as np

# MegaDetector
from megadetector.detection.run_detector import (
    load_detector,
    try_download_known_detector,
    DEFAULT_OUTPUT_CONFIDENCE_THRESHOLD,
)
from megadetector.detection.run_detector_batch import write_results_to_file

# Reutilizamos la implementación de SpeciesNet para el procesamiento de crops
# y la generación de clasificaciones. Habria que ver si fijar la versión de md 
# para asi evitamos problemas por diferencias entre versiones
from megadetector.detection import run_md_and_speciesnet as mdsn

# ==============================================================================
# CONFIGURACIÓN
# ==============================================================================

REMOTE_FOLDER = "pps:" #Cambiar según el nombre de remoto correspondiente

LOCAL_TEMP_DIR = Path("./rclone_local")
FRAMES_DIR = Path("./temp_frames")
RESULTS_DIR = Path("./result")
DETECTIONS_ROOT = Path("./")
METADATA_FILE = RESULTS_DIR / "processing_metadata.csv"
TMP_DETECTIONS = RESULTS_DIR / "_tmp_detections.json"

VIDEO_EXTS = {".mp4", ".avi", ".mov"}
METADATA_FIELDS = ["video_path", "status", "reason", "detection_json", "result_json"]

RCLONE_CMD = shutil.which("rclone")

# Umbral por encima del cual una detección se envía a SpeciesNet.
# Con esto reducimos mucho la clasificación evitando hacerla sobre frames con baja confianza de detección
# La dejamos default pero podríamos cambiarla a futuro.
CLASSIFY_THRESHOLD = mdsn.DEFAULT_DETECTION_CONFIDENCE_THRESHOLD_FOR_CLASSIFICATION

# Segundos entre chequeos de errores en los hilos de crops.
QUEUE_POLL_SECONDS = 5

# ==============================================================================
# ARGUMENTOS
# ==============================================================================


def parse_args():
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
        "--list-paths",
        type=Path,
        default=None,
        help=(
            "Archivo .out con la lista de paths de videos a procesar. "
            "Los paths se interpretan desde REMOTE_FOLDER."
        ),
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Solo con --md: procesa 1 de cada N frames (default: 1 = todos).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Solo con --md: frames por lote en MegaDetector (default: 8).",
    )
    parser.add_argument(
        "--save-detections",
        action="store_true",
        help=(
            "Solo con --md: guarda el detection_results.json de cada video en "
            "DETECTIONS_ROOT/<ruta_sin_extension>/ para reutilizarlo sin --md."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocesa videos aunque ya exista su JSON de salida.",
    )
    args = parser.parse_args()

    if args.stride < 1 or args.batch_size < 1:
        parser.error("--stride y --batch-size deben ser >= 1")

    return args


# ==============================================================================
# UTILIDADES
# ==============================================================================


def prepare_directories():
    """ Crea los directorios auxiliares para el procesamiento. """
    for directory in (LOCAL_TEMP_DIR, FRAMES_DIR, RESULTS_DIR):
        directory.mkdir(exist_ok=True, parents=True)


def check_dependencies():
    """ Revisa que rclone se encuentre en el Path del sistema como comando. """
    if RCLONE_CMD is None:
        raise RuntimeError("No se encontró 'rclone' en el PATH del sistema.")


def clear_directory(directory: Path):
    """ Elimina un directorio junto con su contenido. """
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)


def normalize_video_path(video_path) -> Path:
    """ Quita separadores iniciales para que Path() no descarte componentes."""
    return Path(str(video_path).lstrip("/\\"))


def expected_detection_path(video_path) -> Path:
    """ 
    Genera la ruta esperada para el archivo de detection results.
    Construye la ruta ubicando 'detection_results.json' dentro de un directorio
    dedicado con el nombre del video (sin extensión), conservando la estructura 
    de carpetas relativa a DETECTIONS_ROOT.
    """
    rel = normalize_video_path(video_path)
    return DETECTIONS_ROOT / rel.parent / rel.stem / "detection_results.json"


def find_detection_json(video_path):
    """ 
    Revisa si el expected path corresponde a un archivo json existente en el directorio. 
    Devuelve (Path, None) si el archivo existe, (None, "json_no_encontrado") si el archivo no existe. 
    """
    expected = expected_detection_path(video_path)
    if expected.is_file():
        return expected, None
    return None, "json_no_encontrado"


def safe_output_path(file_path: Path) -> Path:
    # === CAMBIO PRIORIDAD 4 ===
    # Conserva la estructura relativa del remoto dentro de result/.
    #
    # Ejemplo:
    #   SL001/20240905/DCIM/100_BTCF/IMG_001.MP4
    # pasa a:
    #   result/SL001/20240905/DCIM/100_BTCF/IMG_001.json
    return RESULTS_DIR / file_path.with_suffix(".json")




def frame_index_from_name(name):
    """ 
    Extrae el índice numérico del fotograma desde su nombre o ruta.
    Asume que el entero es el último segmento del nombre separado por guiones bajos. 
    """
    try:
        return int(Path(name).stem.split("_")[-1])
    except ValueError:
        return None


def images_to_classify(detector_results):
    """
    Filtra las imágenes que contienen al menos una detección relevante.
    Retorna las imágenes del diccionario cuyas detecciones superan o igualan 
    el umbral de confianza CLASSIFY_THRESHOLD. 
    """
    return [
        image
        for image in detector_results["images"]
        if any(
            det["conf"] >= CLASSIFY_THRESHOLD
            for det in (image.get("detections") or [])
        )
    ]


# ==============================================================================
# METADATOS
# ==============================================================================


def log_result(stats, video_path, status, reason="", detection_json=None, result_json=None):
    """ Registra el estado final de un video en el archivo CSV de metadatos. """
    is_new = not METADATA_FILE.exists()
    with METADATA_FILE.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=METADATA_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow({
            "video_path": video_path,
            "status": status,
            "reason": reason,
            "detection_json": str(Path(detection_json).resolve()) if detection_json else "",
            "result_json": str(Path(result_json).resolve()) if result_json else "",
        })
    stats[status] += 1


# ==============================================================================
# LISTADO, DESCARGA Y EXTRACCIÓN DE FRAMES DE VIDEO
# ==============================================================================


def load_video_list(list_path: Path):
    """
    Carga y filtra las rutas de un archivo de texto con la lista de videos.
    Retorna únicamente las líneas no vacías cuyas extensiones coincidan con VIDEO_EXTS.
    Lanza FileNotFoundError si el archivo no existe.
    """
    if not list_path.exists():
        raise FileNotFoundError(
            f"No existe el archivo de lista de videos: {list_path.resolve()}"
        )
    with list_path.open("r", encoding="utf-8") as file:
        return [
            line.strip()
            for line in file
            if line.strip() and Path(line.strip()).suffix.lower() in VIDEO_EXTS
        ]


def find_remote_videos():
    """
    Lista recursivamente los archivos de video en el almacenamiento remoto mediante rclone.
    Retorna una lista con las rutas relativas de todos los archivos .mp4, .avi y .mov encontrados.
    """
    print("\nBuscando videos en el remoto...")
    result = subprocess.run(
        [
            RCLONE_CMD, "lsf", REMOTE_FOLDER,
            "-R", "--files-only",
            "--include", "*.{mp4,avi,mov}",
            "--ignore-case",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def download_video(job):
    """
    Descarga el video de un job desde el remoto al almacenamiento local usando rclone.
    Devuelve una tupla (éxito, mensaje_de_error).
    """
    try:
        subprocess.run(
            [
                RCLONE_CMD,
                "copyto",
                job.remote_source,
                str(job.local_video.resolve()),
                "--retries", "3",
                "--low-level-retries", "10",
                "--retries-sleep", "5s",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return True, ""
    except subprocess.CalledProcessError as error:
        return False, (error.stderr or "").strip()


def extract_frames(video_path: Path, output_folder: Path, wanted=None, stride=1):
    """
    Extrae fotogramas en formato JPG nombrándolos por su índice real (`frame_XXXXXX.jpg`).
    Permite filtrar por un conjunto específico de índices (`wanted`) o por intervalo (`stride`).
    Devuelve la cantidad total de fotogramas guardados.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {video_path}")
 
    last = max(wanted) if wanted else None
    saved, idx = 0, 0
 
    try:
        while last is None or idx <= last:
            if not cap.grab():
                break
 
            need = (idx in wanted) if wanted is not None else (idx % stride == 0)
            if need:
                ok, frame = cap.retrieve()
                if not ok:
                    break
                frame_file = output_folder / f"frame_{idx:06d}.jpg"
                if not cv2.imwrite(str(frame_file), frame):
                    raise RuntimeError(f"No se pudo guardar el frame: {frame_file}")
                saved += 1
            idx += 1
    finally:
        cap.release()
 
    return saved


# ==============================================================================
# INFERENCIA
# ==============================================================================


def build_output(detector_results, classification_results, classifier_model):
    """
    Combina las clasificaciones de SpeciesNet con los resultados de MegaDetector.
    Mapea nombres comunes a IDs numéricos de categoría, actualiza las detecciones
    con sus clases y agrega metadatos de ejecución al diccionario de resultados.
    """
    next_category_id = 0
    common_name_to_id = {}
    classification_categories = {}
    classification_category_descriptions = {}

    def get_category_id(class_name):
        nonlocal next_category_id
        common_name = mdsn.get_common_name_from_prediction_string(class_name)
        if common_name not in common_name_to_id:
            category_id = str(next_category_id)
            common_name_to_id[common_name] = category_id
            classification_categories[category_id] = common_name
            classification_category_descriptions[category_id] = class_name
            next_category_id += 1
        return common_name_to_id[common_name]

    for image_data in detector_results["images"]:
        detections = image_data.get("detections")
        image_classifications = classification_results.get(image_data["file"])

        if not detections or image_classifications is None:
            continue

        for detection_index, detection in enumerate(detections):
            result = image_classifications.get(detection_index)
            if result is None:
                continue

            if "failure" in result:
                previous = image_data.get("failure", "")
                image_data["failure"] = (
                    f"{previous};{result['failure']}" if previous else result["failure"]
                )
                continue

            detection["classifications"] = [
                [
                    get_category_id(class_name),
                    mdsn.round_float(score, precision=mdsn.CONF_DIGITS),
                ]
                for class_name, score in result["classifications"][:2]
            ]

    info = detector_results.setdefault("info", {})
    info["classifier"] = classifier_model
    info["classification_completion_time"] = time.strftime("%Y-%m-%d %H:%M:%S")

    detector_results["classification_categories"] = classification_categories
    detector_results["classification_category_descriptions"] = (
        classification_category_descriptions
    )
    return detector_results


class PersistentInference:
    """
    Gestiona la carga persistente e inferencia con MegaDetector y SpeciesNet.
    Evita recargar los modelos en memoria entre procesamientos consecutivos.
    """
    def __init__(self, use_md: bool):
        """ Inicializa MD (opcional) y SpeciesNet."""
        self.detector = None

        if use_md:
            print("\nCargando MegaDetector...")
            model_file = try_download_known_detector(
                "MDV5A", force_download=False, verbose=False
            )
            self.detector = load_detector(
                model_file, force_model_download=False, detector_options={}
            )
            print("MegaDetector cargado.")

        # Dos instancias, una para preprocessing en CPU y otra para inferencia.
        print("\nCargando SpeciesNet...")
        self.classifier_model = mdsn.DEFAULT_CLASSIFIER_MODEL
        self.producer_classifier = mdsn.SpeciesNetClassifier(
            self.classifier_model, device="cpu"
        )
        self.consumer_classifier = mdsn.SpeciesNetClassifier(self.classifier_model)
        print("SpeciesNet cargado.")

    # ------------------------------------------------------------------
    def run_md(self, frames_path: Path, detection_json_path: Path, batch_size: int):
        """
        Ejecuta MegaDetector por lotes sobre las imágenes y guarda el JSON de detecciones.
        """
        image_paths = sorted(frames_path.glob("*.jpg"))
        if not image_paths:
            raise RuntimeError(f"No se encontraron frames JPG en {frames_path}")

        results = []
        for start in range(0, len(image_paths), batch_size):
            batch = image_paths[start:start + batch_size]

            images_np = []
            for image_path in batch:
                with Image.open(image_path) as image:
                    images_np.append(np.asarray(image.convert("RGB")))

            results.extend(
                self.detector.generate_detections_one_batch(
                    images_np,
                    [p.name for p in batch],  # solo el nombre del archivo
                    detection_threshold=DEFAULT_OUTPUT_CONFIDENCE_THRESHOLD,
                    augment=False,
                    image_size=None,
                    verbose=False,
                )
            )

        detection_json_path.parent.mkdir(parents=True, exist_ok=True)
        write_results_to_file(
            results, str(detection_json_path), detector_file="MDV5A"
        )
        with detection_json_path.open("r", encoding="utf-8") as file:
            return json.load(file)

    # ------------------------------------------------------------------
    def classify(self, images, frames_path: Path):
        """
        Ejecuta SpeciesNet concurrentemente sobre `images` (filtrado por solo las que tienen detecciones).
        Si un hilo falla, se propaga el error en vez de quedar bloqueado.
        """
        errors = []

        def make_thread(target, args):
            def runner():
                try:
                    target(*args)
                except BaseException as error:
                    errors.append(error)

            return mdsn.Thread(target=runner, daemon=True)

        def check_errors():
            if errors:
                raise RuntimeError(f"Falló un hilo de SpeciesNet: {errors[0]!r}")

        image_queue = mdsn.JoinableQueue(maxsize=mdsn.MAX_IMAGE_QUEUE_SIZE_PER_WORKER)
        batch_queue = mdsn.Queue(maxsize=mdsn.MAX_BATCH_QUEUE_SIZE)
        results_queue = mdsn.Queue()

        producer = make_thread(
            mdsn._crop_producer_func,
            (
                image_queue,
                batch_queue,
                self.classifier_model,
                CLASSIFY_THRESHOLD,
                str(frames_path),
                0,
                self.producer_classifier,
            ),
        )
        consumer = make_thread(
            mdsn._crop_consumer_func,
            (
                batch_queue,
                results_queue,
                self.classifier_model,
                mdsn.DEFAULT_CLASSIFIER_BATCH_SIZE,
                1,
                True,
                "ARG",
                None,
                self.consumer_classifier,
                mdsn.DEFAULT_ROLLUP_TARGET_CONFIDENCE,
            ),
        )

        producer.start()
        consumer.start()

        for item in [*images, None]:
            while True:
                check_errors()
                try:
                    image_queue.put(item, timeout=QUEUE_POLL_SECONDS)
                    break
                except Full:
                    continue

        while True:
            check_errors()
            try:
                classification_results = results_queue.get(timeout=QUEUE_POLL_SECONDS)
                break
            except Empty:
                continue

        producer.join(timeout=30)
        consumer.join(timeout=30)
        return classification_results

    # ------------------------------------------------------------------
    def finalize(self, detector_results, images, frames_path: Path, output_json_path: Path):
        """ Clasifica las imágenes filtradas e integra todo en el JSON final de salida. """
        classification_results = (
            self.classify(images, frames_path) if images else {}
        )
        output = build_output(
            detector_results, classification_results, self.classifier_model
        )
        mdsn.write_json(str(output_json_path), output)


# ==============================================================================
# JOBS
# ==============================================================================


@dataclass
class Job:
    """
    Unidad de trabajo individual en el pipeline para un video específico.
    Almacena las rutas locales/remotas, archivos JSON asociados y el subconjunto 
    de fotogramas a extraer y clasificar.
    """
    number: int
    total: int
    remote_file: str
    remote_source: str
    local_video: Path
    json_output_path: Path
    md_json_path: Path = None        # sin --md
    wanted: set = None               # frames a extraer (None = según stride)


def iter_jobs(remote_files, args, stats):
    """
    Genera instancias de Job para los videos que requieren descarga y procesamiento.
    Filtra y omite los videos ya procesados, sin archivo de detecciones previas o sin
    detecciones relevantes, registrando sus estados en stats antes de omitirlos.
    """
    total = len(remote_files)

    for number, remote_file in enumerate(remote_files, 1):
        try:
            rel = normalize_video_path(remote_file)
            local_video = LOCAL_TEMP_DIR / rel
            json_output_path = safe_output_path(rel)

            if json_output_path.exists() and not args.force:
                log_result(
                    stats,
                    rel,
                    "no_procesado",
                    "resultado_existente",
                    result_json=json_output_path,
                )
                continue

            md_json_path = None
            wanted = None

            if not args.md:
                # find_detection_json devuelve (Path, None) o (None, motivo).
                md_json_path, reason = find_detection_json(rel)

                if md_json_path is None:
                    log_result(
                        stats,
                        rel,
                        "no_procesado",
                        reason or "detection_results.json no encontrado",
                        result_json=json_output_path,
                    )
                    continue

                try:
                    with md_json_path.open("r", encoding="utf-8") as file:
                        detector_results = json.load(file)

                    to_classify = images_to_classify(detector_results)

                    if not to_classify:
                        # No hay frames que clasificar, no hace falta descargar
                        # el video. Se genera directamente el resultado final
                        json_output_path.parent.mkdir(
                            parents=True,
                            exist_ok=True,
                        )

                        output = build_output(
                            detector_results,
                            {},
                            mdsn.DEFAULT_CLASSIFIER_MODEL,
                        )

                        mdsn.write_json(
                            str(json_output_path),
                            output,
                        )

                        log_result(
                            stats,
                            rel,
                            "procesado",
                            "sin_detecciones_para_clasificar",
                            detection_json=md_json_path,
                            result_json=json_output_path,
                        )
                        continue

                    indices = [
                        frame_index_from_name(image["file"])
                        for image in to_classify
                    ]

                    # Si algún nombre no tiene el formato esperado,
                    # se extraen todos los frames.
                    wanted = None if None in indices else set(indices)

                    # Evita conservar el JSON completo mientras el generador
                    # queda suspendido en yield.
                    del detector_results
                    del to_classify
                    del indices

                except Exception as error:
                    log_result(
                        stats,
                        rel,
                        "no_procesado",
                        (
                            f"error leyendo detection JSON: "
                            f"{type(error).__name__}: {error}"
                        ),
                        detection_json=md_json_path,
                        result_json=json_output_path,
                    )
                    continue

            yield Job(
                number=number,
                total=total,
                remote_file=remote_file,
                remote_source=(
                    f"{REMOTE_FOLDER.rstrip('/\\\\')}/"
                    f"{str(remote_file).lstrip('/\\\\')}"
                ),
                local_video=local_video,
                json_output_path=json_output_path,
                md_json_path=md_json_path,
                wanted=wanted,
            )

        except Exception as error:
            try:
                rel = normalize_video_path(remote_file)
            except Exception:
                rel = str(remote_file)

            log_result(
                stats,
                rel,
                "no_procesado",
                f"error preparando video: {type(error).__name__}: {error}",
            )
            continue


def process_job(job, inference, args, stats):
    """
    Ejecuta el flujo completo de procesamiento (extracción, detección y clasificación) para un Job.
    Guarda los resultados finales en JSON, registra el resultado en el CSV de metadatos
    y limpia los archivos temporales de video y frames generados.
    """
    print("\n" + "-" * 70)
    print(f"Video {job.number}/{job.total}")
    print(f"Procesando: {job.remote_file}")
    print("-" * 70)

    saved_detection_path = None

    try:
        print("Extrayendo frames...")
        num_frames = extract_frames(
            job.local_video,
            FRAMES_DIR,
            wanted=job.wanted,
            stride=args.stride,
        )
        print(f"Extraídos {num_frames} frames.")

        if args.md:
            print("\nEjecutando MegaDetector...")
            saved_detection_path = expected_detection_path(job.remote_file)

            detection_path = (
                saved_detection_path
                if args.save_detections
                else TMP_DETECTIONS
            )

            detector_results = inference.run_md(
                FRAMES_DIR,
                detection_path,
                args.batch_size,
            )

            if not args.save_detections:
                TMP_DETECTIONS.unlink(missing_ok=True)
                saved_detection_path = None

            to_classify = images_to_classify(detector_results)

        else:
            with job.md_json_path.open("r", encoding="utf-8") as file:
                detector_results = json.load(file)

            to_classify = images_to_classify(detector_results)
            saved_detection_path = job.md_json_path

        print("\nEjecutando SpeciesNet...")

        job.json_output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        inference.finalize(
            detector_results,
            to_classify,
            FRAMES_DIR,
            job.json_output_path,
        )

        print(
            f"Resultados guardados en:\n"
            f"{job.json_output_path.resolve()}"
        )

        log_result(
            stats,
            job.remote_file,
            "procesado",
            detection_json=saved_detection_path,
            result_json=job.json_output_path,
        )

    except Exception as error:
        print(
            f"Error durante el procesamiento: "
            f"{type(error).__name__}: {error}"
        )

        log_result(
            stats,
            job.remote_file,
            "no_procesado",
            f"error_procesamiento: {type(error).__name__}: {error}",
            detection_json=saved_detection_path,
            result_json=job.json_output_path,
        )

    finally:
        job.local_video.unlink(missing_ok=True)
        clear_directory(FRAMES_DIR)


def process_videos(args):
    """
    Orquesta la ejecución completa del pipeline.
    
    - Obtiene la lista de videos desde el remoto o desde --list-paths.
    - Mantiene los modelos cargados durante toda la ejecución.
    - Descarga el siguiente video mientras se procesa el actual.
    """
    if args.list_paths is not None:
        file_list = load_video_list(args.list_paths)
        print(
            f"\nSe cargaron {len(file_list)} videos desde "
            f"{args.list_paths.resolve()}"
        )
    else:
        file_list = find_remote_videos()
        print(f"\nSe encontraron {len(file_list)} videos para procesar.")

    if not file_list:
        print("No se encontraron videos.")
        return

    stats = Counter()

    # Los modelos se cargan una sola vez
    inference = PersistentInference(args.md)

    jobs = iter_jobs(file_list, args, stats)

    current = None
    current_future = None
    upcoming = None
    upcoming_future = None

    try:
        with ThreadPoolExecutor(max_workers=1) as pool:

            # Obtener el primer Job
            try:
                current = next(jobs, None)
            except Exception as error:
                print(
                    "Error preparando el primer trabajo: "
                    f"{type(error).__name__}: {error}"
                )
                return

            if current is not None:
                current_future = pool.submit(
                    download_video,
                    current,
                )

            while current is not None:

                # Preparar el siguiente Job mientras se descarga el actual
                try:
                    upcoming = next(jobs, None)
                except Exception as error:
                    print(
                        "Error preparando el siguiente trabajo: "
                        f"{type(error).__name__}: {error}"
                    )
                    upcoming = None

                if upcoming is not None:
                    upcoming_future = pool.submit(
                        download_video,
                        upcoming,
                    )
                else:
                    upcoming_future = None

                # Esperar la descarga del Job actual
                download_ok = False
                download_message = ""

                try:
                    download_ok, download_message = current_future.result()
                except Exception as error:
                    download_message = (
                        f"{type(error).__name__}: {error}"
                    )

                if download_ok:
                    try:
                        process_job(
                            current,
                            inference,
                            args,
                            stats,
                        )

                    except Exception as error:
                        reason = (
                            f"error_procesamiento: "
                            f"{type(error).__name__}: {error}"
                        )

                        print(
                            f"Error procesando "
                            f"{current.remote_file}: {reason}"
                        )

                        log_result(
                            stats,
                            current.remote_file,
                            "no_procesado",
                            reason,
                            detection_json=current.md_json_path,
                            result_json=current.json_output_path,
                        )

                else:
                    reason = f"error_descarga: {download_message}"

                    print(
                        f"No se pudo descargar "
                        f"{current.remote_file}: {download_message}"
                    )

                    log_result(
                        stats,
                        current.remote_file,
                        "no_procesado",
                        reason,
                        detection_json=current.md_json_path,
                    )

                # Limpiar el video actual
                try:
                    if current.local_video.exists():
                        current.local_video.unlink()
                except OSError as error:
                    print(
                        f"No se pudo eliminar "
                        f"{current.local_video}: {error}"
                    )

                # Avanzar al siguiente Job
                current = upcoming
                current_future = upcoming_future
                upcoming = None
                upcoming_future = None

    finally:
        # Limpieza global ante cualquier excepción inesperada
        for job in (current, upcoming):
            if job is None:
                continue

            try:
                if job.local_video.exists():
                    job.local_video.unlink()
            except OSError:
                pass

        try:
            clear_directory(FRAMES_DIR)
        except Exception:
            pass

        try:
            clear_directory(LOCAL_TEMP_DIR)
        except Exception:
            pass

    print(
        f"\nMetadatos guardados en:\n"
        f"{METADATA_FILE.resolve()}"
    )

    print(
        f"Procesados: {stats['procesado']} | "
        f"No procesados: {stats['no_procesado']}"
    )

def main():
    args = parse_args()

    print("=" * 70)
    print("PIPELINE MD + SpeciesNet")
    print("=" * 70)
    print(f"\nPlataforma: {sys.platform}")
    print(f"Rclone: {RCLONE_CMD}")
    print(f"MegaDetector desde cero: {args.md}")
    print(f"Lista de videos: {args.list_paths}")
    print(f"Raíz de detecciones: {DETECTIONS_ROOT.resolve()}")

    prepare_directories()
    check_dependencies()
    clear_directory(FRAMES_DIR)
    process_videos(args)

    print("\n" + "=" * 70)
    print("PROCESAMIENTO FINALIZADO")
    print("=" * 70)


if __name__ == "__main__":
    main()