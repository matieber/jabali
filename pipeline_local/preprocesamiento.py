# Nota: este script solo funciona para la estructura asimétrica que nos dieron o, en su defecto, para directorios organizados 
# de la forma "SLxxx/FECHA/NOMBRE_ITEM"
import os
import json
import cv2
from pathlib import Path
from tqdm import tqdm


# configuración de Rutas 
TARGET_SL = "SL004"  # Lote a procesar

VIDEOS_DIR = Path("/mnt/disco/ProyectoJabali/FotosCamarasTrampas")
JSONS_DIR = Path("/mnt/disco/ProyectoJabali/jsons_filtrados")

# almacenamiento interno 
BASE_TEMP_DIR = Path("/media/luciano/LocalDisk/temp_proyecto")
BASE_TEMP_DIR.mkdir(parents=True, exist_ok=True)

TEMP_FRAMES_DIR = BASE_TEMP_DIR / "temp_frames"
TEMP_DETECTIONS_JSON = Path("/media/luciano/LocalDisk/temp_proyecto/temp_detections.json")

# confianza minima de detección de animal para que un json de Mega Detector sea procesado
UMB_CONF = 0.5

def extract_key(path: Path) -> str:
    #extrae la clave unica SLxxx/FECHA/NOMBRE_ITEM
    parts = path.parts
    sl_part = None
    fecha_part = None

    for i, part in enumerate(parts):
        if part.startswith("SL"):
            sl_part = part
            if i + 1 < len(parts):
                fecha_part = parts[i + 1]
            break

    if not sl_part or not fecha_part:
        return None

    item_name = path.stem if path.suffix.lower() in [".mp4", ".m4v"] else path.parent.name
    return f"{sl_part}/{fecha_part}/{item_name}"


def index_videos(base_path: Path, target_sl: str = None) -> dict:
    """Indexa videos contemplando minúsculas y mayúsculas."""
    video_map = {}
    extensions = ["*.mp4", "*.MP4", "*.m4v", "*.M4V", "*.avi", "*.AVI"]

    for ext in extensions:
        for path in base_path.rglob(ext):
            key = extract_key(path)
            if key:
                if target_sl and not key.startswith(f"{target_sl}/"):
                    continue
                video_map[key] = path
    return video_map


def process_video_and_json(video_path: Path, json_path: Path, temp_dir: Path, key_prefix: str):
    """Extrae frames de video donde se detectaron animales (categoría 1)."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    category_map = data.get("detection_categories", {})
    cap = cv2.VideoCapture(str(video_path))
    processed_filepaths = []
    formatted_detections = {}

    temp_dir.mkdir(parents=True, exist_ok=True)

    for item in data.get("images", []):
        raw_detections = item.get("detections", [])
        if not raw_detections:
            continue

        det_list = []
        for d in raw_detections:
            cat_id = str(d.get("category"))
            if cat_id == "1":
                conf = d.get("conf", 0.0) #si no hay un animal en el frame o la confianza es muy baja, no gruarda lo guarda en el json temporal
                if conf >= UMB_CONF:
                    cat_name = category_map.get(cat_id, "animal")
                    det_list.append({
                        "label": cat_name,
                        "conf": d.get("conf", 0.0),
                        "bbox": d.get("bbox", [])
                    })

        if not det_list:
            continue

        file_name = item["file"]
        try:
            frame_idx = int(file_name.replace("frame_", "").replace(".jpg", ""))
        except ValueError:
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if ret:
            safe_prefix = key_prefix.replace("/", "_")
            frame_out_path = temp_dir / f"{safe_prefix}_{file_name}"
            cv2.imwrite(str(frame_out_path), frame)

            filepath_str = str(frame_out_path.resolve())
            processed_filepaths.append(filepath_str)
            formatted_detections[filepath_str] = {"detections": det_list}

    cap.release()
    return processed_filepaths, formatted_detections


def main():
    print(f"idexando videos para la carpeta objetivo '{TARGET_SL}'...")
    video_map = index_videos(VIDEOS_DIR, target_sl=TARGET_SL)
    print(f"Total de videos indexados para {TARGET_SL}: {len(video_map)}")

    print("\nFiltrando JSONs correspondientes...")
    all_jsons = [j for j in JSONS_DIR.rglob("*.json") if "detection" in j.name.lower()]

    # filtrar de antemano para poder calcular el total y mostrar el % preciso
    target_items = []
    for j in all_jsons:
        key = extract_key(j)
        if key and TARGET_SL and key.startswith(f"{TARGET_SL}/") and key in video_map:
            target_items.append((j, key))

    if not target_items:
        raise RuntimeError(
            f"no se pudieron vincular JSONs y videos para '{TARGET_SL}'. "
        )

    all_filepaths = []
    combined_detections = {}

    print(f"\nextrayendo frames ({len(target_items)} videos a procesar):")
    #barra de porcentaje con tqdm
    for json_file, json_key in tqdm(target_items, desc=f"Procesando {TARGET_SL}", unit="video"):
        video_path = video_map[json_key]
        filepaths, detections = process_video_and_json(
            video_path, json_file, TEMP_FRAMES_DIR, json_key
        )
        all_filepaths.extend(filepaths)
        combined_detections.update(detections)

    if not all_filepaths:
        raise RuntimeError(
            f"No se pudo extraer ningún frame para '{TARGET_SL}'. "
            f"Revisa que existan animales (categoría 1) en las detecciones."
        )

    print("\nGuardando JSON temporal de detecciones...")
    predictions_list = [
        {"filepath": fp, "detections": det_data.get("detections", [])}
        for fp, det_data in combined_detections.items()
    ]

    wrapped_detections = {"predictions": predictions_list}

    with open(TEMP_DETECTIONS_JSON, "w", encoding="utf-8") as f:
        json.dump(wrapped_detections, f, ensure_ascii=False, indent=4)

    print(f"\extracción completada")
    print(f"• Frames extraídos: {len(all_filepaths)}")
    print(f"• Detecciones guardadas en: {TEMP_DETECTIONS_JSON}")


if __name__ == "__main__":
    main()