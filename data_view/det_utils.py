"""Utilidades para leer archivos de detecciones de MegaDetector.

Formato esperado (el estandar de MegaDetector, usado por ambos pipelines):

{
  "detection_categories": {"1": "animal", "2": "person", "3": "vehicle"},
  "images": [
    {"file": "frame_00000012.jpg",
     "detections": [{"category": "1", "conf": 0.87, "bbox": [x, y, w, h]}]}
  ]
}

- bbox esta normalizado (0-1) y es [x_min, y_min, ancho, alto].
- El numero de frame se toma del ultimo entero del nombre del archivo
  (frame_00000012.jpg -> 12).
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CATEGORIES = {"1": "animal", "2": "person", "3": "vehicle"}
_FRAME_RE = re.compile(r"(\d+)(?=\.[A-Za-z0-9]+$)")


@dataclass
class Det:
    cat: str
    conf: float
    bbox: tuple  # (x, y, w, h) normalizados


def normalize_rel(video_rel) -> Path:
    """Acepta rutas con / o \\ y quita separadores iniciales."""
    return Path(str(video_rel).replace("\\", "/").lstrip("/"))


def detection_json_path(root, video_rel) -> Path:
    """root/<carpetas>/<nombre_sin_extension>/detection_results.json"""
    rel = normalize_rel(video_rel)
    return Path(root) / rel.parent / rel.stem / "detection_results.json"


def frame_index(name):
    match = _FRAME_RE.search(Path(str(name).replace("\\", "/")).name)
    return int(match.group(1)) if match else None


def load_detections(json_path):
    """Devuelve {indice_de_frame: [Det, ...]} con todos los frames del JSON."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    categories = {str(k): v for k, v in data.get("detection_categories", {}).items()}
    categories = categories or DEFAULT_CATEGORIES

    frames = {}
    for position, image in enumerate(data.get("images", [])):
        idx = frame_index(image.get("file", ""))
        if idx is None:
            idx = position
        detections = []
        for det in image.get("detections") or []:
            category = categories.get(str(det.get("category")), str(det.get("category")))
            detections.append(Det(category, float(det["conf"]), tuple(det["bbox"])))
        frames[idx] = detections
    return frames


def max_conf(detections, category="animal"):
    """Mayor confianza del frame para la categoria ('all' = cualquiera). 0 si no hay."""
    values = [d.conf for d in detections if category == "all" or d.cat == category]
    return max(values, default=0.0)


def conf_series(frames, category="animal", index_range=None):
    """Devuelve (lista_de_frames, lista_de_confianzas) sobre un rango continuo.

    Los frames que no aparecen en el JSON se completan con 0.
    """
    if index_range is None:
        index_range = (min(frames), max(frames)) if frames else (0, -1)
    xs = list(range(index_range[0], index_range[1] + 1))
    ys = [max_conf(frames.get(i, []), category) for i in xs]
    return xs, ys