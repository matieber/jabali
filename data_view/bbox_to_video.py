"""Genera un video que compara las cajas de hasta 2 detecciones sobre el mismo video.

Cada cuadro del video de salida apila el frame original una vez por deteccion
(en el orden indicado), dibujando solo las cajas con confianza >= umbral. Cada
panel lleva su etiqueta en el encabezado.
El indice de frame del JSON (frame_00000012.jpg -> 12) corresponde al frame 12
del video, contando desde 0.
Salida: OUT_DIR/comparacion_<video>.mp4 (o la ruta de --out, que ignora OUT_DIR).
"""

import argparse
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

from data_view.det_utils import load_detections

OUT_DIR = Path("./data/result/data_view/videos")
MAX_DETECTIONS = 2
PANEL_WIDTH = 960

# Colores BGR por categoria
CATEGORY_COLORS = {
    "animal": (60, 200, 60),
    "person": (60, 60, 230),
    "vehicle": (230, 140, 40),
}
DEFAULT_COLOR = (200, 200, 0)
HEADER_BG = (40, 40, 40)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True, type=Path, help="Archivo de video original.")
    p.add_argument("--detection-json", nargs=2, action="append", required=True,
                   metavar=("PATH", "LABEL"),
                   help="detection_results.json y la etiqueta de esa deteccion. "
                        f"Repetible, hasta {MAX_DETECTIONS}.")
    p.add_argument("--threshold", type=float, default=0.1, help="Umbral de confianza (default 0.1).")
    p.add_argument("--start", type=int, default=0, help="Primer frame a incluir (default 0).")
    p.add_argument("--end", type=int, default=None, help="Ultimo frame a incluir, inclusive (default: ultimo).")
    p.add_argument("--out", type=Path, default=None,
                   help=f"Ruta exacta del video de salida. Si se indica, se ignora OUT_DIR ({OUT_DIR}).")
    args = p.parse_args()
    if len(args.detection_json) > MAX_DETECTIONS:
        p.error(f"Se admiten como maximo {MAX_DETECTIONS} --detection-json "
                f"(se indicaron {len(args.detection_json)}).")
    return args


def load_runs(pairs):
    runs, labels = [], set()
    for path, label in pairs:
        path = Path(path)
        if not path.is_file():
            raise SystemExit(f"No existe el archivo de detecciones: {path}")
        if label in labels:
            raise SystemExit(f"Etiqueta repetida: {label}")
        labels.add(label)
        runs.append((label, load_detections(path)))
    return runs


def draw_boxes(img, detections, threshold):
    """Dibuja las detecciones con conf >= umbral. Devuelve la cantidad dibujada."""
    h, w = img.shape[:2]
    thickness = max(2, round(w / 480))
    scale = max(0.5, w / 1400)
    shown = 0
    for det in detections:
        if det.conf < threshold:
            continue
        x, y, bw, bh = det.bbox
        x1, y1 = int(round(x * w)), int(round(y * h))
        x2, y2 = int(round((x + bw) * w)), int(round((y + bh) * h))
        color = CATEGORY_COLORS.get(det.cat, DEFAULT_COLOR)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        text = f"{det.cat} {det.conf:.2f}"
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        ty = y1 if y1 - th - base >= 0 else y1 + th + base
        cv2.rectangle(img, (x1, ty - th - base), (x1 + tw + 4, ty), color, -1)
        cv2.putText(img, text, (x1 + 2, ty - base // 2), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (255, 255, 255), 1, cv2.LINE_AA)
        shown += 1
    return shown


def make_panel(frame, frames, label, threshold, idx):
    h0, w0 = frame.shape[:2]
    height = int(round(PANEL_WIDTH * h0 / w0))
    panel = cv2.resize(frame, (PANEL_WIDTH, height), interpolation=cv2.INTER_AREA)
    has_data = idx in frames
    shown = draw_boxes(panel, frames[idx], threshold) if has_data else 0

    header_h = max(30, round(PANEL_WIDTH / 28))
    header = np.full((header_h, PANEL_WIDTH, 3), HEADER_BG, dtype=np.uint8)
    info = f"{label} | umbral >= {threshold:.2f} | frame {idx} | " + (
        f"{shown} det" if has_data else "sin datos")
    cv2.putText(header, info, (10, int(header_h * 0.7)), cv2.FONT_HERSHEY_SIMPLEX,
                header_h / 55, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([header, panel])


def compose(frame, idx, runs, threshold):
    out = np.vstack([make_panel(frame, frames, label, threshold, idx) for label, frames in runs])
    h, w = out.shape[:2]
    return out[: h - h % 2, : w - w % 2]  # dimensiones pares para el codec


def to_h264(src, dst):
    """Convierte a H.264 + yuv420p (reproducible en VSCode/navegadores). Devuelve True si pudo."""
    if not shutil.which("ffmpeg"):
        return False
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst)]
    return subprocess.run(cmd).returncode == 0


def main():
    args = parse_args()
    if not args.video.is_file():
        raise SystemExit(f"No existe el video: {args.video}")
    runs = load_runs(args.detection_json)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"No se pudo abrir el video: {args.video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    end = total - 1 if args.end is None else min(args.end, total - 1)

    out_path = args.out or OUT_DIR / f"comparacion_{args.video.stem}.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(out_path.stem + ".tmp.mp4")

    writer = None
    written = 0
    idx = 0
    try:
        while idx <= end:
            if not cap.grab():
                break
            if idx >= args.start:
                ok, frame = cap.retrieve()
                if not ok:
                    break
                composite = compose(frame, idx, runs, args.threshold)
                if writer is None:
                    h, w = composite.shape[:2]
                    writer = cv2.VideoWriter(str(tmp_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                    if not writer.isOpened():
                        raise SystemExit("No se pudo crear el video de salida.")
                writer.write(composite)
                written += 1
                if written % 50 == 0:
                    print(f"  {written} cuadros escritos...")
            idx += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    if written == 0:
        raise SystemExit("No se escribio ningun cuadro (revise --start/--end).")

    if to_h264(tmp_path, out_path):
        tmp_path.unlink()
    else:
        tmp_path.replace(out_path)
        print("[aviso] ffmpeg no disponible o fallo la conversion: el video queda en mp4v y "
              "puede no reproducirse en VSCode. Instale ffmpeg o ejecute:\n"
              f"  ffmpeg -i {out_path} -c:v libx264 -pix_fmt yuv420p salida_h264.mp4")
    print(f"Video: {out_path} ({written} cuadros)")


if __name__ == "__main__":
    main()