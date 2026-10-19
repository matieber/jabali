"""
Grafica frame a frame la confianza de una o mas detecciones de un mismo video.

Eje x = numero de frame, eje y = mayor confianza de deteccion del frame
(0 si el detector no detecto nada). Una curva por deteccion, mas el umbral.
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data_view.det_utils import conf_series, load_detections 

DPI = 200
OUT_DIR = Path("./data/result/data_view/figuras")

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--detection-json", nargs=2, action="append", required=True,
                   metavar=("PATH", "LABEL"),
                   help="detection_results.json y la etiqueta de esa deteccion. "
                        "Repetible para comparar varias detecciones del mismo video.")
    p.add_argument("--category", default="animal",
                   help="Categoria a graficar: animal, person, vehicle o all (default: animal).")
    p.add_argument("--threshold", type=float, default=0.1,
                   help="Umbral a marcar y usar en el resumen (default: 0.1).")
    p.add_argument("--out-dir", type=Path, default=None,
                   help=f"Carpeta de salida. Si se indica, se ignora OUT_DIR ({OUT_DIR}).")
    return p.parse_args()


def load_runs(pairs):
    runs = []
    labels = set()
    for path, label in pairs:
        path = Path(path)
        if not path.is_file():
            raise SystemExit(f"No existe el archivo de detecciones: {path}")
        if label in labels:
            raise SystemExit(f"Etiqueta repetida: {label}")
        labels.add(label)
        runs.append((label, path, load_detections(path)))
    return runs


def main():
    args = parse_args()
    runs = load_runs(args.detection_json)
    name = runs[0][1].parent.name
    thr = args.threshold

    out_dir = args.out_dir or OUT_DIR

    lo = min(min(frames, default=0) for _, _, frames in runs)
    hi = max(max(frames, default=-1) for _, _, frames in runs)

    fig, ax = plt.subplots(figsize=(11, 3.8))
    rows = []
    for i, (label, _, frames) in enumerate(runs):
        x, conf = conf_series(frames, args.category, (lo, hi))
        x, conf = np.array(x), np.array(conf)

        missing = sum(1 for f in x if f not in frames)
        if missing:
            print(f"[aviso] {label}: {missing} frames sin entrada en el JSON (se toman como 0)")

        over = int((conf >= thr).sum())
        ax.plot(x, conf, color=f"C{i}", lw=1.3, label=f"{label} ({over} frames ≥ {thr:g})")
        rows.append({
            "label": label,
            "frames": len(x),
            "frames_sobre_umbral": over,
            "conf_media": round(float(conf.mean()), 4),
            "cajas_sobre_umbral": sum(d.conf >= thr for dets in frames.values() for d in dets),
        })

    ax.axhline(thr, color="gray", ls="--", lw=1, label=f"umbral = {thr:g}")
    ax.set_ylim(-0.02, 1.03)
    ax.set_xlim(lo, hi)
    ax.set_xlabel("Número de frame")
    ax.set_ylabel("Confianza máxima del frame")
    ax.set_title(f"{name}  ({rows[0]['frames']} frames)", fontsize=10, loc="left")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{name}_confianza.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"Figura: {out}")

    csv_path = out_dir / f"{name}_resumen.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Resumen: {csv_path}")


if __name__ == "__main__":
    main()