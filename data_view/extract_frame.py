"""Extrae uno o varios frames especificos de un video y los guarda como PNG.

El indice de frame cuenta desde 0 y se lee de forma secuencial (grab/retrieve),
igual que en el script de procesamiento, por lo que frame_00000170.jpg del
pipeline corresponde exactamente a --frame 170.
"""

import argparse
import os
from pathlib import Path

import cv2

OUT_DIR = Path("./data/result/data_view/frames")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True, type=Path, help="Archivo de video.")
    p.add_argument("--frame", type=int, nargs="+", default=[],
                   help="Indice(s) de frame (desde 0). Se pueden indicar varios.")
    p.add_argument("--time", type=float, nargs="+", default=[],
                   help="Instante(s) en segundos; se convierten a frame con los fps del video.")
    p.add_argument("--out-dir", type=Path, default=None,
                   help=f"Carpeta de salida. Si se indica, se ignora OUT_DIR ({OUT_DIR}).")
    return p.parse_args()


def main():
    args = parse_args()
    if not args.video.is_file():
        raise SystemExit(f"No existe el video: {args.video}")

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"No se pudo abrir el video: {args.video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    wanted = set(args.frame)
    wanted.update(int(round(t * fps)) for t in args.time)
    if not wanted:
        raise SystemExit("Indique --frame o --time.")
    bad = [f for f in wanted if f < 0 or (total > 0 and f >= total)]
    if bad:
        raise SystemExit(f"Frames fuera de rango {sorted(bad)} (el video tiene {total} frames, 0 a {total - 1}).")

    out_dir = args.out_dir or OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    pending = sorted(wanted)
    saved = 0

    idx, last = 0, pending[-1]
    wanted_left = set(pending)
    while idx <= last:
        if not cap.grab():
            break
        if idx in wanted_left:
            ok, frame = cap.retrieve()
            if ok:
                path = out_dir / f"{args.video.stem}_frame{idx:08d}.png"
                if not cv2.imwrite(str(path), frame):
                    raise SystemExit(f"No se pudo guardar: {path}")
                print(f"Frame {idx} -> {path}")
                saved += 1
            else:
                print(f"[aviso] no se pudo leer el frame {idx}")
        idx += 1

    cap.release()
    if saved < len(pending):
        print(f"[aviso] se guardaron {saved} de {len(pending)} frames pedidos.")


if __name__ == "__main__":
    main()