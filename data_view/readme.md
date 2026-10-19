# data_view: herramientas de inspección de resultados

Scripts auxiliares para revisar y comparar los resultados del pipeline (`mdv5_sn.py`): ver frames puntuales, comparar detecciones sobre el video, graficar la confianza frame a frame y generar un reporte HTML del CSV de metadatos.

| Script | Para qué sirve | Salida por defecto |
|---|---|---|
| `extract_frame.py` | Extrae frames puntuales de un video como PNG. | `data/result/data_view/frames/` |
| `bbox_to_video.py` | Video que apila las cajas de hasta 2 detecciones sobre el mismo video. | `data/result/data_view/videos/` |
| `confidence_per_frame.py` | Gráfico de confianza por frame de una o más detecciones, más un CSV resumen. | `data/result/data_view/figuras/` |
| `csv_to_html.py` | Dashboard HTML a partir de `processing_metadata.csv`. | Ruta indicada por argumento |

Las carpetas `OUT_DIR` son rutas **relativas al directorio desde donde se ejecuta el comando**. Conviene correr todo desde la raíz del proyecto (donde está `data/`). Cada script permite cambiar la salida con `--out-dir` (o `--out` en `bbox_to_video.py`), que ignora `OUT_DIR`.

## Requisitos

- Python 3.11 o superior.
- `opencv-python`, `numpy` y `matplotlib` (este último solo para `confidence_per_frame.py`).
- `det_utils.py` en la misma carpeta: lo importan `bbox_to_video.py` y `confidence_per_frame.py` (`load_detections` y `conf_series`).
- `csv_to_html.py` solo usa la librería estándar, pero el HTML generado carga Bootstrap, jQuery y DataTables desde CDN, por lo que **necesita internet al abrirlo**.

## Convención de frames

El índice de frame de los JSON (`frame_000012.jpg` → 12) corresponde al frame 12 del video, contando desde 0. Todos los scripts leen el video de forma secuencial (`grab`/`retrieve`), igual que el pipeline, por lo que los índices coinciden.

---

## extract_frame.py

Guarda como PNG uno o varios frames de un video, indicados por índice o por tiempo.

```bash
# Un frame
py data_view/extract_frame.py --video "C:/videos_test/.../IMG_0010.MP4" --frame 170

# Varios frames, por índice y por tiempo (segundos)
py data_view/extract_frame.py --video IMG_0010.MP4 --frame 10 170 --time 2.5
```

| Argumento | Descripción |
|---|---|
| `--video` | Archivo de video (obligatorio). |
| `--frame <n ...>` | Índice(s) de frame, desde 0. |
| `--time <s ...>` | Instante(s) en segundos; se convierten a frame con los fps del video. |
| `--out-dir` | Carpeta de salida (ignora `OUT_DIR`). |

Hay que indicar al menos `--frame` o `--time`. Salida: `<video>_frame<indice de 8 dígitos>.png`. Si algún frame está fuera de rango, el script aborta y muestra el rango válido.

---

## bbox_to_video.py

Genera un video donde cada cuadro apila el frame original una vez por detección, dibujando solo las cajas con confianza ≥ umbral. Cada panel lleva su etiqueta, el umbral, el número de frame y la cantidad de detecciones. Sirve para comparar, por ejemplo, detecciones con distinto stride o distinto modelo sobre el mismo video.

```bash
py data_view/bbox_to_video.py --video IMG_0010.MP4 \
  --detection-json data/result/detections/.../IMG_0010/detection_results.json stride1 \
  --detection-json data/result/detections_stride5/.../IMG_0010/detection_results.json stride5 \
  --threshold 0.2 --start 0 --end 300
```

| Argumento | Descripción |
|---|---|
| `--video` | Video original (obligatorio). |
| `--detection-json PATH LABEL` | `detection_results.json` y su etiqueta. Repetible, **hasta 2**. Las etiquetas no pueden repetirse. |
| `--threshold` | Umbral de confianza (default 0.1). |
| `--start` / `--end` | Rango de frames a incluir; `--end` es inclusivo (default: todo el video). |
| `--out` | Ruta exacta del video de salida (ignora `OUT_DIR`). |

Salida: `comparacion_<video>.mp4`. Los paneles se apilan en el orden en que se indican los `--detection-json`. Si un frame no tiene entrada en el JSON (por ejemplo con stride > 1), el panel muestra `sin datos`.

Colores de las cajas: verde = animal, rojo = persona, azul = vehículo. Se dibuja la categoría y la confianza de MegaDetector, no la especie de SpeciesNet.

---

## confidence_per_frame.py

Grafica la mayor confianza de detección de cada frame (0 si no se detectó nada), con una curva por detección y una línea con el umbral.

```bash
py data_view/confidence_per_frame.py \
  --detection-json data/result/detections/.../IMG_0010/detection_results.json stride1 \
  --detection-json data/result/detections_stride5/.../IMG_0010/detection_results.json stride5 \
  --category animal --threshold 0.2
```

| Argumento | Descripción |
|---|---|
| `--detection-json PATH LABEL` | `detection_results.json` y su etiqueta. Repetible, sin límite. Las etiquetas no pueden repetirse. |
| `--category` | `animal`, `person`, `vehicle` o `all` (default `animal`). |
| `--threshold` | Umbral a marcar y usar en el resumen (default 0.1). |
| `--out-dir` | Carpeta de salida (ignora `OUT_DIR`). |

Salidas, nombradas con la carpeta que contiene el primer JSON (el nombre del video, según la estructura de `detection_results.json`):

- `<video>_confianza.png`: el gráfico.
- `<video>_resumen.csv`: una fila por detección, con `label`, `frames`, `frames_sobre_umbral`, `conf_media` y `cajas_sobre_umbral`.

Notas:

- Los frames sin entrada en un JSON se toman como confianza 0 y se avisa por consola. Es lo esperado al comparar stride 1 contra stride 5.
- `cajas_sobre_umbral` cuenta las cajas de **todas** las categorías, aunque `--category` filtre la curva.

---

## csv_to_html.py

Convierte el CSV de metadatos del pipeline en un dashboard HTML con tarjetas de resumen y una tabla con búsqueda, orden y paginación.

```bash
py data_view/csv_to_html.py data/result/metadata/processing_metadata.csv reporte.html
```

Los dos argumentos (CSV de entrada y HTML de salida) son posicionales y obligatorios.

Qué muestra: total de videos, procesados, fallidos y detecciones totales; y por video el estado, detecciones (entre paréntesis, las clasificadas), tiempo total, tiempo de SpeciesNet, RAM pico, VRAM pico y el motivo si hubo error.

Para tener en cuenta:

- Cualquier estado distinto de `procesado` se cuenta y se muestra como **ERROR**, incluidos los `omitido`.
- El CSV acumula filas de todas las corridas. Si un video aparece en más de una (por ejemplo, un `omitido` tras un `procesado`), se cuenta varias veces.
- No incluye las columnas por etapa de VRAM (`megadetector_gpu_peak_mb`, `speciesnet_gpu_peak_mb`) ni los tiempos de extracción y MegaDetector, aunque sí están en el CSV.