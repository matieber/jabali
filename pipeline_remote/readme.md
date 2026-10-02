# Pipeline MegaDetector + SpeciesNet desde remoto vía RCLONE

Procesa videos de cámaras trampa almacenados en un remoto (configurado con **rclone**) y genera, para cada video, un JSON en formato MegaDetector con las detecciones y la clasificación de especies de **SpeciesNet**. Además registra en un CSV el estado y las métricas de rendimiento (tiempos, RAM, CPU, VRAM) de cada video.

Para cada video, el pipeline:

1. **Descarga** el video desde el remoto con `rclone copyto`.
2. **Extrae frames** como JPG (`frame_XXXXXX.jpg`, nombrados por su índice real en el video) usando OpenCV.
3. **Detecta** animales, personas y vehículos con MegaDetector (MDV5A), o reutiliza detecciones ya guardadas.
4. **Clasifica** con SpeciesNet únicamente las detecciones cuya confianza supera `CLASSIFY_THRESHOLD`.
5. **Guarda** el JSON final y una fila de metadatos en el CSV.
6. **Limpia** el video descargado y los frames temporales.

Mientras se procesa el video actual, el siguiente ya se está descargando en un hilo aparte, de modo que la descarga no frena la inferencia. Los modelos se cargan una sola vez por ejecución.

## Modos de ejecución

### Con `--md` (desde cero)
Se descarga el video, se extraen los frames (según `--stride`), se corre MegaDetector y luego SpeciesNet. Con `--save-detections` se guarda el `detection_results.json` de cada video para poder reutilizarlo después.

### Sin `--md` (reutilizando detecciones)
Se busca un `detection_results.json` previo para cada video. Si existe:

- Si **ninguna** detección supera el umbral de clasificación, el resultado final se genera directamente **sin descargar el video**.
- Si hay detecciones relevantes, solo se extraen y clasifican los **frames que las contienen** (se ignora `--stride`).

Si no existe el JSON de detecciones, el video se registra como `no_procesado` con el motivo `json_no_encontrado`.

> En este modo MegaDetector no se carga, lo que ahorra memoria y tiempo de arranque.

## Requisitos

- Python 3.11 o superior.
- [rclone](https://rclone.org/) instalado, en el `PATH` y con el remoto ya configurado (`rclone config`)
- Librerías de Python:
  - `megadetector` 
  - `speciesnet` 
  - `opencv-python`
  - `psutil` *(opcional, necesario para las métricas de RAM/CPU)*
  - `torch` *(ya viene como dependencia de MegaDetector; se usa para medir el pico de VRAM)*
- GPU con CUDA recomendada para MegaDetector y SpeciesNet.

> **Importante:** 
> - el script usa funciones internas (con prefijo `_`) de megadetector, que pueden cambiar entre versiones por lo que se necesita una versión fija de esta librería anotada en `requirements.txt`.
> - al instalar `megadetector` y `speciesnet` en el mismo entorno, `pip` puede mostrar un error por una incongruencia entre las versiones de la librería ONNX que requiere cada uno. Como este script no utiliza ONNX, ese error es irrelevante para el funcionamiento del pipeline y puede ignorarse.

## Configuración

Las constantes pueden editarse al inicio del script:

| Constante | Descripción |
|---|---|
| `REMOTE_FOLDER` | Nombre del remoto de rclone (ej. `"pps:"`). **Cambiar según tu configuración.** |
| `LOCAL_TEMP_DIR` | Directorio donde rclone descarga los videos. |
| `FRAMES_DIR` | Directorio temporal para los frames extraídos. |
| `RESULTS_DIR` | Directorio de los JSON de resultados y del CSV de metadatos. |
| `DETECTIONS_ROOT` | Raíz donde se guardan/buscan los `detection_results.json`. |
| `VIDEO_EXTS` | Extensiones de video aceptadas (`.mp4`, `.avi`, `.mov`). |
| `CLASSIFY_THRESHOLD` | Confianza mínima de una detección para enviarla a SpeciesNet (por defecto, el valor de SpeciesNet). |
| `DOWNLOAD_TIMEOUT_SECONDS` | Timeout por descarga (20 min). Evita que un rclone colgado detenga todo. |
| `QUEUE_POLL_SECONDS` | Cada cuántos segundos se revisan errores en los hilos de clasificación. |
| `BENCHMARK_SAMPLE_SECONDS` | Intervalo de muestreo de recursos para benchmarking (0.5 s). |

> `FRAMES_DIR` y `LOCAL_TEMP_DIR` se **vacían** automáticamente (`FRAMES_DIR` al iniciar y tras cada video; `LOCAL_TEMP_DIR` al finalizar). No hay que ingresar datos que se quieran conservar a estas carpetas.
## Uso

```bash
# Procesar todos los videos del root seteado, corriendo MegaDetector + SpeciesNet
python rclone_mdsn.py --md

# Igual, guardando las detecciones para reutilizarlas más adelante
python rclone_mdsn.py --md --save-detections

# Procesar 1 de cada 5 frames, con lotes de 16
python rclone_mdsn.py --md --stride 5 --batch-size 16

# Procesar solo los videos listados en un archivo
python rclone_mdsn.py --md --list-paths videos.out

# Reutilizar detecciones existentes (sin MegaDetector)
python rclone_mdsn.py

# Reprocesar aunque ya exista el resultado
python rclone_mdsn.py --force
```

### Argumentos

| Argumento | Descripción |
|---|---|
| `--md` | Ejecuta MegaDetector + SpeciesNet desde cero. Sin este flag se usan detecciones existentes. |
| `--list-paths <ruta a la lista>` | Archivo de texto (uno por línea) con las rutas de los videos a procesar, relativas a `REMOTE_FOLDER`. Se ignoran líneas vacías y extensiones no válidas. Si se omite, se listan todos los videos del remoto con `rclone lsf`. |
| `--stride <n>` | Solo con `--md`: procesa 1 de cada n frames (default: 1, todos). |
| `--batch-size <n>` | Solo con `--md`: frames por lote en MegaDetector (default: 8). |
| `--save-detections` | Solo con `--md`: guarda el `detection_results.json` de cada video en `DETECTIONS_ROOT`. |
| `--force` | Reprocesa videos aunque ya exista su JSON de salida. |

## Estructura de archivos

Dado un video remoto `FotosCamarasTrampas/SL009/20240620/DCIM/100_BTCF/IMG_0007.MP4`:

```
data/
├── rclone_local/                 # descargas temporales (se limpia)
├── temp_frames/                  # frames temporales (se limpia)
├── FotosCamarasTrampas/SL009/20240620/DCIM/100_BTCF/IMG_0007/
│   └── detection_results.json    
└── result/
    ├── processing_metadata.csv   # estado y métricas de todos los videos
    └── FotosCamarasTrampas/SL009/IMG_0007.json   # resultado final (detección + clasificación)
```

Se conserva la estructura de carpetas del remoto tanto en las detecciones como en los resultados.

## Salidas

### JSON de resultados
Formato MegaDetector estándar, con cada detección que incluye `classifications` (las 2 mejores especies y su score), más `classification_categories` y `classification_category_descriptions`. En `info` se agregan el modelo clasificador y la fecha de clasificación.

### CSV de metadatos (`processing_metadata.csv`)
Una fila por video, con las columnas:

- **Estado:** `video_path`, `status`, `reason`, `detection_json`, `result_json`
- **Tiempos (s):** `extract_seconds`, `megadetector_seconds`, `speciesnet_seconds`, `processing_seconds`
- **Conteos:** `frames_extracted`, `detections`, `classified_detections`
- **Recursos:** `ram_start_mb`, `ram_peak_mb`, `ram_end_mb`, `cpu_avg_percent`, `cpu_peak_percent`, `gpu_peak_mb`

Valores posibles de `status`:

| Estado | Significado |
|---|---|
| *procesado* | El video se procesó y se generó su JSON de resultado. |
| *omitido* | Ya existía un resultado y no se usó `--force`. |
| *no_procesado* | Hubo un error (descarga, lectura de JSON, procesamiento) o faltaba el JSON de detecciones. El campo `reason` indica el motivo. |

Si el CSV existente tiene un encabezado distinto al actual, se renombra como `processing_metadata_legacy_<fecha>.csv` y se crea uno nuevo.


## Notas y consideraciones

- **Reanudación:** como los videos con resultado existente se omiten, se puede volver a ejecutar el script tras una interrupción y continuará con los pendientes.
- **Métricas:** si `psutil` no está instalado, las métricas de RAM/CPU quedan en 0. El pico de VRAM solo se registra si hay GPU CUDA disponible.
- **Nombres de frames:** en modo sin `--md`, el índice del frame se obtiene del último segmento del nombre separado por `_` (ej. `frame_000123.jpg`). Si algún nombre no cumple ese formato, se extraen todos los frames del video.
- **Errores en hilos:** si falla un hilo de SpeciesNet, el error se propaga y el video se registra como `no_procesado` en lugar de quedar bloqueado.
- **Descargas fallidas:** se registran como `error_descarga` y el pipeline continúa con el siguiente video.