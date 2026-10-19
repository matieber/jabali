# Pipeline MegaDetector + SpeciesNet para videos de cámaras trampa (rclone / local)

Procesa videos de cámaras trampa, ya sea desde un remoto configurado con **rclone** o desde un **directorio local**, y genera para cada video un JSON en formato MegaDetector con las detecciones y la clasificación de especies de **SpeciesNet**. También permite correr **solo MegaDetector** (sin SpeciesNet) para generar únicamente las detecciones. Además registra en un CSV el estado y las métricas de rendimiento (tiempos, RAM, CPU, VRAM) de cada video.

Para cada video, el pipeline:

1. **Obtiene** el video: lo descarga desde el remoto con `rclone copyto` o, en modo local, lo lee directamente del directorio de origen.
2. **Extrae frames** como JPG (`frame_XXXXXX.jpg`, nombrados por su índice real en el video) usando OpenCV.
3. **Detecta** animales, personas y vehículos con MegaDetector (MDV5A), o reutiliza detecciones ya guardadas.
4. **Clasifica** con SpeciesNet únicamente las detecciones cuya confianza supera `CLASSIFY_THRESHOLD` (omitido con `--md-only`).
5. **Guarda** el JSON final (o el de detecciones, con `--md-only`) y una fila de metadatos en el CSV.
6. **Limpia** el video descargado (solo en modo remoto) y los frames temporales.

Mientras se procesa el video actual, el siguiente ya se está obteniendo en un hilo aparte (descarga con rclone), de modo que la descarga no frena la inferencia. Los modelos se cargan una sola vez por ejecución.

## Modos de ejecución

### Con `--md` (desde cero)
Se obtiene el video, se extraen los frames (según `--stride`), se corre MegaDetector y luego SpeciesNet. Con `--save-detections` se guarda el `detection_results.json` de cada video para poder reutilizarlo después.

### Con `--md-only` (solo MegaDetector)
Implica `--md` y `--save-detections`. Solo se carga MegaDetector (**no se carga SpeciesNet**), se guarda el `detection_results.json` de cada video y **no se genera** el JSON final de clasificación. El video se registra como `procesado` con el motivo `solo_megadetector`. Si ya existe el `detection_results.json` del video en `DETECTIONS_OUTPUT_DIR` y no se usa `--force`, se registra como `omitido` (`deteccion_existente`).

Es útil para separar el trabajo en dos etapas: primero detectar (`--md-only`) y después clasificar sin MegaDetector (modo sin `--md`).

### Sin `--md` (reutilizando detecciones)
Se busca un `detection_results.json` previo para cada video en `DETECTIONS_ROOT`. Si existe:

- Si **ninguna** detección supera el umbral de clasificación, el resultado final se genera directamente **sin obtener el video** (no se descarga).
- Si hay detecciones relevantes, solo se extraen y clasifican los **frames que las contienen** (se ignora `--stride`).

Si no existe el JSON de detecciones, el video se registra como `no_procesado` con el motivo `json_no_encontrado`.

> En este modo MegaDetector no se carga, lo que ahorra memoria y tiempo de arranque.

### Modo local (`--local` / `--local-dir`)
Procesa los videos desde un directorio del equipo en lugar de un remoto, sin usar rclone (no hace falta tenerlo instalado).

- El directorio de origen es `LOCAL_SOURCE_DIR`, o el indicado con `--local-dir` (que implica `--local`).
- Se buscan los videos de forma recursiva; con `--list-paths`, las rutas se interpretan relativas a ese directorio.
- Los videos originales **nunca se borran**; solo se limpian los frames temporales.
- Es combinable con cualquiera de los modos anteriores (`--md`, `--md-only`, sin `--md`).

### Forzar CPU (`--force-cpu`)
Oculta la GPU (`CUDA_VISIBLE_DEVICES=-1`) para MegaDetector y SpeciesNet. Debe pasarse por línea de comandos, ya que se evalúa antes de importar los modelos.

## Requisitos

- Python 3.11 o superior.
- [rclone](https://rclone.org/) instalado, en el `PATH` y con el remoto ya configurado (`rclone config`). **No es necesario en modo local.**
- Librerías de Python:
  - `megadetector` 
  - `speciesnet` 
  - `opencv-python`
  - `Pillow` y `numpy`
  - `psutil` *(opcional, necesario para las métricas de RAM/CPU)*
  - `torch` *(ya viene como dependencia de MegaDetector; se usa para medir el pico de VRAM)*
- GPU con CUDA recomendada para MegaDetector y SpeciesNet (se puede correr en CPU con `--force-cpu`).

> **Importante:** 
> - el script usa funciones internas (con prefijo `_`) de megadetector, que pueden cambiar entre versiones por lo que se necesita una versión fija de esta librería anotada en `requirements.txt`.
> - al instalar `megadetector` y `speciesnet` en el mismo entorno, `pip` puede mostrar un error por una incongruencia entre las versiones de la librería ONNX que requiere cada uno. Como este script no utiliza ONNX, ese error es irrelevante para el funcionamiento del pipeline y puede ignorarse.

## Configuración

Las constantes pueden editarse al inicio del script:

| Constante | Descripción |
|---|---|
| `REMOTE_FOLDER` | Nombre del remoto de rclone (ej. `"pps:"`). **Cambiar según tu configuración.** |
| `LOCAL_SOURCE_DIR` | Directorio local con los videos a procesar en modo `--local` (ej. `C:/videos`). Se puede sobrescribir con `--local-dir`. |
| `LOCAL_TEMP_DIR` | Directorio donde rclone descarga los videos (solo modo remoto). |
| `FRAMES_DIR` | Directorio temporal para los frames extraídos. |
| `RESULTS_DIR` | Directorio raíz de salidas. Contiene las subcarpetas `classifications/`, `metadata/` y `detections/`. |
| `CLASSIFICATIONS_DIR` | Directorio de los JSON finales de clasificación (`RESULTS_DIR/classifications`). |
| `METADATA_DIR` / `METADATA_FILE` | Directorio y archivo del CSV de metadatos (`RESULTS_DIR/metadata/processing_metadata.csv`). |
| `DETECTIONS_OUTPUT_DIR` | Directorio donde `--save-detections` / `--md-only` **guardan** los `detection_results.json` (`RESULTS_DIR/detections`). |
| `DETECTIONS_ROOT` | Directorio de **entrada**: de donde se **leen** los `detection_results.json` al correr sin `--md` (solo SpeciesNet). Es independiente de `DETECTIONS_OUTPUT_DIR`; para reutilizar lo recién guardado, apuntarlo a `result/detections` o copiar allí los archivos. |
| `VIDEO_EXTS` | Extensiones de video aceptadas (`.mp4`, `.avi`, `.mov`). |
| `CLASSIFY_THRESHOLD` | Confianza mínima de una detección para enviarla a SpeciesNet (por defecto, el valor de SpeciesNet). |
| `DOWNLOAD_TIMEOUT_SECONDS` | Timeout por descarga (20 min). Evita que un rclone colgado detenga todo. |
| `QUEUE_POLL_SECONDS` | Cada cuántos segundos se revisan errores en los hilos de clasificación. |
| `BENCHMARK_SAMPLE_SECONDS` | Intervalo de muestreo de recursos para benchmarking (0.5 s). |

> `FRAMES_DIR` se **vacía** automáticamente (al iniciar y tras cada video) y `LOCAL_TEMP_DIR` se vacía al finalizar (solo en modo remoto). No hay que ingresar datos que se quieran conservar en estas carpetas. `LOCAL_SOURCE_DIR` nunca se modifica.

## Uso

```bash
# Procesar todos los videos del remoto, corriendo MegaDetector + SpeciesNet
python mdv5_sn.py --md

# Igual, guardando las detecciones para reutilizarlas más adelante
python mdv5_sn.py --md --save-detections

# Procesar 1 de cada 5 frames, con lotes de 16
python mdv5_sn.py --md --stride 5 --batch-size 16

# Procesar solo los videos listados en un archivo
python mdv5_sn.py --md --list-paths videos.out

# Solo MegaDetector: genera únicamente los detection_results.json
python mdv5_sn.py --md-only

# Reutilizar detecciones existentes (sin MegaDetector)
python mdv5_sn.py

# Reprocesar aunque ya exista el resultado
python mdv5_sn.py --force

# Procesar videos desde un directorio local (sin rclone)
python mdv5_sn.py --md --local-dir "D:/mis_videos"

# Forzar ejecución en CPU
python mdv5_sn.py --md --force-cpu
```

### Argumentos

| Argumento | Descripción |
|---|---|
| `--md` | Ejecuta MegaDetector + SpeciesNet desde cero. Sin este flag se usan detecciones existentes. |
| `--md-only` | Ejecuta solo MegaDetector: implica `--md` y `--save-detections`, no carga SpeciesNet y no genera el JSON final de clasificación. |
| `--list-paths <ruta a la lista>` | Archivo de texto (uno por línea) con las rutas de los videos a procesar, relativas a `REMOTE_FOLDER` (o a `LOCAL_SOURCE_DIR` en modo local). Se ignoran líneas vacías y extensiones no válidas. Si se omite, se listan todos los videos del remoto con `rclone lsf` (o del directorio local, de forma recursiva). |
| `--stride <n>` | Solo con `--md`: procesa 1 de cada n frames (default: 1, todos). |
| `--batch-size <n>` | Solo con `--md`: frames por lote en MegaDetector (default: 8). |
| `--save-detections` | Solo con `--md`: guarda el `detection_results.json` de cada video en `DETECTIONS_OUTPUT_DIR` (`result/detections`). |
| `--force` | Reprocesa videos aunque ya exista su JSON de salida (o, con `--md-only`, su JSON de detecciones). |
| `--force-cpu` | Fuerza CPU: oculta la GPU para MegaDetector y SpeciesNet. |
| `--local` | Procesa videos desde `LOCAL_SOURCE_DIR` sin usar rclone. |
| `--local-dir <ruta>` | Directorio local de videos (sobrescribe `LOCAL_SOURCE_DIR`). Implica `--local`. |

## Estructura de archivos

Dado un video `FotosCamarasTrampas/SL009/20240620/DCIM/100_BTCF/IMG_0007.MP4` (relativo a `REMOTE_FOLDER` o a `LOCAL_SOURCE_DIR`):

```
data/
├── rclone_local/                 # descargas temporales (se limpia; solo modo remoto)
├── temp_frames/                  # frames temporales (se limpia)
└── result/
    ├── detections/
    │   └── FotosCamarasTrampas/SL009/20240620/DCIM/100_BTCF/IMG_0007/
    │       └── detection_results.json    # detecciones de MegaDetector
    ├── classifications/
    │   └── FotosCamarasTrampas/SL009/20240620/DCIM/100_BTCF/IMG_0007.json   
    │                               # resultado final (detección + clasificación)
    └── metadata/
        └── processing_metadata.csv       # estado y métricas de todos los videos
```

Se conserva la estructura de carpetas del origen tanto en `detections/` como en `classifications/`. Sin `--md`, las detecciones se leen desde `DETECTIONS_ROOT` con esa misma estructura (`<DETECTIONS_ROOT>/<carpetas>/<video>/detection_results.json`). Con `--md-only` solo se genera el `detection_results.json` (y el CSV de metadatos).

## Salidas

### JSON de resultados (`result/classifications/`)
Formato MegaDetector estándar, con cada detección que incluye `classifications` (las 2 mejores especies y su score), más `classification_categories` y `classification_category_descriptions`. En `info` se agregan el modelo clasificador y la fecha de clasificación.

### JSON de detecciones (`result/detections/`)
Con `--save-detections` o `--md-only`, el `detection_results.json` de cada video en formato MegaDetector, sin clasificaciones.

### CSV de metadatos (`result/metadata/processing_metadata.csv`)
Una fila por video, con las columnas:

- **Estado:** `video_path`, `status`, `reason`, `detection_json`, `result_json`
- **Tiempos (s):** `extract_seconds`, `megadetector_seconds`, `speciesnet_seconds`, `processing_seconds`
- **Conteos:** `frames_extracted`, `detections`, `classified_detections`
- **Recursos:** `ram_start_mb`, `ram_peak_mb`, `ram_end_mb`, `cpu_avg_percent`, `cpu_peak_percent`
- **VRAM (MB):** `gpu_peak_mb` (pico global del video), `megadetector_gpu_peak_mb` y `speciesnet_gpu_peak_mb` (pico de cada etapa por separado)

Valores posibles de `status`:

| Estado | Significado |
|---|---|
| *procesado* | El video se procesó y se generó su JSON de resultado (o de detecciones, con `--md-only`). También se usa cuando no había detecciones para clasificar (`sin_detecciones_para_clasificar`). |
| *omitido* | Ya existía un resultado (`resultado_existente`) o, con `--md-only`, ya existía el JSON de detecciones (`deteccion_existente`), y no se usó `--force`. |
| *no_procesado* | Hubo un error (descarga, lectura de JSON, procesamiento) o faltaba el JSON de detecciones. El campo `reason` indica el motivo. |

Si el CSV existente tiene un encabezado distinto al actual, se renombra como `processing_metadata_legacy_<fecha>.csv` (en la misma carpeta `metadata/`) y se crea uno nuevo.

## Notas y consideraciones

- **Reanudación:** como los videos con resultado existente se omiten, se puede volver a ejecutar el script tras una interrupción y continuará con los pendientes.
- **Rutas duplicadas:** si la lista de videos contiene rutas repetidas, se descartan automáticamente. Las rutas deben ser relativas; las absolutas o con `..` se rechazan.
- **Métricas:** si `psutil` no está instalado, las métricas de RAM/CPU quedan en 0. El pico de VRAM solo se registra si hay GPU CUDA disponible (y no se usa `--force-cpu`).
- **Nombres de frames:** en modo sin `--md`, el índice del frame se obtiene del último segmento del nombre separado por `_` (ej. `frame_000123.jpg`). Si algún nombre no cumple ese formato, se extraen todos los frames del video.
- **Errores en hilos:** si falla un hilo de SpeciesNet, el error se propaga y el video se registra como `no_procesado` en lugar de quedar bloqueado.
- **Descargas fallidas:** se registran como `error_descarga` y el pipeline continúa con el siguiente video. En modo local, un video inexistente se registra de la misma forma.

## Documentación

Carpeta de Google Drive donde se almacenan las distintas versiones de los informes que se han ido presentando a lo largo del proyecto.
```
  https://drive.google.com/drive/folders/1BDjeB0_tliaYmY3Q7sRNUsPtXjNfCtlc
```