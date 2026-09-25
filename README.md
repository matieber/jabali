## Ejecución
* 1 - Configurar las rutas en `preprocesamiento.py`
* 2 - Correr `preprocesamiento.py`: Guarda los frames temporales y el json input para species net
* 3 - Correr `main.py` : prepara las instancias y corre Species Net

---

## Componentes y Funcionamiento

### Archivos Principales

* **`preprocesamiento.py`**  
  Encargado de la preparación de datos. Indexa los videos y los JSONs de MegaDetector mediante una clave única (`SL---/fecha/video`). Lee las detecciones, filtra únicamente los frames con presencia de animales (categoría `1`) y utiliza OpenCV para extraer solo esos fotogramas a la carpeta `temp_frames/`. Finalmente, reestructura y guarda el archivo `temp_detections.json` con el formato exacto exigido por SpeciesNet.

  > **Configuración requerida:** hay que definir e ingresar las rutas hacia las carpetas de videos (`VIDEOS_DIR`) y JSONs de MegaDetector (`JSONS_DIR`) directamente dentro del script `preprocesamiento.py` antes de la primera ejecución.

* **`main.py`**  
  Corre Species Net. Lee `temp_detections.json` y verifica la existencia física de cada fotograma en el disco (`os.path.isfile`). Si se movieron o eliminaron imágenes manualmente de la carpeta temporal (temp_frames), el script evita procesarlas. Luego prepara el mapa de instancias con la geolocalización correspondiente (`ARG`) y ejecuta SpeciesNet para generar el archivo con los resultados de la clasificación.

---

### Archivos y Carpetas Temporales

* **`temp_frames/` (Carpeta temporal)**  
  Almacena únicamente los fotogramas extraídos en formato `.jpg` que contienen animales confirmados por MegaDetector. Sirve como el conjunto de imágenes sobre el cual SpeciesNet realizará la clasificación.

* **`temp_detections.json` (Archivo temporal)**  
  Contiene la lista filtrada de detecciones (*bounding boxes*, etiquetas y nivel de confianza) asociadas a las imágenes guardadas en `temp_frames/`. Sincroniza la información devuelta por MegaDetector con las entradas requeridas para alimentar el modelo de clasificación.