import os
import json
from pathlib import Path
from tqdm import tqdm
from speciesnet import SpeciesNet, DEFAULT_MODEL
from speciesnet.utils import prepare_instances_dict

# En el main se corre el modelo de SpeciesNet a partir del archivo temp_detections que contiene los frames de los videos
# -------------------------------------------------------------------------
# Configuración
# -------------------------------------------------------------------------
OUTPUT_JSON = "especies_resultados_finales.json"
COUNTRY_CODE = "ARG"
TEMP_DETECTIONS_JSON = "/media/luciano/Local Disk/temp_proyecto/temp_detections.json"

def main():   
    json_path = Path(TEMP_DETECTIONS_JSON)
    print("Preparación de instancias y carga de archivos...")
    if not json_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {TEMP_DETECTIONS_JSON}")

    # cargar el JSON temporal
    print("Leyendo detecciones temporales...")
    with open(json_path, "r", encoding="utf-8") as f:
        raw_json_data = json.load(f)

    predictions_data = raw_json_data.get("predictions", [])

    # extraer rutas y filtrar solo las que EXISTEN en disco (con barra de progreso)
    all_filepaths = []
    filtered_detections_dict = {}

    for item in tqdm(predictions_data, desc="Verificando imágenes en disco"):
        fp = item.get("filepath")
        if fp and os.path.isfile(fp):
            all_filepaths.append(fp)
            filtered_detections_dict[fp] = {"detections": item.get("detections", [])}

    if not all_filepaths:
        raise RuntimeError("No se encontraron imágenes válidas en el disco que coincidan con el JSON.")

    print(f"Total de imágenes listas para clasificar: {len(all_filepaths)}")

    # preparar mapa de instancias
    print("Estructurando instancias para SpeciesNet...")
    instances_dict = prepare_instances_dict(
        filepaths=all_filepaths,
        country=COUNTRY_CODE
    )

    # clasificación
    model = SpeciesNet(DEFAULT_MODEL, components="classifier", geofence=True)

    print("Ejecutando clasificación con SpeciesNet...")    
    model.classify(
        instances_dict=instances_dict,
        detections_dict=filtered_detections_dict,
        run_mode="multi_thread",
        batch_size=8,
        progress_bars=True,  #mantiene activa la barra nativa de inferencia del modelo
        predictions_json=OUTPUT_JSON
    )

    print(f"\n¡Proceso finalizado con éxito! Predicciones guardadas en: {OUTPUT_JSON}")

if __name__ == "__main__":
    main()