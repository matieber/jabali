import json
from pathlib import Path

# =========================================================================
# CONFIGURACIÓN DE RUTAS Y PARÁMETROS
# =========================================================================

# Archivo de entrada (salida que generó SpeciesNet)
INPUT_JSON = Path("/home/luciano/Escritorio/Resultados_SN/especies_resultados_finales_SL008.json")

# Archivo donde se guardarán solo las detecciones de jabalíes
OUTPUT_JSON = Path("/home/luciano/Escritorio/Resultados_SN/jabalies_detectados.json")

# Umbral mínimo de confianza (0.30 capturará perfectamente el 0.9865 de tu imagen)
MIN_CONFIDENCE = 0.6

# Palabras clave para detectar jabalí en la cadena taxonómica
BOAR_KEYWORDS = ["sus scrofa", "wild boar", "jabali", "jabalí", "pig", "suidae"]


def es_jabali(label: str) -> bool:
    """Comprueba si la cadena de clasificación contiene alguna palabra clave."""
    if not label:
        return False
    label_lower = label.lower()
    return any(keyword in label_lower for keyword in BOAR_KEYWORDS)


def filtrar_jabalies(input_path: Path, output_path: Path, min_conf: float):
    if not input_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de entrada: {input_path}")

    print(f"Cargando JSON desde: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Si el JSON viene dentro de un objeto principal, extraer la lista
    predictions = data.get("predictions", data) if isinstance(data, dict) else data

    resultados_jabalies = []
    conteo_jabalies = 0

    print("Filtrando predicciones...")

    for item in predictions:
        filepath = item.get("filepath", "")
        classifications = item.get("classifications", {})

        # Extraer las dos listas paralelas de SpeciesNet
        classes = classifications.get("classes", [])
        scores = classifications.get("scores", [])

        detalles_match = []

        # Iterar en paralelo sobre clases y puntajes
        for cls_str, score in zip(classes, scores):
            conf = float(score)
            
            # Verificar si coincide con jabalí y supera la confianza mínima
            if es_jabali(cls_str) and conf >= min_conf:
                detalles_match.append({
                    "etiqueta_completa": cls_str,
                    "confianza": round(conf, 4)
                })

        # Si se encontró al menos un match válido en el cuadro/imagen
        if detalles_match:
            conteo_jabalies += 1
            resultados_jabalies.append({
                "filepath": filepath,
                "detecciones_jabali": detalles_match
            })

    # Guardar los resultados filtrados
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_imagenes_analizadas": len(predictions),
            "total_hallazgos_jabali": conteo_jabalies,
            "umbral_confianza": min_conf,
            "resultados": resultados_jabalies
        }, f, ensure_ascii=False, indent=4)

    print("\n" + "="*50)
    print("FILTRADO COMPLETADO")
    print("="*50)
    print(f"• Total imágenes analizadas: {len(predictions)}")
    print(f"• Detecciones con jabalí encontradas: {conteo_jabalies}")
    print(f"• Guardado en: {output_path}")


if __name__ == "__main__":
    filtrar_jabalies(INPUT_JSON, OUTPUT_JSON, MIN_CONFIDENCE)


