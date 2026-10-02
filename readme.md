# Procesamiento de videos de cámaras trampa con MegaDetector + SpeciesNet
 
Proyecto para el análisis automatizado de material de cámaras trampa. Combina dos modelos en cascada: **MegaDetector** localiza animales, personas y vehículos, y **SpeciesNet** clasifica la especie de los animales detectados.
 
El repositorio incluye dos versiones del pipeline, según dónde se encuentren los datos de entrada. Cada una tiene su propio README con la instalación, la configuración y el uso en detalle.
 
| Versión | Directorio | Entrada | Qué hace |
|---|---|---|---|
| **Remota** | [`pipeline_remote/`](pipeline_remote/README.md) | Videos en un almacenamiento remoto (rclone) | Descarga cada video, extrae frames, detecta (opcional) y clasifica |
| **Local** | [`pipeline_local/`](pipeline_local/README.md) | Imágenes en disco + JSON de detecciones de MegaDetector | Clasifica con SpeciesNet las detecciones ya existentes |
 
## Cómo funciona (visión general)
 
Ambas versiones comparten la misma lógica de dos etapas:
 
```
Imágenes/frames → MegaDetector (detección) → SpeciesNet (clasificación por especie) → JSON de resultados
```
 
1. **Detección (MegaDetector):** para cada imagen se obtienen las cajas delimitadoras de cada detección, con su categoría (animal, persona o vehículo) y su confianza.
2. **Clasificación (SpeciesNet):** se clasifican las detecciones de animales para asignarles una especie. SpeciesNet puede aplicar un filtro geográfico (*geofence*) según el país configurado.
3. **Resultados:** las clasificaciones se guardan en un JSON.
La diferencia entre versiones está en cómo se obtienen las entradas y qué etapas ejecuta cada una:
 
- **Remota:** parte de videos. Se encarga de descargarlos con rclone, extraer sus frames y, según el modo, correr MegaDetector desde cero o reutilizar detecciones previas. Mientras procesa un video, descarga el siguiente. Además registra el estado y las métricas de recursos de cada video en un CSV.
- **Local:** parte de un JSON de detecciones ya generado (con las rutas de las imágenes) y se limita a la etapa de clasificación. Omite las rutas que no existen en disco.

## Estructura del repositorio
 
```
.
├── README.md
├── requirements.txt
├── pipeline_remote/
│   ├── README.md
│   ├── csv_to_html.py
│   └── rclone_mdsn.py
└── pipeline_local/
    ├── README.md
    ├── preprocesamiento.py
    ├── classifier.py
    ├── run_model.py
    └── main.py
```
 
## Requisitos comunes
 
- Python 3.11 o superior (recomendado un entorno virtual).
- [SpeciesNet](https://github.com/google/cameratrapai) y [MegaDetector](https://github.com/agentmorris/MegaDetector).
- GPU con CUDA recomendada para la inferencia.

 
Las dependencias específicas de cada versión se detallan en su README.
 
## Documentación
 
Carpeta de Google Drive donde se almacenan las distintas versiones de los informes que se han ido presentando a lo largo del proyecto.
```
  https://drive.google.com/drive/folders/1BDjeB0_tliaYmY3Q7sRNUsPtXjNfCtlc
```