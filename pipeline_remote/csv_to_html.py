import csv
import json
import sys


def generar_dashboard_html(ruta_csv_entrada, ruta_html_salida):
    filas = []

    total_videos = 0
    procesados = 0
    no_procesados = 0
    total_detecciones = 0
    tiempo_total_sec = 0.0

    with open(ruta_csv_entrada, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_videos += 1
            status = row.get("status", "").strip().lower()

            video_path = row.get("video_path", "")
            nombre_archivo = (
                video_path.replace("\\", "/").split("/")[-1]
                if video_path
                else "Desconocido"
            )

            det_totales = (
                int(float(row["detections"])) if row.get("detections") else 0
            )
            det_clasificadas = (
                int(float(row["classified_detections"]))
                if row.get("classified_detections")
                else 0
            )

            tiempo_proc = (
                float(row["processing_seconds"])
                if row.get("processing_seconds")
                else 0.0
            )
            ram_pico = (
                float(row["ram_peak_mb"]) if row.get("ram_peak_mb") else 0.0
            )
            gpu_pico = (
                float(row["gpu_peak_mb"]) if row.get("gpu_peak_mb") else 0.0
            )
            cpu_prom = (
                float(row["cpu_avg_percent"])
                if row.get("cpu_avg_percent")
                else 0.0
            )

            if status == "procesado":
                procesados += 1
                total_detecciones += det_totales
                tiempo_total_sec += tiempo_proc
            else:
                no_procesados += 1

            filas.append({
                "archivo": nombre_archivo,
                "path": video_path,
                "status": status,
                "reason": row.get("reason", ""),
                "frames": (
                    int(float(row["frames_extracted"]))
                    if row.get("frames_extracted")
                    else 0
                ),
                "detections": det_totales,
                "classified": det_clasificadas,
                "extract_sec": (
                    round(float(row["extract_seconds"]), 2)
                    if row.get("extract_seconds")
                    else 0
                ),
                "megadetector_sec": (
                    round(float(row["megadetector_seconds"]), 2)
                    if row.get("megadetector_seconds")
                    else 0
                ),
                "speciesnet_sec": (
                    round(float(row["speciesnet_seconds"]), 2)
                    if row.get("speciesnet_seconds")
                    else 0
                ),
                "proc_sec": round(tiempo_proc, 2),
                "ram_peak": round(ram_pico, 1),
                "cpu_avg": round(cpu_prom, 1),
                "gpu_peak": round(gpu_pico, 1),
            })

    datos_json = json.dumps(filas, ensure_ascii=False)

    html_content = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Reporte de Procesamiento de Videos</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdn.datatables.net/1.13.6/css/dataTables.bootstrap5.min.css">
    <style>
        body {{ background-color: #f8f9fa; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }}
        .card-stat {{ border: none; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }}
        .badge-status {{ font-size: 0.85em; padding: 6px 10px; border-radius: 6px; }}
        .table-container {{ background: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }}
        .text-error {{ font-size: 0.85em; color: #dc3545; word-break: break-word; max-width: 300px; }}
        .path-sub {{ font-size: 0.75em; color: #6c757d; display: block; word-break: break-all; }}
    </style>
</head>
<body class="py-4">
    <div class="container-fluid px-4">
        <h2 class="mb-4 text-primary fw-bold">Reporte de Procesamiento de Videos ({total_videos} archivos)</h2>
        
        <div class="row g-3 mb-4">
            <div class="col-md-3">
                <div class="card card-stat bg-white p-3 border-start border-primary border-4">
                    <div class="text-muted small">Total de Videos</div>
                    <div class="fs-3 fw-bold text-dark">{total_videos}</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card card-stat bg-white p-3 border-start border-success border-4">
                    <div class="text-muted small">Procesados Exitosamente</div>
                    <div class="fs-3 fw-bold text-success">{procesados}</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card card-stat bg-white p-3 border-start border-danger border-4">
                    <div class="text-muted small">Fallidos / Errores</div>
                    <div class="fs-3 fw-bold text-danger">{no_procesados}</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card card-stat bg-white p-3 border-start border-warning border-4">
                    <div class="text-muted small">Detecciones Totales</div>
                    <div class="fs-3 fw-bold text-warning">{total_detecciones:,}</div>
                </div>
            </div>
        </div>

        <div class="table-container">
            <table id="tablaVideos" class="table table-hover align-middle w-100">
                <thead class="table-light">
                    <tr>
                        <th>Archivo / Ruta</th>
                        <th>Estado</th>
                        <th>Detecciones (Clasif.)</th>
                        <th>Tiempo (s)</th>
                        <th>T. SpeciesNet</th>
                        <th>RAM Pico (MB)</th>
                        <th>GPU Pico (MB)</th>
                        <th>Detalles / Error</th>
                    </tr>
                </thead>
                <tbody></tbody>
            </table>
        </div>
    </div>

    <script src="https://code.jquery.com/jquery-3.7.0.min.js"></script>
    <script src="https://cdn.datatables.net/1.13.6/js/jquery.dataTables.min.js"></script>
    <script src="https://cdn.datatables.net/1.13.6/js/dataTables.bootstrap5.min.js"></script>
    
    <script>
        const datos = {datos_json};

        $(document).ready(function() {{$('#tablaVideos').DataTable({{
                data: datos,
                pageLength: 25,
                lengthMenu: [[10, 25, 50, 100, -1], [10, 25, 50, 100, "Todos"]],
                order: [[1, 'asc']],
                columns: [
                    {{ 
                        data: 'archivo',
                        render: function(data, type, row) {{
                            return `<strong>${{data}}</strong><span class="path-sub">${{row.path}}</span>`;
                        }}
                    }},
                    {{ 
                        data: 'status',
                        render: function(data) {{
                            if (data === 'procesado') {{
                                return '<span class="badge bg-success badge-status">PROCESADO</span>';
                            }} else {{
                                return '<span class="badge bg-danger badge-status">ERROR</span>';
                            }}
                        }}
                    }},
                    {{ 
                        data: 'detections',
                        render: function(data, type, row) {{
                            if (row.status !== 'procesado') return '-';
                            return `<strong>${{data}}</strong> <span class="text-muted">(${{row.classified}})</span>`;
                        }}
                    }},
                    {{ data: 'proc_sec', render: d => d > 0 ? d + 's' : '-' }},
                    {{ data: 'speciesnet_sec', render: d => d > 0 ? d + 's' : '-' }},
                    {{ data: 'ram_peak', render: d => d > 0 ? d + ' MB' : '-' }},
                    {{ data: 'gpu_peak', render: d => d > 0 ? d + ' MB' : '-' }},
                    {{ 
                        data: 'reason',
                        render: function(data, type, row) {{
                            if (row.status === 'procesado') return '<span class="text-success">OK</span>';
                            return `<div class="text-error" title="${{data}}">${{data}}</div>`;
                        }}
                    }}
                ],
                language: {{
                    url: 'https://cdn.datatables.net/plug-ins/1.13.6/i18n/es-ES.json'
                }}
            }});
        }});
    </script>
</body>
</html>
"""

    with open(ruta_html_salida, "w", encoding="utf-8") as f:
        f.write(html_content)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Uso: python csv_to_html.py <archivo_csv_entrada> <archivo_html_salida>")
        sys.exit(1)

    archivo_csv = sys.argv[1]
    archivo_html = sys.argv[2]

    generar_dashboard_html(archivo_csv, archivo_html)