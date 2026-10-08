# Customer Support Analytics — Power BI

Muestra: primeras 50.000 filas del CSV original · una fila por conversación · NLP: primeros 500 mensajes de clientes

## Abrir

1. Descarga el repositorio completo.
2. Ejecuta `python powerbi/configure_data.py`. Alternativamente, en Transformar datos → Administrar parámetros cambia `DataFolder` a la carpeta `powerbi/data/` con separador final.
3. Abre `powerbi/Analytics.pbip` en Power BI Desktop y pulsa Actualizar.

## Estructura y correspondencia

| Streamlit | Power BI |
|---|---|
| Resumen | Conversaciones, resolución, tendencias |
| Distribución | Canal/idioma/incidencia y desglose |
| Sentimientos | Sentimiento e industria |
| Urgencia y resultados | Urgencia, outcomes y resolución |
| NLP | TF-IDF, LDA, bigramas y léxico sobre 500 mensajes |
| A/B | Todos los pares de canales, Z, IC95 y posterior Beta |
| Calidad | Nulos y cardinalidad de las 50.000 filas |

La muestra conserva el límite inicial de Streamlit: no representa los 629 MB del dataset completo. Se omiten nombres y textos completos. Los resultados A/B no prueban efecto causal.

Las páginas conservan el análisis del proyecto original. Los controles de entrenamiento, conexión, escritura SQL e inferencia en vivo siguen en Python/Streamlit. El informe consume resultados exportados; no reemplaza esos servicios. Los CSV conservan su grano, y las medidas evitan sumar porcentajes o promedios.

## Verificación de esta entrega

El serializador TMDL nativo instalado con Power BI Desktop aceptó el modelo. Se comprobaron las referencias de los campos y los límites de cada visual. Esto valida la estructura; la apertura, actualización y representación de los gráficos se comprueban por separado. Los proyectos sin datos siguen pendientes.

Para regenerar los CSV y el informe desde las fuentes del repositorio: `python powerbi/rebuild_report.py`, seguido de `python powerbi/configure_data.py`. Requiere pandas, numpy y scikit-learn; M5 y atención al cliente también scipy; M5 openpyxl. Los datos externos deben descargarse antes.
