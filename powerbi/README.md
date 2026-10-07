# Atención al cliente · muestra · Power BI

Muestra: primeras 50.000 filas del archivo original Git LFS; una fila por conversación antes de agrupar. La muestra puede contener conversaciones incompletas y no representa todo el dataset.

Datos: 297 filas en la exportación. El CSV omite nombres, teléfonos y correos.

Abra `Analytics.pbip` en Power BI Desktop y cambie el parámetro `DataFolder` a la ruta absoluta de `powerbi/data/` (con separador final). Luego actualice los datos. Alternativamente ejecute `python configure_data.py` para configurar la ruta.

Incluye modelo TMDL, Power Query, medidas DAX explícitas, tarjetas y tabla de detalle. Los archivos JSON y las referencias se verificaron por código; apertura, actualización y renderizado en Power BI Desktop pendientes. No se afirma equivalencia funcional completa con la aplicación Python.

La exportación se reconstruye con `build_powerbi.py` durante la preparación del portafolio. La fuente original permanece en este repositorio.
