# Verificación · Customer Support Analytics

Fecha: 7 de octubre de 2026 (Ecuador). Entorno local Python 3.12.14. Las dependencias de QA se registran aparte; esta comprobación no certifica todas las combinaciones de versiones del proyecto.

## Ejecutado

Siete vistas probadas sin excepciones con una muestra de 50.000 filas originales. No existe cobertura global de pruebas para todos los módulos.

## Dependencias externas y límites

El CSV grande requiere Git LFS. También puedes definir CUSTOMER_SUPPORT_DATA con la ruta de un CSV disponible. El dashboard carga primeras filas, no una muestra aleatoria.

Las conversaciones se deduplican por conv_id para los KPIs. La comparación A/B es observacional y no demuestra causalidad. El NLP usa un subconjunto; las tablas precalculadas de Power BI conservan ese alcance.

## Presentación Power BI

Las definiciones se revisaron para límites y superposiciones, y el diseño móvil sigue el esquema oficial PBIR. La prueba nativa completa en teléfono permanece pendiente. Las fuentes externas deben exportarse antes de actualizar las páginas sin datos.
