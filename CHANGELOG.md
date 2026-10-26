# Changelog

Todos los cambios relevantes de este proyecto se documentan aquí.
El formato sigue [Keep a Changelog](https://keepachangelog.com/es-ES/1.1.0/) y [Versionado Semántico](https://semver.org/lang/es/).

## [1.2.0] - 2026-10-26

### 🔧 Mantenimiento

- **Diseño moderno:** las 7 vistas del dashboard pasan de imágenes estáticas de matplotlib a gráficos interactivos de Plotly con tema oscuro común, encabezado destacado y tarjetas de métricas.
- **Modularización:** fábrica de gráficos única (`views/components.py`), tema central (`views/theme.py`) y cálculos puros reutilizables (`views/analytics.py`) que antes se repetían en cada vista.
- **SOLID:** el enrutamiento por cadena de `if/elif` se reemplaza por un registro de páginas (agregar una vista = agregar una entrada).
- 3 pruebas nuevas para los cálculos de las vistas (12 en total).
- Verificación: las 7 vistas abren sin excepciones con el dataset real.

## [1.1.0] - 2026-10-26

### 🔧 Mantenimiento

- **Corregido:** `fill_nulls_smart` no rellenaba nulos en pandas moderno (uso de `fillna(inplace=True)` sobre una copia). Ahora asigna el resultado explícitamente; la estrategia `mode` usa la moda también en columnas numéricas, como indica su documentación.
- 9 pruebas automatizadas para limpieza, estadística, pruebas A/B y perfilado de datos.
- Flujo de CI que corre las pruebas en cada cambio (no requiere descargar el CSV de LFS).
- Verificación: las 7 vistas del dashboard abren sin excepciones con el dataset real.
- Licencia MIT e insignias en el README.
