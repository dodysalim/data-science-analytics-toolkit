# 📊 Análisis Exploratorio de Datos (EDA) - Customer Support

Este directorio contiene el notebook principal de análisis de datos del proyecto: **`01_customer_support_eda.ipynb`**.

## 🎯 Objetivo del Notebook
Realizar un Análisis Exploratorio de Datos (EDA) sumamente profundo sobre el conjunto de datos de tickets de soporte técnico (`customer_support_data.csv`). El análisis está estructurado metódicamente en **18 pasos** para extraer conclusiones de negocio accionables, evaluar la calidad de los datos y encontrar patrones ocultos en el comportamiento de los clientes.

## 🛠️ Estructura del Análisis (18 Pasos)
El notebook está documentado profesionalmente en español y abarca:
1.  **Limpieza y Formateo:** Conversión de tipos de datos, tratamiento de fechas y manejo de valores nulos o duplicados.
2.  **Calidad de Datos:** Análisis de cardinalidad y verificación de consistencia.
3.  **Análisis Univariado:** Distribuciones de canales, idiomas e industrias.
4.  **Análisis de Sentimiento y Urgencia:** Evaluación de la satisfacción general.
5.  **Análisis Temporal:** Descubrimiento de tendencias semanales y mapas de calor (Heatmaps) de horarios pico.
6.  **Análisis Bivariado:** Cruce de variables como Industria vs Urgencia y Canal vs Sentimiento.
7.  **Análisis de Outcomes (Resultados):** Cuantificación del éxito de las resoluciones vs tickets escalados.
8.  **Feature Engineering Básico:** Relación entre la longitud del texto escrito por el cliente y su nivel de frustración o urgencia.

## 🚀 Uso
Para reproducir el análisis, asegúrate de tener el entorno de Python configurado y ejecuta:
```bash
jupyter notebook 01_customer_support_eda.ipynb
```
*(Asegúrate de que el archivo `data/customer_support_data.csv` se encuentre en la raíz del proyecto, tal como espera el código).*

---
**Autor:** Dody Dueñas
