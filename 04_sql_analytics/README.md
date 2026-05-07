# 📊 SQL Analytics Queries Collection

Colección curada de **queries SQL avanzados** para Análisis de Datos. Compatibles con **PostgreSQL**, **Snowflake**, **BigQuery**, y **SQL Server**.

## 📁 Módulos Incluidos

| # | Módulo | Descripción |
|---|--------|-------------|
| 1 | **Estadísticas Descriptivas** | Media, mediana, percentiles, IQR, distribución de categorías |
| 2 | **Análisis Temporal** | Series de tiempo, crecimiento MoM, medias móviles |
| 3 | **Análisis de Cohortes** | Retención de usuarios por cohorte mensual |
| 4 | **RFM Analysis** | Segmentación de clientes: Recency, Frequency, Monetary |
| 5 | **Ranking & Top-N** | RANK, DENSE_RANK, TOP productos/vendedores por región |
| 6 | **Detección de Anomalías** | Outliers con Z-Score e IQR en SQL puro |
| 7 | **KPIs de Negocio** | Dashboard de métricas, tasa de conversión por canal |

## 💡 Highlights

```sql
-- Análisis RFM en una sola query con segmentación automática
SELECT cliente_id, rfm_total,
    CASE
        WHEN r_score >= 4 AND f_score >= 4 AND m_score >= 4 THEN '🏆 Champions'
        WHEN r_score <= 2 AND f_score <= 2 AND m_score <= 2 THEN '🚨 Lost Customers'
        ELSE '📌 Regular'
    END AS segmento
FROM rfm_scored;
```

## 🗄️ Compatibilidad

| Feature SQL | PostgreSQL | Snowflake | BigQuery | SQL Server |
|-------------|:---:|:---:|:---:|:---:|
| Window Functions | ✅ | ✅ | ✅ | ✅ |
| PERCENTILE_CONT | ✅ | ✅ | ✅ | ✅ |
| CTEs (WITH) | ✅ | ✅ | ✅ | ✅ |
| DATE_TRUNC | ✅ | ✅ | ✅ | ⚠️ |

---
> **Autor:** Dody Dueñas | Data Analyst & Data Scientist
