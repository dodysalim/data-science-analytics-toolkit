# 🗄️ SQL Analytics — Queries Avanzadas de Negocio

Colección de queries SQL de producción para análisis de negocio avanzado, enfocadas en Retail, E-commerce y plataformas digitales.

## 📁 Estructura

```
04_sql_analytics/
├── analytics_queries.sql    # Colección completa de queries
└── README.md
```

## 📋 Queries Incluidas

| Query | Descripción |
|-------|-------------|
| Estadísticas Descriptivas | Media, mediana, percentiles por grupos |
| Análisis RFM | Recencia, Frecuencia y Valor Monetario del cliente |
| Cohort Analysis | Retención mes a mes de cohortes de usuarios |
| Funnel de Conversión | Métricas de cada etapa del embudo de ventas |
| Window Functions | Rankings, acumulados y particiones temporales |
| Churn Prediction SQL | Identificación de clientes en riesgo de abandono |

## 🚀 Uso

Las queries están estructuradas para ser portables. Simplemente reemplaza el nombre de la tabla en la cláusula `FROM`:

```sql
-- Antes
FROM customer_transactions

-- Después (adapta a tu esquema)
FROM schema.tu_tabla
```

Compatible con: **PostgreSQL · MySQL · BigQuery · Snowflake · SQL Server**
