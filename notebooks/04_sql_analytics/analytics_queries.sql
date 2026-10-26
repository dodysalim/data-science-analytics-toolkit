-- ============================================================
--  📊 SQL Analytics Queries Collection
--  Colección de queries SQL para Análisis de Datos avanzado
--  Autor: Dody Dueñas | Data Analyst
--  Fecha: 2026
-- ============================================================


-- ════════════════════════════════════════════════════════════
--  MÓDULO 1: ESTADÍSTICAS DESCRIPTIVAS
-- ════════════════════════════════════════════════════════════

-- 1.1 Estadísticas descriptivas completas de una columna numérica
SELECT
    COUNT(valor)                                     AS total_registros,
    COUNT(*) - COUNT(valor)                          AS valores_nulos,
    ROUND(AVG(valor), 2)                             AS media,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY valor) AS mediana,
    ROUND(STDDEV(valor), 2)                          AS desviacion_estandar,
    ROUND(VARIANCE(valor), 2)                        AS varianza,
    MIN(valor)                                       AS minimo,
    MAX(valor)                                       AS maximo,
    MAX(valor) - MIN(valor)                          AS rango,
    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY valor) AS q1,
    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY valor) AS q3,
    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY valor)
    - PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY valor) AS iqr
FROM tabla_ejemplo;


-- 1.2 Frecuencia y porcentaje de cada categoría
SELECT
    categoria,
    COUNT(*)                                         AS frecuencia,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) AS porcentaje,
    REPEAT('█', CAST(COUNT(*) * 30.0 / MAX(COUNT(*)) OVER() AS INT)) AS barra
FROM tabla_ejemplo
GROUP BY categoria
ORDER BY frecuencia DESC;


-- ════════════════════════════════════════════════════════════
--  MÓDULO 2: ANÁLISIS TEMPORAL (TIME SERIES)
-- ════════════════════════════════════════════════════════════

-- 2.1 Ventas agregadas por período (año-mes)
SELECT
    DATE_TRUNC('month', fecha_venta)                 AS periodo,
    TO_CHAR(DATE_TRUNC('month', fecha_venta), 'YYYY-MM') AS mes_año,
    COUNT(*)                                         AS total_transacciones,
    ROUND(SUM(monto), 2)                             AS ventas_totales,
    ROUND(AVG(monto), 2)                             AS ticket_promedio,
    MIN(monto)                                       AS venta_minima,
    MAX(monto)                                       AS venta_maxima
FROM ventas
GROUP BY DATE_TRUNC('month', fecha_venta)
ORDER BY periodo;


-- 2.2 Crecimiento MoM (Month over Month) con LAG
WITH ventas_mensuales AS (
    SELECT
        DATE_TRUNC('month', fecha_venta) AS mes,
        SUM(monto)                        AS total
    FROM ventas
    GROUP BY DATE_TRUNC('month', fecha_venta)
)
SELECT
    mes,
    ROUND(total, 2)                                    AS ventas_actuales,
    ROUND(LAG(total) OVER (ORDER BY mes), 2)           AS ventas_mes_anterior,
    ROUND(total - LAG(total) OVER (ORDER BY mes), 2)   AS diferencia,
    ROUND(
        (total - LAG(total) OVER (ORDER BY mes))
        / NULLIF(LAG(total) OVER (ORDER BY mes), 0) * 100, 2
    )                                                  AS crecimiento_pct
FROM ventas_mensuales
ORDER BY mes;


-- 2.3 Media móvil de 3 meses (Rolling Average)
WITH ventas_diarias AS (
    SELECT
        fecha_venta::DATE AS fecha,
        SUM(monto)         AS total_dia
    FROM ventas
    GROUP BY fecha_venta::DATE
)
SELECT
    fecha,
    ROUND(total_dia, 2) AS ventas,
    ROUND(
        AVG(total_dia) OVER (ORDER BY fecha ROWS BETWEEN 2 PRECEDING AND CURRENT ROW), 2
    ) AS media_movil_3d,
    ROUND(
        AVG(total_dia) OVER (ORDER BY fecha ROWS BETWEEN 6 PRECEDING AND CURRENT ROW), 2
    ) AS media_movil_7d
FROM ventas_diarias
ORDER BY fecha;


-- ════════════════════════════════════════════════════════════
--  MÓDULO 3: ANÁLISIS DE COHORTES
-- ════════════════════════════════════════════════════════════

-- 3.1 Cohortes de retención de clientes
WITH primera_compra AS (
    SELECT
        cliente_id,
        DATE_TRUNC('month', MIN(fecha_venta)) AS cohort_mes
    FROM ventas
    GROUP BY cliente_id
),
actividad AS (
    SELECT
        v.cliente_id,
        pc.cohort_mes,
        DATE_TRUNC('month', v.fecha_venta) AS mes_actividad,
        DATE_PART('month', AGE(
            DATE_TRUNC('month', v.fecha_venta),
            pc.cohort_mes
        ))                                 AS meses_desde_cohort
    FROM ventas v
    JOIN primera_compra pc ON v.cliente_id = pc.cliente_id
)
SELECT
    cohort_mes,
    meses_desde_cohort,
    COUNT(DISTINCT cliente_id)            AS clientes_activos,
    ROUND(
        COUNT(DISTINCT cliente_id) * 100.0
        / FIRST_VALUE(COUNT(DISTINCT cliente_id)) OVER (
            PARTITION BY cohort_mes ORDER BY meses_desde_cohort
        ), 2
    )                                     AS tasa_retencion_pct
FROM actividad
GROUP BY cohort_mes, meses_desde_cohort
ORDER BY cohort_mes, meses_desde_cohort;


-- ════════════════════════════════════════════════════════════
--  MÓDULO 4: ANÁLISIS RFM (Recency, Frequency, Monetary)
-- ════════════════════════════════════════════════════════════

-- 4.1 Cálculo de métricas RFM por cliente
WITH rfm_base AS (
    SELECT
        cliente_id,
        DATE_PART('day', NOW() - MAX(fecha_venta))  AS recency,
        COUNT(*)                                     AS frequency,
        ROUND(SUM(monto), 2)                         AS monetary
    FROM ventas
    GROUP BY cliente_id
),
rfm_scored AS (
    SELECT
        cliente_id,
        recency,
        frequency,
        monetary,
        NTILE(5) OVER (ORDER BY recency ASC)     AS r_score,  -- 5=más reciente
        NTILE(5) OVER (ORDER BY frequency DESC)  AS f_score,  -- 5=más frecuente
        NTILE(5) OVER (ORDER BY monetary DESC)   AS m_score   -- 5=mayor valor
    FROM rfm_base
)
SELECT
    cliente_id,
    recency,
    frequency,
    monetary,
    r_score,
    f_score,
    m_score,
    (r_score + f_score + m_score)                                  AS rfm_total,
    CASE
        WHEN r_score >= 4 AND f_score >= 4 AND m_score >= 4       THEN '🏆 Champions'
        WHEN r_score >= 3 AND f_score >= 3                        THEN '⭐ Loyal Customers'
        WHEN r_score >= 4 AND f_score <= 2                        THEN '🆕 New Customers'
        WHEN r_score >= 3 AND f_score <= 3 AND m_score >= 3      THEN '💎 Potential Loyalists'
        WHEN r_score <= 2 AND f_score >= 4                        THEN '😴 At Risk'
        WHEN r_score <= 2 AND f_score <= 2 AND m_score <= 2      THEN '🚨 Lost Customers'
        ELSE '📌 Regular'
    END                                                            AS segmento
FROM rfm_scored
ORDER BY rfm_total DESC;


-- ════════════════════════════════════════════════════════════
--  MÓDULO 5: RANKING Y TOP-N
-- ════════════════════════════════════════════════════════════

-- 5.1 Top 10 productos por ventas con ranking
SELECT
    RANK() OVER (ORDER BY SUM(monto) DESC)      AS ranking,
    producto_id,
    nombre_producto,
    COUNT(*)                                     AS cantidad_vendida,
    ROUND(SUM(monto), 2)                         AS total_ventas,
    ROUND(SUM(monto) * 100.0 / SUM(SUM(monto)) OVER(), 2) AS participacion_pct
FROM ventas v
JOIN productos p ON v.producto_id = p.id
GROUP BY producto_id, nombre_producto
ORDER BY total_ventas DESC
LIMIT 10;


-- 5.2 Top vendedor por región usando DENSE_RANK
SELECT *
FROM (
    SELECT
        region,
        vendedor_id,
        nombre_vendedor,
        ROUND(SUM(monto), 2)                    AS ventas_totales,
        DENSE_RANK() OVER (
            PARTITION BY region
            ORDER BY SUM(monto) DESC
        )                                        AS ranking_region
    FROM ventas v
    JOIN vendedores ven ON v.vendedor_id = ven.id
    GROUP BY region, vendedor_id, nombre_vendedor
) ranked
WHERE ranking_region = 1;


-- ════════════════════════════════════════════════════════════
--  MÓDULO 6: DETECCIÓN DE ANOMALÍAS EN SQL
-- ════════════════════════════════════════════════════════════

-- 6.1 Detección de outliers con Z-Score
WITH stats AS (
    SELECT
        AVG(monto)    AS media,
        STDDEV(monto) AS desv_std
    FROM ventas
)
SELECT
    v.*,
    ROUND((v.monto - s.media) / NULLIF(s.desv_std, 0), 2) AS z_score
FROM ventas v, stats s
WHERE ABS((v.monto - s.media) / NULLIF(s.desv_std, 0)) > 3
ORDER BY ABS((v.monto - s.media) / NULLIF(s.desv_std, 0)) DESC;


-- 6.2 Detección de outliers con IQR
WITH quartiles AS (
    SELECT
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY monto) AS q1,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY monto) AS q3
    FROM ventas
),
iqr_calc AS (
    SELECT
        q1,
        q3,
        q3 - q1 AS iqr,
        q1 - 1.5 * (q3 - q1) AS lower_bound,
        q3 + 1.5 * (q3 - q1) AS upper_bound
    FROM quartiles
)
SELECT v.*
FROM ventas v, iqr_calc i
WHERE v.monto < i.lower_bound
   OR v.monto > i.upper_bound;


-- ════════════════════════════════════════════════════════════
--  MÓDULO 7: MÉTRICAS DE NEGOCIO (KPIs)
-- ════════════════════════════════════════════════════════════

-- 7.1 Dashboard de KPIs general
SELECT
    -- Volumen
    COUNT(*)                                          AS total_transacciones,
    COUNT(DISTINCT cliente_id)                        AS clientes_unicos,
    COUNT(DISTINCT producto_id)                       AS productos_vendidos,

    -- Financiero
    ROUND(SUM(monto), 2)                              AS revenue_total,
    ROUND(AVG(monto), 2)                              AS ticket_promedio,
    ROUND(MAX(monto), 2)                              AS mayor_venta,

    -- Temporal
    MIN(fecha_venta)::DATE                            AS primera_venta,
    MAX(fecha_venta)::DATE                            AS ultima_venta,
    DATE_PART('day', MAX(fecha_venta) - MIN(fecha_venta)) AS dias_operacion,

    -- Tasa de actividad
    ROUND(COUNT(*) / NULLIF(DATE_PART('day', MAX(fecha_venta) - MIN(fecha_venta)), 0), 2)
                                                      AS transacciones_por_dia

FROM ventas
WHERE fecha_venta BETWEEN '2025-01-01' AND '2025-12-31';


-- 7.2 Tasa de conversión por canal
SELECT
    canal,
    COUNT(*)                                          AS total_visitas,
    COUNT(CASE WHEN compra = true THEN 1 END)         AS conversiones,
    ROUND(
        COUNT(CASE WHEN compra = true THEN 1 END) * 100.0 / NULLIF(COUNT(*), 0), 2
    )                                                 AS tasa_conversion_pct,
    ROUND(SUM(CASE WHEN compra = true THEN monto END), 2) AS revenue_canal
FROM sesiones
GROUP BY canal
ORDER BY tasa_conversion_pct DESC;
