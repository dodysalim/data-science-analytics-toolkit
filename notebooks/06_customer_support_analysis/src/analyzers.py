"""
📊 Analyzers — Implementaciones concretas de IAnalyzer
========================================================
Principio Abierto/Cerrado (OCP):
  Cada análisis es una clase independiente. Para agregar un nuevo
  análisis, solo se crea una nueva clase. NADA se modifica.

Principio de Segregación de Interfaces (ISP):
  Cada Analyzer implementa solo analyze() y get_name().
  No hay métodos innecesarios.

Principio de Sustitución de Liskov (LSP):
  Todos los analyzers son intercambiables: cualquiera puede pasarse
  donde se espera un IAnalyzer.

Autor: Dody Dueñas
"""

import pandas as pd
import numpy as np
from scipy import stats
from interfaces import IAnalyzer


# ══════════════════════════════════════════════════════════════
#  1. SentimentAnalyzer
# ══════════════════════════════════════════════════════════════
class SentimentAnalyzer(IAnalyzer):
    """
    Analiza la distribución de sentimientos por industria,
    canal y producto. Calcula tasa de sentimiento negativo.
    """

    def get_name(self) -> str:
        return "Análisis de Sentimiento"

    def analyze(self, df: pd.DataFrame) -> dict:
        sentiment_dist = df['overall_sentiment'].value_counts(normalize=True).mul(100).round(2)

        by_industry = df.groupby('industry')['is_negative'].mean().mul(100).round(2)
        by_channel  = df.groupby('channel')['is_negative'].mean().mul(100).round(2)
        by_product  = df.groupby('product')['is_negative'].mean().mul(100).round(2).nlargest(10)

        # Proporción positivo vs negativo por industria
        sentiment_pivot = (
            df.groupby(['industry', 'overall_sentiment'])
            .size().unstack(fill_value=0)
        )
        sentiment_pivot_pct = sentiment_pivot.div(sentiment_pivot.sum(axis=1), axis=0).mul(100).round(2)

        return {
            'distribucion_general': sentiment_dist.to_dict(),
            'tasa_negativa_por_industria_%': by_industry.to_dict(),
            'tasa_negativa_por_canal_%': by_channel.to_dict(),
            'top10_productos_negativos_%': by_product.to_dict(),
            'sentiment_por_industria_%': sentiment_pivot_pct.to_dict()
        }


# ══════════════════════════════════════════════════════════════
#  2. OutcomeAnalyzer
# ══════════════════════════════════════════════════════════════
class OutcomeAnalyzer(IAnalyzer):
    """
    Analiza la distribución de resultados de las conversaciones:
    Resueltas, Escaladas, Pendientes, etc.
    Incluye tasa de resolución por industria y tipo de problema.
    """

    def get_name(self) -> str:
        return "Análisis de Resultados (Outcomes)"

    def analyze(self, df: pd.DataFrame) -> dict:
        outcome_dist = df['outcome'].value_counts(normalize=True).mul(100).round(2)

        # Tasa de resolución global
        resolution_rate = df['is_resolved'].mean() * 100

        # Por industria
        by_industry = df.groupby('industry')['is_resolved'].mean().mul(100).round(2).sort_values(ascending=False)

        # Por issue_type
        by_issue = df.groupby('issue_type')['is_resolved'].mean().mul(100).round(2).sort_values(ascending=False)

        # Por urgencia
        by_urgency = df.groupby('overall_urgency')['is_resolved'].mean().mul(100).round(2)

        # Escalados por urgencia
        escalation_by_urgency = df.groupby('overall_urgency')['is_escalated'].mean().mul(100).round(2)

        return {
            'distribucion_outcomes_%': outcome_dist.to_dict(),
            'tasa_resolucion_global_%': round(resolution_rate, 2),
            'resolucion_por_industria_%': by_industry.to_dict(),
            'resolucion_por_issue_type_%': by_issue.to_dict(),
            'resolucion_por_urgencia_%': by_urgency.to_dict(),
            'escalacion_por_urgencia_%': escalation_by_urgency.to_dict()
        }


# ══════════════════════════════════════════════════════════════
#  3. VolumeAnalyzer
# ══════════════════════════════════════════════════════════════
class VolumeAnalyzer(IAnalyzer):
    """
    Analiza el volumen de conversaciones por hora, día de la semana,
    mes, canal e industria. Útil para planificar recursos del equipo.
    """

    def get_name(self) -> str:
        return "Análisis de Volumen y Temporalidad"

    def analyze(self, df: pd.DataFrame) -> dict:
        # Conversaciones únicas (no turnos)
        convs = df.drop_duplicates(subset='conv_id')

        by_hour       = convs.groupby('hour').size().to_dict()
        by_dow        = convs.groupby('day_of_week').size().to_dict()
        by_month      = convs.groupby('month').size().to_dict()
        by_channel    = convs.groupby('channel').size().to_dict()
        by_industry   = convs.groupby('industry').size().to_dict()
        by_language   = convs.groupby('language').size().to_dict()

        # Hora pico
        peak_hour = max(by_hour, key=by_hour.get)
        peak_day  = max(by_dow, key=by_dow.get)

        # Longitud promedio de conversación (turnos)
        conv_lengths = df.groupby('conv_id')['turn_index'].count()
        avg_turns = conv_lengths.mean()
        max_turns = conv_lengths.max()

        return {
            'total_conversaciones': len(convs),
            'total_turnos': len(df),
            'promedio_turnos_por_conv': round(avg_turns, 2),
            'max_turnos_conv': int(max_turns),
            'hora_pico': peak_hour,
            'dia_pico': peak_day,
            'vol_por_hora': by_hour,
            'vol_por_dia_semana': by_dow,
            'vol_por_mes': by_month,
            'vol_por_canal': by_channel,
            'vol_por_industria': by_industry,
            'vol_por_idioma': by_language
        }


# ══════════════════════════════════════════════════════════════
#  4. UrgencyAnalyzer
# ══════════════════════════════════════════════════════════════
class UrgencyAnalyzer(IAnalyzer):
    """
    Analiza la distribución de urgencia, correlación con sentimiento
    y relación con el resultado de la conversación.
    """

    def get_name(self) -> str:
        return "Análisis de Urgencia"

    def analyze(self, df: pd.DataFrame) -> dict:
        urgency_dist = df['overall_urgency'].value_counts(normalize=True).mul(100).round(2)

        # Urgencia crítica por industria
        critical_by_industry = df.groupby('industry')['is_critical'].mean().mul(100).round(2).sort_values(ascending=False)

        # Relación urgencia - sentimiento
        urgency_sentiment = (
            df.groupby(['overall_urgency', 'overall_sentiment'])
            .size().unstack(fill_value=0)
        )
        urgency_sentiment_pct = urgency_sentiment.div(urgency_sentiment.sum(axis=1), axis=0).mul(100).round(2)

        # Urgencia vs texto largo (si urgencia alta, el cliente escribe más)
        avg_text_by_urgency = df.groupby('overall_urgency')['text_length'].mean().round(2)

        return {
            'distribucion_urgencia_%': urgency_dist.to_dict(),
            'urgencia_critica_por_industria_%': critical_by_industry.to_dict(),
            'urgencia_vs_sentimiento_%': urgency_sentiment_pct.to_dict(),
            'longitud_texto_por_urgencia': avg_text_by_urgency.to_dict()
        }


# ══════════════════════════════════════════════════════════════
#  5. IntentAnalyzer
# ══════════════════════════════════════════════════════════════
class IntentAnalyzer(IAnalyzer):
    """
    Analiza las intenciones principales del cliente (primary_intent),
    su relación con el tipo de problema y la tasa de resolución.
    """

    def get_name(self) -> str:
        return "Análisis de Intenciones del Cliente"

    def analyze(self, df: pd.DataFrame) -> dict:
        intent_dist = df['primary_intent'].value_counts(normalize=True).mul(100).round(2)

        # Intenciones más escaladas
        escalation_by_intent = df.groupby('primary_intent')['is_escalated'].mean().mul(100).round(2).sort_values(ascending=False)

        # Resolución por intención
        resolution_by_intent = df.groupby('primary_intent')['is_resolved'].mean().mul(100).round(2).sort_values(ascending=False)

        # Intenciones más negativas
        negative_by_intent = df.groupby('primary_intent')['is_negative'].mean().mul(100).round(2).sort_values(ascending=False)

        # Top intenciones por industria
        top_intent_by_industry = (
            df.groupby(['industry', 'primary_intent']).size()
            .reset_index(name='count')
            .sort_values(['industry', 'count'], ascending=[True, False])
            .groupby('industry').head(3)
            .set_index(['industry', 'primary_intent'])['count']
            .to_dict()
        )

        return {
            'distribucion_intenciones_%': intent_dist.to_dict(),
            'escalacion_por_intencion_%': escalation_by_intent.to_dict(),
            'resolucion_por_intencion_%': resolution_by_intent.to_dict(),
            'negatividad_por_intencion_%': negative_by_intent.to_dict(),
            'top3_intenciones_por_industria': {str(k): v for k, v in top_intent_by_industry.items()}
        }
