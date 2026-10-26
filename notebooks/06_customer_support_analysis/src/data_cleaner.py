"""
🧹 Data Cleaner — Implementación de IDataCleaner
==================================================
Principio de Responsabilidad Única (SRP):
  Esta clase se ocupa SOLO de limpiar el dataset de Customer Support.

Principio de Sustitución de Liskov (LSP):
  CustomerSupportCleaner puede reemplazar IDataCleaner en cualquier
  parte del sistema sin romper su comportamiento.

Autor: Dody Dueñas
"""

import pandas as pd
import numpy as np
import logging
from interfaces import IDataCleaner

logger = logging.getLogger(__name__)


class CustomerSupportCleaner(IDataCleaner):
    """
    Limpieza específica para el dataset de Customer Support.

    Operaciones:
    1. Eliminar duplicados por conv_id + turn_index
    2. Parsear timestamps
    3. Normalizar texto de columnas categóricas
    4. Filtrar conversaciones con menos de 2 turnos
    5. Crear columnas derivadas útiles para el análisis
    """

    # Columnas categóricas a normalizar (strip + lowercase)
    CATEGORICAL_COLS = [
        'role', 'industry', 'product', 'issue_type', 'language',
        'channel', 'overall_sentiment', 'overall_urgency',
        'outcome', 'primary_intent'
    ]

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Pipeline de limpieza en pasos encadenados.
        Cada paso es un método privado con responsabilidad única.
        """
        logger.info(f"Iniciando limpieza: {len(df):,} filas")

        df = (
            df
            .pipe(self._remove_duplicates)
            .pipe(self._parse_timestamps)
            .pipe(self._normalize_categoricals)
            .pipe(self._filter_short_conversations)
            .pipe(self._add_derived_features)
        )

        logger.info(f"Limpieza completa: {len(df):,} filas restantes")
        return df

    # ──────────────────────────────────────────────
    #  Paso 1: Duplicados
    # ──────────────────────────────────────────────
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        original = len(df)
        df = df.drop_duplicates(subset=['conv_id', 'turn_index'])
        removed = original - len(df)
        if removed:
            logger.info(f"  Duplicados eliminados: {removed:,}")
        return df

    # ──────────────────────────────────────────────
    #  Paso 2: Timestamps
    # ──────────────────────────────────────────────
    def _parse_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        if df['timestamp'].dtype == object:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        invalid = df['timestamp'].isnull().sum()
        if invalid:
            logger.warning(f"  Timestamps inválidos descartados: {invalid:,}")
            df = df.dropna(subset=['timestamp'])
        return df

    # ──────────────────────────────────────────────
    #  Paso 3: Normalizar categóricas
    # ──────────────────────────────────────────────
    def _normalize_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self.CATEGORICAL_COLS:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.lower()
        return df

    # ──────────────────────────────────────────────
    #  Paso 4: Filtrar conversaciones cortas
    # ──────────────────────────────────────────────
    def _filter_short_conversations(self, df: pd.DataFrame) -> pd.DataFrame:
        conv_sizes = df.groupby('conv_id')['turn_index'].count()
        valid_convs = conv_sizes[conv_sizes >= 2].index
        original = len(df)
        df = df[df['conv_id'].isin(valid_convs)]
        logger.info(f"  Conversaciones con < 2 turnos eliminadas: {original - len(df):,} filas")
        return df

    # ──────────────────────────────────────────────
    #  Paso 5: Features derivadas
    # ──────────────────────────────────────────────
    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Agrega columnas calculadas útiles para el análisis.
        """
        # Componentes temporales
        df['hour']        = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.day_name()
        df['month']       = df['timestamp'].dt.month
        df['year']        = df['timestamp'].dt.year

        # Longitud del texto
        df['text_length'] = df['text'].fillna('').str.len()
        df['word_count']  = df['text'].fillna('').str.split().str.len()

        # Turno del rol (1 = primer turno de ese rol en la conv)
        df = df.sort_values(['conv_id', 'turn_index'])
        df['role_turn'] = df.groupby(['conv_id', 'role']).cumcount() + 1

        # Indicadores binarios
        df['is_resolved']  = (df['outcome'] == 'resolved').astype(int)
        df['is_escalated'] = (df['outcome'] == 'escalated').astype(int)
        df['is_negative']  = (df['overall_sentiment'] == 'negative').astype(int)
        df['is_critical']  = (df['overall_urgency'] == 'critical').astype(int)

        logger.info("  Features derivadas agregadas: hour, day_of_week, text_length, is_resolved, etc.")
        return df
