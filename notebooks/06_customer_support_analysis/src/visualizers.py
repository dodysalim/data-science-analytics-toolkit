"""
📈 Visualizers — Implementaciones concretas de IVisualizer
===========================================================
Principio de Responsabilidad Única (SRP):
  Cada visualizador se ocupa de UN tipo de gráfico.

Principio Abierto/Cerrado (OCP):
  Para agregar un nuevo gráfico, solo se crea una nueva clase.

Autor: Dody Dueñas
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from interfaces import IVisualizer

# Paleta y estilo global profesional
PALETTE = "viridis"
PALETTE_DIV = "RdYlGn"
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'figure.facecolor': '#f8f9fa',
    'axes.facecolor': '#ffffff',
    'axes.edgecolor': '#cccccc',
    'axes.titlesize': 13,
    'axes.labelsize': 11,
})
sns.set_theme(style="whitegrid", palette=PALETTE)


# ══════════════════════════════════════════════════════════════
#  1. SentimentVisualizer
# ══════════════════════════════════════════════════════════════
class SentimentVisualizer(IVisualizer):
    """Gráficos de distribución de sentimiento."""

    def get_title(self) -> str:
        return "Distribución de Sentimiento"

    def plot(self, df: pd.DataFrame, output_dir: str = ".") -> None:
        fig, axes = plt.subplots(1, 3, figsize=(20, 7))
        fig.suptitle('Analisis de Sentimiento - Customer Support', fontsize=16, y=1.01, fontweight='bold')

        # 1. Distribución general (pie)
        sentiment_counts = df['overall_sentiment'].value_counts()
        colors_pie = ['#e74c3c', '#95a5a6', '#2ecc71']
        axes[0].pie(sentiment_counts.values, labels=sentiment_counts.index,
                    autopct='%1.1f%%', colors=colors_pie, startangle=90,
                    wedgeprops={'edgecolor': 'white', 'linewidth': 2})
        axes[0].set_title('Distribución General de Sentimiento')

        # 2. Tasa negativa por industria
        neg_by_ind = df.groupby('industry')['is_negative'].mean().mul(100).sort_values()
        colors_bar = ['#e74c3c' if v > 50 else '#3498db' for v in neg_by_ind.values]
        axes[1].barh(neg_by_ind.index, neg_by_ind.values, color=colors_bar, edgecolor='white')
        axes[1].axvline(50, color='gray', linestyle='--', alpha=0.7, label='50%')
        axes[1].set_xlabel('% Conversaciones Negativas')
        axes[1].set_title('Tasa de Sentimiento Negativo por Industria')
        axes[1].legend()
        for i, v in enumerate(neg_by_ind.values):
            axes[1].text(v + 0.5, i, f'{v:.1f}%', va='center', fontsize=9)

        # 3. Heatmap sentimiento x canal
        pivot = pd.crosstab(df['channel'], df['overall_sentiment'], normalize='index').mul(100)
        sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn', ax=axes[2],
                    linewidths=0.5, cbar_kws={'label': '%'})
        axes[2].set_title('Sentimiento por Canal (%)')

        plt.tight_layout()
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        out = Path(output_dir) / 'sentiment_analysis.png'
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  [OK] Guardado: {out}")


# ══════════════════════════════════════════════════════════════
#  2. OutcomeVisualizer
# ══════════════════════════════════════════════════════════════
class OutcomeVisualizer(IVisualizer):
    """Gráficos de resultados de conversaciones."""

    def get_title(self) -> str:
        return "Resultados de Conversaciones"

    def plot(self, df: pd.DataFrame, output_dir: str = ".") -> None:
        fig, axes = plt.subplots(1, 3, figsize=(20, 7))
        fig.suptitle('Analisis de Resultados (Outcomes) - Customer Support', fontsize=16, y=1.01, fontweight='bold')

        # 1. Distribución global de outcomes
        outcomes = df['outcome'].value_counts()
        colors_out = sns.color_palette(PALETTE, len(outcomes))
        axes[0].bar(outcomes.index, outcomes.values, color=colors_out, edgecolor='white')
        axes[0].set_title('Distribución de Outcomes')
        axes[0].set_xlabel('Outcome')
        axes[0].set_ylabel('Cantidad de Conversaciones')
        axes[0].tick_params(axis='x', rotation=30)
        for i, v in enumerate(outcomes.values):
            axes[0].text(i, v + 10, str(v), ha='center', fontsize=9)

        # 2. Tasa de resolución por industria
        res_by_ind = df.groupby('industry')['is_resolved'].mean().mul(100).sort_values(ascending=False)
        colors_res = ['#2ecc71' if v >= 60 else '#e67e22' if v >= 40 else '#e74c3c' for v in res_by_ind.values]
        axes[1].bar(res_by_ind.index, res_by_ind.values, color=colors_res, edgecolor='white')
        axes[1].set_title('Tasa de Resolución por Industria (%)')
        axes[1].set_ylabel('% Resueltos')
        axes[1].tick_params(axis='x', rotation=30)
        axes[1].axhline(res_by_ind.mean(), color='navy', linestyle='--', label=f'Promedio: {res_by_ind.mean():.1f}%')
        axes[1].legend()
        for i, v in enumerate(res_by_ind.values):
            axes[1].text(i, v + 0.5, f'{v:.1f}%', ha='center', fontsize=9)

        # 3. Outcome x urgencia (stacked bar)
        pivot = pd.crosstab(df['overall_urgency'], df['outcome'], normalize='index').mul(100)
        pivot.plot(kind='bar', stacked=True, ax=axes[2], colormap=PALETTE, edgecolor='white')
        axes[2].set_title('Outcomes por Nivel de Urgencia (%)')
        axes[2].set_xlabel('Urgencia')
        axes[2].set_ylabel('%')
        axes[2].tick_params(axis='x', rotation=0)
        axes[2].legend(loc='upper right', fontsize=8)

        plt.tight_layout()
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        out = Path(output_dir) / 'outcome_analysis.png'
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  [OK] Guardado: {out}")


# ══════════════════════════════════════════════════════════════
#  3. VolumeVisualizer
# ══════════════════════════════════════════════════════════════
class VolumeVisualizer(IVisualizer):
    """Gráficos de volumen y temporalidad."""

    DAYS_ORDER = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    def get_title(self) -> str:
        return "Volumen y Temporalidad"

    def plot(self, df: pd.DataFrame, output_dir: str = ".") -> None:
        convs = df.drop_duplicates(subset='conv_id')

        fig = plt.figure(figsize=(20, 12))
        fig.suptitle('Analisis de Volumen y Temporalidad - Customer Support', fontsize=16, fontweight='bold')
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

        # 1. Volumen por hora
        ax1 = fig.add_subplot(gs[0, 0])
        vol_hour = convs.groupby('hour').size()
        ax1.plot(vol_hour.index, vol_hour.values, marker='o', color='steelblue', lw=2)
        ax1.fill_between(vol_hour.index, vol_hour.values, alpha=0.3, color='steelblue')
        ax1.set_title('Volumen por Hora del Día')
        ax1.set_xlabel('Hora')
        ax1.set_ylabel('Conversaciones')

        # 2. Volumen por día de la semana
        ax2 = fig.add_subplot(gs[0, 1])
        vol_dow = convs.groupby('day_of_week').size().reindex(
            [d for d in self.DAYS_ORDER if d in convs['day_of_week'].unique()]
        )
        colors_dow = sns.color_palette(PALETTE, len(vol_dow))
        ax2.bar(vol_dow.index, vol_dow.values, color=colors_dow, edgecolor='white')
        ax2.set_title('Volumen por Día de la Semana')
        ax2.tick_params(axis='x', rotation=30)
        ax2.set_ylabel('Conversaciones')

        # 3. Volumen por canal
        ax3 = fig.add_subplot(gs[0, 2])
        vol_chan = convs.groupby('channel').size().sort_values(ascending=True)
        ax3.barh(vol_chan.index, vol_chan.values, color=sns.color_palette(PALETTE, len(vol_chan)), edgecolor='white')
        ax3.set_title('Volumen por Canal')
        ax3.set_xlabel('Conversaciones')

        # 4. Longitud de conversaciones (distribución)
        ax4 = fig.add_subplot(gs[1, 0])
        conv_lengths = df.groupby('conv_id')['turn_index'].count()
        ax4.hist(conv_lengths, bins=30, color='mediumseagreen', edgecolor='white', alpha=0.85)
        ax4.axvline(conv_lengths.mean(), color='red', linestyle='--', label=f'Media: {conv_lengths.mean():.1f}')
        ax4.set_title('Distribución de Turnos por Conversación')
        ax4.set_xlabel('Número de Turnos')
        ax4.legend()

        # 5. Volumen por industria
        ax5 = fig.add_subplot(gs[1, 1])
        vol_ind = convs.groupby('industry').size().sort_values(ascending=False)
        ax5.bar(vol_ind.index, vol_ind.values, color=sns.color_palette(PALETTE, len(vol_ind)), edgecolor='white')
        ax5.set_title('Volumen por Industria')
        ax5.tick_params(axis='x', rotation=30)

        # 6. Longitud de texto por rol
        ax6 = fig.add_subplot(gs[1, 2])
        df.groupby('role')['text_length'].plot(kind='density', ax=ax6, legend=True, lw=2)
        ax6.set_title('Distribución de Longitud de Texto por Rol')
        ax6.set_xlabel('Longitud del texto (caracteres)')
        ax6.legend(['Agente', 'Cliente'])

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        out = Path(output_dir) / 'volume_analysis.png'
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  [OK] Guardado: {out}")


# ══════════════════════════════════════════════════════════════
#  4. IntentVisualizer
# ══════════════════════════════════════════════════════════════
class IntentVisualizer(IVisualizer):
    """Gráficos de intenciones del cliente."""

    def get_title(self) -> str:
        return "Intenciones del Cliente"

    def plot(self, df: pd.DataFrame, output_dir: str = ".") -> None:
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        fig.suptitle('Analisis de Intenciones del Cliente - Customer Support', fontsize=16, fontweight='bold')

        # 1. Top intenciones
        intent_counts = df['primary_intent'].value_counts().nlargest(12)
        colors = sns.color_palette(PALETTE, len(intent_counts))
        axes[0].barh(intent_counts.index[::-1], intent_counts.values[::-1], color=colors[::-1], edgecolor='white')
        axes[0].set_title('Top 12 Intenciones del Cliente')
        axes[0].set_xlabel('Cantidad de Conversaciones')

        # 2. Heatmap intención x industria
        pivot = pd.crosstab(df['primary_intent'], df['industry'])
        pivot_norm = pivot.div(pivot.sum(axis=0), axis=1).mul(100)
        sns.heatmap(pivot_norm, annot=True, fmt='.0f', cmap='YlOrRd',
                    ax=axes[1], linewidths=0.5, cbar_kws={'label': '%'})
        axes[1].set_title('Distribución de Intenciones por Industria (%)')
        axes[1].set_xlabel('Industria')
        axes[1].set_ylabel('Intención')

        plt.tight_layout()
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        out = Path(output_dir) / 'intent_analysis.png'
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  [OK] Guardado: {out}")
