import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

PALETTE = ["#2563eb","#0891b2","#16a34a","#d97706","#dc2626","#7c3aed","#db2777","#059669"]

def fig_bar(series, titulo, color="#2563eb", h=False):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    if h:
        ax.barh(series.index[::-1], series.values[::-1], color=color, edgecolor="white")
    else:
        ax.bar(series.index, series.values, color=color, edgecolor="white")
        plt.xticks(rotation=35, ha="right", fontsize=8)
    ax.set_title(titulo, fontweight="bold")
    sns.despine(); fig.tight_layout()
    return fig

def fig_pie(series, titulo):
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.pie(series.values, labels=series.index, autopct="%1.1f%%",
           colors=PALETTE[:len(series)],
           wedgeprops={"width": 0.55, "edgecolor": "white"})
    ax.set_title(titulo, fontweight="bold")
    return fig
