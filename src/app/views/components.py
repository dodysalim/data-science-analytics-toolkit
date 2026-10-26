"""Fábrica de gráficos Plotly con el tema del dashboard."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from .theme import GRID, MUTED, PALETTE, SENTIMENT_COLORS, TEXT, ACCENT

_LAYOUT = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
               font=dict(family="Inter, system-ui, sans-serif", color=TEXT, size=12),
               title=dict(font=dict(size=15, color=TEXT), x=0.01, xanchor="left"),
               legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=MUTED)),
               hoverlabel=dict(bgcolor="#121B2A", bordercolor=ACCENT, font=dict(color=TEXT)),
               margin=dict(l=10, r=10, t=50, b=10))
_AXES = dict(gridcolor=GRID, zeroline=False, linecolor=GRID, tickfont=dict(color=MUTED), title_font=dict(color=MUTED))


def style(fig, **layout):
    fig.update_layout(**{**_LAYOUT, **layout})
    fig.update_xaxes(**_AXES)
    fig.update_yaxes(**_AXES)
    return fig


def show(fig):
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})


def fig_bar(series: pd.Series, titulo: str, color: str = ACCENT, h: bool = False, text_fmt: str = None):
    data = series.iloc[::-1] if h else series
    fig = go.Figure(go.Bar(x=data.values if h else data.index, y=data.index if h else data.values,
                           orientation="h" if h else "v", marker_color=color,
                           text=[text_fmt.format(v) for v in data.values] if text_fmt else None,
                           textposition="outside" if text_fmt else None))
    return style(fig, title=titulo, bargap=0.3)


def fig_pie(series: pd.Series, titulo: str, colors=None):
    colors = colors or [SENTIMENT_COLORS.get(k, PALETTE[i % len(PALETTE)]) for i, k in enumerate(series.index)]
    fig = go.Figure(go.Pie(labels=series.index, values=series.values, hole=0.55, textinfo="label+percent",
                           marker=dict(colors=colors, line=dict(color="#0B1220", width=2))))
    return style(fig, title=titulo, showlegend=False)


def fig_stacked_share(shares: pd.DataFrame, titulo: str, horizontal: bool = False, colors=None):
    fig = go.Figure()
    for i, col in enumerate(shares.columns):
        color = (colors or {}).get(col) or SENTIMENT_COLORS.get(col) or PALETTE[i % len(PALETTE)]
        kwargs = dict(x=shares[col], y=shares.index, orientation="h") if horizontal else dict(x=shares.index, y=shares[col])
        fig.add_trace(go.Bar(name=str(col), marker_color=color, **kwargs,
                             hovertemplate="%{fullData.name}: %{" + ("x" if horizontal else "y") + ":.1%}<extra></extra>"))
    fig = style(fig, title=titulo, barmode="stack", legend=dict(orientation="h", y=-0.2, font=dict(color=MUTED)))
    fig.update_layout(**({"xaxis_tickformat": ".0%"} if horizontal else {"yaxis_tickformat": ".0%"}))
    return fig


def fig_area(series: pd.Series, titulo: str, color: str = ACCENT, y_title: str = "", x_title: str = ""):
    fig = go.Figure(go.Scatter(x=series.index, y=series.values, mode="lines+markers", fill="tozeroy",
                               line=dict(color=color, width=3, shape="spline"), marker=dict(size=6),
                               fillcolor=_alpha(color, 0.15)))
    return style(fig, title=titulo, xaxis_title=x_title, yaxis_title=y_title)


def fig_heatmap(table: pd.DataFrame, titulo: str):
    fig = px.imshow(table, text_auto=".2f", aspect="auto",
                    color_continuous_scale=["#121B2A", "#2E5E7E", "#3FA3A8", ACCENT])
    return style(fig, title=titulo, height=max(380, 34 * len(table)), coloraxis_colorbar=dict(title="Proporción"))


def fig_posteriors(curves, titulo: str):
    """`curves`: lista de (nombre, x, densidad, color)."""
    fig = go.Figure()
    for name, x, y, color in curves:
        fig.add_trace(go.Scatter(x=x, y=y, name=name, mode="lines", fill="tozeroy",
                                 line=dict(color=color, width=2), fillcolor=_alpha(color, 0.35)))
    return style(fig, title=titulo, xaxis_title="Tasa de resolucion", xaxis_tickformat=".1%")


def _alpha(hex_color: str, a: float) -> str:
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"
