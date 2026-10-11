# -*- coding: utf-8 -*-

"""
comparison.py

Fungsi untuk membandingkan hasil beberapa metode TSP.
Tidak bergantung pada Streamlit.
"""

import pandas as pd
import plotly.graph_objects as go

# Palet tema
ROSEWOOD = "#6B0B0C"      # batang terbaik
MUTED = "#A9C7D1"         # batang lainnya (Botticelli yang lebih pekat)
COFFEE = "#2D120D"
GRID = "#EDE7C4"          # garis bantu tipis (turunan Lemon Chiffon)


def _bar_chart(comparison_df, column, title, y_title, best, fmt):
    """
    Bar chart satu warna: batang terbaik Rosewood, lainnya Botticelli lembut.

    best : "min" (nilai terkecil terbaik) atau "max" (terbesar terbaik)
    fmt  : fungsi format label di atas batang
    """

    values = list(comparison_df[column])

    fig = go.Figure()

    if values:

        target = min(values) if best == "min" else max(values)

        colors = [ROSEWOOD if v == target else MUTED for v in values]

        fig.add_trace(
            go.Bar(
                x=comparison_df["Method"],
                y=values,
                marker=dict(color=colors, line=dict(width=0)),
                text=[fmt(v) for v in values],
                textposition="outside",
                cliponaxis=False,
                textfont=dict(color=COFFEE, size=13),
                hovertemplate="%{x}<br>" + y_title + ": %{y}<extra></extra>"
            )
        )

    fig.update_layout(
        title=dict(
            text=(
                f"{title}<br>"
                "<sup>Batang Rosewood = terbaik</sup>"
            ),
            font=dict(family="Georgia, serif", size=18, color=COFFEE)
        ),
        xaxis_title=None,
        yaxis_title=y_title,
        height=420,
        bargap=0.45,
        showlegend=False,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color=COFFEE),
        margin=dict(l=20, r=20, t=80, b=40)
    )

    fig.update_xaxes(
        automargin=True,
        showgrid=False,
        linecolor=GRID,
        tickangle=-15 if len(values) > 3 else 0
    )

    fig.update_yaxes(
        rangemode="tozero",
        gridcolor=GRID,
        zeroline=False,
        showline=False
    )

    return fig


# ============================================================
# 1. HITUNG IMPROVEMENT
# ============================================================

def calculate_improvement(initial_distance, final_distance):
    """
    Menghitung persentase perbaikan jarak.

    Improvement (%) =
        ((Initial - Final) / Initial) * 100

    Return None jika metode tidak memiliki initial solution
    (misalnya metode Constructive yang membangun rute dari awal).
    """

    if initial_distance is None:
        return None

    if initial_distance == 0:
        return 0.0

    return (
        (initial_distance - final_distance)
        / initial_distance
    ) * 100


# ============================================================
# 2. MEMBUAT TABEL COMPARISON
# ============================================================

def create_comparison_table(results):
    """
    Membuat DataFrame hasil perbandingan.

    results berupa list dictionary.

    Contoh:
    [
        {
            "method": "Nearest Neighbor",
            "initial_distance": None,      # Constructive: tidak ada initial
            "final_distance": 120,
            "execution_time": 0.002
        },
        {
            "method": "2-opt",
            "initial_distance": 150,
            "final_distance": 120,
            "execution_time": 0.004
        }
    ]

    Initial Distance dan Improvement (%) bernilai kosong (NaN)
    untuk metode yang tidak memiliki initial solution.
    """

    rows = []

    for result in results:

        initial_raw = result.get("initial_distance")

        initial = (
            None
            if initial_raw is None
            else float(initial_raw)
        )

        final = float(
            result.get("final_distance", 0)
        )

        time_value = float(
            result.get("execution_time", 0)
        )

        improvement = calculate_improvement(
            initial,
            final
        )

        rows.append(
            {
                "Method": result.get(
                    "method",
                    "Unknown"
                ),

                "Initial Distance": (
                    None
                    if initial is None
                    else round(initial, 2)
                ),

                "Final Distance": round(
                    final,
                    2
                ),

                "Improvement (%)": (
                    None
                    if improvement is None
                    else round(improvement, 2)
                ),

                "Execution Time (ms)": round(
                    time_value * 1000,
                    3
                )
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# 3. SORTING HASIL
# ============================================================

def sort_by_distance(comparison_df):
    """
    Mengurutkan hasil berdasarkan final distance.
    Jarak paling kecil berada di atas.
    """

    return comparison_df.sort_values(
        by="Final Distance",
        ascending=True
    ).reset_index(drop=True)


def sort_by_time(comparison_df):
    """
    Mengurutkan hasil berdasarkan waktu eksekusi.
    Waktu paling cepat berada di atas.
    """

    return comparison_df.sort_values(
        by="Execution Time (ms)",
        ascending=True
    ).reset_index(drop=True)


# ============================================================
# 4. METODE TERBAIK BERDASARKAN DISTANCE
# ============================================================

def get_best_distance(comparison_df):
    """
    Mengambil metode dengan final distance terkecil.
    """

    if comparison_df.empty:
        return None

    idx = comparison_df[
        "Final Distance"
    ].idxmin()

    return comparison_df.loc[idx]


# ============================================================
# 5. METODE TERCEPAT
# ============================================================

def get_fastest_method(comparison_df):
    """
    Mengambil metode dengan waktu eksekusi tercepat.
    """

    if comparison_df.empty:
        return None

    idx = comparison_df[
        "Execution Time (ms)"
    ].idxmin()

    return comparison_df.loc[idx]


# ============================================================
# 6. GRAFIK FINAL DISTANCE
# ============================================================

def plot_distance_comparison(comparison_df):
    """
    Grafik perbandingan final distance.
    Semakin kecil semakin baik.
    """

    return _bar_chart(
        comparison_df,
        "Final Distance",
        "Final Distance Comparison",
        "Final Distance",
        "min",
        lambda v: f"{v:.2f}"
    )


# ============================================================
# 7. GRAFIK EXECUTION TIME
# ============================================================

def plot_time_comparison(comparison_df):
    """
    Grafik perbandingan waktu eksekusi.
    Semakin kecil semakin cepat.
    """

    return _bar_chart(
        comparison_df,
        "Execution Time (ms)",
        "Execution Time Comparison",
        "Time (ms)",
        "min",
        lambda v: f"{v:.3f}"
    )


# ============================================================
# 8. GRAFIK IMPROVEMENT
# ============================================================

def plot_improvement_comparison(comparison_df):
    """
    Grafik persentase improvement.

    Semakin besar improvement,
    semakin besar penurunan jarak dari initial solution.

    Hanya metode yang memiliki initial solution yang ditampilkan.
    """

    data = comparison_df.dropna(
        subset=["Improvement (%)"]
    )

    if data.empty:

        fig = go.Figure()

        fig.add_annotation(
            text=(
                "Tidak ada metode dengan initial solution "
                "(improvement tidak dapat dihitung)."
            ),
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(color=COFFEE)
        )

        fig.update_layout(
            title="Solution Improvement",
            height=300,
            paper_bgcolor="white",
            plot_bgcolor="white",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )

        return fig

    return _bar_chart(
        data,
        "Improvement (%)",
        "Solution Improvement",
        "Improvement (%)",
        "max",
        lambda v: f"{v:.2f}%"
    )
