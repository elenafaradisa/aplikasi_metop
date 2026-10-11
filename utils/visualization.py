import plotly.graph_objects as go


# Palet tema
ROSEWOOD = "#6B0B0C"
COFFEE = "#2D120D"
NODE_BLUE = "#4F8FA3"
HOME_GOLD = "#B7791F"
PLOT_BG = "#FFFDF0"      
GRID = "#DCEBEF"        


def style_figure(fig):
    """Gaya seragam: latar krem lembut, grid Botticelli, teks Coffee Bean."""

    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor=PLOT_BG,
        font=dict(color=COFFEE),
        title_font=dict(family="Georgia, serif", size=18, color=COFFEE)
    )

    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID)

    return fig


# ============================================================
# 1. VISUALISASI SEBARAN NODE
# ============================================================

def plot_nodes(df, home=0, title="Node Distribution"):
    """
    Membuat visualisasi posisi seluruh node.

    Parameter
    ---------
    df : DataFrame
        Data dengan kolom: node, x, y.

    home : int
        Index node yang digunakan sebagai titik awal.

    title : str
        Judul grafik.

    Return
    ------
    plotly.graph_objects.Figure
    """

    fig = go.Figure()

    # Semua node
    colors = [
        HOME_GOLD if i == home else NODE_BLUE
        for i in range(len(df))
    ]

    fig.add_trace(
        go.Scatter(
            x=df["x"],
            y=df["y"],
            mode="markers+text",
            text=df["node"].astype(str),
            textposition="top center",
            marker=dict(
                size=14,
                color=colors,
                line=dict(
                    color="white",
                    width=2
                )
            ),
            hovertemplate=(
                "Node: %{text}<br>"
                "X: %{x}<br>"
                "Y: %{y}"
                "<extra></extra>"
            )
        )
    )

    fig.update_layout(
        title=title,
        height=450,
        margin=dict(
            l=20,
            r=20,
            t=50,
            b=20
        ),
        xaxis_title="X",
        yaxis_title="Y",
        yaxis=dict(
            scaleanchor="x",
            scaleratio=1
        ),
        showlegend=False
    )

    return style_figure(fig)


# ============================================================
# 2. VISUALISASI RUTE
# ============================================================

def plot_route(
    df,
    tour,
    home=0,
    title="TSP Route",
    show_arrows=True
):
    """
    Membuat visualisasi rute TSP.

    Parameter
    ---------
    df : DataFrame
        Data dengan kolom node, x, y.

    tour : list
        Urutan node yang dikunjungi.
        Contoh: [0, 2, 4, 1, 3, 0]

    home : int
        Index node awal.

    title : str
        Judul grafik.

    show_arrows : bool
        Menampilkan arah perjalanan.

    Return
    ------
    plotly.graph_objects.Figure
    """

    # Ambil koordinat berdasarkan index node
    xs = [df.iloc[i]["x"] for i in tour]
    ys = [df.iloc[i]["y"] for i in tour]

    fig = go.Figure()

    # --------------------------------------------------------
    # Garis rute
    # --------------------------------------------------------

    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            line=dict(
                color=ROSEWOOD,
                width=3
            ),
            hoverinfo="skip",
            name="Route"
        )
    )

    # --------------------------------------------------------
    # Arah perjalanan
    # --------------------------------------------------------

    if show_arrows and len(tour) <= 80:

        for i in range(len(tour) - 1):

            x_start = xs[i]
            y_start = ys[i]

            x_end = xs[i + 1]
            y_end = ys[i + 1]

            fig.add_annotation(
                x=x_end,
                y=y_end,
                ax=x_start,
                ay=y_start,
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=3,
                arrowsize=1,
                arrowwidth=1.5,
                arrowcolor=ROSEWOOD,
                standoff=8
            )

    # --------------------------------------------------------
    # Node
    # --------------------------------------------------------

    node_colors = [
        HOME_GOLD if i == home else NODE_BLUE
        for i in range(len(df))
    ]

    fig.add_trace(
        go.Scatter(
            x=df["x"],
            y=df["y"],
            mode="markers+text",
            text=df["node"].astype(str),
            textposition="top center",
            marker=dict(
                size=14,
                color=node_colors,
                line=dict(
                    color="white",
                    width=2
                )
            ),
            hovertemplate=(
                "Node: %{text}<br>"
                "X: %{x}<br>"
                "Y: %{y}"
                "<extra></extra>"
            ),
            name="Node"
        )
    )

    fig.update_layout(
        title=title,
        height=500,
        margin=dict(
            l=20,
            r=20,
            t=50,
            b=20
        ),
        xaxis_title="X",
        yaxis_title="Y",
        yaxis=dict(
            scaleanchor="x",
            scaleratio=1
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    return style_figure(fig)


# ============================================================
# 3. VISUALISASI PERBANDINGAN RUTE
# ============================================================

def plot_route_comparison(
    df,
    initial_tour,
    final_tour,
    home=0,
    initial_title="Initial Route",
    final_title="Final Route"
):
    """
    Menghasilkan dua Figure:
    - rute awal
    - rute akhir

    Dipakai oleh app.py untuk ditampilkan berdampingan.
    """

    initial_fig = plot_route(
        df=df,
        tour=initial_tour,
        home=home,
        title=initial_title
    )

    final_fig = plot_route(
        df=df,
        tour=final_tour,
        home=home,
        title=final_title
    )

    return initial_fig, final_fig


# ============================================================
# 4. VISUALISASI HISTORY / ITERASI
# ============================================================

def plot_iteration(
    df,
    tour,
    iteration,
    home=0
):
    """
    Membuat visualisasi rute pada suatu iterasi.
    """

    return plot_route(
        df=df,
        tour=tour,
        home=home,
        title=f"Iteration {iteration}"
    )
