```python
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

import importlib
import inspect
import random
import time


# ============================================================
# IMPORT BASE
# ============================================================

from algorithms.base import (
    build_distance_matrix,
    tour_distance,
    generate_initial_tour,
    validate_tour
)

from utils.load_data import load_data


# ============================================================
# KONFIGURASI HALAMAN
# ============================================================

st.set_page_config(
    page_title="TSP Learning & Optimization",
    page_icon="🧭",
    layout="wide"
)


# ============================================================
# DAFTAR METODE
# ============================================================

INITIAL_METHODS = {
    "Random": None,
    "Nearest Neighbor": "nearest_neighbor",
    "Nearest Insertion": "nearest_insertion",
    "Farthest Insertion": "farthest_insertion",
    "Arbitrary Insertion": "arbitrary_insertion"
}

OPTIMIZATION_METHODS = {
    "None": None,
    "2-opt": "two_opt",
    "3-opt": "three_opt",
    "Simulated Annealing": "simulated_annealing",
    "Tabu Search": "tabu_search"
}


# ============================================================
# LOKASI FILE ALGORITMA
# ============================================================

SEARCH_MODULES = [
    "algorithms.constructive",
    "algorithms.simulated_anneling",
    "algorithms.simulated_annealing",
    "algorithms.tabu_search",
    "algorithms.two_opt",
    "algorithms.three_opt",
    "algorithms.local_search"
]


# Beberapa kemungkinan nama fungsi
ALIASES = {
    "nearest_neighbor": [
        "nearest_neighbor",
        "nearest_neighbour",
        "nn"
    ],

    "nearest_insertion": [
        "nearest_insertion",
        "ni"
    ],

    "farthest_insertion": [
        "farthest_insertion",
        "fi"
    ],

    "arbitrary_insertion": [
        "arbitrary_insertion",
        "random_insertion",
        "ai"
    ],

    "two_opt": [
        "two_opt",
        "2opt",
        "opt2"
    ],

    "three_opt": [
        "three_opt",
        "3opt",
        "opt3"
    ],

    "simulated_annealing": [
        "simulated_annealing",
        "simulated_anneling",
        "sa"
    ],

    "tabu_search": [
        "tabu_search",
        "tabu",
        "ts"
    ]
}


# ============================================================
# DETEKSI ALGORITMA
# ============================================================

def discover_algorithms():
    """
    Mencari fungsi algoritma yang sudah tersedia.

    Kalau file/fungsi belum ada:
    tidak error, tetapi dianggap belum tersedia.
    """

    found = {}

    for module_name in SEARCH_MODULES:

        try:
            module = importlib.import_module(module_name)

        except Exception:
            # File belum ada / belum bisa di-import
            continue

        for function_name, function in inspect.getmembers(
            module,
            inspect.isfunction
        ):

            if function_name.startswith("_"):
                continue

            normalized_function = function_name.lower().replace("_", "")

            for algorithm_name, aliases in ALIASES.items():

                normalized_aliases = [
                    alias.lower().replace("_", "")
                    for alias in aliases
                ]

                if normalized_function in normalized_aliases:

                    found[algorithm_name] = {
                        "module": module_name,
                        "function": function_name,
                        "function_object": function
                    }

    return found


def get_algorithm(key):
    """
    Mengambil fungsi algoritma berdasarkan key.
    """

    algorithms = discover_algorithms()

    return algorithms.get(key)


# ============================================================
# STATUS ALGORITMA
# ============================================================

def algorithm_status(key):

    if key is None:
        return True

    if key == "random":
        return True

    return get_algorithm(key) is not None


# ============================================================
# SESSION STATE
# ============================================================

if "df" not in st.session_state:
    st.session_state.df = None

if "runs" not in st.session_state:
    st.session_state.runs = []


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div style="
        padding: 20px;
        border-radius: 15px;
        background: linear-gradient(135deg, #6C5CE7, #00B894);
        color: white;
        margin-bottom: 20px;
    ">
        <h1>🧭 TSP Learning & Optimization</h1>
        <p>
            Input Data → Initial Solution → Optimization →
            Visualization → Comparison
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Pengaturan")

    metric = st.radio(
        "Jenis jarak",
        ["euclidean", "manhattan"],
        format_func=lambda x: x.capitalize()
    )

    st.divider()

    st.subheader("📦 Status Algoritma")

    all_methods = {
        **INITIAL_METHODS,
        **OPTIMIZATION_METHODS
    }

    checked = set()

    for method_name, key in all_methods.items():

        if method_name in checked:
            continue

        checked.add(method_name)

        if key is None:
            st.markdown(
                f"🟢 **{method_name}**"
            )

        elif algorithm_status(key):
            st.markdown(
                f"🟢 **{method_name}**"
            )

        else:
            st.markdown(
                f"⏳ **{method_name}** — belum tersedia"
            )


# ============================================================
# DATASET
# ============================================================

st.header("📂 1. Problem Setting")

source = st.radio(
    "Sumber data",
    [
        "Generate Random",
        "Upload CSV / Excel",
        "Input Manual"
    ],
    horizontal=True
)


new_data = None


# ------------------------------------------------------------
# RANDOM
# ------------------------------------------------------------

if source == "Generate Random":

    col1, col2, col3, col4 = st.columns(4)

    n = col1.number_input(
        "Jumlah node",
        min_value=3,
        max_value=500,
        value=10
    )

    seed = col2.number_input(
        "Seed",
        min_value=0,
        max_value=99999,
        value=42
    )

    xmax = col3.number_input(
        "Maksimum X",
        min_value=10,
        value=100
    )

    ymax = col4.number_input(
        "Maksimum Y",
        min_value=10,
        value=100
    )

    if st.button(
        "🎲 Generate Data",
        type="primary"
    ):

        rng = random.Random(int(seed))

        new_data = pd.DataFrame({
            "x": [
                round(rng.uniform(0, xmax), 2)
                for _ in range(int(n))
            ],

            "y": [
                round(rng.uniform(0, ymax), 2)
                for _ in range(int(n))
            ]
        })


# ------------------------------------------------------------
# UPLOAD
# ------------------------------------------------------------

elif source == "Upload CSV / Excel":

    uploaded_file = st.file_uploader(
        "Upload file",
        type=["csv", "xlsx", "xls"]
    )

    st.caption(
        "CSV dapat menggunakan header x,y atau tanpa header."
    )

    if uploaded_file is not None:

        if st.button(
            "📥 Gunakan File",
            type="primary"
        ):
            new_data = uploaded_file


# ------------------------------------------------------------
# MANUAL
# ------------------------------------------------------------

else:

    if "manual_data" not in st.session_state:

        st.session_state.manual_data = pd.DataFrame({
            "node": ["A", "B", "C", "D", "E"],
            "x": [10, 60, 90, 70, 30],
            "y": [20, 80, 40, 10, 50]
        })

    edited_data = st.data_editor(
        st.session_state.manual_data,
        num_rows="dynamic",
        use_container_width=True
    )

    if st.button(
        "✍️ Gunakan Data Manual",
        type="primary"
    ):

        new_data = edited_data


# ============================================================
# MEMUAT DATA
# ============================================================

if new_data is not None:

    try:

        st.session_state.df = load_data(new_data)
        st.session_state.runs = []

        st.success("Data berhasil dimuat.")

    except Exception as e:

        st.error(
            f"Gagal memuat data: {e}"
        )


# ============================================================
# DATA AKTIF
# ============================================================

df = st.session_state.df


if df is not None:

    st.divider()

    st.subheader("Data Aktif")

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # NODE AWAL
    # --------------------------------------------------------

    home_label = st.selectbox(
        "Node awal / Home",
        df["node"].tolist()
    )

    home = int(
        df.index[
            df["node"] == home_label
        ][0]
    )

    # --------------------------------------------------------
    # KOORDINAT
    # --------------------------------------------------------

    coords = list(
        zip(
            df["x"],
            df["y"]
        )
    )

    labels = df["node"].tolist()

    # --------------------------------------------------------
    # DISTANCE MATRIX
    # --------------------------------------------------------

    distance_matrix = build_distance_matrix(
        coords,
        metric
    )

    st.subheader(
        f"📏 Distance Matrix — {metric.capitalize()}"
    )

    st.dataframe(
        pd.DataFrame(
            distance_matrix,
            index=labels,
            columns=labels
        ).round(2),
        use_container_width=True
    )

    # --------------------------------------------------------
    # VISUALISASI NODE
    # --------------------------------------------------------

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["x"],
            y=df["y"],
            mode="markers+text",
            text=df["node"],
            textposition="top center",
            marker=dict(
                size=14
            )
        )
    )

    fig.update_layout(
        title="Sebaran Node",
        height=450,
        xaxis_title="X",
        yaxis_title="Y",
        yaxis=dict(
            scaleanchor="x"
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# TABS
# ============================================================

tab_learn, tab_experiment, tab_results = st.tabs(
    [
        "📚 Learn Methods",
        "🧪 Experiment",
        "📊 Results"
    ]
)


# ============================================================
# LEARN METHODS
# ============================================================

with tab_learn:

    st.header("📚 Learn Methods")

    method_info = {

        "Nearest Neighbor": (
            "Greedy Heuristic",
            "Memilih node terdekat dari node yang sedang dikunjungi."
        ),

        "Nearest Insertion": (
            "Greedy Heuristic",
            "Memilih node yang paling dekat dengan tour lalu menyisipkannya "
            "pada posisi yang memberikan tambahan jarak paling kecil."
        ),

        "Farthest Insertion": (
            "Greedy Heuristic",
            "Memilih node yang paling jauh dari tour kemudian menyisipkannya "
            "pada posisi terbaik."
        ),

        "Arbitrary Insertion": (
            "Greedy Heuristic",
            "Memilih node secara arbitrary kemudian menyisipkannya "
            "pada posisi terbaik."
        ),

        "2-opt": (
            "Local Search",
            "Menghapus dua edge dan membalik sebagian tour untuk mencari "
            "rute yang lebih pendek."
        ),

        "3-opt": (
            "Local Search",
            "Menghapus tiga edge dan mencoba beberapa kemungkinan "
            "penyambungan kembali."
        ),

        "Simulated Annealing": (
            "Metaheuristic",
            "Dapat menerima solusi yang lebih buruk dengan probabilitas tertentu "
            "agar dapat keluar dari local optimum."
        ),

        "Tabu Search": (
            "Metaheuristic",
            "Menggunakan tabu list untuk mencegah kembali ke move atau solusi "
            "tertentu dalam beberapa iterasi."
        )
    }

    selected_method = st.selectbox(
        "Pilih metode",
        list(method_info.keys())
    )

    category, description = method_info[
        selected_method
    ]

    st.markdown(
        f"### {selected_method}"
    )

    st.caption(
        f"Kategori: {category}"
    )

    st.write(description)

    st.info(
        "Ilustrasi langkah algoritma akan ditampilkan "
        "setelah modul algoritmanya selesai dibuat."
    )


# ============================================================
# EXPERIMENT
# ============================================================

with tab_experiment:

    st.header("🧪 Experiment")

    if df is None:

        st.info(
            "Masukkan data terlebih dahulu pada Problem Setting."
        )

    else:

        col1, col2 = st.columns(2)

        # ----------------------------------------------------
        # INITIAL SOLUTION
        # ----------------------------------------------------

        with col1:

            st.subheader("1️⃣ Initial Solution")

            initial_name = st.selectbox(
                "Pilih metode initial solution",
                list(INITIAL_METHODS.keys())
            )

            initial_key = INITIAL_METHODS[
                initial_name
            ]

            if initial_key is not None:

                if algorithm_status(initial_key):

                    st.success(
                        "Metode tersedia."
                    )

                else:

                    st.warning(
                        "⏳ Algoritma masih dibuat."
                    )

            else:

                st.success(
                    "Random tersedia dari base.py."
                )

        # ----------------------------------------------------
        # OPTIMIZATION
        # ----------------------------------------------------

        with col2:

            st.subheader("2️⃣ Optimization")

            optimization_name = st.selectbox(
                "Pilih metode optimasi",
                list(OPTIMIZATION_METHODS.keys())
            )

            optimization_key = OPTIMIZATION_METHODS[
                optimization_name
            ]

            if optimization_key is not None:

                if algorithm_status(optimization_key):

                    st.success(
                        "Metode tersedia."
                    )

                else:

                    st.warning(
                        "⏳ Algoritma masih dibuat."
                    )

            else:

                st.success(
                    "Tanpa optimasi."
                )

        # ----------------------------------------------------
        # PARAMETER
        # ----------------------------------------------------

        st.divider()

        st.subheader("⚙️ Parameter")

        if optimization_key is not None:

            algorithm = get_algorithm(
                optimization_key
            )

            if algorithm is None:

                st.info(
                    "Parameter akan muncul setelah "
                    "file algoritma tersedia."
                )

            else:

                function = algorithm["function_object"]

                signature = inspect.signature(
                    function
                )

                st.code(
                    f"{function.__name__}{signature}"
                )

        # ----------------------------------------------------
        # RUN
        # ----------------------------------------------------

        initial_ready = algorithm_status(
            initial_key
        )

        optimization_ready = algorithm_status(
            optimization_key
        )

        ready = (
            initial_ready
            and optimization_ready
        )

        if st.button(
            "▶️ Jalankan",
            type="primary",
            disabled=not ready,
            use_container_width=True
        ):

            st.info(
                "Algoritma akan dijalankan "
                "setelah modul algoritmanya tersedia."
            )


# ============================================================
# RESULTS
# ============================================================

with tab_results:

    st.header("📊 Results")

    if not st.session_state.runs:

        st.info(
            "Belum ada hasil eksperimen."
        )

    else:

        results_df = pd.DataFrame(
            st.session_state.runs
        )

        st.dataframe(
            results_df,
            use_container_width=True,
            hide_index=True
        )

        st.subheader(
            "⚖️ Method Comparison"
        )

        st.caption(
            "Minimal dua metode dapat dibandingkan berdasarkan "
            "final distance dan execution time."
        )

        # Grafik comparison akan aktif
        # setelah algoritma selesai dibuat.
