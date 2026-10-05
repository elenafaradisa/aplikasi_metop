import streamlit as st
import pandas as pd
import random
import time

from algorithms.base import (
    build_distance_matrix,
    tour_distance,
    generate_initial_tour,
    validate_tour
)

from algorithms.constructive import (
    nearest_neighbor,
    nearest_insertion,
    farthest_insertion,
    arbitrary_insertion
)

from algorithms.local_search import (
    two_opt,
    three_opt
)

from algorithms.simulated_annealing import (
    simulated_annealing
)

from algorithms.tabu_search import (
    tabu_search
)

from utils.load_data import load_data

from utils.visualization import (
    plot_nodes,
    plot_route,
    plot_route_comparison
)

from utils.comparison import (
    create_comparison_table,
    sort_by_distance,
    get_best_distance,
    get_fastest_method,
    plot_distance_comparison,
    plot_time_comparison,
    plot_improvement_comparison
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="TSP Learning & Optimization",
    page_icon="🧭",
    layout="wide"
)


# ============================================================
# ALGORITHM INFORMATION
# ============================================================

ALGORITHMS = {

    # --------------------------------------------------------
    # CONSTRUCTIVE
    # --------------------------------------------------------

    "Nearest Neighbor": {
        "type": "Constructive",
        "function": nearest_neighbor,
        "description":
            "Membangun rute dengan memilih node terdekat secara bertahap.",
        "needs_home": True,
        "needs_initial_route": False,
        "parameters": {}
    },

    "Nearest Insertion": {
        "type": "Constructive",
        "function": nearest_insertion,
        "description":
            "Membangun rute dengan menyisipkan node pada posisi terbaik.",
        "needs_home": True,
        "needs_initial_route": False,
        "parameters": {}
    },

    "Farthest Insertion": {
        "type": "Constructive",
        "function": farthest_insertion,
        "description":
            "Membangun rute dengan memprioritaskan node yang paling jauh.",
        "needs_home": True,
        "needs_initial_route": False,
        "parameters": {}
    },

    "Arbitrary Insertion": {
        "type": "Constructive",
        "function": arbitrary_insertion,
        "description":
            "Memilih node secara acak lalu menyisipkannya pada posisi terbaik.",
        "needs_home": True,
        "needs_initial_route": False,
        "parameters": {
            "seed": {
                "type": "int",
                "default": 42,
                "min": 0,
                "max": 99999,
                "step": 1,
                "description":
                    "Seed untuk mengontrol proses acak."
            }
        }
    },


    # --------------------------------------------------------
    # LOCAL SEARCH
    # --------------------------------------------------------

    "2-opt": {
        "type": "Local Search",
        "function": two_opt,
        "description":
            "Memperbaiki rute dengan membalik segmen rute.",
        "needs_home": False,
        "needs_initial_route": True,
        "parameters": {
            "max_iter": {
                "type": "int",
                "default": 100,
                "min": 1,
                "max": 10000,
                "step": 1,
                "description":
                    "Batas maksimum iterasi pencarian."
            },
            "strategy": {
                "type": "select",
                "options": ["best", "first"],
                "default": "best",
                "description":
                    "Best mencari perbaikan terbaik; First mengambil perbaikan pertama."
            }
        }
    },

    "3-opt": {
        "type": "Local Search",
        "function": three_opt,
        "description":
            "Memperbaiki rute dengan mengevaluasi perubahan 3-opt.",
        "needs_home": False,
        "needs_initial_route": True,
        "parameters": {
            "max_iter": {
                "type": "int",
                "default": 100,
                "min": 1,
                "max": 10000,
                "step": 1,
                "description":
                    "Batas maksimum iterasi pencarian."
            },
            "strategy": {
                "type": "select",
                "options": ["best", "first"],
                "default": "best",
                "description":
                    "Best mencari perbaikan terbaik; First mengambil perbaikan pertama."
            }
        }
    },


    # --------------------------------------------------------
    # METAHEURISTIC
    # --------------------------------------------------------

    "Simulated Annealing": {
        "type": "Metaheuristic",
        "function": simulated_annealing,
        "description":
            "Mencari solusi dengan menerima beberapa solusi lebih buruk secara probabilistik.",
        "needs_home": False,
        "needs_initial_route": True,
        "parameters": {
            "initial_temp": {
                "type": "float",
                "default": 1000.0,
                "min": 0.01,
                "max": 100000.0,
                "step": 10.0,
                "description":
                    "Suhu awal pencarian."
            },
            "cooling_rate": {
                "type": "float",
                "default": 0.95,
                "min": 0.01,
                "max": 0.999,
                "step": 0.01,
                "description":
                    "Laju penurunan suhu setiap iterasi."
            },
            "min_temp": {
                "type": "float",
                "default": 0.01,
                "min": 0.0001,
                "max": 100.0,
                "step": 0.01,
                "description":
                    "Suhu minimum sebelum pencarian berhenti."
            },
            "max_iter": {
                "type": "int",
                "default": 10,
                "min": 1,
                "max": 10000,
                "step": 1,
                "description":
                    "Batas maksimum iterasi."
            },
            "seed": {
                "type": "int",
                "default": 42,
                "min": 0,
                "max": 99999,
                "step": 1,
                "description":
                    "Seed untuk menjaga hasil acak tetap konsisten."
            },
            "verbose_history": {
                "type": "bool",
                "default": False,
                "description":
                    "Menyimpan informasi history tambahan."
            }
        }
    },

    "Tabu Search": {
        "type": "Metaheuristic",
        "function": tabu_search,
        "description":
            "Mencari solusi dengan menyimpan perpindahan sebelumnya dalam tabu list.",
        "needs_home": False,
        "needs_initial_route": True,
        "parameters": {
            "max_iter": {
                "type": "int",
                "default": 100,
                "min": 1,
                "max": 10000,
                "step": 1,
                "description":
                    "Batas maksimum iterasi."
            },
            "tabu_tenure": {
                "type": "int",
                "default": 3,
                "min": 1,
                "max": 100,
                "step": 1,
                "description":
                    "Lama perpindahan berada dalam tabu list."
            },
            "verbose_history": {
                "type": "bool",
                "default": False,
                "description":
                    "Menyimpan informasi history tambahan."
            }
        }
    }
}


CONSTRUCTIVE = [
    name for name, info in ALGORITHMS.items()
    if info["type"] == "Constructive"
]

LOCAL_SEARCH = [
    name for name, info in ALGORITHMS.items()
    if info["type"] == "Local Search"
]

METAHEURISTIC = [
    name for name, info in ALGORITHMS.items()
    if info["type"] == "Metaheuristic"
]


# ============================================================
# SESSION STATE
# ============================================================

if "df" not in st.session_state:
    st.session_state.df = None

if "independent_results" not in st.session_state:
    st.session_state.independent_results = []

if "hybrid_results" not in st.session_state:
    st.session_state.hybrid_results = []

if "manual_data" not in st.session_state:
    st.session_state.manual_data = pd.DataFrame({
        "node": ["A", "B", "C", "D", "E", "F"],
        "x": [10, 60, 90, 70, 30, 20],
        "y": [20, 80, 40, 50, 90, 10]
    })


# ============================================================
# HELPER
# ============================================================

def route_to_labels(df, route):

    if route is None:
        return "-"

    return " → ".join(
        str(df.iloc[int(i)]["node"])
        for i in route
    )


def normalize_route(route, n):

    if route is None:
        return None

    route = list(route)

    if len(route) == n:
        route.append(route[0])

    return route


def extract_route(result):

    if isinstance(result, dict):

        for key in [
            "route",
            "tour",
            "best_tour",
            "final_tour"
        ]:
            if key in result:
                return result[key]

    if isinstance(result, tuple):

        if len(result) > 0:
            if isinstance(result[0], (list, tuple)):
                return list(result[0])

    if isinstance(result, (list, tuple)):
        return list(result)

    return None


def extract_distance(result, route, dist_matrix):

    if isinstance(result, dict):

        for key in [
            "distance",
            "best_distance",
            "final_distance"
        ]:
            if key in result:
                try:
                    return float(result[key])
                except:
                    pass

    if isinstance(result, tuple):

        for value in result[1:]:

            if isinstance(value, (int, float)):
                return float(value)

    if route is not None:
        return tour_distance(route, dist_matrix)

    return float("inf")


def extract_history(result):

    if isinstance(result, dict):
        return result.get("history")

    if isinstance(result, tuple):

        for value in result:

            if isinstance(value, list):

                if value and isinstance(value[0], dict):
                    return value

    return None


def render_parameters(method_name, key_prefix):

    params = {}

    configs = ALGORITHMS[method_name]["parameters"]

    for name, config in configs.items():

        st.caption(
            f"**{name.replace('_', ' ').title()}** — "
            f"{config['description']}"
        )

        widget_key = f"{key_prefix}_{method_name}_{name}"

        if config["type"] == "int":

            params[name] = st.number_input(
                name.replace("_", " ").title(),
                min_value=config["min"],
                max_value=config["max"],
                value=config["default"],
                step=config["step"],
                key=widget_key
            )

        elif config["type"] == "float":

            params[name] = st.number_input(
                name.replace("_", " ").title(),
                min_value=config["min"],
                max_value=config["max"],
                value=config["default"],
                step=config["step"],
                key=widget_key
            )

        elif config["type"] == "select":

            params[name] = st.selectbox(
                name.replace("_", " ").title(),
                config["options"],
                index=config["options"].index(
                    config["default"]
                ),
                key=widget_key
            )

        elif config["type"] == "bool":

            params[name] = st.checkbox(
                name.replace("_", " ").title(),
                value=config["default"],
                key=widget_key
            )

    return params


def run_algorithm(
    method_name,
    dist_matrix,
    initial_route=None,
    home=None,
    parameters=None
):

    parameters = parameters or {}

    function = ALGORITHMS[method_name]["function"]

    # --------------------------------------------------------
    # CONSTRUCTIVE
    # --------------------------------------------------------

    if method_name == "Nearest Neighbor":

        return function(
            dist_matrix,
            start_node=home
        )

    if method_name == "Nearest Insertion":

        return function(
            dist_matrix,
            start_node=home
        )

    if method_name == "Farthest Insertion":

        return function(
            dist_matrix,
            start_node=home
        )

    if method_name == "Arbitrary Insertion":

        return function(
            dist_matrix,
            start_node=home,
            **parameters
        )


    # --------------------------------------------------------
    # LOCAL SEARCH
    # --------------------------------------------------------

    if method_name == "2-opt":

        return function(
            initial_route=initial_route,
            dist_matrix=dist_matrix,
            **parameters
        )

    if method_name == "3-opt":

        return function(
            initial_route=initial_route,
            dist_matrix=dist_matrix,
            **parameters
        )


    # --------------------------------------------------------
    # METAHEURISTIC
    # --------------------------------------------------------

    if method_name == "Simulated Annealing":

        return function(
            dist_matrix=dist_matrix,
            initial_tour=initial_route,
            **parameters
        )

    if method_name == "Tabu Search":

        return function(
            initial_route=initial_route,
            dist_matrix=dist_matrix,
            **parameters
        )

    raise ValueError(
        f"Algoritma {method_name} belum didukung."
    )


def make_initial_route(df, mode, home_index, seed):

    if mode == "Generate Random":

        return generate_initial_tour(
            len(df),
            home=home_index,
            seed=int(seed)
        )

    return None


def make_manual_route(df, home_index, order):

    if order is None:
        return None

    if len(order) != len(df) - 1:
        return None

    route = [home_index]

    for node in order:

        idx = df.index[
            df["node"] == node
        ][0]

        route.append(int(idx))

    route.append(home_index)

    return route


def run_one(
    method_name,
    df,
    dist_matrix,
    home_index=None,
    initial_route=None,
    parameters=None
):

    start = time.perf_counter()

    result = run_algorithm(
        method_name=method_name,
        dist_matrix=dist_matrix,
        initial_route=initial_route,
        home=home_index,
        parameters=parameters
    )

    elapsed = time.perf_counter() - start

    route = extract_route(result)

    route = normalize_route(
        route,
        len(df)
    )

    distance = extract_distance(
        result,
        route,
        dist_matrix
    )

    history = extract_history(result)

    if initial_route is not None:

        initial_distance = tour_distance(
            initial_route,
            dist_matrix
        )

    else:

        initial_distance = None

    return {
        "method": method_name,
        "route": route,
        "initial_route": initial_route,
        "initial_distance": initial_distance,
        "final_distance": distance,
        "execution_time": elapsed,
        "history": history
    }


# ============================================================
# HEADER
# ============================================================

st.title("🧭 TSP Learning & Optimization")

st.caption(
    "Pelajari, jalankan, dan bandingkan metode "
    "Travelling Salesman Problem."
)


# ============================================================
# MODE
# ============================================================

tab_independent, tab_hybrid = st.tabs([
    "Independent",
    "Hybrid"
])


# ============================================================
# DATASET
# ============================================================

st.header("📂 Dataset")

input_type = st.radio(
    "Input dataset:",
    ["Generate", "Upload", "Manual"],
    horizontal=True
)

new_data = None


if input_type == "Generate":

    c1, c2, c3, c4 = st.columns(4)

    n_nodes = c1.number_input(
        "Jumlah node",
        min_value=3,
        max_value=500,
        value=10
    )

    dataset_seed = c2.number_input(
        "Seed",
        min_value=0,
        value=42
    )

    max_x = c3.number_input(
        "Maksimum X",
        min_value=1,
        value=100
    )

    max_y = c4.number_input(
        "Maksimum Y",
        min_value=1,
        value=100
    )

    if st.button(
        "Generate Dataset",
        type="primary"
    ):

        rng = random.Random(
            int(dataset_seed)
        )

        new_data = pd.DataFrame({

            "node": [
                chr(65 + i)
                if i < 26
                else f"N{i + 1}"
                for i in range(int(n_nodes))
            ],

            "x": [
                round(
                    rng.uniform(0, max_x),
                    2
                )
                for _ in range(int(n_nodes))
            ],

            "y": [
                round(
                    rng.uniform(0, max_y),
                    2
                )
                for _ in range(int(n_nodes))
            ]
        })


elif input_type == "Upload":

    uploaded = st.file_uploader(
        "Upload CSV / Excel",
        type=["csv", "xlsx", "xls"]
    )

    if uploaded is not None:

        if st.button(
            "Gunakan Dataset",
            type="primary"
        ):
            new_data = uploaded


else:

    edited = st.data_editor(
        st.session_state.manual_data,
        num_rows="dynamic",
        use_container_width=True
    )

    if st.button(
        "Gunakan Dataset",
        type="primary"
    ):
        new_data = edited


if new_data is not None:

    try:

        st.session_state.df = load_data(
            new_data
        )

        st.session_state.independent_results = []
        st.session_state.hybrid_results = []

        st.success(
            "Dataset berhasil digunakan."
        )

        st.rerun()

    except Exception as e:

        st.error(
            f"Gagal membaca dataset: {e}"
        )


# ============================================================
# IF DATA EXISTS
# ============================================================

if st.session_state.df is None:

    st.info(
        "Masukkan dataset terlebih dahulu."
    )

    st.stop()


df = st.session_state.df

coords = list(
    zip(
        df["x"],
        df["y"]
    )
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Pengaturan")

    metric = st.radio(
        "Distance Method",
        ["Euclidean", "Manhattan"]
    )

    metric_key = metric.lower()

    st.divider()


# ============================================================
# INDEPENDENT
# ============================================================

with tab_independent:

    with st.sidebar:

        st.subheader("Algorithm Options")

        selected_independent = st.multiselect(
            "Select algorithm(s):",
            list(ALGORITHMS.keys()),
            key="independent_algorithms"
        )


        st.divider()

        st.subheader("Initial Solution")

        has_constructive = any(
            ALGORITHMS[m]["needs_home"]
            for m in selected_independent
        )

        has_initial_route = any(
            ALGORITHMS[m]["needs_initial_route"]
            for m in selected_independent
        )


        # ----------------------------------------------------
        # HOME
        # ----------------------------------------------------

        home_index = 0

        if has_constructive:

            st.markdown(
                "Node awal / Home"
            )

            home_label = st.selectbox(
                "Pilih node awal:",
                df["node"].tolist(),
                key="independent_home"
            )

            home_index = int(
                df.index[
                    df["node"] == home_label
                ][0]
            )

            st.caption(
                "Digunakan oleh metode Constructive "
                "untuk menentukan node awal."
            )


        # ----------------------------------------------------
        # INITIAL ROUTE
        # ----------------------------------------------------

        initial_route = None

        if has_initial_route:

            st.markdown(
                "Starting Route"
            )

            st.caption(
                "Local Search dan Metaheuristic "
                "membutuhkan satu rute sebagai solusi awal."
            )

            route_mode = st.radio(
                "Sumber initial route:",
                ["Generate Random", "Pilih manual"],
                horizontal=True,
                key="independent_route_mode"
            )


            if route_mode == "Generate Random":

                seed = st.number_input(
                    "Seed",
                    min_value=0,
                    max_value=99999,
                    value=42,
                    step=1,
                    key="independent_route_seed"
                )

                # Home internal hanya digunakan
                # untuk membentuk closed tour.
                initial_route = generate_initial_tour(
                    len(df),
                    home=(
                        home_index
                        if has_constructive
                        else 0
                    ),
                    seed=int(seed)
                )


            else:

                route_home = (
                    home_index
                    if has_constructive
                    else 0
                )

                remaining = [
                    node
                    for node in df["node"].tolist()
                    if node != df.iloc[route_home]["node"]
                ]

                order = st.multiselect(
                    "Urutan node:",
                    remaining,
                    key="independent_manual_route"
                )

                if len(order) == len(remaining):

                    initial_route = make_manual_route(
                        df,
                        route_home,
                        order
                    )

                else:

                    st.caption(
                        "Pilih semua node untuk membentuk rute."
                    )


        # ----------------------------------------------------
        # PARAMETERS
        # ----------------------------------------------------

        st.divider()

        st.subheader(
            "Algorithm Parameters"
        )

        independent_params = {}

        for method in selected_independent:

            with st.expander(
                method
            ):

                st.caption(
                    ALGORITHMS[method]["description"]
                )

                independent_params[method] = (
                    render_parameters(
                        method,
                        "independent"
                    )
                )


    # ========================================================
    # DATASET VIEW
    # ========================================================

    st.subheader("Dataset")

    c1, c2 = st.columns(2)

    with c1:

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

    dist_matrix = build_distance_matrix(
        coords,
        metric=metric_key
    )

    with c2:

        fig = plot_nodes(
            df,
            home=home_index if has_constructive else 0,
            title="Node Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # ========================================================
    # RUN INDEPENDENT
    # ========================================================

    if st.button(
        "▶ Jalankan Independent",
        type="primary",
        use_container_width=True
    ):

        if not selected_independent:

            st.warning(
                "Pilih minimal satu algoritma."
            )

        elif (
            has_initial_route
            and initial_route is None
        ):

            st.warning(
                "Initial route belum lengkap."
            )

        else:

            results = []

            for method in selected_independent:

                try:

                    # Setiap algoritma berjalan sendiri.
                    result = run_one(
                        method,
                        df,
                        dist_matrix,
                        home_index=(
                            home_index
                            if ALGORITHMS[method]["needs_home"]
                            else None
                        ),
                        initial_route=(
                            initial_route
                            if ALGORITHMS[method]["needs_initial_route"]
                            else None
                        ),
                        parameters=independent_params.get(
                            method,
                            {}
                        )
                    )

                    results.append(result)

                except Exception as e:

                    st.error(
                        f"{method} gagal dijalankan: {e}"
                    )

            st.session_state.independent_results = results


    # ========================================================
    # INDEPENDENT RESULTS
    # ========================================================

    results = st.session_state.independent_results

    if results:

        st.divider()

        st.subheader(
            "Independent Results"
        )

        for i, result in enumerate(results):

            method = result["method"]

            st.markdown(
                f"### {method}"
            )

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Initial Distance",
                (
                    f"{result['initial_distance']:.2f}"
                    if result["initial_distance"] is not None
                    else "-"
                )
            )

            c2.metric(
                "Final Distance",
                f"{result['final_distance']:.2f}"
            )

            c3.metric(
                "Execution Time",
                f"{result['execution_time'] * 1000:.3f} ms"
            )

            st.write(
                "**Final Route:**",
                route_to_labels(
                    df,
                    result["route"]
                )
            )

            if result["route"] is not None:

                fig = plot_route(
                    df,
                    result["route"],
                    home=result["route"][0],
                    title=f"{method} — Final Route"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    key=f"ind_route_{i}"
                )


        # ====================================================
        # COMPARISON
        # ====================================================

        if len(results) >= 2:

            st.divider()

            st.subheader(
                "Comparison"
            )

            comparison_df = create_comparison_table(
                results
            )

            comparison_df = sort_by_distance(
                comparison_df
            )

            st.dataframe(
                comparison_df,
                use_container_width=True,
                hide_index=True
            )

            c1, c2 = st.columns(2)

            best = get_best_distance(
                comparison_df
            )

            fastest = get_fastest_method(
                comparison_df
            )

            with c1:

                if best is not None:

                    st.metric(
                        "Best Distance",
                        best["Method"],
                        f"{best['Final Distance']:.2f}"
                    )

            with c2:

                if fastest is not None:

                    st.metric(
                        "Fastest",
                        fastest["Method"],
                        f"{fastest['Execution Time (ms)']:.3f} ms"
                    )

            c1, c2 = st.columns(2)

            with c1:

                st.plotly_chart(
                    plot_distance_comparison(
                        comparison_df
                    ),
                    use_container_width=True
                )

            with c2:

                st.plotly_chart(
                    plot_time_comparison(
                        comparison_df
                    ),
                    use_container_width=True
                )

            st.plotly_chart(
                plot_improvement_comparison(
                    comparison_df
                ),
                use_container_width=True
            )


# ============================================================
# HYBRID
# ============================================================

with tab_hybrid:

    st.subheader(
        "Hybrid / Sequential Optimization"
    )

    st.caption(
        "Output dari suatu algoritma digunakan sebagai "
        "initial route untuk algoritma berikutnya."
    )


    # --------------------------------------------------------
    # SIDEBAR HYBRID
    # --------------------------------------------------------

    with st.sidebar:

        st.divider()

        st.subheader(
            "Hybrid Configuration"
        )

        hybrid_methods = st.multiselect(
            "Urutan algoritma:",
            list(ALGORITHMS.keys()),
            key="hybrid_algorithms",
            help=(
                "Urutan pilihan menentukan urutan proses hybrid."
            )
        )


        # ----------------------------------------------------
        # HOME
        # ----------------------------------------------------

        hybrid_home = 0

        hybrid_has_constructive = any(
            ALGORITHMS[m]["needs_home"]
            for m in hybrid_methods
        )

        if hybrid_has_constructive:

            st.markdown(
                "Node awal / Home"
            )

            hybrid_home_label = st.selectbox(
                "Pilih node awal:",
                df["node"].tolist(),
                key="hybrid_home"
            )

            hybrid_home = int(
                df.index[
                    df["node"] == hybrid_home_label
                ][0]
            )

            st.caption(
                "Digunakan sebagai node awal "
                "oleh metode Constructive."
            )


        # ----------------------------------------------------
        # INITIAL ROUTE
        # ----------------------------------------------------

        first_needs_route = (
            len(hybrid_methods) > 0
            and ALGORITHMS[
                hybrid_methods[0]
            ]["needs_initial_route"]
        )

        hybrid_initial_route = None


        if first_needs_route:

            st.markdown(
                "Initial Solution"
            )

            st.caption(
                "Algoritma pertama membutuhkan "
                "initial route sebagai titik awal."
            )

            hybrid_route_mode = st.radio(
                "Sumber initial route:",
                [
                    "Generate Random",
                    "Pilih manual"
                ],
                horizontal=True,
                key="hybrid_route_mode"
            )


            if hybrid_route_mode == "Generate Random":

                hybrid_seed = st.number_input(
                    "Seed",
                    min_value=0,
                    max_value=99999,
                    value=42,
                    step=1,
                    key="hybrid_seed"
                )

                hybrid_initial_route = (
                    generate_initial_tour(
                        len(df),
                        home=0,
                        seed=int(hybrid_seed)
                    )
                )


            else:

                remaining = [
                    node
                    for node in df["node"].tolist()
                    if node != df.iloc[0]["node"]
                ]

                hybrid_order = st.multiselect(
                    "Urutan node:",
                    remaining,
                    key="hybrid_manual_route"
                )

                if len(hybrid_order) == len(remaining):

                    hybrid_initial_route = (
                        make_manual_route(
                            df,
                            0,
                            hybrid_order
                        )
                    )


        # ----------------------------------------------------
        # PARAMETERS FOR EACH STEP
        # ----------------------------------------------------

        hybrid_params = {}

        for i, method in enumerate(hybrid_methods):

            with st.expander(
                f"Step {i + 1} — {method}"
            ):

                st.caption(
                    ALGORITHMS[method]["description"]
                )

                hybrid_params[method] = (
                    render_parameters(
                        method,
                        f"hybrid_{i}"
                    )
                )


    # ========================================================
    # PIPELINE PREVIEW
    # ========================================================

    if hybrid_methods:

        st.markdown(
            "### Hybrid Pipeline"
        )

        pipeline = " → ".join(
            hybrid_methods
        )

        st.info(
            pipeline
        )


    # ========================================================
    # RUN HYBRID
    # ========================================================

    if st.button(
        "▶ Jalankan Hybrid",
        type="primary",
        use_container_width=True
    ):

        if not hybrid_methods:

            st.warning(
                "Pilih minimal satu algoritma."
            )

        elif (
            first_needs_route
            and hybrid_initial_route is None
        ):

            st.warning(
                "Initial route belum lengkap."
            )

        else:

            dist_matrix = build_distance_matrix(
                coords,
                metric=metric_key
            )

            current_route = hybrid_initial_route

            hybrid_results = []


            for step, method in enumerate(
                hybrid_methods
            ):

                try:

                    result = run_one(
                        method,
                        df,
                        dist_matrix,

                        home_index=(
                            hybrid_home
                            if ALGORITHMS[method]["needs_home"]
                            else None
                        ),

                        initial_route=(
                            current_route
                            if ALGORITHMS[method]["needs_initial_route"]
                            else None
                        ),

                        parameters=hybrid_params.get(
                            method,
                            {}
                        )
                    )


                    hybrid_results.append(
                        result
                    )


                    # ------------------------------------------------
                    # OUTPUT STEP INI MENJADI INPUT STEP BERIKUTNYA
                    # ------------------------------------------------

                    current_route = result["route"]


                except Exception as e:

                    st.error(
                        f"Step {step + 1} — "
                        f"{method} gagal: {e}"
                    )

                    break


            st.session_state.hybrid_results = (
                hybrid_results
            )


    # ========================================================
    # HYBRID RESULT
    # ========================================================

    hybrid_results = (
        st.session_state.hybrid_results
    )

    if hybrid_results:

        st.divider()

        st.subheader(
            "Hybrid Results"
        )


        for i, result in enumerate(
            hybrid_results
        ):

            st.markdown(
                f"### Step {i + 1} — "
                f"{result['method']}"
            )

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Initial Distance",
                (
                    f"{result['initial_distance']:.2f}"
                    if result["initial_distance"] is not None
                    else "-"
                )
            )

            c2.metric(
                "Final Distance",
                f"{result['final_distance']:.2f}"
            )

            c3.metric(
                "Execution Time",
                f"{result['execution_time'] * 1000:.3f} ms"
            )

            st.write(
                "**Route:**",
                route_to_labels(
                    df,
                    result["route"]
                )
            )

            if result["route"] is not None:

                fig = plot_route(
                    df,
                    result["route"],
                    home=result["route"][0],
                    title=(
                        f"Step {i + 1} — "
                        f"{result['method']}"
                    )
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    key=f"hybrid_route_{i}"
                )


        # ----------------------------------------------------
        # FINAL HYBRID RESULT
        # ----------------------------------------------------

        final = hybrid_results[-1]

        st.success(
            f"Final Hybrid Distance: "
            f"{final['final_distance']:.2f}"
        )

        st.write(
            "**Final Hybrid Route:**",
            route_to_labels(
                df,
                final["route"]
            )
        )

