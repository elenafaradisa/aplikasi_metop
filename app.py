import streamlit as st
import pandas as pd
import random
import time
import importlib
import inspect

from algorithms.base import (
    build_distance_matrix,
    tour_distance,
    generate_initial_tour,
    validate_tour
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
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="TSP Learning & Optimization",
    page_icon="🧭",
    layout="wide"
)


# ============================================================
# ALGORITHM REGISTRY
# ============================================================

ALGORITHMS = {

    # --------------------------------------------------------
    # Constructive / Greedy
    # --------------------------------------------------------

    "Nearest Neighbor": {
        "key": "nearest_neighbor",
        "category": "Constructive / Greedy",
        "module": "algorithms.constructive",
        "description": "Membangun rute dengan memilih node terdekat secara bertahap.",
        "needs_home": True,
        "needs_initial_route": False,
        "parameters": {}
    },

    "Nearest Insertion": {
        "key": "nearest_insertion",
        "category": "Constructive / Greedy",
        "module": "algorithms.constructive",
        "description": "Membangun rute dengan menyisipkan node yang paling dekat ke rute.",
        "needs_home": True,
        "needs_initial_route": False,
        "parameters": {}
    },

    "Farthest Insertion": {
        "key": "farthest_insertion",
        "category": "Constructive / Greedy",
        "module": "algorithms.constructive",
        "description": "Membangun rute dengan memprioritaskan node yang paling jauh.",
        "needs_home": True,
        "needs_initial_route": False,
        "parameters": {}
    },

    "Arbitrary Insertion": {
        "key": "arbitrary_insertion",
        "category": "Constructive / Greedy",
        "module": "algorithms.constructive",
        "description": "Memilih node secara acak lalu menyisipkannya pada posisi terbaik.",
        "needs_home": True,
        "needs_initial_route": False,
        "parameters": {
            "seed": {
                "type": "int",
                "default": 42,
                "min": 0,
                "max": 99999,
                "step": 1,
                "description": "Seed untuk mengatur urutan pemilihan node secara acak."
            }
        }
    },


    # --------------------------------------------------------
    # Local Search
    # --------------------------------------------------------

    "2-opt": {
        "key": "two_opt",
        "category": "Local Search",
        "module": "algorithms.local_search",
        "description": "Memperbaiki rute dengan membalik segmen rute menggunakan 2-opt.",
        "needs_home": False,
        "needs_initial_route": True,
        "parameters": {
            "max_iter": {
                "type": "int",
                "default": 100,
                "min": 1,
                "max": 10000,
                "step": 1,
                "description": "Jumlah maksimum iterasi pencarian."
            },
            "strategy": {
                "type": "select",
                "options": ["best", "first"],
                "default": "best",
                "description": "Best memilih perbaikan terbaik; First memilih perbaikan pertama yang ditemukan."
            }
        }
    },

    "3-opt": {
        "key": "three_opt",
        "category": "Local Search",
        "module": "algorithms.local_search",
        "description": "Memperbaiki rute dengan mengevaluasi kombinasi perubahan 3-opt.",
        "needs_home": False,
        "needs_initial_route": True,
        "parameters": {
            "max_iter": {
                "type": "int",
                "default": 100,
                "min": 1,
                "max": 10000,
                "step": 1,
                "description": "Jumlah maksimum iterasi pencarian."
            },
            "strategy": {
                "type": "select",
                "options": ["best", "first"],
                "default": "best",
                "description": "Best memilih perbaikan terbaik; First memilih perbaikan pertama yang ditemukan."
            }
        }
    },


    # --------------------------------------------------------
    # Metaheuristic
    # --------------------------------------------------------

    "Simulated Annealing": {
        "key": "simulated_annealing",
        "category": "Metaheuristic",
        "module": "algorithms.simulated_annealing",
        "description": "Mencari solusi lebih baik dengan menerima beberapa solusi yang lebih buruk secara probabilistik.",
        "needs_home": False,
        "needs_initial_route": True,
        "parameters": {
            "initial_temp": {
                "type": "float",
                "default": 1000.0,
                "min": 0.01,
                "max": 100000.0,
                "step": 10.0,
                "description": "Suhu awal proses pencarian."
            },
            "cooling_rate": {
                "type": "float",
                "default": 0.95,
                "min": 0.01,
                "max": 0.999,
                "step": 0.01,
                "description": "Laju penurunan suhu setiap iterasi."
            },
            "min_temp": {
                "type": "float",
                "default": 0.01,
                "min": 0.0001,
                "max": 100.0,
                "step": 0.01,
                "description": "Batas suhu minimum sebelum pencarian berhenti."
            },
            "max_iter": {
                "type": "int",
                "default": 10,
                "min": 1,
                "max": 10000,
                "step": 1,
                "description": "Jumlah maksimum iterasi."
            },
            "seed": {
                "type": "int",
                "default": 42,
                "min": 0,
                "max": 99999,
                "step": 1,
                "description": "Seed untuk menjaga hasil acak tetap dapat direproduksi."
            },
            "verbose_history": {
                "type": "bool",
                "default": False,
                "description": "Mengatur apakah informasi history tambahan disimpan."
            }
        }
    },

    "Tabu Search": {
        "key": "tabu_search",
        "category": "Metaheuristic",
        "module": "algorithms.tabu_search",
        "description": "Mencari solusi dengan menyimpan perpindahan sebelumnya dalam tabu list.",
        "needs_home": False,
        "needs_initial_route": True,
        "parameters": {
            "max_iter": {
                "type": "int",
                "default": 100,
                "min": 1,
                "max": 10000,
                "step": 1,
                "description": "Jumlah maksimum iterasi."
            },
            "tabu_tenure": {
                "type": "int",
                "default": 3,
                "min": 1,
                "max": 100,
                "step": 1,
                "description": "Berapa lama suatu perpindahan tetap berada dalam tabu list."
            },
            "verbose_history": {
                "type": "bool",
                "default": False,
                "description": "Mengatur apakah informasi history tambahan disimpan."
            }
        }
    }
}


# ============================================================
# METHOD GROUPS
# ============================================================

CONSTRUCTIVE_METHODS = [
    "Nearest Neighbor",
    "Nearest Insertion",
    "Farthest Insertion",
    "Arbitrary Insertion"
]

INITIAL_ROUTE_METHODS = [
    "2-opt",
    "3-opt",
    "Simulated Annealing",
    "Tabu Search"
]


# ============================================================
# SESSION STATE
# ============================================================

if "df" not in st.session_state:
    st.session_state.df = None

if "results" not in st.session_state:
    st.session_state.results = []

if "manual_data" not in st.session_state:
    st.session_state.manual_data = pd.DataFrame({
        "node": ["A", "B", "C", "D", "E", "F"],
        "x": [10, 60, 90, 70, 30, 20],
        "y": [20, 80, 40, 10, 50, 90]
    })


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_algorithm_function(method_name):
    """
    Mengambil fungsi algoritma berdasarkan registry.
    """

    info = ALGORITHMS.get(method_name)

    if info is None:
        return None

    try:
        module = importlib.import_module(info["module"])
        return getattr(module, info["key"], None)

    except Exception:
        return None


def algorithm_available(method_name):
    """
    Mengecek apakah fungsi algoritma tersedia.
    """

    return get_algorithm_function(method_name) is not None


def render_parameter_widget(method_name):
    """
    Membuat widget parameter berdasarkan konfigurasi algoritma.
    """

    info = ALGORITHMS[method_name]
    parameter_config = info.get("parameters", {})

    params = {}

    if not parameter_config:
        st.caption("Metode ini tidak memiliki parameter tambahan.")

        return params

    for name, config in parameter_config.items():

        st.caption(
            f"**{name.replace('_', ' ').title()}** — "
            f"{config['description']}"
        )

        param_type = config["type"]

        if param_type == "int":

            params[name] = st.number_input(
                name.replace("_", " ").title(),
                min_value=config["min"],
                max_value=config["max"],
                value=config["default"],
                step=config["step"],
                key=f"{method_name}_{name}"
            )

        elif param_type == "float":

            params[name] = st.number_input(
                name.replace("_", " ").title(),
                min_value=config["min"],
                max_value=config["max"],
                value=config["default"],
                step=config["step"],
                key=f"{method_name}_{name}"
            )

        elif param_type == "select":

            params[name] = st.selectbox(
                name.replace("_", " ").title(),
                config["options"],
                index=config["options"].index(config["default"]),
                key=f"{method_name}_{name}"
            )

        elif param_type == "bool":

            params[name] = st.checkbox(
                name.replace("_", " ").title(),
                value=config["default"],
                key=f"{method_name}_{name}"
            )

    return params


def run_algorithm(
    method_name,
    dist_matrix,
    initial_route=None,
    home=None,
    parameters=None
):
    """
    Menjalankan algoritma sesuai signature fungsi yang tersedia.
    """

    function = get_algorithm_function(method_name)

    if function is None:
        raise ValueError(
            f"Algoritma {method_name} belum tersedia."
        )

    if parameters is None:
        parameters = {}

    signature = inspect.signature(function)

    kwargs = {}

    for name in signature.parameters:

        lname = name.lower()

        if lname in {
            "dist_matrix",
            "distance_matrix",
            "dist",
            "matrix"
        }:
            kwargs[name] = dist_matrix

        elif lname in {
            "tour",
            "initial_tour",
            "route",
            "solution"
        }:
            kwargs[name] = initial_route

        elif lname in {
            "home",
            "start",
            "start_node"
        }:
            kwargs[name] = home

        elif name in parameters:
            kwargs[name] = parameters[name]

    return function(**kwargs)


def extract_route(result):
    """
    Mengambil route dari berbagai format return algoritma.
    """

    # --------------------------------------------------------
    # Dictionary result
    # --------------------------------------------------------

    if isinstance(result, dict):

        for key in [
            "route",
            "tour",
            "best_tour",
            "final_tour"
        ]:
            if key in result:
                return result[key]

        return None

    # --------------------------------------------------------
    # Tuple result
    # Contoh Arbitrary Insertion:
    # (route, distance)
    #
    # Simulated Annealing:
    # (best_tour, best_distance, history)
    # --------------------------------------------------------

    if isinstance(result, tuple):

        if len(result) > 0:

            first = result[0]

            if isinstance(first, (list, tuple)):
                return list(first)

    # --------------------------------------------------------
    # Direct route
    # --------------------------------------------------------

    if isinstance(result, (list, tuple)):
        return list(result)

    return None


def extract_distance(result, route, dist_matrix):
    """
    Mengambil distance dari hasil algoritma.
    Jika tidak tersedia, hitung ulang dari route.
    """

    if isinstance(result, dict):

        for key in [
            "distance",
            "best_distance",
            "final_distance"
        ]:

            if key in result:

                try:
                    return float(result[key])
                except Exception:
                    pass

    if isinstance(result, tuple):

        # Arbitrary Insertion
        if len(result) >= 2:

            if isinstance(result[1], (int, float)):

                return float(result[1])

        # Simulated Annealing
        if len(result) >= 2:

            if isinstance(result[1], (int, float)):

                return float(result[1])

    if route is not None:

        return tour_distance(
            route,
            dist_matrix
        )

    return float("inf")


def extract_history(result):
    """
    Mengambil history bila algoritma menyediakannya.
    """

    if isinstance(result, dict):
        return result.get("history")

    if isinstance(result, tuple):

        for item in result:

            if isinstance(item, list):

                if len(item) > 0 and isinstance(item[0], dict):
                    return item

    return None


def normalize_route(route, n):
    """
    Memastikan route berbentuk closed tour.
    """

    if route is None:
        return None

    route = list(route)

    # Jika belum closed
    if len(route) == n:
        route.append(route[0])

    return route


def route_to_labels(df, route):
    """
    Mengubah index route menjadi nama node.
    """

    if route is None:
        return "-"

    return " → ".join(
        str(df.iloc[index]["node"])
        for index in route
    )


def get_initial_route_for_algorithm(
    df,
    home_index,
    route_mode,
    seed,
    selected_order
):
    """
    Membentuk initial route untuk Local Search / Metaheuristic.
    """

    n = len(df)

    if route_mode == "Generate Random":

        return generate_initial_tour(
            n,
            home=home_index,
            seed=int(seed)
        )

    if route_mode == "Pilih manual":

        if selected_order is None:
            return None

        node_labels = df["node"].tolist()

        if len(selected_order) != n - 1:
            return None

        route = [home_index]

        for node in selected_order:

            idx = df.index[
                df["node"] == node
            ][0]

            route.append(idx)

        route.append(home_index)

        return route

    return None


# ============================================================
# PAGE HEADER
# ============================================================

st.title("🧭 TSP Learning & Optimization")

st.caption(
    "Pelajari dan bandingkan metode Constructive, "
    "Local Search, dan Metaheuristic untuk "
    "Travelling Salesman Problem."
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Pengaturan")

    st.divider()


    # ========================================================
    # DISTANCE METHOD
    # ========================================================

    st.subheader("Distance Method")

    metric = st.radio(
        "Select distance method:",
        ["Euclidean", "Manhattan"],
        label_visibility="collapsed"
    )

    metric_key = metric.lower()


    st.divider()


    # ========================================================
    # ALGORITHM SELECTION
    # ========================================================

    st.subheader("Algorithm Options")

    selected_methods = st.multiselect(
        "Select algorithm(s):",
        list(ALGORITHMS.keys()),
        label_visibility="collapsed",
        help="Pilih satu atau beberapa algoritma untuk dibandingkan."
    )

    if not selected_methods:

        st.caption(
            "Pilih minimal satu algoritma."
        )


    st.divider()


    # ========================================================
    # INITIAL SOLUTION
    # ========================================================

    st.subheader("Initial Solution")

    home_index = 0

    initial_route = None

    selected_home = None

    route_mode = None

    seed = 42

    selected_order = None


    # --------------------------------------------------------
    # Apakah ada constructive method?
    # --------------------------------------------------------

    needs_home = any(
        ALGORITHMS[m]["needs_home"]
        for m in selected_methods
    )

    needs_initial_route = any(
        ALGORITHMS[m]["needs_initial_route"]
        for m in selected_methods
    )


    if st.session_state.df is None:

        st.info(
            "Masukkan dataset terlebih dahulu "
            "untuk mengatur initial solution."
        )

    else:

        df_sidebar = st.session_state.df

        node_labels = df_sidebar["node"].tolist()


        # ====================================================
        # HOME / START NODE
        # HANYA UNTUK CONSTRUCTIVE
        # ====================================================

        if needs_home:

            # Tidak bold agar tidak membingungkan
            # dengan bagian Initial Solution.

            st.markdown(
                "Node awal / Home"
            )

            selected_home = st.selectbox(
                "Pilih node awal:",
                node_labels,
                key="home_node"
            )

            home_index = int(
                df_sidebar.index[
                    df_sidebar["node"] == selected_home
                ][0]
            )


            st.caption(
                "Constructive method membentuk rute "
                "langsung dari node awal yang dipilih."
            )


        # ====================================================
        # INITIAL ROUTE
        # UNTUK LOCAL SEARCH / METAHEURISTIC
        # ====================================================

        if needs_initial_route:

            st.markdown("Starting Route")

            st.caption(
                "Metode Local Search dan Metaheuristic "
                "membutuhkan satu rute lengkap sebagai solusi awal."
            )


            route_mode = st.radio(
                "Sumber initial route:",
                [
                    "Generate Random",
                    "Pilih manual"
                ],
                horizontal=True,
                key="route_mode"
            )


            if route_mode == "Generate Random":

                seed = st.number_input(
                    "Seed",
                    min_value=0,
                    max_value=99999,
                    value=42,
                    step=1,
                    key="initial_seed"
                )

                # Untuk metode yang tidak menggunakan
                # start node, home internal dibuat dari
                # node pertama dataset.
                route_home = (
                    home_index
                    if needs_home
                    else 0
                )

                initial_route = generate_initial_tour(
                    len(df_sidebar),
                    home=route_home,
                    seed=int(seed)
                )

                st.caption(
                    "Seed digunakan untuk menghasilkan "
                    "urutan initial route secara konsisten."
                )

                st.caption(
                    f"Initial route: "
                    f"{route_to_labels(df_sidebar, initial_route)}"
                )


            else:

                # Jika tidak ada constructive method,
                # home tidak ditampilkan ke user.
                route_home = (
                    home_index
                    if needs_home
                    else 0
                )

                remaining_nodes = [
                    node
                    for node in node_labels
                    if node != node_labels[route_home]
                ]

                selected_order = st.multiselect(
                    "Urutan node:",
                    remaining_nodes,
                    placeholder="Pilih urutan node",
                    key="manual_route"
                )

                if len(selected_order) == len(remaining_nodes):

                    initial_route = (
                        get_initial_route_for_algorithm(
                            df_sidebar,
                            route_home,
                            "Pilih manual",
                            42,
                            selected_order
                        )
                    )

                    st.caption(
                        f"Initial route: "
                        f"{route_to_labels(df_sidebar, initial_route)}"
                    )

                else:

                    initial_route = None

                    st.warning(
                        "Pilih semua node tersisa "
                        "untuk membentuk initial route lengkap."
                    )


    st.divider()


    # ========================================================
    # ALGORITHM PARAMETERS
    # ========================================================

    st.subheader("Algorithm Parameters")

    algorithms_parameters = {}


    if not selected_methods:

        st.caption(
            "Parameter algoritma akan muncul setelah "
            "algoritma dipilih."
        )

    else:

        for method_name in selected_methods:

            info = ALGORITHMS[method_name]

            with st.expander(
                f"⚙️ {method_name}",
                expanded=False
            ):

                st.caption(
                    info["description"]
                )

                algorithms_parameters[
                    method_name
                ] = render_parameter_widget(
                    method_name
                )


# ============================================================
# DATASET CONFIGURATION
# ============================================================

st.header("📂 Dataset Configuration")


input_type = st.radio(
    "Input dataset:",
    [
        "Generate",
        "Upload",
        "Manual"
    ],
    horizontal=True
)


new_data = None


# ============================================================
# GENERATE DATASET
# ============================================================

if input_type == "Generate":

    st.subheader("Generate Dataset")

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
                    rng.uniform(
                        0,
                        max_x
                    ),
                    2
                )
                for _ in range(int(n_nodes))
            ],

            "y": [
                round(
                    rng.uniform(
                        0,
                        max_y
                    ),
                    2
                )
                for _ in range(int(n_nodes))
            ]
        })


# ============================================================
# UPLOAD DATASET
# ============================================================

elif input_type == "Upload":

    st.subheader("Upload Dataset")

    uploaded_file = st.file_uploader(
        "Upload CSV atau Excel",
        type=[
            "csv",
            "xlsx",
            "xls"
        ]
    )


    if uploaded_file is not None:

        if st.button(
            "Gunakan Dataset",
            type="primary"
        ):

            new_data = uploaded_file


# ============================================================
# MANUAL DATASET
# ============================================================

elif input_type == "Manual":

    st.subheader("Input Manual")

    edited_data = st.data_editor(
        st.session_state.manual_data,
        num_rows="dynamic",
        use_container_width=True
    )


    if st.button(
        "Gunakan Dataset",
        type="primary"
    ):

        new_data = edited_data


# ============================================================
# PROCESS DATASET
# ============================================================

if new_data is not None:

    try:

        processed_data = load_data(
            new_data
        )

        st.session_state.df = (
            processed_data
        )

        st.session_state.results = []

        st.success(
            "Dataset berhasil digunakan."
        )

        st.rerun()

    except Exception as e:

        st.error(
            f"Gagal membaca dataset: {e}"
        )


# ============================================================
# ACTIVE DATASET
# ============================================================

if st.session_state.df is not None:

    df = st.session_state.df

    coords = list(
        zip(
            df["x"],
            df["y"]
        )
    )


    # ========================================================
    # DISTANCE MATRIX
    # ========================================================

    dist_matrix = build_distance_matrix(
        coords,
        metric=metric_key
    )


    st.divider()


    # ========================================================
    # DATASET DISPLAY
    # ========================================================

    st.subheader("Dataset")

    left, right = st.columns(
        [1, 1]
    )


    with left:

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            f"Jumlah node: {len(df)}"
        )


    with right:

        # Jika home tersedia, gunakan home.
        # Jika tidak, gunakan node pertama
        # hanya untuk visualisasi dataset.

        display_home = (
            home_index
            if needs_home
            else 0
        )

        fig_nodes = plot_nodes(
            df,
            home=display_home,
            title="Sebaran Node"
        )

        st.plotly_chart(
            fig_nodes,
            use_container_width=True
        )


    # ========================================================
    # DISTANCE MATRIX
    # ========================================================

    st.subheader(
        f"Distance Matrix — {metric}"
    )

    distance_df = pd.DataFrame(
        dist_matrix,
        index=df["node"],
        columns=df["node"]
    )

    st.dataframe(
        distance_df.round(2),
        use_container_width=True
    )


    # ========================================================
    # RUN ALGORITHM
    # ========================================================

    st.divider()

    st.header("🚀 Results")


    if not selected_methods:

        st.info(
            "Pilih minimal satu algoritma "
            "pada sidebar."
        )

    else:

        if st.button(
            "▶️ Jalankan Algoritma",
            type="primary",
            use_container_width=True
        ):

            st.session_state.results = []


            # ------------------------------------------------
            # LOOP ALGORITHM
            # ------------------------------------------------

            for method in selected_methods:

                if not algorithm_available(method):

                    st.warning(
                        f"{method} belum tersedia."
                    )

                    continue


                info = ALGORITHMS[method]


                # ------------------------------------------------
                # INITIAL ROUTE
                # ------------------------------------------------

                if info["needs_initial_route"]:

                    if initial_route is None:

                        st.error(
                            f"{method} membutuhkan "
                            "initial route lengkap."
                        )

                        continue

                    route_input = initial_route

                    initial_distance = (
                        tour_distance(
                            route_input,
                            dist_matrix
                        )
                    )

                else:

                    route_input = None

                    initial_distance = None


                # ------------------------------------------------
                # HOME
                # ------------------------------------------------

                if info["needs_home"]:

                    algorithm_home = (
                        home_index
                    )

                else:

                    # Tidak ditampilkan sebagai input.
                    # Hanya nilai default bila signature
                    # algoritma ternyata memerlukannya.

                    algorithm_home = 0


                # ------------------------------------------------
                # PARAMETERS
                # ------------------------------------------------

                params = algorithms_parameters.get(
                    method,
                    {}
                )


                # ------------------------------------------------
                # RUN
                # ------------------------------------------------

                try:

                    start_time = (
                        time.perf_counter()
                    )


                    result = run_algorithm(
                        method_name=method,
                        dist_matrix=dist_matrix,
                        initial_route=route_input,
                        home=algorithm_home,
                        parameters=params
                    )


                    elapsed = (
                        time.perf_counter()
                        - start_time
                    )


                    # ------------------------------------------------
                    # EXTRACT RESULT
                    # ------------------------------------------------

                    final_route = extract_route(
                        result
                    )


                    final_route = normalize_route(
                        final_route,
                        len(df)
                    )


                    final_distance = (
                        extract_distance(
                            result,
                            final_route,
                            dist_matrix
                        )
                    )


                    # ------------------------------------------------
                    # INITIAL DISTANCE
                    # ------------------------------------------------

                    if initial_distance is None:

                        initial_distance = (
                            final_distance
                        )


                    # ------------------------------------------------
                    # HISTORY
                    # ------------------------------------------------

                    history = extract_history(
                        result
                    )


                    # ------------------------------------------------
                    # VALIDATE
                    # ------------------------------------------------

                    if final_route is not None:

                        try:

                            validate_tour(
                                final_route,
                                len(df),
                                home=final_route[0]
                            )

                        except Exception:

                            pass


                    # ------------------------------------------------
                    # SAVE RESULT
                    # ------------------------------------------------

                    st.session_state.results.append({

                        "method": method,

                        "route": final_route,

                        "initial_distance":
                            initial_distance,

                        "final_distance":
                            final_distance,

                        "execution_time":
                            elapsed,

                        "history":
                            history
                    })


                except Exception as e:

                    st.error(
                        f"{method} gagal dijalankan: {e}"
                    )


    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    if st.session_state.results:

        st.subheader(
            "Hasil Algoritma"
        )


        for i, result in enumerate(
            st.session_state.results
        ):

            method = result["method"]

            route = result["route"]

            initial_distance = (
                result["initial_distance"]
            )

            final_distance = (
                result["final_distance"]
            )

            elapsed = (
                result["execution_time"]
            )


            st.markdown(
                f"### {method}"
            )


            # ------------------------------------------------
            # METRICS
            # ------------------------------------------------

            c1, c2, c3 = st.columns(3)


            c1.metric(
                "Initial Distance",
                f"{initial_distance:.2f}"
            )


            c2.metric(
                "Final Distance",
                f"{final_distance:.2f}"
            )


            c3.metric(
                "Execution Time",
                f"{elapsed * 1000:.3f} ms"
            )


            # ------------------------------------------------
            # ROUTE
            # ------------------------------------------------

            st.write(
                "**Final Route:**",
                route_to_labels(
                    df,
                    route
                )
            )


            # ------------------------------------------------
            # ROUTE VISUALIZATION
            # ------------------------------------------------

            if route is not None:

                route_home = route[0]

                fig_route = plot_route(
                    df=df,
                    tour=route,
                    home=route_home,
                    title=f"Final Route — {method}"
                )

                st.plotly_chart(
                    fig_route,
                    use_container_width=True,
                    key=f"route_{i}"
                )


            # ------------------------------------------------
            # INITIAL VS FINAL
            # ------------------------------------------------

            if (
                method in INITIAL_ROUTE_METHODS
                and initial_route is not None
                and route is not None
            ):

                with st.expander(
                    "Lihat Initial Route vs Final Route"
                ):

                    initial_fig, final_fig = (
                        plot_route_comparison(
                            df=df,
                            initial_tour=initial_route,
                            final_tour=route,
                            home=initial_route[0],
                            initial_title="Initial Route",
                            final_title="Final Route"
                        )
                    )


                    c1, c2 = st.columns(2)


                    with c1:

                        st.plotly_chart(
                            initial_fig,
                            use_container_width=True,
                            key=f"initial_{i}"
                        )


                    with c2:

                        st.plotly_chart(
                            final_fig,
                            use_container_width=True,
                            key=f"final_{i}"
                        )


            st.divider()


        # ====================================================
        # COMPARISON
        # ====================================================

        if len(
            st.session_state.results
        ) >= 2:

            st.subheader(
                "⚖️ Method Comparison"
            )


            comparison_df = (
                create_comparison_table(
                    st.session_state.results
                )
            )


            comparison_df = (
                sort_by_distance(
                    comparison_df
                )
            )


            # ------------------------------------------------
            # TABLE
            # ------------------------------------------------

            st.dataframe(
                comparison_df,
                use_container_width=True,
                hide_index=True
            )


            # ------------------------------------------------
            # BEST METHOD
            # ------------------------------------------------

            best_method = (
                get_best_distance(
                    comparison_df
                )
            )


            fastest_method = (
                get_fastest_method(
                    comparison_df
                )
            )


            if best_method is not None:

                c1, c2 = st.columns(2)


                with c1:

                    st.metric(
                        "Best Distance",
                        str(
                            best_method["Method"]
                        ),
                        f"{best_method['Final Distance']:.2f}"
                    )


                with c2:

                    st.metric(
                        "Fastest Method",
                        str(
                            fastest_method["Method"]
                        ),
                        f"{fastest_method['Execution Time (ms)']:.3f} ms"
                    )


            # ------------------------------------------------
            # COMPARISON CHARTS
            # ------------------------------------------------

            st.markdown(
                "#### Perbandingan Performa"
            )


            c1, c2 = st.columns(2)


            with c1:

                fig_distance = (
                    plot_distance_comparison(
                        comparison_df
                    )
                )

                st.plotly_chart(
                    fig_distance,
                    use_container_width=True
                )


            with c2:

                fig_time = (
                    plot_time_comparison(
                        comparison_df
                    )
                )

                st.plotly_chart(
                    fig_time,
                    use_container_width=True
                )


            # ------------------------------------------------
            # IMPROVEMENT
            # ------------------------------------------------

            st.markdown(
                "#### Improvement dari Initial Solution"
            )


            fig_improvement = (
                plot_improvement_comparison(
                    comparison_df
                )
            )


            st.plotly_chart(
                fig_improvement,
                use_container_width=True
            )
