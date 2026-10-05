import streamlit as st
import pandas as pd
import random
import time
import importlib
import inspect
import plotly.graph_objects as go

from algorithms.base import (
    build_distance_matrix,
    tour_distance,
    generate_initial_tour,
    validate_tour
)

from utils.load_data import load_data


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
    # ---------------- Constructive / Greedy ----------------
    "Nearest Neighbor": {
        "key": "nearest_neighbor",
        "category": "Constructive / Greedy",
        "module": "algorithms.constructive"
    },

    "Nearest Insertion": {
        "key": "nearest_insertion",
        "category": "Constructive / Greedy",
        "module": "algorithms.constructive"
    },

    "Farthest Insertion": {
        "key": "farthest_insertion",
        "category": "Constructive / Greedy",
        "module": "algorithms.constructive"
    },

    "Arbitrary Insertion": {
        "key": "arbitrary_insertion",
        "category": "Constructive / Greedy",
        "module": "algorithms.constructive"
    },

    # ---------------- Local Search ----------------
    "2-opt": {
        "key": "two_opt",
        "category": "Local Search",
        "module": "algorithms.local_search"
    },

    "3-opt": {
        "key": "three_opt",
        "category": "Local Search",
        "module": "algorithms.local_search"
    },

    # ---------------- Metaheuristic ----------------
    "Simulated Annealing": {
        "key": "simulated_annealing",
        "category": "Metaheuristic",
        "module": "algorithms.simulated_annealing"
    },

    "Tabu Search": {
        "key": "tabu_search",
        "category": "Metaheuristic",
        "module": "algorithms.tabu_search"
    }
}


# ============================================================
# CONSTRUCTIVE & FULL-ROUTE METHODS
# ============================================================

CONSTRUCTIVE_METHODS = [
    "Nearest Neighbor",
    "Nearest Insertion",
    "Farthest Insertion",
    "Arbitrary Insertion"
]

FULL_ROUTE_METHODS = [
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
    Mencari fungsi algoritma dari module masing-masing.
    """
    info = ALGORITHMS.get(method_name)
    if not info:
        return None

    module_name = info["module"]
    function_name = info["key"]

    try:
        module = importlib.import_module(module_name)
    except ImportError:
        return None

    return getattr(module, function_name, None)


def algorithm_available(method_name):
    """Cek keberadaan modul/fungsi algoritma."""
    return get_algorithm_function(method_name) is not None


def get_algorithm_requirements(method_name):
    """
    Memeriksa parameter fungsi untuk menentukan apakah memerlukan
    home node, initial tour, atau kustom parameter lainnya.
    """
    func = get_algorithm_function(method_name)
    reqs = {"home": False, "tour": False, "has_params": False}

    if func is None:
        # Jika belum diimplementasikan, gunakan logika kategori sederhana
        if method_name in CONSTRUCTIVE_METHODS:
            reqs["home"] = True
        else:
            reqs["tour"] = True
            reqs["home"] = True
        return reqs

    sig = inspect.signature(func)
    auto_params = {
        "dist_matrix", "distance_matrix", "dist", "matrix", 
        "coords", "coordinates", "points", "tour", "initial_tour", 
        "route", "solution", "home", "start", "start_node", "n", "num_nodes"
    }

    for name in sig.parameters:
        lname = name.lower()
        if lname in {"home", "start", "start_node"}:
            reqs["home"] = True
        elif lname in {"tour", "initial_tour", "route", "solution"}:
            reqs["tour"] = True
        elif lname not in auto_params:
            reqs["has_params"] = True

    return reqs


def render_algorithm_parameters(method_name):
    """Menampilkan widget parameter dinamis berdasarkan signature fungsi algoritma."""
    func = get_algorithm_function(method_name)
    if func is None:
        st.caption(f"⏳ Parameters for {method_name} are not available yet.")
        return {}

    sig = inspect.signature(func)
    params = {}

    auto_params = {
        "dist_matrix", "distance_matrix", "dist", "matrix", 
        "coords", "coordinates", "points", "tour", "initial_tour", 
        "route", "solution", "home", "start", "start_node", "n", "num_nodes"
    }

    for name, parameter in sig.parameters.items():
        if name.lower() in auto_params:
            continue

        label = name.replace("_", " ").title()

        if parameter.default is inspect.Parameter.empty:
            params[name] = st.number_input(f"{label} ({method_name})", value=10)
        elif isinstance(parameter.default, bool):
            params[name] = st.checkbox(f"{label} ({method_name})", value=parameter.default)
        elif isinstance(parameter.default, int):
            params[name] = st.number_input(f"{label} ({method_name})", value=parameter.default, step=1)
        elif isinstance(parameter.default, float):
            params[name] = st.number_input(f"{label} ({method_name})", value=float(parameter.default))
        elif isinstance(parameter.default, str):
            params[name] = st.text_input(f"{label} ({method_name})", value=parameter.default)

    return params


def run_algorithm(method_name, dist_matrix, coords, initial_route, home, parameters):
    function = get_algorithm_function(method_name)
    if function is None:
        raise ValueError(f"Algoritma {method_name} belum tersedia.")

    signature = inspect.signature(function)
    kwargs = {}

    for name in signature.parameters:
        lname = name.lower()
        if lname in {"dist_matrix", "distance_matrix", "dist", "matrix"}:
            kwargs[name] = dist_matrix
        elif lname in {"coords", "coordinates", "points"}:
            kwargs[name] = coords
        elif lname in {"tour", "initial_tour", "route", "solution"}:
            kwargs[name] = initial_route
        elif lname in {"home", "start", "start_node"}:
            kwargs[name] = home
        elif name in parameters:
            kwargs[name] = parameters[name]

    return function(**kwargs)


def route_figure(df, tour, title="Route"):
    if tour is None:
        return None

    x = [df.iloc[i]["x"] for i in tour]
    y = [df.iloc[i]["y"] for i in tour]
    labels = [str(df.iloc[i]["node"]) for i in tour]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x, y=y,
            mode="lines+markers+text",
            text=labels,
            textposition="top center",
            line=dict(width=2),
            marker=dict(size=10)
        )
    )
    fig.update_layout(
        title=title,
        height=450,
        showlegend=False,
        xaxis_title="X",
        yaxis_title="Y",
        yaxis=dict(scaleanchor="x")
    )
    return fig


# ============================================================
# PAGE HEADER
# ============================================================

st.title("🧭 TSP Learning & Optimization")
st.caption("Pelajari dan bandingkan metode Constructive, Local Search, dan Metaheuristic untuk Travelling Salesman Problem.")


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("⚙️ Pengaturan")
    st.divider()

    # 1. DISTANCE
    st.subheader("Distance Method")
    metric = st.radio(
        "Select distance method:",
        ["Euclidean", "Manhattan"],
        label_visibility="collapsed"
    )
    metric_key = metric.lower()

    st.divider()

    # 2. ALGORITHM SELECTION
    st.subheader("Algorithm Options")
    selected_methods = st.multiselect(
        "Select algorithm(s):",
        list(ALGORITHMS.keys()),
        label_visibility="collapsed",
        help="You can select multiple algorithms to compare."
    )

    if not selected_methods:
        st.caption("Select at least one algorithm to proceed.")

    st.divider()

    # CEK KEBUTUHAN INPUT
    requirements = [get_algorithm_requirements(m) for m in selected_methods]
    needs_home = any(req["home"] for req in requirements)
    needs_tour = any(req["tour"] for req in requirements)

    # 3. INITIAL SOLUTION
    st.subheader("Initial Solution")
    home_index = 0
    initial_route = None

    if st.session_state.df is None:
        st.info("Masukkan dataset terlebih dahulu untuk mengatur initial solution.")
    else:
        df_sidebar = st.session_state.df
        node_labels = df_sidebar["node"].tolist()

        if needs_home:
            st.markdown("**Node awal / Home**")
            selected_home = st.selectbox("Pilih node awal:", node_labels)
            home_index = int(df_sidebar.index[df_sidebar["node"] == selected_home][0])

        if needs_tour:
            st.markdown("**Starting Route**")
            st.caption("Metode Local Search / Metaheuristic membutuhkan satu rute lengkap sebagai solusi awal.")

            route_mode = st.radio(
                "Sumber initial route:",
                ["Generate Random", "Pilih manual"],
                horizontal=True
            )

            if route_mode == "Generate Random":
                seed = st.number_input("Seed", min_value=0, max_value=99999, value=42, step=1)
                initial_route = generate_initial_tour(
                    len(df_sidebar),
                    home_index,
                    seed=int(seed)
                )
            else:
                remaining_nodes = [node for node in node_labels if node != node_labels[home_index]]
                selected_order = st.multiselect("Urutan node:", remaining_nodes, placeholder="Pilih urutan node")

                if len(selected_order) == len(remaining_nodes):
                    initial_route = [home_index]
                    for node in selected_order:
                        idx = df_sidebar.index[df_sidebar["node"] == node][0]
                        initial_route.append(idx)
                    initial_route.append(home_index)

                    st.caption(
                        f"**Rute Terbentuk:** {node_labels[home_index]} → "
                        f"{' → '.join(map(str, selected_order))} → "
                        f"{node_labels[home_index]}"
                    )
                else:
                    initial_route = None
                    st.warning("Pilih semua node tersisa untuk membentuk initial route yang lengkap.")

        elif needs_home and not needs_tour:
            st.caption("Constructive method akan membentuk rute langsung dari node awal yang dipilih.")

    st.divider()

    # 4. PARAMETERS CONFIGURATION
    st.subheader("Algorithm Parameters")
    algorithms_parameters = {}

    if not selected_methods:
        st.caption("Select an algorithm above to display its parameters.")
    else:
        for method_name in selected_methods:
            req = get_algorithm_requirements(method_name)
            if req["has_params"]:
                with st.expander(f"⚙️ {method_name} Parameters", expanded=False):
                    algorithms_parameters[method_name] = render_algorithm_parameters(method_name)
            else:
                st.caption(f"ℹ️ {method_name} uses default settings (no parameters required).")

    st.divider()


# ============================================================
# MAIN PAGE - DATASET CONFIGURATION
# ============================================================

st.header("📂 Dataset Configuration")

input_type = st.radio(
    "Input dataset:",
    ["Generate", "Upload", "Manual"],
    horizontal=True
)

new_data = None

if input_type == "Generate":
    st.subheader("Generate Dataset")
    c1, c2, c3, c4 = st.columns(4)
    n_nodes = c1.number_input("Jumlah node", min_value=3, max_value=500, value=10)
    seed = c2.number_input("Seed", min_value=0, value=42)
    max_x = c3.number_input("Maksimum X", min_value=1, value=100)
    max_y = c4.number_input("Maksimum Y", min_value=1, value=100)

    if st.button("Generate Dataset", type="primary"):
        rng = random.Random(int(seed))
        new_data = pd.DataFrame({
            "x": [round(rng.uniform(0, max_x), 2) for _ in range(int(n_nodes))],
            "y": [round(rng.uniform(0, max_y), 2) for _ in range(int(n_nodes))]
        })

elif input_type == "Upload":
    st.subheader("Upload Dataset")
    uploaded_file = st.file_uploader("Upload CSV atau Excel", type=["csv", "xlsx", "xls"])
    if uploaded_file is not None and st.button("Gunakan Dataset", type="primary"):
        new_data = uploaded_file

elif input_type == "Manual":
    st.subheader("Input Manual")
    edited_data = st.data_editor(st.session_state.manual_data, num_rows="dynamic", use_container_width=True)
    if st.button("Gunakan Dataset", type="primary"):
        new_data = edited_data

# Process Dataset
if new_data is not None:
    try:
        processed_data = load_data(new_data)
        st.session_state.df = processed_data
        st.session_state.results = []
        st.success("Dataset berhasil digunakan.")
        st.rerun()
    except Exception as e:
        st.error(f"Gagal membaca dataset: {e}")


# ============================================================
# ACTIVE DATASET VISUALIZATION & RUN ALGORITHM
# ============================================================

if st.session_state.df is not None:
    df = st.session_state.df
    coords = list(zip(df["x"], df["y"]))

    dist_matrix = build_distance_matrix(coords, metric=metric_key)

    st.divider()
    st.subheader("Dataset")

    left, right = st.columns([1, 1])

    with left:
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.caption(f"Jumlah node: {len(df)}")

    with right:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df["x"], y=df["y"],
                mode="markers+text",
                text=df["node"],
                textposition="top center",
                marker=dict(size=12)
            )
        )
        fig.update_layout(
            title="Sebaran Node",
            height=400,
            showlegend=False,
            xaxis_title="X",
            yaxis_title="Y",
            yaxis=dict(scaleanchor="x")
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader(f"Distance Matrix — {metric}")
    distance_df = pd.DataFrame(dist_matrix, index=df["node"], columns=df["node"])
    st.dataframe(distance_df.round(2), use_container_width=True)

    # RUN ALGORITHM
    st.divider()
    st.header("🚀 Results")

    if not selected_methods:
        st.info("Pilih minimal satu algoritma pada sidebar.")
    else:
        if st.button("▶️ Jalankan Algoritma", type="primary", use_container_width=True):
            st.session_state.results = []  # Reset hasil eksekusi sebelumnya

            for method in selected_methods:
                if not algorithm_available(method):
                    st.warning(f"⏳ {method} belum tersedia. Algoritma masih dalam proses pengerjaan.")
                    continue

                if method in CONSTRUCTIVE_METHODS:
                    route_input = None
                else:
                    if initial_route is None:
                        st.error(f"{method} membutuhkan initial route lengkap.")
                        continue
                    route_input = initial_route

                # Parameter pengguna
                params = algorithms_parameters.get(method, {})

                try:
                    start_time = time.perf_counter()
                    result = run_algorithm(
                        method_name=method,
                        dist_matrix=dist_matrix,
                        coords=coords,
                        initial_route=route_input,
                        home=home_index,
                        parameters=params
                    )
                    elapsed = time.perf_counter() - start_time

                    # Parse hasil
                    if isinstance(result, dict):
                        final_route = result.get("tour", result.get("best_tour", result.get("final_tour")))
                    else:
                        final_route = result

                    # Closed Tour Normalization
                    if final_route is not None and len(final_route) == len(df):
                        final_route = list(final_route)
                        final_route.append(final_route[0])

                    final_distance = tour_distance(final_route, dist_matrix)

                    st.session_state.results.append({
                        "method": method,
                        "route": final_route,
                        "distance": final_distance,
                        "time": elapsed
                    })

                except Exception as e:
                    st.error(f"{method} gagal dijalankan: {e}")

    # DISPLAY RESULTS
    if st.session_state.results:
        st.subheader("Hasil Algoritma")

        for i, result in enumerate(st.session_state.results):
            method = result["method"]
            route = result["route"]
            distance = result["distance"]
            elapsed = result["time"]

            st.markdown(f"### {method}")
            c1, c2, c3 = st.columns(3)
            c1.metric("Final Distance", f"{distance:.2f}")
            c2.metric("Execution Time", f"{elapsed * 1000:.2f} ms")
            c3.metric("Status", "Selesai")

            st.write(
                "**Final Route:**",
                " → ".join(str(df.iloc[node]["node"]) for node in route)
            )

            fig = route_figure(df, route, title=f"Final Route — {method}")
            st.plotly_chart(fig, use_container_width=True, key=f"result_{i}")
            st.divider()

        # COMPARISON TABLE & CHARTS
        if len(st.session_state.results) >= 2:
            st.subheader("⚖️ Method Comparison")

            comparison = pd.DataFrame([
                {
                    "Method": r["method"],
                    "Final Distance": round(r["distance"], 2),
                    "Execution Time (ms)": round(r["time"] * 1000, 2)
                }
                for r in st.session_state.results
            ])

            st.dataframe(comparison, use_container_width=True, hide_index=True)

            c1, c2 = st.columns(2)
            with c1:
                fig_distance = go.Figure()
                fig_distance.add_trace(go.Bar(x=comparison["Method"], y=comparison["Final Distance"]))
                fig_distance.update_layout(title="Final Distance", yaxis_title="Distance", xaxis_title="Method")
                st.plotly_chart(fig_distance, use_container_width=True)

            with c2:
                fig_time = go.Figure()
                fig_time.add_trace(go.Bar(x=comparison["Method"], y=comparison["Execution Time (ms)"]))
                fig_time.update_layout(title="Execution Time", yaxis_title="Time (ms)", xaxis_title="Method")
                st.plotly_chart(fig_time, use_container_width=True)
