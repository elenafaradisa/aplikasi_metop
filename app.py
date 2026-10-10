import streamlit as st
import pandas as pd
import math
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

# Spasi sidebar: garis di bawah judul lebih dekat ke judul
st.markdown(
    """
    <style>
    .st-key-sidebar_title { gap: 0.15rem !important; }
    .st-key-sidebar_title hr { margin: 0.15rem 0 0.35rem 0 !important; }
    .st-key-sidebar_title h2 { padding: 0 0 0.15rem 0 !important; }
    [data-testid="stSidebar"] hr { margin: 0.6rem 0 !important; }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# ALGORITHM CONFIGURATION
# ============================================================

ALGORITHMS = {

    # --------------------------------------------------------
    # CONSTRUCTIVE
    # --------------------------------------------------------

    "Nearest Neighbor": {
        "type": "Constructive",
        "needs_start_node": True,
        "needs_initial_route": False,
        "function": nearest_neighbor,
        "description":
            "Membangun rute dengan memilih node terdekat secara bertahap.",
        "parameters": {}
    },

    "Nearest Insertion": {
        "type": "Constructive",
        "needs_start_node": True,
        "needs_initial_route": False,
        "function": nearest_insertion,
        "description":
            "Membangun rute dengan menyisipkan node pada posisi terbaik.",
        "parameters": {}
    },

    "Farthest Insertion": {
        "type": "Constructive",
        "needs_start_node": True,
        "needs_initial_route": False,
        "function": farthest_insertion,
        "description":
            "Membangun rute dengan memprioritaskan node yang paling jauh.",
        "parameters": {}
    },

    "Arbitrary Insertion": {
        "type": "Constructive",
        "needs_start_node": True,
        "needs_initial_route": False,
        "function": arbitrary_insertion,
        "description":
            "Memilih node secara acak lalu menyisipkannya pada posisi terbaik.",
        "parameters": {
            "seed": {
                "type": "int",
                "default": 42,
                "min": 0,
                "max": 99999,
                "step": 1,
                "help":
                    "Seed untuk mengontrol proses acak."
            }
        }
    },


    # --------------------------------------------------------
    # LOCAL SEARCH
    # --------------------------------------------------------

    "2-opt": {
        "type": "Local Search",
        "needs_start_node": False,
        "needs_initial_route": True,
        "function": two_opt,
        "description":
            "Memperbaiki rute dengan membalik segmen rute.",
        "parameters": {
            "max_iter": {
                "type": "int",
                "default": 100,
                "min": 1,
                "max": 10000,
                "step": 1,
                "help":
                    "Batas maksimum iterasi pencarian."
            },
            "strategy": {
                "type": "select",
                "options": ["best", "first"],
                "default": "best",
                "help":
                    "Best memilih perbaikan terbaik, sedangkan First memilih perbaikan pertama."
            }
        }
    },

    "3-opt": {
        "type": "Local Search",
        "needs_start_node": False,
        "needs_initial_route": True,
        "function": three_opt,
        "description":
            "Memperbaiki rute dengan mengevaluasi perubahan 3-opt.",
        "parameters": {
            "max_iter": {
                "type": "int",
                "default": 100,
                "min": 1,
                "max": 10000,
                "step": 1,
                "help":
                    "Batas maksimum iterasi pencarian."
            },
            "strategy": {
                "type": "select",
                "options": ["best", "first"],
                "default": "best",
                "help":
                    "Best memilih perbaikan terbaik, sedangkan First memilih perbaikan pertama."
            }
        }
    },


    # --------------------------------------------------------
    # METAHEURISTIC
    # --------------------------------------------------------

    "Simulated Annealing": {
        "type": "Metaheuristic",
        "needs_start_node": False,
        "needs_initial_route": True,
        "function": simulated_annealing,
        "description":
            "Mencari solusi dengan menerima beberapa solusi yang lebih buruk secara probabilistik.",
        "parameters": {
            "initial_temp": {
                "type": "float",
                "default": 1000.0,
                "min": 0.01,
                "max": 100000.0,
                "step": 10.0,
                "help":
                    "Suhu awal proses pencarian."
            },
            "cooling_rate": {
                "type": "float",
                "default": 0.95,
                "min": 0.01,
                "max": 0.999,
                "step": 0.01,
                "help":
                    "Seberapa cepat suhu diturunkan."
            },
            "min_temp": {
                "type": "float",
                "default": 0.01,
                "min": 0.0001,
                "max": 100.0,
                "step": 0.01,
                "help":
                    "Suhu minimum sebelum pencarian berhenti."
            },
            "max_iter": {
                "type": "int",
                "default": 10,
                "min": 1,
                "max": 10000,
                "step": 1,
                "help":
                    "Batas maksimum iterasi."
            },
            "seed": {
                "type": "int",
                "default": 42,
                "min": 0,
                "max": 99999,
                "step": 1,
                "help":
                    "Seed untuk menjaga proses acak tetap konsisten."
            },
            "verbose_history": {
                "type": "bool",
                "default": False,
                "help":
                    "Menyimpan history proses pencarian."
            }
        }
    },


    "Tabu Search": {
        "type": "Metaheuristic",
        "needs_start_node": False,
        "needs_initial_route": True,
        "function": tabu_search,
        "description":
            "Mencari solusi dengan menggunakan tabu list untuk menghindari perpindahan yang sama.",
        "parameters": {
            "max_iter": {
                "type": "int",
                "default": 100,
                "min": 1,
                "max": 10000,
                "step": 1,
                "help":
                    "Batas maksimum iterasi."
            },
            "tabu_tenure": {
                "type": "int",
                "default": 3,
                "min": 1,
                "max": 100,
                "step": 1,
                "help":
                    "Lama perpindahan disimpan dalam tabu list."
            },
            "verbose_history": {
                "type": "bool",
                "default": False,
                "help":
                    "Menyimpan history proses pencarian."
            }
        }
    }
}


# ============================================================
# SESSION STATE
# ============================================================

if "df" not in st.session_state:
    st.session_state.df = None

if "independent_results" not in st.session_state:
    st.session_state.independent_results = []

if "independent_run_id" not in st.session_state:
    st.session_state.independent_run_id = 0


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

        for value in result:

            if isinstance(value, (list, tuple)):
                return list(value)

    if isinstance(result, (list, tuple)):
        return list(result)

    return None


def extract_history(result):

    if isinstance(result, dict):
        return result.get("history")

    if isinstance(result, tuple):

        for value in result:

            if isinstance(value, list):

                if value and isinstance(value[0], dict):
                    return value

    return None


# ------------------------------------------------------------
# INITIAL SOLUTION RESOLVER
# ------------------------------------------------------------

def resolve_initial(
    method,
    initial_solutions,
    selected_algorithms,
    _seen=None
):
    """Kembalikan (start_node, initial_route) untuk satu algoritma."""

    _seen = set(_seen or [])

    if method in _seen:

        raise ValueError(
            f"Referensi 'Sama dengan' berputar pada {method}."
        )

    _seen.add(method)

    solution = initial_solutions.get(method)

    if not solution:
        return None, None

    if solution["type"] == "start_node":
        return solution["start_node"], None

    if solution["type"] == "route":
        return None, solution["route"]

    if solution["type"] == "same":

        source = solution["source"]

        if source not in selected_algorithms:

            raise ValueError(
                f"{method} meminta initial dari {source}, "
                f"tetapi {source} tidak dipilih."
            )

        return resolve_initial(
            source,
            initial_solutions,
            selected_algorithms,
            _seen
        )

    return None, None


# ------------------------------------------------------------
# DETAIL PERHITUNGAN PER ITERASI
# (diturunkan dari history algoritma + distance matrix;
#  tidak mengubah / memanggil ulang algoritma)
#
#   method_kind(nama)                                -> jenis algoritma
#   build_iteration_table(kind, history, dist, labels) -> tabel semua iterasi
#   build_iteration_view(kind, history, k, dist, labels) -> detail iterasi ke-k
# ------------------------------------------------------------

KIND_BY_METHOD = {
    "Nearest Neighbor": "nn",
    "Nearest Insertion": "ni",
    "Farthest Insertion": "fi",
    "Arbitrary Insertion": "ai",
    "2-opt": "local",
    "3-opt": "local",
    "Simulated Annealing": "sa",
    "Tabu Search": "tabu",
}

INSERTION_RULE = {
    "ni": "node dengan jarak terdekat ke tour saat ini",
    "fi": "node dengan jarak terjauh dari tour saat ini",
    "ai": "node dipilih secara acak",
}

START_RULE = {
    "ni": "node terdekat dari start node",
    "fi": "node terjauh dari start node",
    "ai": "node dipilih secara acak",
}


def method_kind(method):
    return KIND_BY_METHOD.get(method, "generic")


# ============================================================
# HELPER FORMAT
# ============================================================

def _f(value, digits=2):

    if value is None:
        return "-"

    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _signed(value):

    if value is None:
        return "-"

    return f"{float(value):+.2f}"


def _route_of(item):

    if not isinstance(item, dict):
        return None

    for key in ("route", "tour", "route_after", "best_route", "best_tour"):

        value = item.get(key)

        if hasattr(value, "tolist"):
            value = value.tolist()

        if isinstance(value, (list, tuple)) and len(value) >= 1:
            return [int(v) for v in value]

    return None


def route_text(route, labels):

    if route is None:
        return "-"

    return " → ".join(labels[int(i)] for i in route)


def _edge(a, b, labels):

    return f"{labels[a]}–{labels[b]}"


def _edges(edges, labels):

    return ", ".join(_edge(a, b, labels) for a, b in edges)


def two_opt_move(route, i, j):
    """
    Move 2-opt (reverse segmen i..j).
    Return: (edge dibuang, edge ditambah, rute baru)
    """

    removed = [
        (route[i - 1], route[i]),
        (route[j], route[j + 1]),
    ]

    added = [
        (route[i - 1], route[j]),
        (route[i], route[j + 1]),
    ]

    new_route = route[:i] + route[i:j + 1][::-1] + route[j + 1:]

    return removed, added, new_route


def attr_text(attr, labels):
    """Atribut tabu (frozenset berisi edge) -> teks 'A–B, C–D'."""

    try:
        edges = sorted(tuple(sorted(edge)) for edge in attr)
        return ", ".join(_edge(a, b, labels) for a, b in edges)
    except Exception:
        return str(attr)


def _attr_of_move(route, i, j):
    """Atribut tabu = edge yang DITAMBAHKAN move (disimpan ke tabu list)."""

    _, added, _ = two_opt_move(route, i, j)

    return frozenset(frozenset(edge) for edge in added)


# Blok tampilan -------------------------------------------------

def _lines(heading, pairs):
    return {"kind": "lines", "heading": heading, "lines": pairs}


def _table(heading, df, caption=None):
    return {"kind": "table", "heading": heading, "df": df, "caption": caption}


def _text(text):
    return {"kind": "text", "text": text}


def _summary(k, history, labels, distance_note=""):

    item = history[k]

    return [
        ("Iteration", str(item.get("iteration", k))),
        ("Route", route_text(_route_of(item), labels)),
        ("Distance", _f(item.get("distance")) + distance_note),
    ]


def _before_after(k, history):

    before = history[k - 1].get("distance")
    after = history[k].get("distance")

    delta = (
        None
        if before is None or after is None
        else after - before
    )

    return [
        ("Distance sebelum", _f(before)),
        ("Distance sesudah", _f(after)),
        ("Δ Distance", _signed(delta)),
    ]


# ============================================================
# CONSTRUCTIVE (NN, NI, FI, AI)
# ============================================================

def _derive_nn(history, k, dist):

    n = len(dist)
    cur = _route_of(history[k])

    if k == 0:
        return {"type": "initial", "start": cur[0]}

    prev = _route_of(history[k - 1])

    # semua node sudah dikunjungi -> tutup tour
    if len(prev) == n and len(cur) == n + 1 and cur[-1] == cur[0]:

        return {
            "type": "close",
            "current": prev[-1],
            "selected": cur[-1],
            "distance": dist[prev[-1]][cur[-1]],
        }

    current = prev[-1]
    selected = cur[-1]

    candidates = [
        {"node": v, "distance": dist[current][v]}
        for v in range(n)
        if v not in prev
    ]

    return {
        "type": "move",
        "current": current,
        "selected": selected,
        "distance": dist[current][selected],
        "candidates": candidates,
    }


def _derive_insertion(kind, history, k, dist):

    n = len(dist)
    cur = _route_of(history[k])

    if k == 0:

        start, second = cur[0], cur[1]

        candidates = (
            []
            if kind == "ai"
            else [
                {"node": v, "distance": dist[start][v]}
                for v in range(n)
                if v != start
            ]
        )

        return {
            "type": "initial_pair",
            "start": start,
            "selected": second,
            "candidates": candidates,
        }

    prev = _route_of(history[k - 1])
    prev_nodes = set(prev)

    selected = next(v for v in cur[:-1] if v not in prev_nodes)

    position = cur.index(selected)
    after, before = cur[position - 1], cur[position + 1]

    options = []

    for i in range(len(prev) - 1):

        a, b = prev[i], prev[i + 1]

        d_as = dist[a][selected]
        d_sb = dist[selected][b]
        d_ab = dist[a][b]

        options.append({
            "after": a,
            "before": b,
            "d_as": d_as,
            "d_sb": d_sb,
            "d_ab": d_ab,
            "increase": d_as + d_sb - d_ab,
            "chosen": (a == after and b == before),
        })

    candidates = []

    if kind != "ai":

        tour_nodes = prev[:-1]

        for v in range(n):

            if v in prev_nodes:
                continue

            nearest = min(tour_nodes, key=lambda t: (dist[v][t], t))

            candidates.append({
                "node": v,
                "distance": dist[v][nearest],
                "nearest_tour_node": nearest,
            })

    chosen = next(o for o in options if o["chosen"])

    return {
        "type": "insert",
        "selected": selected,
        "after": after,
        "before": before,
        "increase": chosen["increase"],
        "options": options,
        "candidates": candidates,
    }


def _constructive_view(kind, history, k, dist, labels):

    L = labels

    if kind == "nn":
        d = _derive_nn(history, k, dist)
    else:
        d = _derive_insertion(kind, history, k, dist)

    blocks = []

    # ---------------- NN ----------------

    if d["type"] == "initial":

        blocks.append(_text(
            f"Mulai dari start node **{L[d['start']]}**. "
            "Belum ada jarak yang dihitung."
        ))

        return {
            "summary": _summary(k, history, L),
            "blocks": blocks,
        }

    if d["type"] == "close":

        pairs = [
            ("Posisi sekarang", L[d["current"]]),
            ("Kembali ke start node", L[d["selected"]]),
            (
                "Perhitungan",
                f"d({L[d['current']]}, {L[d['selected']]}) = "
                f"{_f(d['distance'])}",
            ),
        ] + _before_after(k, history)

        blocks.append(_lines("Perhitungan / keputusan", pairs))

        return {
            "summary": _summary(k, history, L),
            "blocks": blocks,
        }

    if d["type"] == "move":

        m = len(d["candidates"])

        pairs = [
            ("Posisi sekarang", L[d["current"]]),
            ("Node yang dipilih", L[d["selected"]]),
            (
                "Alasan",
                f"d({L[d['current']]}, {L[d['selected']]}) = "
                f"{_f(d['distance'])} adalah yang terkecil "
                f"dari {m} kandidat (jika sama, pilih node berindeks lebih kecil)",
            ),
        ] + _before_after(k, history)

        blocks.append(_lines("Perhitungan / keputusan", pairs))

        rows = [
            {
                "Node": L[c["node"]],
                f"Jarak dari {L[d['current']]}": _f(c["distance"]),
                "Dipilih": "✅" if c["node"] == d["selected"] else "",
                "_sort": c["distance"],
            }
            for c in d["candidates"]
        ]

        df = (
            pd.DataFrame(rows)
            .sort_values("_sort", kind="stable")
            .drop(columns="_sort")
        )

        blocks.append(_table(
            f"Kandidat: jarak dari {L[d['current']]} ke setiap node "
            "yang belum dikunjungi",
            df,
            "Diurutkan dari jarak terkecil.",
        ))

        return {
            "summary": _summary(
                k, history, L,
                " (sementara, sudah termasuk kembali ke start node)"
            ),
            "blocks": blocks,
        }

    # ---------------- Insertion ----------------

    if d["type"] == "initial_pair":

        pairs = [
            ("Start node", L[d["start"]]),
            ("Node kedua", f"{L[d['selected']]} ({START_RULE[kind]})"),
            (
                "Tour awal",
                f"{L[d['start']]} → {L[d['selected']]} → {L[d['start']]}",
            ),
            (
                "Distance",
                f"2 × d({L[d['start']]}, {L[d['selected']]}) = "
                f"{_f(history[k].get('distance'))}",
            ),
        ]

        blocks.append(_lines("Perhitungan / keputusan", pairs))

        if d["candidates"]:

            rows = [
                {
                    "Node": L[c["node"]],
                    f"Jarak dari {L[d['start']]}": _f(c["distance"]),
                    "Dipilih": "✅" if c["node"] == d["selected"] else "",
                    "_sort": c["distance"],
                }
                for c in d["candidates"]
            ]

            df = (
                pd.DataFrame(rows)
                .sort_values("_sort", kind="stable")
                .drop(columns="_sort")
            )

            blocks.append(_table(
                f"Kandidat node kedua (jarak dari {L[d['start']]})",
                df,
            ))

        return {
            "summary": _summary(k, history, L),
            "blocks": blocks,
        }

    # type == insert

    s = L[d["selected"]]
    a, b = L[d["after"]], L[d["before"]]

    pairs = [
        ("Node yang dipilih", f"{s} ({INSERTION_RULE[kind]})"),
        ("Disisipkan setelah", a),
        ("Disisipkan antara", f"{a} dan {b}"),
        (
            "Tambahan jarak",
            f"d({a},{s}) + d({s},{b}) − d({a},{b}) = "
            f"{_f(dist[d['after']][d['selected']])} + "
            f"{_f(dist[d['selected']][d['before']])} − "
            f"{_f(dist[d['after']][d['before']])} = "
            f"{_f(d['increase'])}",
        ),
    ] + _before_after(k, history)

    blocks.append(_lines("Perhitungan / keputusan", pairs))

    if d["candidates"]:

        rows = [
            {
                "Node": L[c["node"]],
                "Jarak ke tour (terdekat)": _f(c["distance"]),
                "Node tour terdekat": L[c["nearest_tour_node"]],
                "Dipilih": "✅" if c["node"] == d["selected"] else "",
                "_sort": c["distance"],
            }
            for c in d["candidates"]
        ]

        df = pd.DataFrame(rows).sort_values(
            "_sort",
            ascending=(kind == "ni"),
            kind="stable",
        ).drop(columns="_sort")

        blocks.append(_table(
            "Langkah 1 — memilih node yang akan disisipkan",
            df,
            (
                "Nearest Insertion memilih jarak terkecil ke tour."
                if kind == "ni"
                else "Farthest Insertion memilih jarak terbesar ke tour."
            ),
        ))

    elif kind == "ai":

        blocks.append(_text(
            f"**Langkah 1 — memilih node:** node **{s}** dipilih secara "
            "acak dari node yang belum masuk tour (hasilnya tetap sama "
            "selama seed sama)."
        ))

    rows = [
        {
            "Disisipkan antara": f"{L[o['after']]} dan {L[o['before']]}",
            "Perhitungan": (
                f"{_f(o['d_as'])} + {_f(o['d_sb'])} − {_f(o['d_ab'])}"
            ),
            "Tambahan jarak": _f(o["increase"]),
            "Terbaik": "✅" if o["chosen"] else "",
        }
        for o in d["options"]
    ]

    blocks.append(_table(
        f"Langkah 2 — mencari posisi terbaik untuk {s}",
        pd.DataFrame(rows),
        f"Tambahan jarak = d(i,{s}) + d({s},j) − d(i,j); "
        "dipilih yang tambahannya paling kecil.",
    ))

    return {
        "summary": _summary(k, history, L),
        "blocks": blocks,
    }


# ============================================================
# TABU SEARCH
# ============================================================

def _tabu_list_table(tabu_list, labels, k, new_attr=None):

    rows = []

    for attr, expiry in (tabu_list or {}).items():

        rows.append({
            "Edge baru yang dilarang dibongkar": attr_text(attr, labels),
            "Tabu sampai iterasi": int(expiry),
            "Status": (
                "baru masuk" if (new_attr is not None and attr == new_attr)
                else ("aktif" if expiry >= k else "kadaluarsa")
            ),
        })

    return pd.DataFrame(rows)


def _tabu_view(history, k, dist, labels):

    L = labels

    if k == 0:

        return {
            "summary": _summary(k, history, L),
            "blocks": [_text(
                "Iterasi 0 adalah solusi awal. Tabu list masih kosong, "
                f"dan aspiration level awal = {_f(history[0].get('distance'))}."
            )],
        }

    prev, cur = history[k - 1], history[k]

    route_before = _route_of(prev)
    distance_before = prev.get("distance")

    best_before = min(h.get("distance") for h in history[:k])
    best_after = min(h.get("distance") for h in history[:k + 1])

    move = cur.get("move") or {}
    mi, mj = move.get("i"), move.get("j")

    blocks = []

    # --- kondisi awal iterasi
    blocks.append(_lines("Kondisi awal iterasi", [
        ("Current Route", route_text(route_before, L)),
        ("Current Distance", _f(distance_before)),
        (
            "Aspiration level (jarak terbaik sejauh ini)",
            _f(best_before),
        ),
    ]))

    # --- tabu list sebelum
    tl_before = prev.get("tabu_list") or {}

    active = {a: e for a, e in tl_before.items() if e >= k}

    if active:

        blocks.append(_table(
            "Tabu List (aktif pada iterasi ini)",
            _tabu_list_table(active, L, k),
            "Move yang membongkar (membuang) edge ini ditolak, kecuali "
            "memenuhi aspiration criterion (jarak < aspiration level).",
        ))

    else:

        blocks.append(_text("**Tabu List:** kosong (belum ada move yang dilarang)."))

    # --- semua kandidat
    candidates = cur.get("candidates") or []

    if candidates:

        rows = []

        for c in candidates:

            removed, added, new_route = two_opt_move(route_before, c["i"], c["j"])

            rows.append({
                "Move (i, j)": f"({c['i']}, {c['j']})",
                "Edge dibuang": _edges(removed, L),
                "Edge ditambah": _edges(added, L),
                "New Route": route_text(new_route, L),
                "New Distance": _f(c["distance"]),
                "Δ": _signed(c["distance"] - distance_before),
                "Tabu?": "Ya" if c.get("is_tabu") else "Tidak",
                "Aspirasi?": "Ya" if c.get("aspiration_met") else "Tidak",
                "Boleh dipilih?": "Ya" if c.get("admissible") else "Tidak",
                "Dipilih": "✅" if (c["i"] == mi and c["j"] == mj) else "",
                "_sort": c["distance"],
            })

        df = (
            pd.DataFrame(rows)
            .sort_values("_sort", kind="stable")
            .drop(columns="_sort")
        )

        blocks.append(_table(
            f"Candidate Moves ({len(candidates)} kandidat 2-opt dievaluasi)",
            df,
            "Diurutkan dari New Distance terkecil. Move dipilih dari "
            "kandidat 'Boleh dipilih' dengan jarak terkecil.",
        ))

    else:

        blocks.append(_text(
            "_Daftar kandidat tidak tersedia pada run ini "
            "(verbose_history aktif hanya untuk dataset kecil)._"
        ))

    # --- move terpilih
    if mi is not None:

        removed, added, _ = two_opt_move(route_before, mi, mj)

        chosen = next(
            (c for c in candidates if c["i"] == mi and c["j"] == mj),
            None,
        )

        if chosen is None:
            reason = "-"
        elif chosen.get("admissible") and chosen.get("is_tabu"):
            reason = (
                "move ini tabu, tetapi memenuhi aspiration criterion "
                "(lebih baik dari aspiration level), jadi tetap boleh dipilih"
            )
        elif chosen.get("admissible"):
            reason = (
                "kandidat tidak tabu dengan New Distance terkecil "
                "(boleh lebih buruk dari Current Distance)"
            )
        else:
            reason = (
                "semua kandidat tabu dan tidak ada yang memenuhi aspirasi, "
                "sehingga dipilih kandidat dengan jarak terkecil (fallback)"
            )

        blocks.append(_lines("Selected Move", [
            ("Move (i, j)", f"({mi}, {mj})"),
            ("Edge dibuang", _edges(removed, L)),
            ("Edge ditambah", _edges(added, L)),
            ("Alasan", reason),
            ("New Route", route_text(_route_of(cur), L)),
            ("New Distance", _f(cur.get("distance"))),
            ("Δ Distance", _signed(cur.get("distance") - distance_before)),
            (
                "Best so far",
                f"{_f(best_before)} → {_f(best_after)}"
                + ("  (rekor baru!)" if best_after < best_before else ""),
            ),
        ]))

        new_attr = _attr_of_move(route_before, mi, mj)

        blocks.append(_table(
            "Tabu List sesudah iterasi",
            _tabu_list_table(cur.get("tabu_list"), L, k + 1, new_attr),
            "Edge baru yang terbentuk oleh move ini masuk tabu list selama tabu tenure.",
        ))

    return {
        "summary": _summary(k, history, L),
        "blocks": blocks,
    }


# ============================================================
# SIMULATED ANNEALING
# ============================================================

def _sa_view(history, k, dist, labels):

    L = labels

    item = history[k]

    if k == 0:

        return {
            "summary": _summary(k, history, L),
            "blocks": [_text(
                "Iterasi 0 adalah solusi awal "
                f"dengan suhu awal {_f(item.get('temperature'))}."
            )],
        }

    prev = history[k - 1]

    route_before = _route_of(prev)
    distance_before = prev.get("distance")

    move = item.get("move") or {}

    blocks = []

    pairs = [
        ("Suhu (T)", _f(item.get("temperature"), 4)),
        ("Current Route", route_text(route_before, L)),
        ("Current Distance", _f(distance_before)),
    ]

    if move:

        removed, added, new_route = two_opt_move(
            route_before, move["i"], move["j"]
        )

        candidate_distance = distance_before + item["delta"]

        pairs += [
            ("Candidate move (dipilih acak)", f"({move['i']}, {move['j']})"),
            ("Edge dibuang", _edges(removed, L)),
            ("Edge ditambah", _edges(added, L)),
            ("Candidate Route", route_text(new_route, L)),
            ("Candidate Distance", _f(candidate_distance)),
            ("Δ = Candidate − Current", _signed(item["delta"])),
        ]

        if item["delta"] < 0:
            pairs.append((
                "Probabilitas diterima",
                "1.0000 (lebih baik, pasti diterima)",
            ))
        else:
            pairs.append((
                "Probabilitas diterima",
                f"exp(−Δ/T) = exp(−{_f(item['delta'])}/"
                f"{_f(item.get('temperature'), 4)}) = "
                f"{_f(item.get('probability'), 4)}",
            ))

    pairs += [
        (
            "Keputusan",
            "DITERIMA" if item.get("accepted") else "DITOLAK (rute tetap)",
        ),
        ("New Route", route_text(_route_of(item), L)),
        ("New Distance", _f(item.get("distance"))),
        ("Best distance sejauh ini", _f(item.get("best_distance"))),
    ]

    blocks.append(_lines("Perhitungan / keputusan", pairs))

    return {
        "summary": _summary(k, history, L),
        "blocks": blocks,
    }


# ============================================================
# LOCAL SEARCH (2-opt, 3-opt)
# ============================================================

def _local_edge_rows(move, labels):

    rows = []

    for item in move.get("removed_edges", []):

        a, b = item["edge"]

        rows.append({
            "Keterangan": "Dibuang",
            "Edge": _edge(a, b, labels),
            "Jarak": _f(item["distance"]),
        })

    for item in move.get("added_edges", []):

        a, b = item["edge"]

        rows.append({
            "Keterangan": "Ditambah",
            "Edge": _edge(a, b, labels),
            "Jarak": _f(item["distance"]),
        })

    return pd.DataFrame(rows)


def _local_view(history, k, labels, strategy=None):

    L = labels

    item = history[k]

    move = item.get("move")

    # ---------------- iterasi 0: rute awal ----------------

    if k == 0 or not move:

        return {
            "summary": _summary(k, history, L),
            "blocks": [_text(
                "Iterasi 0 adalah **rute awal (initial route)**. "
                "Belum ada move yang diterapkan; pencarian dimulai "
                "dari rute ini."
            )],
        }

    prev = history[k - 1]

    removed = move.get("removed_edges", [])
    added = move.get("added_edges", [])

    positions = move.get("positions", {})

    before = move.get("distance_before", prev.get("distance"))
    after = move.get("distance_after", item.get("distance"))

    blocks = []

    # ---------------- kondisi awal ----------------

    blocks.append(_lines("Kondisi awal iterasi", [
        ("Current Route", route_text(_route_of(prev), L)),
        ("Current Distance", _f(before)),
    ]))

    # ---------------- move terpilih ----------------

    pairs = [
        ("Jenis move", str(move.get("type", "-"))),
        (
            "Posisi pada rute",
            ", ".join(f"{key} = {value}" for key, value in positions.items())
            + " (urutan dimulai dari 0)",
        ),
    ]

    if "reversed_segment" in move:

        segment = list(move["reversed_segment"])

        pairs.append((
            "Segmen yang dibalik",
            f"{route_text(segment, L)}  →  {route_text(segment[::-1], L)}",
        ))

    if "reconnection" in move:

        pairs.append((
            "Penyambungan ulang",
            f"{move['reconnection']} (variant {move.get('variant')})",
        ))

        for name, segment in (move.get("segments") or {}).items():

            pairs.append((f"Segmen {name}", route_text(list(segment), L)))

    blocks.append(_lines("Move yang dipilih", pairs))

    if "reconnection" in move:

        blocks.append(_text(
            "_Keterangan: A, B, C, D adalah potongan rute; "
            "tanda ' berarti potongan itu dibalik urutannya._"
        ))

    # ---------------- edge ----------------

    blocks.append(_table(
        "Edge yang dibuang dan ditambahkan",
        _local_edge_rows(move, L),
    ))

    # ---------------- perhitungan ----------------

    removed_cost = move.get("removed_cost")
    added_cost = move.get("added_cost")
    delta = move.get("delta")

    blocks.append(_lines("Perhitungan", [
        (
            "Total jarak edge dibuang",
            " + ".join(_f(e["distance"]) for e in removed)
            + f" = {_f(removed_cost)}",
        ),
        (
            "Total jarak edge ditambah",
            " + ".join(_f(e["distance"]) for e in added)
            + f" = {_f(added_cost)}",
        ),
        (
            "Δ = ditambah − dibuang",
            f"{_f(added_cost)} − {_f(removed_cost)} = {_signed(delta)}",
        ),
    ]))

    # ---------------- hasil ----------------

    if strategy == "best":

        reason = (
            "Δ negatif (rute lebih pendek). Strategy best: dipilih move "
            "dengan perbaikan terbesar dari semua kemungkinan move."
        )

    elif strategy == "first":

        reason = (
            "Δ negatif (rute lebih pendek). Strategy first: dipilih move "
            "pertama yang ditemukan dan memperbaiki rute."
        )

    else:

        reason = "Δ negatif, sehingga rute menjadi lebih pendek."

    blocks.append(_lines("Hasil iterasi", [
        ("New Route", route_text(_route_of(item), L)),
        ("Distance sebelum", _f(before)),
        ("Distance sesudah", _f(after)),
        ("Δ Distance", _signed(None if before is None or after is None else after - before)),
        ("Alasan diterima", reason),
    ]))

    if k == len(history) - 1:

        blocks.append(_text(
            "Ini iterasi terakhir. Pencarian berhenti karena tidak ada "
            "move yang memperbaiki rute lagi (local optimum) atau karena "
            "batas **max_iter** tercapai."
        ))

    return {
        "summary": _summary(k, history, L),
        "blocks": blocks,
    }


# ============================================================
# GENERIC (2-opt / 3-opt / lainnya)
# ============================================================

def _generic_view(history, k, labels):

    item = history[k]

    pairs = []

    for key, value in item.items():

        if key == "iteration":
            continue

        if key in ("route", "tour"):
            continue

        if isinstance(value, (list, dict)) and not value:
            continue

        if isinstance(value, float):
            value = _f(value)

        pairs.append((key.replace("_", " ").title(), str(value)))

    blocks = [_lines("Informasi iterasi", pairs)] if pairs else []

    return {
        "summary": _summary(k, history, labels),
        "blocks": blocks,
    }


# ============================================================
# PUBLIC: DETAIL SATU ITERASI
# ============================================================

def build_iteration_view(kind, history, k, dist, labels, strategy=None):
    """
    Detail perhitungan iterasi ke-k.

    Return dict:
        {
            "summary": [(label, nilai), ...],
            "blocks":  [ {"kind": "lines" | "table" | "text", ...}, ... ]
        }
    """

    item = history[k]

    if not isinstance(item, dict):

        return {
            "summary": [("Iteration", str(k)), ("Info", str(item))],
            "blocks": [],
        }

    try:

        if kind in ("nn", "ni", "fi", "ai") and _route_of(item) is not None:
            return _constructive_view(kind, history, k, dist, labels)

        if kind == "local":
            return _local_view(history, k, labels, strategy)

        if kind == "tabu":
            return _tabu_view(history, k, dist, labels)

        if kind == "sa":
            return _sa_view(history, k, dist, labels)

    except Exception as error:   # jangan sampai UI gagal total

        return {
            "summary": _summary(k, history, labels)
            if _route_of(item) is not None
            else [("Iteration", str(k))],
            "blocks": [_text(
                f"_Detail perhitungan tidak dapat diturunkan: {error}_"
            )],
        }

    return _generic_view(history, k, labels)


# ============================================================
# PUBLIC: TABEL KUMPULAN ITERASI
# ============================================================

def build_iteration_table(kind, history, dist, labels):
    """Tabel ringkas semua iterasi (satu baris per iterasi)."""

    rows = []

    best = None

    for k, item in enumerate(history):

        if not isinstance(item, dict):
            rows.append({"Iterasi": str(k), "Info": str(item)})
            continue

        route = _route_of(item)

        row = {"Iterasi": str(item.get("iteration", k))}

        if route is not None:
            row["Route"] = route_text(route, labels)

        distance = item.get("distance")

        row["Distance"] = _f(distance)

        if distance is not None:
            best = distance if best is None else min(best, distance)

        try:

            if kind == "nn" and k > 0:

                d = _derive_nn(history, k, dist)

                row["Node dipilih"] = labels[d["selected"]]

            elif kind in ("ni", "fi", "ai"):

                d = _derive_insertion(kind, history, k, dist)

                if d["type"] == "initial_pair":

                    row["Node dipilih"] = labels[d["selected"]]

                else:

                    row["Node dipilih"] = labels[d["selected"]]
                    row["Disisipkan antara"] = (
                        f"{labels[d['after']]} dan {labels[d['before']]}"
                    )
                    row["Tambahan jarak"] = _f(d["increase"])

            elif kind == "local":

                mv = item.get("move")

                if mv:

                    removed = [e["edge"] for e in mv.get("removed_edges", [])]
                    added = [e["edge"] for e in mv.get("added_edges", [])]

                    row["Move"] = (
                        f"{_edges(removed, labels)} → {_edges(added, labels)}"
                    )

                    if mv.get("reconnection"):
                        row["Penyambungan"] = mv["reconnection"]

                    row["Δ"] = _signed(mv.get("delta"))

            elif kind == "tabu":

                row["Best so far"] = _f(best)

                if k > 0:

                    mv = item.get("move") or {}

                    removed, added, _ = two_opt_move(
                        _route_of(history[k - 1]), mv["i"], mv["j"]
                    )

                    row["Move"] = (
                        f"{_edges(removed, labels)} → {_edges(added, labels)}"
                    )

                    row["Δ"] = _signed(
                        distance - history[k - 1].get("distance")
                    )

                    row["Ukuran tabu list"] = str(
                        len(item.get("tabu_list") or {})
                    )

            elif kind == "sa":

                row["Best so far"] = _f(item.get("best_distance"))

                if k > 0:

                    mv = item.get("move") or {}

                    removed, added, _ = two_opt_move(
                        _route_of(history[k - 1]), mv["i"], mv["j"]
                    )

                    row["Move"] = (
                        f"{_edges(removed, labels)} → {_edges(added, labels)}"
                    )

                    row["Δ"] = _signed(item.get("delta"))
                    row["Probabilitas"] = _f(item.get("probability"), 4)
                    row["Diterima"] = "Ya" if item.get("accepted") else "Tidak"
                    row["Suhu"] = _f(item.get("temperature"))

        except Exception:
            pass

        rows.append(row)

    return pd.DataFrame(rows).fillna("-")


# ------------------------------------------------------------
# PENJELASAN SINGKAT ALGORITMA
# ------------------------------------------------------------

ALGORITHM_EXPLANATIONS = {

    "Nearest Neighbor":
        "Nearest Neighbor membangun rute dari start node, lalu selalu "
        "berpindah ke node terdekat yang belum dikunjungi sampai semua "
        "node terkunjungi, kemudian kembali ke start node. Cepat dan "
        "sederhana, tetapi sifatnya greedy sehingga rute di akhir "
        "sering menjadi panjang.",

    "Nearest Insertion":
        "Nearest Insertion memulai tour kecil (start node dan node "
        "terdekatnya). Tiap iterasi, node di luar tour yang paling dekat "
        "dengan tour dipilih, lalu disisipkan pada posisi yang menambah "
        "jarak paling kecil.",

    "Farthest Insertion":
        "Farthest Insertion mirip Nearest Insertion, tetapi node yang "
        "dipilih adalah yang paling jauh dari tour. Dengan begitu "
        "kerangka rute terbentuk lebih awal, lalu node lain disisipkan "
        "pada posisi dengan tambahan jarak minimum.",

    "Arbitrary Insertion":
        "Arbitrary Insertion memilih node yang akan disisipkan secara "
        "acak (dikontrol seed), lalu menyisipkannya pada posisi dengan "
        "tambahan jarak paling kecil. Hasilnya bergantung pada seed.",

    "2-opt":
        "2-opt memperbaiki rute awal dengan menghapus dua edge lalu "
        "menyambungnya kembali (membalik segmen di antaranya) bila total "
        "jarak menjadi lebih pendek. Diulang sampai tidak ada perbaikan "
        "atau iterasi maksimum tercapai.",

    "3-opt":
        "3-opt memperbaiki rute awal dengan menghapus tiga edge dan "
        "menyambungnya kembali dengan kombinasi terbaik. Pencariannya "
        "lebih luas dari 2-opt, tetapi lebih lambat.",

    "Simulated Annealing":
        "Simulated Annealing memilih satu move acak pada tiap iterasi. "
        "Move yang lebih baik selalu diterima, sedangkan yang lebih buruk "
        "diterima dengan peluang exp(−Δ/T). Suhu T diturunkan bertahap "
        "sehingga algoritma makin selektif dan bisa keluar dari optimum "
        "lokal.",

    "Tabu Search":
        "Tabu Search mengevaluasi seluruh kandidat move 2-opt pada tiap "
        "iterasi, lalu memilih yang terbaik di antara yang tidak tabu "
        "(walau lebih buruk dari rute sekarang). Edge baru yang terbentuk "
        "dari move terpilih dimasukkan ke tabu list selama tabu tenure, "
        "sehingga move yang membongkarnya (undo) ditolak agar tidak "
        "berputar balik; move tabu tetap boleh dipilih bila lebih baik "
        "dari rute terbaik (aspiration criterion).",
}


# Detail kandidat (verbose_history) hanya dipaksa aktif untuk dataset
# kecil-menengah agar history tidak membengkak.
DETAIL_NODE_LIMIT = 100


# ------------------------------------------------------------
# HISTORY
# ------------------------------------------------------------

def _is_int(value):

    return (
        hasattr(value, "__index__")
        and not isinstance(value, bool)
    )


def normalize_history(history):
    """
    Samakan bentuk history: list of dict.
    Jika algoritma hanya mengembalikan list angka (mis. 2-opt / 3-opt),
    ubah menjadi [{"iteration": i, "distance": nilai}, ...].
    """

    if not history:
        return history

    if all(
        isinstance(h, (int, float)) and not isinstance(h, bool)
        for h in history
    ):

        return [
            {"iteration": i, "distance": float(h)}
            for i, h in enumerate(history)
        ]

    return history


def find_route_in_step(df, item):
    """Cari rute pada satu iterasi (rute parsial juga diterima)."""

    if not isinstance(item, dict):
        return None

    for key in (
        "route_after",
        "new_route",
        "current_route",
        "current_tour",
        "route",
        "tour",
        "best_route",
        "best_tour"
    ):

        value = item.get(key)

        if hasattr(value, "tolist"):
            value = value.tolist()

        if (
            isinstance(value, (list, tuple))
            and len(value) >= 2
            and all(
                _is_int(v) and 0 <= int(v) < len(df)
                for v in value
            )
        ):
            return [int(v) for v in value]

    return None


def render_view(view):
    """Tampilkan hasil build_iteration_view()."""

    st.markdown(
        "#### Detail Iterasi"
    )

    st.markdown(
        "  \n".join(
            f"**{label}** : {value}"
            for label, value in view["summary"]
        )
    )

    for block in view["blocks"]:

        if block["kind"] == "text":

            st.markdown(block["text"])

        elif block["kind"] == "lines":

            st.markdown(f"##### {block['heading']}")

            st.markdown(
                "  \n".join(
                    f"**{label}** : {value}"
                    for label, value in block["lines"]
                )
            )

        elif block["kind"] == "table":

            st.markdown(f"##### {block['heading']}")

            if block.get("caption"):
                st.caption(block["caption"])

            table = block["df"]

            if table is not None and len(table) > 0:

                st.dataframe(
                    table,
                    use_container_width=True,
                    hide_index=True,
                    height=min(38 * (len(table) + 1) + 3, 420)
                )

            else:

                st.caption("(kosong)")


def highlight_selected_row(table, selected):
    """Beri warna pada baris iterasi yang sedang dipilih."""

    if selected < 0:
        return table

    def _color(row):

        style = (
            "background-color: rgba(108, 92, 231, 0.28)"
            if row.name == selected
            else ""
        )

        return [style] * len(row)

    try:
        return table.style.apply(_color, axis=1)
    except Exception:
        return table


# ------------------------------------------------------------
# HASIL SATU ALGORITMA
# ------------------------------------------------------------

def render_independent_result(
    df,
    dist_matrix,
    result,
    index,
    key_prefix="independent",
    run_id=None
):
    """
    Urutan tampilan:
    1. Nama algoritma + penjelasan singkat
    2. Hasil terbaik / hasil akhir
    3. [Visualisasi]  |  [Tabel iterasi + slider Pilih Iterasi]
    4. Expander: perhitungan detail iterasi yang dipilih
    """

    method = result["method"]
    history = result.get("history") or []
    kind = method_kind(method)
    labels = df["node"].astype(str).tolist()
    if run_id is None:
        run_id = st.session_state.get("independent_run_id", 0)

    is_improvement = ALGORITHMS[method]["type"] in (
        "Local Search",
        "Metaheuristic"
    )

    # -------------------- 1. judul + penjelasan --------------------

    st.subheader(method)

    st.info(
        ALGORITHM_EXPLANATIONS.get(
            method,
            ALGORITHMS[method]["description"]
        )
    )

    # -------------------- 2. hasil terbaik / akhir ------------------

    st.markdown(
        "#### Hasil Terbaik" if is_improvement else "#### Hasil Akhir"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Initial Distance",
            "-"
            if result["initial_distance"] is None
            else f"{result['initial_distance']:.2f}"
        )

    with col2:

        st.metric(
            "Final Distance",
            f"{result['final_distance']:.2f}"
        )

    with col3:

        st.metric(
            "Execution Time",
            f"{result['execution_time'] * 1000:.3f} ms"
        )

    start_label = str(df.iloc[int(result["route"][0])]["node"])

    start_note = result.get("start_note")

    st.write(
        "**Start Node:**",
        start_label + (f" ({start_note})" if start_note else "")
    )

    if result["initial_route"] is not None:

        st.write(
            "**Initial Route:**",
            route_to_labels(df, result["initial_route"])
        )

    st.write(
        "**Route:**",
        route_to_labels(df, result["route"])
    )

    distances = [
        h.get("distance")
        for h in history
        if isinstance(h, dict) and h.get("distance") is not None
    ]

    if is_improvement and len(distances) > 1:

        best_position = distances.index(min(distances))

        st.caption(
            f"Jarak terbaik ditemukan pada iterasi "
            f"{history[best_position].get('iteration', best_position)}. "
            f"Iterasi terakhir: "
            f"{history[-1].get('iteration', len(history) - 1)} "
            f"(distance {distances[-1]:.2f})."
        )

    # -------------------- 3. visualisasi | tabel -------------------

    selected = -1

    col_viz, col_table = st.columns(2)

    with col_table:

        st.markdown("**Tabel Iterasi**")

        if history:

            table_slot = st.container()

            options = [-1] + list(range(len(history)))

            def _option_label(value):

                if value == -1:
                    return "Hasil akhir"

                item = history[value]

                number = (
                    item.get("iteration", value)
                    if isinstance(item, dict)
                    else value
                )

                return f"Iterasi {number}"

            selected = st.select_slider(
                "Pilih Iterasi",
                options=options,
                value=-1,
                format_func=_option_label,
                key=f"{key_prefix}_iter_{run_id}_{index}"
            )

            table = build_iteration_table(
                kind,
                history,
                dist_matrix,
                labels
            )

            with table_slot:

                st.dataframe(
                    highlight_selected_row(table, selected),
                    use_container_width=True,
                    hide_index=True,
                    height=340
                )

        else:

            st.info(
                "Algoritma ini tidak mengembalikan history iterasi."
            )

    with col_viz:

        plot_route_data = result["route"]

        plot_title = f"{method} — Final Route"

        if selected >= 0:

            step_item = history[selected]

            step_route = find_route_in_step(df, step_item)

            step_number = (
                step_item.get("iteration", selected)
                if isinstance(step_item, dict)
                else selected
            )

            if step_route is not None:

                plot_route_data = step_route

                plot_title = f"{method} — Iterasi {step_number}"

            else:

                st.caption(
                    f"Iterasi {step_number} tidak menyimpan rute; "
                    "menampilkan hasil akhir."
                )

        st.plotly_chart(
            plot_route(
                df,
                plot_route_data,
                home=plot_route_data[0],
                title=plot_title
            ),
            use_container_width=True,
            key=f"{key_prefix}_route_{index}"
        )

    # -------------------- 4. detail (expander) ---------------------

    with st.expander(
        "🔍 Lihat langkah per iterasi (perhitungan detail)",
        expanded=False
    ):

        if not history:

            st.info("Tidak ada history untuk ditampilkan.")

        elif selected < 0:

            st.info(
                "Geser slider **Pilih Iterasi** di atas untuk melihat "
                "perhitungan detail iterasi tertentu. Grafik dan tabel "
                "ikut menyesuaikan."
            )

        else:

            render_view(
                build_iteration_view(
                    kind,
                    history,
                    selected,
                    dist_matrix,
                    labels,
                    strategy=(result.get("parameters") or {}).get("strategy")
                )
            )


# ============================================================
# RUN ALGORITHM
# ============================================================

def run_algorithm(
    method,
    dist_matrix,
    start_node=None,
    initial_route=None,
    parameters=None
):

    parameters = parameters or {}

    # --------------------------------------------------------
    # CONSTRUCTIVE
    # --------------------------------------------------------

    if method == "Nearest Neighbor":

        return nearest_neighbor(
            dist_matrix,
            start_node
        )

    elif method == "Nearest Insertion":

        return nearest_insertion(
            dist_matrix,
            start_node
        )

    elif method == "Farthest Insertion":

        return farthest_insertion(
            dist_matrix,
            start_node
        )

    elif method == "Arbitrary Insertion":

        return arbitrary_insertion(
            dist_matrix,
            start_node,
            seed=parameters.get("seed")
        )


    # --------------------------------------------------------
    # LOCAL SEARCH
    # --------------------------------------------------------

    elif method == "2-opt":

        return two_opt(
            initial_route,
            dist_matrix,
            max_iter=parameters.get("max_iter", 100),
            strategy=parameters.get("strategy", "best")
        )

    elif method == "3-opt":

        return three_opt(
            initial_route,
            dist_matrix,
            max_iter=parameters.get("max_iter", 100),
            strategy=parameters.get("strategy", "best")
        )


    # --------------------------------------------------------
    # SIMULATED ANNEALING
    # --------------------------------------------------------

    elif method == "Simulated Annealing":

        return simulated_annealing(
            dist_matrix,
            initial_route=initial_route,
            initial_temp=parameters.get(
                "initial_temp",
                1000.0
            ),
            cooling_rate=parameters.get(
                "cooling_rate",
                0.95
            ),
            min_temp=parameters.get(
                "min_temp",
                0.01
            ),
            max_iter=parameters.get(
                "max_iter",
                10
            ),
            seed=parameters.get(
                "seed",
                42
            )
        )


    # --------------------------------------------------------
    # TABU SEARCH
    # --------------------------------------------------------

    elif method == "Tabu Search":

        return tabu_search(
            initial_route,
            dist_matrix,
            max_iter=parameters.get(
                "max_iter",
                100
            ),
            tabu_tenure=parameters.get(
                "tabu_tenure",
                3
            ),
            verbose_history=parameters.get(
                "verbose_history",
                False
            )
        )

    else:

        raise ValueError(
            f"Algoritma '{method}' tidak ditemukan."
        )


# ============================================================
# RUN ONE INDEPENDENT ALGORITHM
# ============================================================

def execute_independent_algorithm(
    method,
    df,
    dist_matrix,
    start_node=None,
    initial_route=None,
    parameters=None
):

    start_time = time.perf_counter()

    result = run_algorithm(
        method=method,
        dist_matrix=dist_matrix,
        start_node=start_node,
        initial_route=initial_route,
        parameters=parameters
    )

    execution_time = (
        time.perf_counter()
        - start_time
    )

    route = extract_route(result)

    if route is None:

        raise ValueError(
            "Output algoritma tidak berisi rute."
        )

    # Validasi: rute harus tertutup, memuat semua node tepat satu kali,
    # dan dimulai dari node awal yang dipilih.
    expected_home = (
        start_node
        if start_node is not None
        else (
            initial_route[0]
            if initial_route is not None
            else route[0]
        )
    )

    validate_tour(
        list(route),
        len(dist_matrix),
        home=expected_home
    )

    final_distance = tour_distance(
        route,
        dist_matrix
    )

    # Initial distance hanya relevan
    # untuk Local Search / Metaheuristic.
    if initial_route is not None:

        initial_distance = tour_distance(
            initial_route,
            dist_matrix
        )

    else:

        initial_distance = None

    return {
        "method": method,
        "route": route,
        "initial_route": initial_route,
        "initial_distance": initial_distance,
        "final_distance": final_distance,
        "execution_time": execution_time,
        "history": normalize_history(
            extract_history(result)
        ),
        "parameters": dict(parameters or {})
    }


# ============================================================
# PENGATURAN SIDEBAR (TERSIMPAN PER ALGORITMA)
# ============================================================
#
# Sidebar hanya menampilkan pengaturan SATU algoritma (yang dipilih
# pada radio "Pilih algoritma"). Karena widget yang tidak ditampilkan
# dihapus oleh Streamlit, nilai semua pengaturan disimpan manual di
# st.session_state.independent_cfg agar tidak hilang saat berpindah
# algoritma.

RANDOM_NODE = "🎲 Random"

for _namespace in ("independent", "hybrid"):

    if f"{_namespace}_cfg" not in st.session_state:
        st.session_state[f"{_namespace}_cfg"] = {}


def _cfg_store(ns):

    return st.session_state.setdefault(f"{ns}_cfg", {})


def cfg_get(method, name, default=None, ns="independent"):

    return _cfg_store(ns).get(method, {}).get(name, default)


def cfg_set(method, name, value, ns="independent"):

    _cfg_store(ns).setdefault(method, {})[name] = value

    return value


def _cfg_key(method, name, ns="independent"):

    return f"{ns}_cfg_{method}_{name}"


def cfg_selectbox(
    method, name, label, options, default=None, ns="independent", **kwargs
):

    stored = cfg_get(
        method,
        name,
        options[0] if default is None else default,
        ns
    )

    index = options.index(stored) if stored in options else 0

    value = st.selectbox(
        label,
        options,
        index=index,
        key=_cfg_key(method, name, ns),
        **kwargs
    )

    return cfg_set(method, name, value, ns)


def cfg_number(method, name, label, default, ns="independent", **kwargs):

    value = st.number_input(
        label,
        value=cfg_get(method, name, default, ns),
        key=_cfg_key(method, name, ns),
        **kwargs
    )

    return cfg_set(method, name, value, ns)


def cfg_checkbox(method, name, label, default, ns="independent", **kwargs):

    value = st.checkbox(
        label,
        value=cfg_get(method, name, default, ns),
        key=_cfg_key(method, name, ns),
        **kwargs
    )

    return cfg_set(method, name, value, ns)


def cfg_multiselect(method, name, label, options, ns="independent", **kwargs):

    stored = [
        v
        for v in cfg_get(method, name, [], ns)
        if v in options
    ]

    value = st.multiselect(
        label,
        options,
        default=stored,
        key=_cfg_key(method, name, ns),
        **kwargs
    )

    return cfg_set(method, name, value, ns)


def pick_random_node(n_nodes, seed):
    """Pilih index node secara acak (reproducible dengan seed)."""

    return random.Random(int(seed)).randrange(n_nodes)


def previous_route_algorithms(method, selected_algorithms):
    """Algoritma sebelumnya (yang juga memakai initial route)."""

    position = selected_algorithms.index(method)

    return [
        m
        for m in selected_algorithms[:position]
        if ALGORITHMS[m]["needs_initial_route"]
    ]


# ------------------------------------------------------------
# PANEL: INITIAL SOLUTION (satu algoritma)
# ------------------------------------------------------------

def render_initial_panel(method, selected_algorithms, df):

    config = ALGORITHMS[method]

    labels = df["node"].tolist()

    start_options = labels + [RANDOM_NODE]


    # ---------------- CONSTRUCTIVE → START NODE ----------------

    if config["needs_start_node"]:

        choice = cfg_selectbox(
            method,
            "start_choice",
            "Start Node",
            start_options,
            default=labels[0],
            help=(
                "Node awal untuk metode Constructive. "
                "Pilih Random untuk memilih node awal secara acak."
            )
        )

        if choice == RANDOM_NODE:

            seed = cfg_number(
                method,
                "start_seed",
                "Seed (Random)",
                42,
                min_value=0,
                max_value=99999,
                step=1
            )

            st.caption(
                f"Dengan seed {int(seed)}, start node = "
                f"**{labels[pick_random_node(len(df), seed)]}**."
            )

        return


    # ------------- LOCAL SEARCH / METAHEURISTIC → ROUTE -------------

    previous = previous_route_algorithms(method, selected_algorithms)

    route_options = ["Generate Random", "Manual"]

    if previous:
        route_options.append(f"Sama dengan {previous[-1]}")

    route_type = cfg_selectbox(
        method,
        "route_type",
        "Starting Route",
        route_options,
        default="Generate Random",
        help=(
            "Local Search dan Metaheuristic membutuhkan "
            "satu rute sebagai solusi awal."
        )
    )


    # ---------------- SAMA DENGAN ALGORITMA SEBELUMNYA ----------------

    if route_type.startswith("Sama dengan"):

        st.caption(
            f"Memakai rute awal yang sama dengan **{previous[-1]}**."
        )


    # ---------------- GENERATE RANDOM ----------------

    elif route_type == "Generate Random":

        choice = cfg_selectbox(
            method,
            "random_start",
            "Start Node",
            start_options,
            default=labels[0]
        )

        seed = cfg_number(
            method,
            "initial_seed",
            "Seed",
            42,
            min_value=0,
            max_value=99999,
            step=1,
            help=(
                "Seed untuk mengacak rute "
                "(dan memilih start node jika Random)."
            )
        )

        start = (
            pick_random_node(len(df), seed)
            if choice == RANDOM_NODE
            else labels.index(choice)
        )

        preview = generate_initial_tour(
            len(df),
            home=start,
            seed=int(seed)
        )

        st.caption("Initial Route")

        st.code(route_to_labels(df, preview))


    # ---------------- MANUAL ----------------

    else:

        route_start = cfg_selectbox(
            method,
            "manual_start",
            "Route Start",
            labels,
            default=labels[0]
        )

        remaining_nodes = [
            node
            for node in labels
            if node != route_start
        ]

        route_order = cfg_multiselect(
            method,
            "manual_order",
            "Node Order",
            remaining_nodes
        )

        if len(route_order) == len(remaining_nodes):

            st.caption("Final Initial Route")

            st.code(
                " → ".join(
                    [route_start] + route_order + [route_start]
                )
            )

        else:

            st.caption(
                "Pilih seluruh node untuk membentuk rute."
            )


# ------------------------------------------------------------
# PANEL: PARAMETER (satu algoritma)
# ------------------------------------------------------------

def render_parameter_panel(method, ns="independent"):

    for parameter, config in ALGORITHMS[method]["parameters"].items():

        label = parameter.replace("_", " ").title()

        name = f"param_{parameter}"

        if config["type"] == "int":

            cfg_number(
                method,
                name,
                label,
                int(config["default"]),
                ns=ns,
                min_value=int(config["min"]),
                max_value=int(config["max"]),
                step=int(config["step"]),
                help=config["help"]
            )

        elif config["type"] == "float":

            cfg_number(
                method,
                name,
                label,
                float(config["default"]),
                ns=ns,
                min_value=float(config["min"]),
                max_value=float(config["max"]),
                step=float(config["step"]),
                help=config["help"]
            )

        elif config["type"] == "select":

            cfg_selectbox(
                method,
                name,
                label,
                config["options"],
                default=config["default"],
                ns=ns,
                help=config["help"]
            )

        elif config["type"] == "bool":

            cfg_checkbox(
                method,
                name,
                label,
                config["default"],
                ns=ns,
                help=config["help"]
            )


# ------------------------------------------------------------
# BANGUN initial_solutions & parameter untuk SEMUA algoritma terpilih
# (dibaca dari pengaturan tersimpan, bukan dari widget yang tampil)
# ------------------------------------------------------------

def build_initial_solutions(selected_algorithms, df):

    labels = df["node"].tolist()

    n = len(df)

    solutions = {}

    for method in selected_algorithms:

        config = ALGORITHMS[method]


        # ---------------- CONSTRUCTIVE ----------------

        if config["needs_start_node"]:

            choice = cfg_get(method, "start_choice", labels[0])

            note = None

            if choice == RANDOM_NODE:

                seed = int(cfg_get(method, "start_seed", 42))

                start = pick_random_node(n, seed)

                note = f"dipilih acak (seed {seed})"

            elif choice in labels:

                start = labels.index(choice)

            else:

                start = 0

            solutions[method] = {
                "type": "start_node",
                "start_node": start,
                "note": note
            }

            continue


        # ---------------- LOCAL SEARCH / METAHEURISTIC ----------------

        previous = previous_route_algorithms(method, selected_algorithms)

        route_type = cfg_get(method, "route_type", "Generate Random")

        if previous and route_type == f"Sama dengan {previous[-1]}":

            solutions[method] = {
                "type": "same",
                "source": previous[-1]
            }

        elif route_type == "Manual":

            route_start = cfg_get(method, "manual_start", labels[0])

            if route_start not in labels:
                route_start = labels[0]

            remaining = [x for x in labels if x != route_start]

            order = [
                x
                for x in cfg_get(method, "manual_order", [])
                if x in remaining
            ]

            if len(order) == len(remaining):

                route = [
                    labels.index(x)
                    for x in [route_start] + order + [route_start]
                ]

                solutions[method] = {
                    "type": "route",
                    "route": route,
                    "note": None
                }

        else:

            choice = cfg_get(method, "random_start", labels[0])

            seed = int(cfg_get(method, "initial_seed", 42))

            note = None

            if choice == RANDOM_NODE:

                start = pick_random_node(n, seed)

                note = f"dipilih acak (seed {seed})"

            elif choice in labels:

                start = labels.index(choice)

            else:

                start = 0

            solutions[method] = {
                "type": "route",
                "route": generate_initial_tour(
                    n,
                    home=start,
                    seed=seed
                ),
                "note": note
            }

    return solutions


def build_algorithm_parameters(selected_algorithms, ns="independent"):

    parameters = {}

    for method in selected_algorithms:

        config = ALGORITHMS[method]["parameters"]

        if not config:
            continue

        parameters[method] = {
            parameter: cfg_get(
                method,
                f"param_{parameter}",
                settings["default"],
                ns
            )
            for parameter, settings in config.items()
        }

    return parameters


def start_note_of(method, initial_solutions):
    """Catatan start node (mis. 'dipilih acak') mengikuti rantai 'Sama dengan'."""

    seen = set()

    while method not in seen:

        seen.add(method)

        solution = initial_solutions.get(method) or {}

        if solution.get("type") == "same":

            method = solution["source"]

            continue

        return solution.get("note")

    return None


# ============================================================
# TAB KHUSUS: TABU SEARCH (DATA TUGAS)
# ============================================================
#
# Tab ini berdiri sendiri: TIDAK memakai dataset pada bagian
# "Input Data" dan TIDAK terpengaruh sidebar. Algoritma yang dipakai
# tetap Tabu Search yang sama dengan tab lainnya.
#
# >>> ISI DATA TUGAS DI BAWAH INI <<<
#
#   nodes           : label setiap node, misalnya ["A", "B", "C", "D", "E", "F", "G"]
#   distance_matrix : matriks jarak n x n (list of list)         -> dipakai jika diisi
#   coords          : koordinat [(x, y), ...] sesuai urutan nodes -> dipakai untuk
#                     visualisasi, dan untuk menghitung jarak jika distance_matrix kosong
#   initial_route   : rute awal dari soal (label node), misalnya
#                     ["A", "B", "C", "D", "E", "F", "G", "A"]  (boleh tanpa kembali ke awal)
#   tabu_tenure     : tabu tenure dari soal
#   max_iter        : jumlah iterasi
#   metric          : "euclidean" / "manhattan" (hanya jika jarak dihitung dari coords)
#
# Cukup isi nodes + (distance_matrix atau coords). Sisanya opsional.

TABU_TASK_DATA = {
    "nodes": None,
    "distance_matrix": None,
    "coords": None,
    "initial_route": None,
    "tabu_tenure": 3,
    "max_iter": 10,
    "metric": "euclidean",
}


# Data contoh (DUMMY) hanya untuk melihat struktur tab ini.
TABU_TASK_EXAMPLE = {
    "nodes": ["A", "B", "C", "D", "E", "F"],
    "distance_matrix": None,
    "coords": [(10, 20), (60, 80), (90, 40), (70, 10), (30, 50), (20, 90)],
    "initial_route": ["A", "B", "C", "D", "E", "F", "A"],
    "tabu_tenure": 3,
    "max_iter": 10,
    "metric": "euclidean",
}


TABU_TASK_FORMAT_HELP = '''TABU_TASK_DATA = {
    "nodes": ["A", "B", "C", "D", "E", "F", "G"],
    "distance_matrix": [
        [0, 12, 10, 19, 8, 15, 11],
        [12, 0, 3, 7, 2, 9, 14],
        # ... sampai 7 baris
    ],
    # atau isi coords (opsional bila distance_matrix sudah ada):
    "coords": None,   # [(x, y), ...]
    "initial_route": ["A", "B", "C", "D", "E", "F", "G", "A"],
    "tabu_tenure": 3,
    "max_iter": 10,
    "metric": "euclidean",
}'''


if "tabu_task_output" not in st.session_state:
    st.session_state.tabu_task_output = None

if "tabu_task_run_id" not in st.session_state:
    st.session_state.tabu_task_run_id = 0


def task_data_filled(data):

    return bool(data.get("nodes")) and (
        data.get("distance_matrix") is not None
        or data.get("coords") is not None
    )


def prepare_task_data(data):
    """
    Ubah TABU_TASK_DATA menjadi (df, dist_matrix, initial_route).
    Jika coords kosong, node diletakkan melingkar hanya untuk visualisasi.
    """

    nodes = [str(x) for x in data["nodes"]]

    n = len(nodes)

    if n < 3:
        raise ValueError("Jumlah node minimal adalah 3.")

    if len(set(nodes)) != n:
        raise ValueError("Label node tidak boleh duplikat.")

    # ---- koordinat (untuk visualisasi)
    coords = data.get("coords")

    if coords is not None:

        if len(coords) != n:
            raise ValueError(
                "Jumlah coords harus sama dengan jumlah nodes."
            )

        xs = [float(c[0]) for c in coords]
        ys = [float(c[1]) for c in coords]

    else:

        xs = [
            round(50 + 40 * math.cos(2 * math.pi * i / n), 2)
            for i in range(n)
        ]

        ys = [
            round(50 + 40 * math.sin(2 * math.pi * i / n), 2)
            for i in range(n)
        ]

    # ---- matriks jarak
    matrix = data.get("distance_matrix")

    if matrix is not None:

        if len(matrix) != n or any(len(row) != n for row in matrix):
            raise ValueError(
                f"distance_matrix harus berukuran {n} x {n}."
            )

        dist = [[float(v) for v in row] for row in matrix]

    else:

        dist = build_distance_matrix(
            list(zip(xs, ys)),
            metric=data.get("metric", "euclidean")
        )

    df_task = pd.DataFrame({
        "node": nodes,
        "x": xs,
        "y": ys
    })

    # ---- rute awal dari soal
    route = None

    if data.get("initial_route"):

        position = {label: i for i, label in enumerate(nodes)}

        try:

            route = [position[str(x)] for x in data["initial_route"]]

        except KeyError as error:

            raise ValueError(
                f"Node {error} pada initial_route tidak ada di nodes."
            )

        if len(route) == n:
            route.append(route[0])

        validate_tour(route, n, home=route[0])

    return df_task, dist, route


def render_hybrid_placeholder():

    st.subheader(
        "Hybrid"
    )

    st.info(
        "Masukkan dataset terlebih dahulu untuk menggunakan Hybrid."
    )


def render_tabu_task():

    st.subheader("Tabu Search — Data Tugas")

    st.caption(
        "Tab khusus untuk data dari soal tugas. Tab ini tidak memakai "
        "dataset pada bagian Input Data dan tidak terpengaruh sidebar. "
        "Tabu Search yang dipakai sama dengan tab lainnya."
    )

    data = TABU_TASK_DATA

    # ---------------- data belum diisi ----------------

    if not task_data_filled(data):

        st.info(
            "Data tugas belum diisi. Isi `TABU_TASK_DATA` di app.py "
            "(cari tulisan 'ISI DATA TUGAS')."
        )

        use_example = st.checkbox(
            "Lihat tampilan dengan data contoh (dummy)",
            key="tabutask_example"
        )

        if not use_example:

            with st.expander("Contoh format TABU_TASK_DATA"):

                st.code(TABU_TASK_FORMAT_HELP, language="python")

            return

        data = TABU_TASK_EXAMPLE

        st.warning(
            "Menampilkan data contoh (dummy), bukan data tugas."
        )

    # ---------------- siapkan data ----------------

    try:

        task_df, task_dist, task_route = prepare_task_data(data)

    except Exception as error:

        st.error(f"Data tugas tidak valid: {error}")

        return

    labels = task_df["node"].tolist()

    with st.expander("Data tugas", expanded=False):

        col_a, col_b = st.columns(2)

        with col_a:

            st.markdown("**Node & koordinat**")

            st.dataframe(
                task_df,
                use_container_width=True,
                hide_index=True
            )

        with col_b:

            st.markdown("**Distance matrix**")

            st.dataframe(
                pd.DataFrame(task_dist, index=labels, columns=labels).round(2),
                use_container_width=True
            )

        if data.get("coords") is None:

            st.caption(
                "Koordinat tidak diisi, sehingga node digambar melingkar "
                "hanya untuk visualisasi (jarak tetap dari distance_matrix)."
            )

    # ---------------- pengaturan ----------------

    st.markdown("#### Pengaturan")

    col1, col2 = st.columns(2)

    with col1:

        source_options = (
            ["Dari soal"] if task_route is not None else []
        ) + ["Generate Random", "Manual"]

        source = st.selectbox(
            "Initial solution",
            source_options,
            key="tabutask_source"
        )

        initial_route = None

        if source == "Dari soal":

            initial_route = task_route

        elif source == "Generate Random":

            start_label = st.selectbox(
                "Start Node",
                labels,
                key="tabutask_random_start"
            )

            seed = st.number_input(
                "Seed",
                min_value=0,
                max_value=99999,
                value=42,
                step=1,
                key="tabutask_seed"
            )

            initial_route = generate_initial_tour(
                len(labels),
                home=labels.index(start_label),
                seed=int(seed)
            )

        else:

            manual_start = st.selectbox(
                "Route Start",
                labels,
                key="tabutask_manual_start"
            )

            remaining = [x for x in labels if x != manual_start]

            order = st.multiselect(
                "Node Order",
                remaining,
                key="tabutask_manual_order"
            )

            if len(order) == len(remaining):

                initial_route = [
                    labels.index(x)
                    for x in [manual_start] + order + [manual_start]
                ]

        if initial_route is not None:

            st.caption("Initial Route")

            st.code(route_to_labels(task_df, initial_route))

            st.caption(
                f"Initial Distance: "
                f"{tour_distance(initial_route, task_dist):.2f}"
            )

        else:

            st.caption("Lengkapi Node Order untuk membentuk rute.")

    with col2:

        tabu_tenure = st.number_input(
            "Tabu Tenure",
            min_value=1,
            max_value=100,
            value=int(data.get("tabu_tenure", 3)),
            step=1,
            key="tabutask_tenure"
        )

        max_iter = st.number_input(
            "Max Iteration",
            min_value=1,
            max_value=1000,
            value=int(data.get("max_iter", 10)),
            step=1,
            key="tabutask_max_iter"
        )

        n_show = st.number_input(
            "Jumlah iterasi yang diilustrasikan",
            min_value=1,
            max_value=100,
            value=3,
            step=1,
            key="tabutask_show",
            help="Tugas mewajibkan minimal 3 iterasi."
        )

    if st.button(
        "▶ Run Tabu Search (Tugas)",
        type="primary",
        use_container_width=True,
        disabled=initial_route is None,
        key="tabutask_run"
    ):

        try:

            st.session_state.tabu_task_run_id += 1

            result = execute_independent_algorithm(
                method="Tabu Search",
                df=task_df,
                dist_matrix=task_dist,
                start_node=None,
                initial_route=initial_route,
                parameters={
                    "max_iter": int(max_iter),
                    "tabu_tenure": int(tabu_tenure),
                    "verbose_history": True
                }
            )

            st.session_state.tabu_task_output = {
                "df": task_df,
                "dist": task_dist,
                "result": result,
                "n_show": int(n_show),
                "dummy": data is TABU_TASK_EXAMPLE
            }

        except Exception as error:

            st.error(f"Tabu Search gagal dijalankan: {error}")


    # ---------------- hasil ----------------

    output = st.session_state.tabu_task_output

    if output is None:
        return

    st.divider()

    if output["dummy"]:

        st.warning(
            "Hasil di bawah memakai data contoh (dummy)."
        )

    out_df = output["df"]
    out_dist = output["dist"]
    out_result = output["result"]
    out_labels = out_df["node"].astype(str).tolist()

    render_independent_result(
        out_df,
        out_dist,
        out_result,
        0,
        key_prefix="tabutask",
        run_id=st.session_state.tabu_task_run_id
    )

    # ---------------- ilustrasi N iterasi pertama ----------------

    history = out_result["history"] or []

    # jumlah iterasi mengikuti isian terbaru (tanpa perlu Run ulang)
    shown = min(int(n_show), len(history) - 1)

    st.divider()

    st.subheader(f"Ilustrasi Tabu Search — {shown} iterasi pertama")

    st.caption(
        "Setiap iterasi menampilkan current route, tabu list, seluruh "
        "candidate move, selected move, new route, dan tabu list sesudahnya."
    )

    for k in range(1, shown + 1):

        with st.expander(f"ITERATION {k}", expanded=True):

            render_view(
                build_iteration_view(
                    "tabu",
                    history,
                    k,
                    out_dist,
                    out_labels
                )
            )

    st.subheader("Hasil Akhir")

    st.success(
        f"Rute terbaik: {route_to_labels(out_df, out_result['route'])} "
        f"— distance {out_result['final_distance']:.2f} "
        f"(initial {out_result['initial_distance']:.2f})"
    )


# ============================================================
# HYBRID
# ============================================================
#
# Tahap 1: metode Constructive membangun initial solution.
# Tahap 2: metode Optimization (Local Search / Metaheuristic)
#          memperbaiki rute dari tahap 1.

if "hybrid_results" not in st.session_state:
    st.session_state.hybrid_results = []

if "hybrid_run_id" not in st.session_state:
    st.session_state.hybrid_run_id = 0


HYBRID_MAX_COMBINATIONS = 12


def render_hybrid_result(df, dist_matrix, combo, index, run_id):
    """Hasil satu kombinasi: ringkasan, rute awal vs akhir, lalu tiap tahap."""

    stage1 = combo["stage1"]
    stage2 = combo["stage2"]

    start_label = str(df.iloc[int(stage1["route"][0])]["node"])

    gain = (
        (combo["initial_distance"] - combo["final_distance"])
        / combo["initial_distance"] * 100
        if combo["initial_distance"]
        else 0.0
    )

    st.info(
        f"Alur: Start Node **{start_label}** → "
        f"**{stage1['method']}** (distance {stage1['final_distance']:.2f}) → "
        f"**{stage2['method']}** (distance {stage2['final_distance']:.2f})"
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Initial (Constructive)", f"{combo['initial_distance']:.2f}")
    col2.metric("Final (Hybrid)", f"{combo['final_distance']:.2f}")
    col3.metric("Improvement", f"{gain:.2f}%")
    col4.metric("Total Execution Time", f"{combo['execution_time'] * 1000:.3f} ms")

    initial_fig, final_fig = plot_route_comparison(
        df,
        stage1["route"],
        stage2["route"],
        home=stage1["route"][0],
        initial_title=f"Initial — {stage1['method']}",
        final_title=f"Final — {combo['label']}"
    )

    col_a, col_b = st.columns(2)

    with col_a:

        st.plotly_chart(
            initial_fig,
            use_container_width=True,
            key=f"hybrid_{run_id}_{index}_cmp_initial"
        )

    with col_b:

        st.plotly_chart(
            final_fig,
            use_container_width=True,
            key=f"hybrid_{run_id}_{index}_cmp_final"
        )

    st.divider()

    st.markdown("### Tahap 1 — Initial Solution")

    render_independent_result(
        df,
        dist_matrix,
        stage1,
        index,
        key_prefix=f"hybrid_{index}_s1",
        run_id=run_id
    )

    st.divider()

    st.markdown("### Tahap 2 — Optimization")

    render_independent_result(
        df,
        dist_matrix,
        stage2,
        index,
        key_prefix=f"hybrid_{index}_s2",
        run_id=run_id
    )


def render_hybrid(df, dist_matrix):

    st.subheader("Hybrid")

    st.caption(
        "Gabungkan metode Constructive (membangun initial solution) dengan "
        "metode Optimization (memperbaikinya). Rute hasil tahap 1 menjadi "
        "initial route tahap 2."
    )

    labels = df["node"].tolist()

    constructive_methods = [
        m for m, c in ALGORITHMS.items()
        if c["type"] == "Constructive"
    ]

    optimizer_methods = [
        m for m, c in ALGORITHMS.items()
        if c["type"] != "Constructive"
    ]


    # ---------------- 1. kombinasi ----------------

    st.markdown("#### 1. Pilih Kombinasi")

    col1, col2 = st.columns(2)

    with col1:

        inits = st.multiselect(
            "Initial Solution (Constructive)",
            constructive_methods,
            key="hybrid_inits"
        )

    with col2:

        optimizers = st.multiselect(
            "Optimization",
            optimizer_methods,
            key="hybrid_optimizers"
        )

    pairs = [(i, o) for i in inits for o in optimizers]

    pair_labels = [f"{i} → {o}" for i, o in pairs]

    chosen_pairs = []

    if pairs:

        chosen_labels = st.multiselect(
            "Kombinasi yang dijalankan",
            pair_labels,
            default=pair_labels,
            key="hybrid_combinations",
            help="Hapus kombinasi yang tidak ingin dijalankan."
        )

        chosen_pairs = [
            pair
            for pair, label in zip(pairs, pair_labels)
            if label in chosen_labels
        ]

        if len(chosen_pairs) > HYBRID_MAX_COMBINATIONS:

            st.warning(
                f"Maksimal {HYBRID_MAX_COMBINATIONS} kombinasi sekaligus. "
                "Kurangi pilihan kombinasi."
            )

            chosen_pairs = []

    else:

        st.caption(
            "Pilih minimal satu Initial Solution dan satu Optimization."
        )


    # ---------------- 2. start node + parameter ----------------

    col_start, col_params = st.columns(2)

    with col_start:

        st.markdown("#### 2. Start Node")

        start_choice = st.selectbox(
            "Start Node",
            labels + [RANDOM_NODE],
            key="hybrid_start_choice",
            help="Node awal yang dipakai metode Constructive."
        )

        if start_choice == RANDOM_NODE:

            start_seed = st.number_input(
                "Seed (Random)",
                min_value=0,
                max_value=99999,
                value=42,
                step=1,
                key="hybrid_start_seed"
            )

            start = pick_random_node(len(df), start_seed)

            start_note = f"dipilih acak (seed {int(start_seed)})"

            st.caption(f"Start node = **{labels[start]}**.")

        else:

            start = labels.index(start_choice)

            start_note = None

    involved = list(dict.fromkeys(inits + optimizers))

    methods_with_parameters = [
        m for m in involved
        if ALGORITHMS[m]["parameters"]
    ]

    with col_params:

        st.markdown("#### 3. Parameters")

        if not methods_with_parameters:

            st.caption(
                "Algoritma yang dipilih tidak memiliki parameter."
            )

        else:

            active_method = st.radio(
                "Pilih algoritma:",
                methods_with_parameters,
                horizontal=True,
                key="hybrid_active_parameters"
            )

            render_parameter_panel(active_method, ns="hybrid")


    # ---------------- run ----------------

    if st.button(
        "▶ Run Hybrid",
        type="primary",
        use_container_width=True,
        disabled=not chosen_pairs,
        key="hybrid_run_button"
    ):

        st.session_state.hybrid_run_id += 1

        parameters_all = build_algorithm_parameters(involved, ns="hybrid")

        def _parameters_of(method):

            parameters = dict(parameters_all.get(method, {}))

            if (
                "verbose_history" in ALGORITHMS[method]["parameters"]
                and len(df) <= DETAIL_NODE_LIMIT
            ):

                parameters["verbose_history"] = True

            return parameters

        results = []

        for init_method, opt_method in chosen_pairs:

            label = f"{init_method} → {opt_method}"

            try:

                stage1 = execute_independent_algorithm(
                    method=init_method,
                    df=df,
                    dist_matrix=dist_matrix,
                    start_node=start,
                    initial_route=None,
                    parameters=_parameters_of(init_method)
                )

                stage1["start_note"] = start_note

                stage2 = execute_independent_algorithm(
                    method=opt_method,
                    df=df,
                    dist_matrix=dist_matrix,
                    start_node=None,
                    initial_route=stage1["route"],
                    parameters=_parameters_of(opt_method)
                )

                results.append({
                    "label": label,
                    "method": label,
                    "stage1": stage1,
                    "stage2": stage2,
                    "initial_distance": stage1["final_distance"],
                    "final_distance": stage2["final_distance"],
                    "execution_time": (
                        stage1["execution_time"]
                        + stage2["execution_time"]
                    ),
                    "route": stage2["route"]
                })

            except Exception as error:

                st.error(f"{label} gagal dijalankan: {error}")

        st.session_state.hybrid_results = results


    # ---------------- results ----------------

    results = st.session_state.hybrid_results

    if not results:
        return

    run_id = st.session_state.hybrid_run_id

    st.divider()

    st.subheader("Results")

    result_tabs = st.tabs([r["label"] for r in results])

    for i, (result_tab, combo) in enumerate(zip(result_tabs, results)):

        with result_tab:

            render_hybrid_result(df, dist_matrix, combo, i, run_id)


    # ---------------- comparison ----------------

    if len(results) >= 2:

        # Pembanding: hasil Constructive saja (tanpa optimasi)
        baselines = {}

        for r in results:

            s1 = r["stage1"]

            baselines.setdefault(s1["method"], {
                "method": f"{s1['method']} (tanpa optimasi)",
                "initial_distance": None,
                "final_distance": s1["final_distance"],
                "execution_time": s1["execution_time"]
            })

        entries = list(results) + list(baselines.values())

        st.divider()

        st.subheader("Comparison")

        st.caption(
            "Hybrid dibandingkan satu sama lain dan dengan hasil "
            "Constructive saja (tanpa optimasi)."
        )

        comparison_df = sort_by_distance(
            create_comparison_table(entries)
        )

        st.dataframe(
            comparison_df,
            use_container_width=True,
            hide_index=True
        )

        col_best, col_fast = st.columns(2)

        best = get_best_distance(comparison_df)

        fastest = get_fastest_method(comparison_df)

        with col_best:

            if best is not None:

                st.metric(
                    "Best Distance",
                    best["Method"],
                    f"{best['Final Distance']:.2f}"
                )

        with col_fast:

            if fastest is not None:

                st.metric(
                    "Fastest",
                    fastest["Method"],
                    f"{fastest['Execution Time (ms)']:.3f} ms"
                )

        col_c, col_d = st.columns(2)

        with col_c:

            st.plotly_chart(
                plot_distance_comparison(comparison_df),
                use_container_width=True,
                key=f"hybrid_{run_id}_cmp_distance"
            )

        with col_d:

            st.plotly_chart(
                plot_time_comparison(comparison_df),
                use_container_width=True,
                key=f"hybrid_{run_id}_cmp_time"
            )

        st.plotly_chart(
            plot_improvement_comparison(comparison_df),
            use_container_width=True,
            key=f"hybrid_{run_id}_cmp_improvement"
        )


# ============================================================
# HEADER
# ============================================================

st.title("🧭 TSP Learning & Optimization")

st.caption(
    "Media interaktif untuk membedah, menguji, dan membandingkan "
    "berbagai metode heuristik dalam menyelesaikan "
    "Travelling Salesman Problem."
)


# ============================================================
# INPUT DATA
# ============================================================

st.header("📂 Input Data")

input_type = st.radio(
    "Pilih sumber data:",
    ["Generate", "Upload", "Manual"],
    horizontal=True
)


new_data = None


# ------------------------------------------------------------
# GENERATE
# ------------------------------------------------------------

if input_type == "Generate":

    col1, col2, col3 = st.columns(3)

    with col1:

        n_nodes = st.number_input(
            "Jumlah node",
            min_value=3,
            max_value=500,
            value=10
        )

    with col2:

        data_seed = st.number_input(
            "Seed",
            min_value=0,
            value=42
        )

    with col3:

        coordinate_range = st.number_input(
            "Coordinate Range",
            min_value=1,
            value=100
        )


    if st.button(
        "Generate Dataset",
        type="primary"
    ):

        rng = random.Random(
            int(data_seed)
        )

        new_data = pd.DataFrame({

            "node": [
                chr(65 + i)
                if i < 26
                else f"N{i + 1}"
                for i in range(
                    int(n_nodes)
                )
            ],

            "x": [
                round(
                    rng.uniform(
                        0,
                        coordinate_range
                    ),
                    2
                )
                for _ in range(
                    int(n_nodes)
                )
            ],

            "y": [
                round(
                    rng.uniform(
                        0,
                        coordinate_range
                    ),
                    2
                )
                for _ in range(
                    int(n_nodes)
                )
            ]
        })


# ------------------------------------------------------------
# UPLOAD
# ------------------------------------------------------------

elif input_type == "Upload":

    uploaded_file = st.file_uploader(
        "Upload CSV / Excel",
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


# ------------------------------------------------------------
# MANUAL
# ------------------------------------------------------------

else:

    if "manual_data" not in st.session_state:

        st.session_state.manual_data = pd.DataFrame({
            "node": ["A", "B", "C", "D", "E"],
            "x": [10, 50, 90, 70, 30],
            "y": [20, 80, 40, 60, 90]
        })


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
# LOAD DATA
# ============================================================

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


# Stop kalau belum ada data

if st.session_state.df is None:

    st.divider()

    tab_independent, tab_hybrid, tab_tabu_task = st.tabs([
        "Independent",
        "Hybrid",
        "Tabu Search (Tugas)"
    ])

    with tab_independent:

        st.info(
            "Masukkan dataset terlebih dahulu."
        )

    with tab_hybrid:

        render_hybrid_placeholder()

    # Tab ini memakai data tugas sendiri, tidak butuh dataset di atas
    with tab_tabu_task:

        render_tabu_task()

    st.stop()


df = st.session_state.df


# ============================================================
# VIEW DATA
# ============================================================

st.divider()

st.header("📊 Data & Visualization")

col1, col2 = st.columns(2)


with col1:

    st.subheader("Dataset")

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )


with col2:

    st.subheader("Node Distribution")

    # home=-1: tidak ada node yang ditandai merah, karena
    # start node ditentukan per algoritma pada sidebar.
    fig_nodes = plot_nodes(
        df,
        home=-1
    )

    st.plotly_chart(
        fig_nodes,
        use_container_width=True
    )


# ============================================================
# DISTANCE METHOD
# ============================================================

st.divider()

st.header("📐 Distance Matrix")

distance_method = st.radio(
    "Distance Method",
    ["Euclidean", "Manhattan"],
    horizontal=True
)


distance_metric = (
    "euclidean"
    if distance_method == "Euclidean"
    else "manhattan"
)


coords = list(
    zip(
        df["x"],
        df["y"]
    )
)


dist_matrix = build_distance_matrix(
    coords,
    metric=distance_metric
)


with st.expander(
    "View Distance Matrix"
):

    st.dataframe(
        pd.DataFrame(
            dist_matrix,
            index=df["node"],
            columns=df["node"]
        ),
        use_container_width=True
    )


# ============================================================
# MAIN MODE
# ============================================================

st.divider()

tab_independent, tab_hybrid, tab_tabu_task = st.tabs([
    "Independent",
    "Hybrid",
    "Tabu Search (Tugas)"
])


# ============================================================
# INDEPENDENT SIDEBAR
# ============================================================

with st.sidebar:

    # Judul + garis dalam satu wadah agar jaraknya rapat
    try:
        title_box = st.container(key="sidebar_title")
    except TypeError:          # Streamlit versi lama (tanpa parameter key)
        title_box = st.container()

    with title_box:

        st.header("Independent")

        st.divider()

    # --------------------------------------------------------
    # ALGORITHM
    # --------------------------------------------------------

    st.subheader("Algorithm")

    selected_algorithms = st.multiselect(
        "Pilih satu atau beberapa algoritma:",
        list(ALGORITHMS.keys()),
        key="independent_algorithm_selection"
    )

    # Selalu didefinisikan agar tidak NameError saat Run
    initial_solutions = {}
    algorithm_parameters = {}


    if selected_algorithms:

        # ====================================================
        # INITIAL SOLUTION  (pilih algoritma → ubah)
        # ====================================================

        st.divider()

        st.subheader("Initial Solution")

        active_initial = st.radio(
            "Pilih algoritma:",
            selected_algorithms,
            horizontal=True,
            key="independent_active_initial"
        )

        render_initial_panel(
            active_initial,
            selected_algorithms,
            df
        )


        # ====================================================
        # PARAMETERS  (pilih algoritma → ubah)
        # ====================================================

        st.divider()

        st.subheader("Parameters")

        methods_with_parameters = [
            method
            for method in selected_algorithms
            if ALGORITHMS[method]["parameters"]
        ]

        if not methods_with_parameters:

            st.caption(
                "Algoritma yang dipilih tidak memiliki parameter."
            )

        else:

            active_parameters = st.radio(
                "Pilih algoritma:",
                methods_with_parameters,
                horizontal=True,
                key="independent_active_parameters"
            )

            render_parameter_panel(active_parameters)


        # Pengaturan SEMUA algoritma terpilih dibaca dari
        # penyimpanan (bukan hanya yang sedang tampil).
        initial_solutions = build_initial_solutions(
            selected_algorithms,
            df
        )

        algorithm_parameters = build_algorithm_parameters(
            selected_algorithms
        )


    # --------------------------------------------------------
    # RUN
    # --------------------------------------------------------

    st.divider()

    run_independent = st.button(
        "▶ Run Independent",
        type="primary",
        use_container_width=True,
        disabled=not selected_algorithms,
        key="independent_run_button"
    )


# ============================================================
# INDEPENDENT CONTENT
# ============================================================

with tab_independent:

    st.subheader(
        "Independent Algorithms"
    )

    st.caption(
        "Setiap algoritma berjalan secara independen "
        "menggunakan input awalnya masing-masing. "
        "Atur algoritma, initial solution, dan parameter di sidebar, "
        "lalu klik Run."
    )


    if not selected_algorithms:

        st.info(
            "Pilih minimal satu algoritma pada sidebar."
        )


    # ========================================================
    # RUN
    # ========================================================

    if run_independent and selected_algorithms:

        results = []

        # Reset pilihan iterasi pada hasil sebelumnya
        st.session_state.independent_run_id += 1


        # ----------------------------------------------------
        # RUN EACH ALGORITHM INDEPENDENTLY
        # ----------------------------------------------------

        for method in selected_algorithms:

            try:

                start_node, initial_route = resolve_initial(
                    method,
                    initial_solutions,
                    selected_algorithms
                )

                if (
                    ALGORITHMS[method]["needs_start_node"]
                    and start_node is None
                ):

                    raise ValueError(
                        "Start Node belum dipilih."
                    )

                if (
                    ALGORITHMS[method]["needs_initial_route"]
                    and initial_route is None
                ):

                    raise ValueError(
                        "Starting Route belum lengkap."
                    )

                parameters = dict(
                    algorithm_parameters.get(
                        method,
                        {}
                    )
                )

                # Agar detail per iterasi (mis. semua kandidat
                # Tabu Search) tersedia untuk dataset kecil-menengah.
                if (
                    "verbose_history"
                    in ALGORITHMS[method]["parameters"]
                    and len(df) <= DETAIL_NODE_LIMIT
                ):

                    parameters["verbose_history"] = True

                result = execute_independent_algorithm(
                    method=method,
                    df=df,
                    dist_matrix=dist_matrix,
                    start_node=start_node,
                    initial_route=initial_route,
                    parameters=parameters
                )

                result["start_note"] = start_note_of(
                    method,
                    initial_solutions
                )

                results.append(
                    result
                )


            except Exception as e:

                st.error(
                    f"{method} gagal dijalankan: {e}"
                )


        st.session_state.independent_results = (
            results
        )


    # ========================================================
    # RESULTS
    # ========================================================

    results = st.session_state.independent_results


    if results:

        st.divider()

        st.subheader(
            "Results"
        )


        for i, result in enumerate(results):

            render_independent_result(
                df,
                dist_matrix,
                result,
                i
            )

            if i < len(results) - 1:

                st.divider()


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


            col1, col2 = st.columns(2)


            with col1:

                best = get_best_distance(
                    comparison_df
                )

                if best is not None:

                    st.metric(
                        "Best Distance",
                        best["Method"],
                        f"{best['Final Distance']:.2f}"
                    )


            with col2:

                fastest = get_fastest_method(
                    comparison_df
                )

                if fastest is not None:

                    st.metric(
                        "Fastest",
                        fastest["Method"],
                        f"{fastest['Execution Time (ms)']:.3f} ms"
                    )


            col1, col2 = st.columns(2)


            with col1:

                st.plotly_chart(
                    plot_distance_comparison(
                        comparison_df
                    ),
                    use_container_width=True,
                    key="independent_cmp_distance"
                )


            with col2:

                st.plotly_chart(
                    plot_time_comparison(
                        comparison_df
                    ),
                    use_container_width=True,
                    key="independent_cmp_time"
                )


            st.plotly_chart(
                plot_improvement_comparison(
                    comparison_df
                ),
                use_container_width=True,
                key="independent_cmp_improvement"
            )


# ============================================================
# HYBRID
# ============================================================

with tab_hybrid:

    render_hybrid(df, dist_matrix)


# ============================================================
# TABU SEARCH (DATA TUGAS)
# ============================================================

with tab_tabu_task:

    render_tabu_task()
