try:
    from .base import (
        build_distance_matrix,
        tour_distance,
        generate_initial_tour,
        validate_tour,
    )
except ImportError:  # dijalankan langsung / tanpa package
    from base import (
        build_distance_matrix,
        tour_distance,
        generate_initial_tour,
        validate_tour,
    )


def get_neighbors(tour):
    """Neighborhood 2-opt (reverse segmen i..j), home pada posisi 0 dan -1 tetap.

    Setiap neighbor menyertakan 'tabu_attr': pasangan kota pada ujung segmen
    yang dibalik (tidak peduli urutan), dipakai sebagai atribut tabu.
    """
    neighbors = []

    for i in range(1, len(tour) - 2):
        for j in range(i + 1, len(tour) - 1):
            new_tour = list(tour)
            new_tour[i:j + 1] = reversed(new_tour[i:j + 1])

            a, b = tour[i], tour[j]
            neighbors.append({
                "tour": new_tour,
                "i": i,
                "j": j,
                "tabu_attr": (min(a, b), max(a, b)),
            })

    return neighbors


def tabu_search(
    dist_matrix,
    initial_tour=None,
    tabu_tenure=3,
    max_iter=10,
    seed=None,
    verbose_history=False,
):
    n = len(dist_matrix)

    if initial_tour is None:
        current_tour = generate_initial_tour(n, seed=seed)
    else:
        validate_tour(initial_tour, n)
        current_tour = list(initial_tour)

    current_distance = tour_distance(current_tour, dist_matrix)
    best_tour = list(current_tour)
    best_distance = current_distance          # aspiration level awal

    tabu_list = {}   # tabu_attr -> iterasi kadaluarsa
    history = [{
        "iteration": 0,
        "tour": list(current_tour),
        "distance": current_distance,
        "best_distance": best_distance,
        "move": None,
        "tabu_list": dict(tabu_list),
        "candidates": [],
    }]

    for it in range(1, max_iter + 1):
        neighbors = get_neighbors(current_tour)   # semua kandidat 2-opt
        if not neighbors:                         # tur terlalu kecil
            break

        candidate_log = []
        best_candidate = None
        best_candidate_distance = None

        # Evaluasi seluruh neighborhood, pilih yang admissible terbaik
        for nb in neighbors:
            d = tour_distance(nb["tour"], dist_matrix)
            is_tabu = nb["tabu_attr"] in tabu_list and tabu_list[nb["tabu_attr"]] >= it
            aspiration_met = d < best_distance
            admissible = (not is_tabu) or aspiration_met

            if verbose_history:
                candidate_log.append({
                    "i": nb["i"], "j": nb["j"],
                    "tabu_attr": nb["tabu_attr"],
                    "distance": d,
                    "is_tabu": is_tabu,
                    "aspiration_met": aspiration_met,
                    "admissible": admissible,
                })

            if admissible and (best_candidate is None or d < best_candidate_distance):
                best_candidate = nb
                best_candidate_distance = d

        # Fallback langka: semua neighbor tabu dan tidak ada yang memenuhi aspiration
        if best_candidate is None:
            best_candidate = min(
                neighbors, key=lambda nb: tour_distance(nb["tour"], dist_matrix)
            )
            best_candidate_distance = tour_distance(best_candidate["tour"], dist_matrix)

        # Pindah ke kandidat terbaik (boleh lebih buruk dari current)
        current_tour = best_candidate["tour"]
        current_distance = best_candidate_distance

        # Update tabu list lalu buang yang sudah kadaluarsa
        tabu_list[best_candidate["tabu_attr"]] = it + tabu_tenure
        tabu_list = {k: v for k, v in tabu_list.items() if v > it}

        # Update solusi terbaik global
        if current_distance < best_distance:
            best_distance = current_distance
            best_tour = list(current_tour)

        history.append({
            "iteration": it,
            "tour": list(current_tour),
            "distance": current_distance,
            "best_distance": best_distance,
            "move": {"i": best_candidate["i"], "j": best_candidate["j"]},
            "tabu_list": dict(tabu_list),
            "candidates": candidate_log,
        })

    return best_tour, best_distance, history


if __name__ == "__main__":
    coords = [(0, 0), (2, 4), (5, 2), (6, 6), (1, 5), (3, 1)]
    dm = build_distance_matrix(coords, metric="euclidean")

    initial = generate_initial_tour(len(coords), home=0, seed=42)
    print("Tur awal  :", initial, f"| jarak = {tour_distance(initial, dm):.4f}")

    best_tour, best_distance, history = tabu_search(
        dm,
        initial_tour=initial,
        tabu_tenure=3,
        max_iter=15,
        verbose_history=True,
    )

    print("\nRiwayat iterasi:")
    for h in history:
        print(
            f"it={h['iteration']:>2} | move={h['move']} | "
            f"jarak={h['distance']:.4f} | best={h['best_distance']:.4f} | "
            f"tabu={h['tabu_list']}"
        )

    print("\nTur terbaik  :", best_tour)
    print(f"Jarak terbaik: {best_distance:.4f}")
