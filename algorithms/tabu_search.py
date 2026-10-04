from .base import generate_initial_tour, validate_tour, tour_distance


# Neighbor generator -- sama persis style-nya dengan get_neighbors()
# di simulated_annealing.py (closed tour: index 0 dan index terakhir
# sama-sama home, jadi keduanya nggak pernah ikut dibalik).
def get_neighbors(tour):
    neighbors = []

    for i in range(1, len(tour) - 2):
        for j in range(i + 1, len(tour) - 1):

            new_tour = list(tour)
            new_tour[i:j + 1] = reversed(new_tour[i:j + 1])

            neighbors.append({
                "tour": new_tour,
                "i": i,
                "j": j
            })

    return neighbors


# Edge yang dibuang oleh move (i, j) -- dipakai sebagai "atribut tabu".
# Pakai city (bukan posisi), karena posisi city bisa geser antar iterasi
# tapi identitas kota tidak.
def removed_edges(tour, i, j):
    e1 = frozenset((tour[i - 1], tour[i]))
    e2 = frozenset((tour[j], tour[j + 1]))
    return frozenset((e1, e2))


def tabu_search(
    dist_matrix,
    initial_tour=None,
    tabu_tenure=3,
    max_iter=10,
    seed=None,
    verbose_history=False,
):
    """
    Tabu Search untuk TSP.

    Step 1 : home city = tour[0] (dan tour[-1], karena closed tour).
    Step 2 : initial solution (random via generate_initial_tour, atau
             dikasih langsung lewat initial_tour). Total distance awal
             jadi aspiration level. tabu_list mulai kosong.
    Step 3 : i = 1 (loop "for it in range(1, max_iter + 1)" di bawah)
    Step 4 : tukar 2 arc jadi 2 arc baru (2-opt, lihat get_neighbors) ->
             hitung total distance tiap kandidat.
    Step 5 : evaluasi SEMUA kandidat di neighborhood, pilih yang
             admissible terbaik jadi current solution, update tabu_list.
    Step 6 : i += 1, ulangi Step 4.
    Step 7 : stop kalau iterasi == max_iter. Tur terbaik = hasil akhir.

    dist_matrix     : matrix jarak n x n (dari base.build_distance_matrix)
    initial_tour    : closed tour [home, ..., home]. None -> random.
    tabu_tenure     : berapa iterasi sebuah move dilarang diulang
    max_iter        : jumlah iterasi (Step 7)
    verbose_history : True -> tiap iterasi simpan semua kandidat yang
                       dievaluasi (buat tab ilustrasi / debugging).
                       False -> cuma ringkasan per iterasi.

    Return: (best_tour, best_distance, history)
    """
    n = len(dist_matrix)

    if initial_tour is None:
        current_tour = generate_initial_tour(n, seed=seed)
    else:
        validate_tour(initial_tour, n)
        current_tour = list(initial_tour)

    current_distance = tour_distance(current_tour, dist_matrix)

    best_tour = list(current_tour)
    best_distance = current_distance          # aspiration level awal (Step 2)

    tabu_list = {}   # atribut (removed_edges) -> iterasi kadaluarsa

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

        neighbors = get_neighbors(current_tour)   # Step 4: semua kandidat 2-opt

        candidate_log = []
        best_candidate = None
        best_candidate_distance = None
        best_candidate_attr = None

        # Step 5: evaluasi seluruh neighborhood, pilih admissible terbaik
        for nb in neighbors:

            attr = removed_edges(current_tour, nb["i"], nb["j"])
            d = tour_distance(nb["tour"], dist_matrix)

            is_tabu = attr in tabu_list and tabu_list[attr] >= it
            aspiration_met = d < best_distance          # aspiration criterion
            admissible = (not is_tabu) or aspiration_met

            if verbose_history:
                candidate_log.append({
                    "i": nb["i"],
                    "j": nb["j"],
                    "distance": d,
                    "is_tabu": is_tabu,
                    "aspiration_met": aspiration_met,
                    "admissible": admissible,
                })

            if admissible and (best_candidate is None or d < best_candidate_distance):
                best_candidate = nb
                best_candidate_distance = d
                best_candidate_attr = attr

        # fallback langka: semua kandidat tabu & tak satupun aspiration_met
        if best_candidate is None:
            best_candidate = min(
                neighbors,
                key=lambda nb: tour_distance(nb["tour"], dist_matrix),
            )
            best_candidate_distance = tour_distance(best_candidate["tour"], dist_matrix)
            best_candidate_attr = removed_edges(
                current_tour, best_candidate["i"], best_candidate["j"]
            )

        current_tour = best_candidate["tour"]
        current_distance = best_candidate_distance

        # update tabu list: tambah move yang baru dipakai, buang yang kadaluarsa
        tabu_list[best_candidate_attr] = it + tabu_tenure
        tabu_list = {k: v for k, v in tabu_list.items() if v >= it}

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
