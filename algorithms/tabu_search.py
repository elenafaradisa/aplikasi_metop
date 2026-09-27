from utils import tour_distance, generate_initial_tour, validate_tour, get_neighbors

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
    best_distance = current_distance          # aspiration level awal (Step 2)
 
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
        neighbors = get_neighbors(current_tour)   # Step 4: semua kandidat 2-opt
        candidate_log = []
        best_candidate = None
        best_candidate_distance = None
 
        # Step 5: evaluasi seluruh neighborhood, pilih yang admissible terbaik
        for nb in neighbors:
            d = tour_distance(nb["tour"], dist_matrix)
            is_tabu = nb["tabu_attr"] in tabu_list and tabu_list[nb["tabu_attr"]] >= it
            aspiration_met = d < best_distance
            admissible = (not is_tabu) or aspiration_met

            if verbose_history:
                candidate_log.append({
                    "i": nb["i"], "j": nb["j"],
                    "distance": d,
                    "is_tabu": is_tabu,
                    "aspiration_met": aspiration_met,
                    "admissible": admissible,
                })
 
            if admissible and (best_candidate is None or d < best_candidate_distance):
                best_candidate = nb
                best_candidate_distance = d
 
        if best_candidate is None:   # fallback langka: semua tabu, tanpa aspiration
            best_candidate = min(neighbors, key=lambda nb: tour_distance(nb["tour"], dist_matrix))
            best_candidate_distance = tour_distance(best_candidate["tour"], dist_matrix)
 
        current_tour = best_candidate["tour"]
        current_distance = best_candidate_distance
 
        tabu_list[best_candidate["tabu_attr"]] = it + tabu_tenure
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
 
 
if __name__ == "__main__":
    from utils import build_distance_matrix, generate_initial_tour
 
    coords = [(0, 0), (2, 4), (5, 2), (6, 6), (1, 5), (3, 1)]
    dm = build_distance_matrix(coords, metric="euclidean")
