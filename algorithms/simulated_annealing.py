import random
import math

from .base import (
    tour_distance,
    validate_tour,
    generate_initial_tour
)

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


def simulated_annealing(
    dist_matrix,
    initial_tour=None,
    initial_temp=1000,
    cooling_rate=0.95,
    min_temp=0.01,
    max_iter=10,
    seed=None,
    verbose_history=False,
):
    n = len(dist_matrix)

    rng = random.Random(seed)

    if initial_tour is None:
        current_tour = generate_initial_tour(n, seed=seed)
    else:
        validate_tour(initial_tour, n)
        current_tour = list(initial_tour)

    current_distance = tour_distance(current_tour, dist_matrix)

    best_tour = list(current_tour)
    best_distance = current_distance

    temperature = initial_temp

    history = [{
        "iteration": 0,
        "tour": list(current_tour),
        "distance": current_distance,
        "best_distance": best_distance,
        "temperature": temperature,
        "move": None,
        "delta": None,
        "probability": None,
        "accepted": True,
    }]

    for it in range(1, max_iter + 1):

        if temperature <= min_temp:
            break

        neighbors = get_neighbors(current_tour)

        candidate = rng.choice(neighbors)

        candidate_tour = candidate["tour"]
        candidate_distance = tour_distance(
            candidate_tour,
            dist_matrix
        )

        delta = candidate_distance - current_distance

        if delta < 0:
            probability = 1.0
        else:
            probability = math.exp(-delta / temperature)

        accepted = rng.random() < probability

        if accepted:
            current_tour = list(candidate_tour)
            current_distance = candidate_distance

        if current_distance < best_distance:
            best_distance = current_distance
            best_tour = list(current_tour)

        history.append({
            "iteration": it,
            "tour": list(current_tour),
            "distance": current_distance,
            "best_distance": best_distance,
            "temperature": temperature,
            "move": {
                "i": candidate["i"],
                "j": candidate["j"],
            },
            "delta": delta,
            "probability": probability,
            "accepted": accepted,
        })

        temperature *= cooling_rate

    return best_tour, best_distance, history
