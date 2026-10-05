import numbers

from .base import tour_distance, validate_tour

# Toleransi float: move hanya diterima jika memperbaiki > _EPS.
# Mencegah looping tak berujung akibat selisih pembulatan (delta ~ -1e-15).

_EPS = 1e-9
_STRATEGIES = ("best", "first")


# Helper bersama
def _prepare(initial_route, dist_matrix, max_iter, strategy):
    """Validasi input; kembalikan salinan tur awal (input tidak dimutasi)."""
    if strategy not in _STRATEGIES:
        raise ValueError(f"strategy harus salah satu dari {_STRATEGIES}.")

    n = len(dist_matrix)
    if any(len(row) != n for row in dist_matrix):
        raise ValueError("dist_matrix harus berbentuk persegi (n x n).")

    if (
        isinstance(max_iter, bool)
        or not isinstance(max_iter, numbers.Integral)
        or max_iter < 0
    ):
        raise ValueError("max_iter harus integer >= 0.")

    route = list(initial_route)
    # start/home = node pertama tur awal; tidak pernah dipindahkan
    validate_tour(route, n, home=route[0] if route else 0)
    return route


def _local_search(route, dist_matrix, max_iter, strategy, find_move, apply_move):
    first_improvement = strategy == "first"
    best_route = route
    best_distance = tour_distance(best_route, dist_matrix)
    history = [best_distance]  # history[0] = jarak tur awal

    for _ in range(int(max_iter)):
        move = find_move(best_route, dist_matrix, first_improvement)
        if move is None:
            break  # local optimum

        best_route = apply_move(best_route, move)
        best_distance = tour_distance(best_route, dist_matrix)
        history.append(best_distance)

    return {
        "route": best_route,
        "distance": best_distance,
        "history": history,
    }


# 2-opt
def _find_2opt_move(route, d, first_improvement=False):
    n = len(route) - 1
    best = None
    best_delta = -_EPS

    for i in range(1, n - 1):
        a, b = route[i - 1], route[i]
        for j in range(i + 1, n):
            c, e = route[j], route[j + 1]
            delta = d[a][c] + d[b][e] - d[a][b] - d[c][e]
            if delta < best_delta:
                best_delta, best = delta, (delta, i, j)
                if first_improvement:
                    return best
    return best


def _apply_2opt(route, move):
    _, i, j = move
    return route[:i] + route[i:j + 1][::-1] + route[j + 1:]


def two_opt(initial_route, dist_matrix, max_iter=100, strategy="best"):
    route = _prepare(initial_route, dist_matrix, max_iter, strategy)
    return _local_search(
        route, dist_matrix, max_iter, strategy, _find_2opt_move, _apply_2opt
    )


# 3-opt
def _find_3opt_move(route, d, first_improvement=False):
    n = len(route) - 1
    best = None
    best_delta = -_EPS

    for i in range(1, n - 1):
        a, b1 = route[i - 1], route[i]
        da_b1 = d[a][b1]
        for j in range(i + 1, n):
            b2, c1 = route[j - 1], route[j]
            removed_ab = da_b1 + d[b2][c1]
            for k in range(j + 1, n + 1):
                c2, e = route[k - 1], route[k]
                removed = removed_ab + d[c2][e]

                added = (
                    d[a][b2] + d[b1][c1] + d[c2][e],   # 0
                    d[a][b1] + d[b2][c2] + d[c1][e],   # 1
                    d[a][c2] + d[c1][b2] + d[b1][e],   # 2
                    d[a][b2] + d[b1][c2] + d[c1][e],   # 3
                    d[a][c1] + d[c2][b1] + d[b2][e],   # 4
                    d[a][c1] + d[c2][b2] + d[b1][e],   # 5
                    d[a][c2] + d[c1][b1] + d[b2][e],   # 6
                )

                for variant, cost in enumerate(added):
                    delta = cost - removed
                    if delta < best_delta:
                        best_delta = delta
                        best = (delta, i, j, k, variant)
                        if first_improvement:
                            return best
    return best


def _apply_3opt(route, move):
    _, i, j, k, variant = move
    A, B, C, D = route[:i], route[i:j], route[j:k], route[k:]
    if variant == 0:
        return A + B[::-1] + C + D
    if variant == 1:
        return A + B + C[::-1] + D
    if variant == 2:
        return A + C[::-1] + B[::-1] + D
    if variant == 3:
        return A + B[::-1] + C[::-1] + D
    if variant == 4:
        return A + C + B + D
    if variant == 5:
        return A + C + B[::-1] + D
    if variant == 6:
        return A + C[::-1] + B + D
    raise ValueError(f"variant tidak dikenal: {variant}")


def three_opt(initial_route, dist_matrix, max_iter=100, strategy="best"):
    route = _prepare(initial_route, dist_matrix, max_iter, strategy)
    return _local_search(
        route, dist_matrix, max_iter, strategy, _find_3opt_move, _apply_3opt
    )