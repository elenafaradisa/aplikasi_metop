# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# Import Library

# %% colab={"base_uri": "https://localhost:8080/", "height": 384} id="AgCZkt-3rCx-" outputId="bd2e7d4e-50ce-4433-dc0a-6d7b2ae62eef"
import pandas as pd
import random

from .base import build_distance_matrix, tour_distance

# Validasi Start Node

def validate_start_node(start_node, n):

    if not isinstance(start_node, int):
        raise TypeError(
            "start_node harus berupa integer."
        )

    if start_node < 0 or start_node >= n:
        raise ValueError(
            f"start_node harus berada pada rentang 0 sampai {n - 1}."
        )


# Nearest Neighbor

def nearest_neighbor(distance_matrix, start_node):

    n = len(distance_matrix)

    validate_start_node(
        start_node,
        n
    )

    route = [start_node]
    visited = {start_node}

    current_node = start_node

    while len(visited) < n:

        candidates = [
            node
            for node in range(n)
            if node not in visited
        ]

        next_node = min(
            candidates,
            key=lambda node: (
                distance_matrix[current_node][node],
                node
            )
        )

        route.append(next_node)
        visited.add(next_node)

        current_node = next_node

    # Kembali ke start node
    route.append(start_node)

    total_distance = tour_distance(
        route,
        distance_matrix
    )

    return route, total_distance


# Nearest Insertion

def nearest_insertion(distance_matrix, start_node):

    n = len(distance_matrix)

    validate_start_node(
        start_node,
        n
    )

    # Pilih node terdekat dari start node
    candidates = [
        node
        for node in range(n)
        if node != start_node
    ]

    nearest_node = min(
        candidates,
        key=lambda node: (
            distance_matrix[start_node][node],
            node
        )
    )

    # Initial closed tour
    route = [
        start_node,
        nearest_node,
        start_node
    ]

    unvisited = (
        set(range(n))
        - {start_node, nearest_node}
    )

    while unvisited:

        # Pilih node unvisited yang paling dekat
        # dengan current tour
        selected_node = min(
            unvisited,
            key=lambda node: (
                min(
                    distance_matrix[node][tour_node]
                    for tour_node in route[:-1]
                ),
                node
            )
        )

        best_position = None
        best_increase = float("inf")

        # Cari posisi insertion dengan
        # minimum tambahan jarak
        for i in range(len(route) - 1):

            node_i = route[i]
            node_j = route[i + 1]

            increase = (
                distance_matrix[node_i][selected_node]
                + distance_matrix[selected_node][node_j]
                - distance_matrix[node_i][node_j]
            )

            if (
                increase < best_increase
                or (
                    increase == best_increase
                    and (
                        best_position is None
                        or i + 1 < best_position
                    )
                )
            ):
                best_increase = increase
                best_position = i + 1

        route.insert(
            best_position,
            selected_node
        )

        unvisited.remove(selected_node)

    total_distance = tour_distance(
        route,
        distance_matrix
    )

    return route, total_distance


# Farthest Insertion

def farthest_insertion(distance_matrix, start_node):

    n = len(distance_matrix)

    validate_start_node(
        start_node,
        n
    )

    # Pilih node terjauh dari start node
    candidates = [
        node
        for node in range(n)
        if node != start_node
    ]

    farthest_node = max(
        candidates,
        key=lambda node: (
            distance_matrix[start_node][node],
            -node
        )
    )

    # Initial closed tour
    route = [
        start_node,
        farthest_node,
        start_node
    ]

    unvisited = (
        set(range(n))
        - {start_node, farthest_node}
    )

    while unvisited:

        # Pilih node yang paling jauh
        # dari current tour
        selected_node = max(
            unvisited,
            key=lambda node: (
                min(
                    distance_matrix[node][tour_node]
                    for tour_node in route[:-1]
                ),
                -node
            )
        )

        best_position = None
        best_increase = float("inf")

        # Cari posisi insertion dengan
        # minimum tambahan jarak
        for i in range(len(route) - 1):

            node_i = route[i]
            node_j = route[i + 1]

            increase = (
                distance_matrix[node_i][selected_node]
                + distance_matrix[selected_node][node_j]
                - distance_matrix[node_i][node_j]
            )

            if (
                increase < best_increase
                or (
                    increase == best_increase
                    and (
                        best_position is None
                        or i + 1 < best_position
                    )
                )
            ):
                best_increase = increase
                best_position = i + 1

        route.insert(
            best_position,
            selected_node
        )

        unvisited.remove(selected_node)

    total_distance = tour_distance(
        route,
        distance_matrix
    )

    return route, total_distance


# Arbitraty Insertion


def arbitrary_insertion(distance_matrix, start_node, seed=None):

    n = len(distance_matrix)

    validate_start_node(
        start_node,
        n
    )

    rng = random.Random(seed)

    # Pilih node kedua secara acak
    candidates = [
        node
        for node in range(n)
        if node != start_node
    ]

    second_node = rng.choice(candidates)

    # Initial closed tour
    route = [
        start_node,
        second_node,
        start_node
    ]

    unvisited = (
        set(range(n))
        - {start_node, second_node}
    )

    while unvisited:

        # Pilih node unvisited secara acak
        # (sorted agar hasil reproducible dengan seed)
        selected_node = rng.choice(
            sorted(unvisited)
        )

        best_position = None
        best_increase = float("inf")

        # Cari posisi insertion dengan
        # minimum tambahan jarak
        for i in range(len(route) - 1):

            node_i = route[i]
            node_j = route[i + 1]

            increase = (
                distance_matrix[node_i][selected_node]
                + distance_matrix[selected_node][node_j]
                - distance_matrix[node_i][node_j]
            )

            if (
                increase < best_increase
                or (
                    increase == best_increase
                    and (
                        best_position is None
                        or i + 1 < best_position
                    )
                )
            ):
                best_increase = increase
                best_position = i + 1

        route.insert(
            best_position,
            selected_node
        )

        unvisited.remove(selected_node)

    total_distance = tour_distance(
        route,
        distance_matrix
    )

    return route, total_distance

