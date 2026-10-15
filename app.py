
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

            st.header(
                "Results"
            )

            spacer(0.5)


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

                st.header(
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
