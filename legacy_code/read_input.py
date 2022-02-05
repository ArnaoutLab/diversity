def read_shared_features(
    filepath, manager, species_column, species_order, species_subset=None, **kwargs
):
    """Reads species features into a shareable list."""
    LOGGER.debug(
        "read_features(%s, %s, %s, %s, species_subset=%s, kwargs=%s)",
        filepath,
        manager,
        species_column,
        species_order,
        species_subset,
        kwargs,
    )
    delimiter = get_file_delimiter(filepath)
    if species_subset is None:
        skip_row = None
    else:
        skip_row = lambda x: x not in species_subset
    return manager.ShareableList(
        read_csv(
            filepath,
            sep=delimiter,
            index_col=species_column,
            skiprows=skip_row,
            **kwargs
        )[species_order].to_numpy()
    )


def read_shared_counts(
    filepath,
    manager,
    subcommunity_column,
    species_column,
    count_column,
    species_ordering,
):
    """Reads and pivots counts and returns it as a shared array."""
    LOGGER.debug(
        "read_counts(%s, %s, %s, %s, %s)",
        filepath,
        manager,
        subcommunity_column,
        species_column,
        count_column,
    )
    delimiter = get_file_delimiter(filepath)
    counts = pivot_table(
        read_csv(
            filepath,
            sep=delimiter,
            usecols=[subcommunity_column, species_column, count_column],
        ),
        subcommunity_column,
        species_column,
        [count_column],
        index_ordering=species_ordering,
    )
    shared_counts = SharedArrayView(
        manager.empty_spec(shape=counts.shape, dtype=dtype("f8"))
    )
    shared_counts[:] = counts
    return shared_counts
