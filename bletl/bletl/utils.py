"""Contains helper functions that do not depend on other modules within this package."""
import pandas


def __to_typed_cols__(dfin:pandas.DataFrame, ocol_ncol_type:list):
    """Can be used to filter & convert data frame columns.

    Args:
        dfin: raw data frame to start from
        ocol_ncol_type: list of tuples ('original column', 'new column', datatype)

    Returns:
        DataFrame
    """
    dfout = pandas.DataFrame()
    for ocol, ncol, typ in ocol_ncol_type:
        if ocol is None or not ocol in dfin:
            dfout[ncol] = None
        else:
            dfout[ncol] = dfin[ocol].astype(typ)
    return dfout


def _unindex(dataframe):
    """Resets the index of the DataFrame.

    Args:
        dataframe (pandas.DataFrame): guess what

    Returns:
        index_names (FrozenList): all the index names. May be (None,)
        dataframe (pandas.DataFrame): frame without indices
    """
    return dataframe.index.names, dataframe.reset_index()


def _reindex(dataframe:pandas.DataFrame, index_names) -> pandas.DataFrame:
    """Applies an indexing scheme to a DataFrame

    Args:
        dataframe (pandas.DataFrame): guess what
        index_names (tuple): all the index names. May be (None,)

    Returns:
        dataframe (pandas.DataFrame): frame with the indexing scheme
    """
    if index_names[0] is not None:
        return dataframe.set_index(index_names)
    else:
        return dataframe


def _concatenate_fragments(fragments:list, start_times:list) -> pandas.DataFrame:
    """Concatenate multiple dataframes while shifting time and cycles.

    Args:
        fragments (list): list of DataFrames to concatenate
        start_times (list): list of the experiment start times for each data fragment

    Returns:
        dataframe (pandas.DataFrame): time/cycle-aware concatenation of fragments
    """
    index_names, stack = _unindex(fragments[0])
    columns = set(stack.columns)

    for fragment, fragment_start in zip(fragments[1:], start_times[1:]):
        assert isinstance(fragment, pandas.DataFrame), 'fragments must be a list of DataFrames'
        index_names_f, fragment = _unindex(fragment)
        assert set(index_names_f) == set(index_names), 'indices must match across all fragments'
        assert set(fragment.columns) == columns, 'columns must match across all fragments'

        # shift time and cycle columns in the fragment
        if 'time' in columns:
            fragment['time'] += (fragment_start - start_times[0]).total_seconds() / 3600
        if 'cycle' in columns:
            fragment['cycle'] += max(stack['cycle'])

        # append the fragment to the stack
        stack = pandas.concat((stack, fragment))

    # re-apply the original indexing scheme
    return _reindex(stack, index_names)
