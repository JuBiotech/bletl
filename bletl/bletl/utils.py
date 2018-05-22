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
        dfout[ncol] = dfin[ocol].astype(typ)
    return dfout


