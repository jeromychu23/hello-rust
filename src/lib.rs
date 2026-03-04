use pyo3::prelude::*;
use pyo3::types::PyList;

#[pyfunction]
fn add(a: i64, b: i64) -> PyResult<i64> {
    Ok(a + b)
}

/// Pivot a long-form dataframe into one row per group with one output column per prefix.
///
/// Parameters
/// ----------
/// df:
///     A Polars DataFrame.
/// group_cols:
///     Columns used to group rows (supports multiple columns).
/// target_col:
///     The value column to extract into new output columns.
/// prefixes:
///     Keyword values to split on.
/// key_col:
///     The column that contains keyword values.
///
/// Example
/// -------
/// group_cols=["sn", "pn"], target_col="date", prefixes=["Install", "Remove"], key_col="type"
/// -> output columns: Install_date, Remove_date
#[pyfunction]
fn pivot_by_prefix(
    py: Python<'_>,
    df: &Bound<'_, PyAny>,
    group_cols: Vec<String>,
    target_col: String,
    prefixes: Vec<String>,
    key_col: String,
) -> PyResult<Py<PyAny>> {
    let pl = py.import("polars")?;
    let exprs = PyList::empty(py);

    for prefix in prefixes {
        let target_expr = pl.getattr("col")?.call1((target_col.as_str(),))?;
        let key_expr = pl.getattr("col")?.call1((key_col.as_str(),))?;
        let keyword = pl.getattr("lit")?.call1((prefix.as_str(),))?;
        let condition = key_expr.call_method1("eq", (keyword,))?;

        let alias_name = format!("{}_{}", prefix, target_col);
        let agg_expr = target_expr
            .call_method1("filter", (condition,))?
            .call_method0("first")?
            .call_method1("alias", (alias_name,))?;

        exprs.append(agg_expr)?;
    }

    let grouped = df.call_method1("group_by", (group_cols,))?;
    let result = grouped.call_method1("agg", (exprs,))?;
    Ok(result.unbind())
}

#[pymodule]
fn hello_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(add, m)?)?;
    m.add_function(wrap_pyfunction!(pivot_by_prefix, m)?)?;
    Ok(())
}
