use pyo3::class::basic::CompareOp;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::{HashMap, HashSet};

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

#[derive(Clone, Copy)]
enum Plan {
    Backward,
    Forward,
}

fn parse_plan(plan: &str) -> PyResult<Plan> {
    match plan {
        "backward" => Ok(Plan::Backward),
        "forward" => Ok(Plan::Forward),
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "plan must be either 'backward' or 'forward'",
        )),
    }
}

fn value_to_key(value: Option<Bound<'_, PyAny>>) -> PyResult<String> {
    match value {
        None => Ok("<missing>".to_string()),
        Some(v) if v.is_none() => Ok("<none>".to_string()),
        Some(v) => {
            if let Ok(s) = v.extract::<String>() {
                Ok(format!("S:{s}"))
            } else {
                Ok(format!("R:{}", v.str()?))
            }
        }
    }
}

fn composite_key(row: &Bound<'_, PyDict>, cols: &[String]) -> PyResult<String> {
    let mut parts = Vec::with_capacity(cols.len());
    for col in cols {
        let val = row.get_item(col)?;
        parts.push(value_to_key(val)?);
    }
    Ok(parts.join("\u{1F}"))
}

fn matches_target_key(
    row: &Bound<'_, PyDict>,
    target_key: &HashMap<String, String>,
) -> PyResult<bool> {
    for (k, v) in target_key {
        let actual = row.get_item(k)?.and_then(|x| x.extract::<String>().ok());
        if actual.as_deref() != Some(v.as_str()) {
            return Ok(false);
        }
    }
    Ok(true)
}

fn compare_bool(left: &Bound<'_, PyAny>, right: &Bound<'_, PyAny>, op: CompareOp) -> bool {
    left.rich_compare(right, op)
        .and_then(|result| result.is_truthy())
        .unwrap_or(false)
}

fn is_valid_target(value: Option<Bound<'_, PyAny>>) -> bool {
    match value {
        None => false,
        Some(v) => !v.is_none(),
    }
}

fn choose_candidate_by_plan(
    py: Python<'_>,
    rows: &[Py<PyDict>],
    candidates: &[usize],
    target_col: &str,
    ref_target: &Bound<'_, PyAny>,
    plan: Plan,
) -> PyResult<Option<usize>> {
    let mut best_idx: Option<usize> = None;
    let mut best_val: Option<Py<PyAny>> = None;

    for idx in candidates {
        let row = rows[*idx].bind(py);
        let Some(val) = row.get_item(target_col)? else {
            continue;
        };
        if val.is_none() {
            continue;
        }

        let in_direction = match plan {
            Plan::Backward => compare_bool(&val, ref_target, CompareOp::Le),
            Plan::Forward => compare_bool(&val, ref_target, CompareOp::Ge),
        };
        if !in_direction {
            continue;
        }

        match &best_val {
            None => {
                best_idx = Some(*idx);
                best_val = Some(val.unbind());
            }
            Some(current_best) => {
                let better = match plan {
                    Plan::Backward => compare_bool(&val, current_best.bind(py), CompareOp::Gt),
                    Plan::Forward => compare_bool(&val, current_best.bind(py), CompareOp::Lt),
                };
                if better {
                    best_idx = Some(*idx);
                    best_val = Some(val.unbind());
                }
            }
        }
    }

    Ok(best_idx)
}

/// For each row, walk through parent links (single-path DFS over ancestor chain) until a parent
/// column starts with the configured prefix, then replace that row's target column with the matched
/// ancestor row's target value.
///
/// plan:
/// - "backward": choose the nearest candidate with target_col <= current node target_col.
/// - "forward": choose the nearest candidate with target_col >= current node target_col.
///
/// If no valid candidate is found, the original value is kept unchanged.
#[pyfunction]
fn propagate_target_from_ancestor(
    py: Python<'_>,
    df: &Bound<'_, PyAny>,
    self_cols: Vec<String>,
    parent_cols: Vec<String>,
    target_col: String,
    target_key: HashMap<String, String>,
    para: HashMap<String, String>,
    plan: String,
) -> PyResult<Py<PyAny>> {
    if self_cols.len() != parent_cols.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "self_cols and parent_cols must have the same length",
        ));
    }
    if para.len() != 1 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "para must contain exactly one key/value pair",
        ));
    }

    let plan = parse_plan(&plan)?;
    let (para_col, para_prefix) = para.iter().next().expect("checked len=1");
    let rows_list = df.call_method0("to_dicts")?.cast_into::<PyList>()?;

    let mut rows: Vec<Py<PyDict>> = Vec::with_capacity(rows_list.len());
    for item in rows_list.iter() {
        rows.push(item.cast_into::<PyDict>()?.unbind());
    }

    let mut active_indices = Vec::new();
    for (i, row) in rows.iter().enumerate() {
        if matches_target_key(&row.bind(py), &target_key)? {
            active_indices.push(i);
        }
    }

    let mut self_index: HashMap<String, Vec<usize>> = HashMap::new();
    for i in &active_indices {
        let row = rows[*i].bind(py);
        let key = composite_key(&row, &self_cols)?;
        self_index.entry(key).or_default().push(*i);
    }

    for start_idx in active_indices {
        let start_row = rows[start_idx].bind(py);
        let Some(start_target) = start_row.get_item(&target_col)? else {
            continue;
        };
        if !is_valid_target(Some(start_target.clone())) {
            continue;
        }

        let mut current_parent = composite_key(&start_row, &parent_cols)?;
        let mut current_target = start_target;
        let mut visited = HashSet::new();
        let mut replacement: Option<Py<PyAny>> = None;

        loop {
            if !visited.insert(current_parent.clone()) {
                break;
            }

            let Some(candidates) = self_index.get(&current_parent) else {
                break;
            };

            let Some(chosen_idx) = choose_candidate_by_plan(
                py,
                &rows,
                candidates,
                &target_col,
                &current_target,
                plan,
            )?
            else {
                break;
            };

            let candidate = rows[chosen_idx].bind(py);
            let Some(candidate_target) = candidate.get_item(&target_col)? else {
                break;
            };
            if !is_valid_target(Some(candidate_target.clone())) {
                break;
            }

            let para_val = candidate
                .get_item(para_col)?
                .and_then(|x| x.extract::<String>().ok());
            if para_val
                .as_deref()
                .is_some_and(|val| val.starts_with(para_prefix))
            {
                replacement = Some(candidate_target.unbind());
                break;
            }

            current_parent = composite_key(&candidate, &parent_cols)?;
            current_target = candidate_target;
        }

        if let Some(new_value) = replacement {
            rows[start_idx]
                .bind(py)
                .set_item(target_col.as_str(), new_value.bind(py))?;
        }
    }

    let pl = py.import("polars")?;
    let result = pl.getattr("DataFrame")?.call1((rows_list,))?;
    Ok(result.unbind())
}

#[pymodule]
fn hello_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(add, m)?)?;
    m.add_function(wrap_pyfunction!(pivot_by_prefix, m)?)?;
    m.add_function(wrap_pyfunction!(propagate_target_from_ancestor, m)?)?;
    Ok(())
}
