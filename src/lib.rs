use pyo3::class::basic::CompareOp;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::{HashMap, HashSet};

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
    prefixes: Vec<String>,
    key_col: String,
    target_col: String,
) -> PyResult<Py<PyAny>> {
    let pl = py.import("polars")?;
    let cycle_start = prefixes[0].clone();
    let sort_key = PyList::empty(py);
    for col in &group_cols {
        sort_key.append(col)?;
    }
    sort_key.append(target_col.as_str())?;
    let sorted_df = df.call_method1("sort", (sort_key,))?;

    // 2. 建立 cycle flag:
    // (pl.col(key_col) == pl.lit(cycle_start)).cast(pl.Int64)
    let key_expr = pl.getattr("col")?.call1((key_col.as_str(),))?;
    let cycle_lit = pl.getattr("lit")?.call1((cycle_start.as_str(),))?;
    let cycle_flag = key_expr
        .call_method1("eq", (cycle_lit,))?
        .call_method1("cast", (pl.getattr("Int64")?,))?;
    let over_cols = PyList::empty(py);
    for col in &group_cols {
        over_cols.append(col)?;
    }

    let cycle_expr = cycle_flag
        .call_method0("cum_sum")?
        .call_method1("over", (over_cols,))?
        .call_method1("alias", ("_cycle_id",))?;
    let with_expr = PyList::empty(py);
    with_expr.append(cycle_expr)?;

    let working_df = sorted_df.call_method1("with_columns", (with_expr,))?;

    let new_group_cols = PyList::empty(py);
    for col in &group_cols {
        new_group_cols.append(col)?;
    }
    let _ = new_group_cols.append("_cycle_id");

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
    let grouped = working_df.call_method1("group_by", (new_group_cols,))?;
    let result = grouped.call_method1("agg", (exprs,))?;
    Ok(result.unbind())
}

#[derive(Clone, Copy)]
enum Plan {
    Backward,
    Forward,
}

// 判斷要用哪個plan
fn parse_plan(plan: &str) -> PyResult<Plan> {
    match plan {
        "backward" => Ok(Plan::Backward),
        "forward" => Ok(Plan::Forward),
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "plan must be either 'backward' or 'forward'",
        )),
    }
}

fn key_from_row(row: &Bound<'_, PyDict>, key_col: &str) -> PyResult<String> {
    let value = row.get_item(key_col)?.ok_or_else(|| {
        PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!("missing key column: {key_col}"))
    })?;
    value.extract::<String>()
}

fn matches_kv(row: &Bound<'_, PyDict>, kv_map: &HashMap<String, String>) -> PyResult<bool> {
    for (k, v) in kv_map {
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

fn earliest_blocker_on_node_after_target(
    py: Python<'_>,
    rows: &[Py<PyDict>],
    self_index_all: &HashMap<String, Vec<usize>>,
    node_key: &str,
    ref_target: &Bound<'_, PyAny>,
    target_col: &str,
    block_key: &HashMap<String, String>,
) -> PyResult<Option<Py<PyAny>>> {
    let Some(indices) = self_index_all.get(node_key) else {
        return Ok(None);
    };

    let mut earliest: Option<Py<PyAny>> = None;

    for idx in indices {
        let row = rows[*idx].bind(py);
        if !matches_kv(&row, block_key)? {
            continue;
        }

        let Some(block_target) = row.get_item(target_col)? else {
            continue;
        };
        if !is_valid_target(Some(block_target.clone())) {
            continue;
        }

        if !compare_bool(&block_target, ref_target, CompareOp::Gt) {
            continue;
        }

        let should_take = match &earliest {
            None => true,
            Some(current) => compare_bool(&block_target, current.bind(py), CompareOp::Lt),
        };
        if should_take {
            earliest = Some(block_target.unbind());
        }
    }

    Ok(earliest)
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
///
/// block_key:
/// - Optional key/value matcher applied at node level.
/// - For each active row, blocker search starts on self node and continues on traversed
///   parent nodes.
/// - Blockers are filtered by target_col > anchor target, then the earliest blocker among
///   visited nodes is tracked.
/// - Before accepting a matched target candidate, compare candidate target_col vs blocker:
///   if target is earlier than blocker, keep target; otherwise break and keep original value.
fn find_target_rows(
    py: Python<'_>,
    rows: &[Py<PyDict>],
    self_key_col: &str,
    parent_key_col: &str,
    target_col: &str,
    target_key: &HashMap<String, String>,
    block_key: Option<&HashMap<String, String>>,
    stop_col: &str,
    stop_val: &str,
    plan: Plan,
) -> PyResult<()> {
    let mut self_index_all: HashMap<String, Vec<usize>> = HashMap::new();
    for (i, row) in rows.iter().enumerate() {
        let key = key_from_row(&row.bind(py), self_key_col)?;
        self_index_all.entry(key).or_default().push(i);
    }

    let mut active_indices = Vec::new();
    for (i, row) in rows.iter().enumerate() {
        if matches_kv(&row.bind(py), target_key)? {
            active_indices.push(i);
        }
    }

    let mut self_index: HashMap<String, Vec<usize>> = HashMap::new();
    for i in &active_indices {
        let row = rows[*i].bind(py);
        let key = key_from_row(&row, self_key_col)?;
        self_index.entry(key).or_default().push(*i);
    }

    for start_idx in active_indices {
        let start_row = rows[start_idx].bind(py);
        let Some(start_target) = start_row.get_item(target_col)? else {
            continue;
        };
        if !is_valid_target(Some(start_target.clone())) {
            continue;
        }
        let anchor_target = start_target.clone();
        let start_self = key_from_row(&start_row, self_key_col)?;
        let mut earliest_blocker: Option<Py<PyAny>> = None;

        if let Some(kv) = block_key {
            if let Some(blocker) = earliest_blocker_on_node_after_target(
                py,
                rows,
                &self_index_all,
                &start_self,
                &anchor_target,
                target_col,
                kv,
            )? {
                earliest_blocker = Some(blocker);
            }
        }

        let mut current_parent = key_from_row(&start_row, parent_key_col)?;
        let mut current_target = start_target;
        let mut visited = HashSet::new();
        let mut replacement: Option<Py<PyAny>> = None;

        loop {
            if !visited.insert(current_parent.clone()) {
                break;
            }

            if let Some(kv) = block_key {
                if let Some(node_blocker) = earliest_blocker_on_node_after_target(
                    py,
                    rows,
                    &self_index_all,
                    &current_parent,
                    &anchor_target,
                    target_col,
                    kv,
                )? {
                    let should_take = match &earliest_blocker {
                        None => true,
                        Some(current) => {
                            compare_bool(node_blocker.bind(py), current.bind(py), CompareOp::Lt)
                        }
                    };
                    if should_take {
                        earliest_blocker = Some(node_blocker);
                    }
                }
            }

            let Some(candidates) = self_index.get(&current_parent) else {
                break;
            };

            let Some(chosen_idx) =
                choose_candidate_by_plan(py, rows, candidates, target_col, &current_target, plan)?
            else {
                break;
            };

            let candidate = rows[chosen_idx].bind(py);
            let Some(candidate_target) = candidate.get_item(target_col)? else {
                break;
            };
            if !is_valid_target(Some(candidate_target.clone())) {
                break;
            }

            let para_val = candidate
                .get_item(stop_col)?
                .and_then(|x| x.extract::<String>().ok());
            if para_val
                .as_deref()
                .is_some_and(|val| val.starts_with(stop_val))
            {
                let target_before_block = match &earliest_blocker {
                    None => true,
                    Some(blocker_target) => {
                        compare_bool(&candidate_target, blocker_target.bind(py), CompareOp::Lt)
                    }
                };
                if target_before_block {
                    replacement = Some(candidate_target.unbind());
                }
                break;
            }

            current_parent = key_from_row(&candidate, parent_key_col)?;
            current_target = candidate_target;
        }

        if let Some(new_value) = replacement {
            rows[start_idx]
                .bind(py)
                .set_item(target_col, new_value.bind(py))?;
        }
    }

    Ok(())
}

#[pyfunction]
#[pyo3(signature = (df, self_cols, parent_cols, target_col, target_key, stop_point, plan, block_key=None))]
fn find_target_value_from_ancestor(
    py: Python<'_>,
    df: &Bound<'_, PyAny>,
    self_cols: Vec<String>,
    parent_cols: Vec<String>,
    target_col: String,
    target_key: HashMap<String, String>,
    stop_point: HashMap<String, String>,
    plan: String,
    block_key: Option<HashMap<String, String>>,
) -> PyResult<Py<PyAny>> {
    // 檢查key col數量是否一樣
    if self_cols.len() != parent_cols.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "self_cols and parent_cols must have the same amount of arugments",
        ));
    }
    // 檢查是否只有一個stopPoint
    if stop_point.len() != 1 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "stop_point must contain exactly one key/value pair",
        ));
    }
    // 判斷backward || forward
    let plan = parse_plan(&plan)?;
    // 取得stopPoint HashMap裡面的col跟value
    let (stop_col, stop_val) = stop_point.iter().next().expect("check len=1");
    let pl = py.import("polars")?;

    // 將key_col合併的closure function
    // 等於:
    // pl.concat_str(
    //     [pl.col("serial").cast(pl.String).fill_null("<null>"), pl.col("part").cast(pl.String).fill_null("<null>")],
    //     separator="##"
    // )
    let build_key_expr = |cols: &[String]| -> PyResult<Py<PyAny>> {
        let exprs = PyList::empty(py);
        for col in cols {
            let expr = pl
                .getattr("col")?
                .call1((col.as_str(),))?
                .call_method1("cast", (pl.getattr("String")?,))?
                .call_method1("fill_null", ("<null>",))?;
            exprs.append(expr)?;
        }
        let kwargs = PyDict::new(py);
        kwargs.set_item("separator", "##")?;
        let key_expr = pl.getattr("concat_str")?.call((exprs,), Some(&kwargs))?;
        Ok(key_expr.unbind())
    };

    // 新增欄位名稱
    let self_key_col = "__self_key";
    let parent_key_col = "__parent_key";

    // 套用closeure function
    let self_key_expr = build_key_expr(&self_cols)?
        .bind(py)
        .call_method1("alias", (self_key_col,))?
        .unbind();
    let parent_key_expr = build_key_expr(&parent_cols)?
        .bind(py)
        .call_method1("alias", (parent_key_col,))?
        .unbind();
    // 把function加到PyList
    let key_exprs = PyList::empty(py);
    key_exprs.append(self_key_expr.bind(py))?;
    key_exprs.append(parent_key_expr.bind(py))?;
    // 新增欄位
    let df_with_keys = df.call_method1("with_columns", (key_exprs,))?;

    // 把dataframe轉 to_dicts=>[dict, dict]的型別再轉list
    let row_list = df_with_keys
        .call_method0("to_dicts")?
        .cast_into::<PyList>()?;

    // 用with_capacity先設定好內存(Heap)空間
    let mut rows: Vec<Py<PyDict>> = Vec::with_capacity(row_list.len());
    // 把row_list的每一個item都加到rows
    for item in row_list.iter() {
        rows.push(item.cast_into::<PyDict>()?.unbind());
    }

    // 呼叫function
    find_target_rows(
        py,
        &rows,
        self_key_col,
        parent_key_col,
        target_col.as_str(),
        &target_key,
        block_key.as_ref(),
        stop_col,
        stop_val,
        plan,
    )?;

    // 把row_list轉回DataFrame，並drop cols
    let result = pl.getattr("DataFrame")?.call1((row_list,))?;
    let result = result.call_method1("drop", ((self_key_col, parent_key_col),))?;
    Ok(result.unbind())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rows_with_keys(py: Python<'_>, rows_data: Vec<[(&str, &str); 5]>) -> Vec<Py<PyDict>> {
        rows_data
            .into_iter()
            .map(|pairs| {
                let d = PyDict::new(py);
                let mut id_val: Option<&str> = None;
                let mut parent_val: Option<&str> = None;
                for (k, v) in pairs {
                    if k == "id" {
                        id_val = Some(v);
                    }
                    if k == "parent_id" {
                        parent_val = Some(v);
                    }
                    d.set_item(k, v).expect("set test item");
                }
                d.set_item("__self_key", id_val.expect("id exists"))
                    .expect("set self key");
                d.set_item("__parent_key", parent_val.expect("parent_id exists"))
                    .expect("set parent key");
                d.unbind()
            })
            .collect()
    }

    #[test]
    fn blocker_terminates_chain_resolution() {
        Python::attach(|py| {
            let rows: Vec<Py<PyDict>> = rows_with_keys(
                py,
                vec![
                    [
                        ("id", "a1"),
                        ("parent_id", "root"),
                        ("event", "Install"),
                        ("kind", "target"),
                        ("ts", "1"),
                    ],
                    [
                        ("id", "a2"),
                        ("parent_id", "a1"),
                        ("event", "Remove"),
                        ("kind", "target"),
                        ("ts", "2"),
                    ],
                    [
                        ("id", "a3"),
                        ("parent_id", "a2"),
                        ("event", "Remove"),
                        ("kind", "other"),
                        ("ts", "3"),
                    ],
                    [
                        ("id", "leaf"),
                        ("parent_id", "a3"),
                        ("event", "Other"),
                        ("kind", "target"),
                        ("ts", "4"),
                    ],
                ],
            );

            let target_key = HashMap::from([("kind".to_string(), "target".to_string())]);
            let stop_col = "event";
            let stop_val = "Install";

            find_target_rows(
                py,
                &rows,
                "__self_key",
                "__parent_key",
                "ts",
                &target_key,
                None,
                stop_col,
                stop_val,
                Plan::Backward,
            )
            .expect("propagation without blocker");

            let leaf_ts_no_block: String = rows[3]
                .bind(py)
                .get_item("ts")
                .expect("leaf ts item")
                .expect("leaf ts exists")
                .extract()
                .expect("leaf ts string");
            assert_eq!(leaf_ts_no_block, "3");

            rows[3].bind(py).set_item("ts", "4").expect("reset leaf ts");

            let blocker = HashMap::from([("event".to_string(), "Remove".to_string())]);
            find_target_rows(
                py,
                &rows,
                "__self_key",
                "__parent_key",
                "ts",
                &target_key,
                Some(&blocker),
                stop_col,
                stop_val,
                Plan::Backward,
            )
            .expect("propagation with blocker");

            let leaf_ts_blocked: String = rows[3]
                .bind(py)
                .get_item("ts")
                .expect("leaf ts item")
                .expect("leaf ts exists")
                .extract()
                .expect("leaf ts string");
            assert_eq!(leaf_ts_blocked, "4");
        });
    }

    #[test]
    fn node_level_self_blocker_prevents_propagation() {
        Python::attach(|py| {
            let rows: Vec<Py<PyDict>> = rows_with_keys(
                py,
                vec![
                    [
                        ("id", "parent"),
                        ("parent_id", "root"),
                        ("event", "Install"),
                        ("kind", "target"),
                        ("ts", "1"),
                    ],
                    [
                        ("id", "leaf"),
                        ("parent_id", "parent"),
                        ("event", "Install"),
                        ("kind", "target"),
                        ("ts", "2"),
                    ],
                    [
                        ("id", "leaf"),
                        ("parent_id", "parent"),
                        ("event", "Remove"),
                        ("kind", "other"),
                        ("ts", "3"),
                    ],
                ],
            );

            let target_key = HashMap::from([("kind".to_string(), "target".to_string())]);
            let blocker = HashMap::from([("event".to_string(), "Remove".to_string())]);

            find_target_rows(
                py,
                &rows,
                "__self_key",
                "__parent_key",
                "ts",
                &target_key,
                Some(&blocker),
                "event",
                "Install",
                Plan::Backward,
            )
            .expect("propagation with self-history blocker");

            let leaf_install_ts: String = rows[1]
                .bind(py)
                .get_item("ts")
                .expect("leaf install ts item")
                .expect("leaf install ts exists")
                .extract()
                .expect("leaf install ts string");
            assert_eq!(leaf_install_ts, "2");
        });
    }

    #[test]
    fn node_level_self_blocker_not_later_does_not_block() {
        Python::attach(|py| {
            let rows: Vec<Py<PyDict>> = rows_with_keys(
                py,
                vec![
                    [
                        ("id", "parent"),
                        ("parent_id", "root"),
                        ("event", "Install"),
                        ("kind", "target"),
                        ("ts", "1"),
                    ],
                    [
                        ("id", "leaf"),
                        ("parent_id", "parent"),
                        ("event", "Install"),
                        ("kind", "target"),
                        ("ts", "2"),
                    ],
                    [
                        ("id", "leaf"),
                        ("parent_id", "parent"),
                        ("event", "Remove"),
                        ("kind", "other"),
                        ("ts", "1"),
                    ],
                ],
            );

            let target_key = HashMap::from([("kind".to_string(), "target".to_string())]);
            let blocker = HashMap::from([("event".to_string(), "Remove".to_string())]);

            find_target_rows(
                py,
                &rows,
                "__self_key",
                "__parent_key",
                "ts",
                &target_key,
                Some(&blocker),
                "event",
                "Install",
                Plan::Backward,
            )
            .expect("propagation with non-later self blocker");

            let leaf_install_ts: String = rows[1]
                .bind(py)
                .get_item("ts")
                .expect("leaf install ts item")
                .expect("leaf install ts exists")
                .extract()
                .expect("leaf install ts string");
            assert_eq!(leaf_install_ts, "1");
        });
    }

    #[test]
    fn node_level_parent_blocker_prevents_propagation() {
        Python::attach(|py| {
            let rows: Vec<Py<PyDict>> = rows_with_keys(
                py,
                vec![
                    [
                        ("id", "ancestor"),
                        ("parent_id", "root"),
                        ("event", "Install"),
                        ("kind", "target"),
                        ("ts", "9"),
                    ],
                    [
                        ("id", "parent"),
                        ("parent_id", "ancestor"),
                        ("event", "Install"),
                        ("kind", "target"),
                        ("ts", "8"),
                    ],
                    [
                        ("id", "parent"),
                        ("parent_id", "ancestor"),
                        ("event", "Remove"),
                        ("kind", "other"),
                        ("ts", "10"),
                    ],
                    [
                        ("id", "leaf"),
                        ("parent_id", "parent"),
                        ("event", "Install"),
                        ("kind", "target"),
                        ("ts", "7"),
                    ],
                ],
            );

            let target_key = HashMap::from([("kind".to_string(), "target".to_string())]);
            let blocker = HashMap::from([("event".to_string(), "Remove".to_string())]);

            find_target_rows(
                py,
                &rows,
                "__self_key",
                "__parent_key",
                "ts",
                &target_key,
                Some(&blocker),
                "event",
                "Install",
                Plan::Backward,
            )
            .expect("propagation with parent-node blocker");

            let leaf_ts: String = rows[3]
                .bind(py)
                .get_item("ts")
                .expect("leaf ts item")
                .expect("leaf ts exists")
                .extract()
                .expect("leaf ts string");
            assert_eq!(leaf_ts, "7");
        });
    }

    #[test]
    fn blocker_checks_use_start_target_as_anchor_across_ancestors() {
        Python::attach(|py| {
            let rows: Vec<Py<PyDict>> = rows_with_keys(
                py,
                vec![
                    [
                        ("id", "gp"),
                        ("parent_id", "root"),
                        ("event", "Install"),
                        ("kind", "target"),
                        ("ts", "2022-01-01T00:00:00.000Z"),
                    ],
                    [
                        ("id", "gp"),
                        ("parent_id", "root"),
                        ("event", "Remove"),
                        ("kind", "other"),
                        ("ts", "2023-08-01T00:00:00.000Z"),
                    ],
                    [
                        ("id", "p"),
                        ("parent_id", "gp"),
                        ("event", "Other"),
                        ("kind", "target"),
                        ("ts", "2023-07-15T12:03:00.000Z"),
                    ],
                    [
                        ("id", "p"),
                        ("parent_id", "gp"),
                        ("event", "Remove"),
                        ("kind", "other"),
                        ("ts", "2023-07-15T12:00:00.000Z"),
                    ],
                    [
                        ("id", "leaf"),
                        ("parent_id", "p"),
                        ("event", "Other"),
                        ("kind", "target"),
                        ("ts", "2024-02-05T16:30:00.000Z"),
                    ],
                ],
            );

            let target_key = HashMap::from([("kind".to_string(), "target".to_string())]);
            let blocker = HashMap::from([("event".to_string(), "Remove".to_string())]);

            find_target_rows(
                py,
                &rows,
                "__self_key",
                "__parent_key",
                "ts",
                &target_key,
                Some(&blocker),
                "event",
                "Install",
                Plan::Backward,
            )
            .expect("propagation with anchored blocker checks");

            let leaf_ts: String = rows[4]
                .bind(py)
                .get_item("ts")
                .expect("leaf ts item")
                .expect("leaf ts exists")
                .extract()
                .expect("leaf ts string");
            assert_eq!(leaf_ts, "2022-01-01T00:00:00.000Z");
        });
    }
}

#[pymodule]
fn hello_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(pivot_by_prefix, m)?)?;
    m.add_function(wrap_pyfunction!(find_target_value_from_ancestor, m)?)?;
    Ok(())
}
