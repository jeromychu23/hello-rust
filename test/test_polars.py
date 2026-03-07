from pathlib import Path

import polars as pl

import hello_rust


def _normalize(df: pl.DataFrame) -> pl.DataFrame:
    return df.sort(["id", "ts"]).select(sorted(df.columns))


def test_propagate_target_from_ancestor_from_local_csv() -> None:
    csv_path = Path(__file__).parent / "data" / "pivot_input.csv"
    source_df = pl.read_csv(csv_path, try_parse_dates=False)
    
    source_df = (
        source_df
        .with_columns(pl.col("date").alias("install_date"))
        .with_columns(pl.col("date").alias("remove_date"))
    )

    result_df = hello_rust.propagate_target_from_ancestor(
        source_df,
        self_cols=["id"],
        parent_cols=["parent_id"],
        target_col="install_date",
        target_key={"event": "Install"},
        para={"parent_id": "parent0"},
        plan="forward",
        block_key={"event": "Remove"},
    )
    
    result_df = hello_rust.propagate_target_from_ancestor(
        result_df,
        self_cols=["id"],
        parent_cols=["parent_id"],
        target_col="remove_date",
        target_key={"event": "Remove"},
        para={"event": "Remove"},
        plan="backward",
        block_key={"event": "Install"},
    )

    # expected_df = source_df.with_columns(
    #     pl.when(pl.col("id") == "leaf").then(pl.lit(10)).otherwise(pl.col("ts")).alias("ts")
    # )

    print("result_df:")
    try:
        print(result_df)
    except UnicodeEncodeError:
        print(result_df.to_dicts())

    # print("expected_df:")
    # try:
    #     print(expected_df)
    # except UnicodeEncodeError:
    #     print(expected_df.to_dicts())

    # compare removed for now; print output only
    # test command .\.venv\Scripts\python -m pytest -s -q test/test_polars.py
