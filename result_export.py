"""Build the final Excel export table for avatar usage-rate results."""

from __future__ import annotations

import pandas as pd


EXPORT_COLUMNS = ["排名", "角色中文名", "使用率", "使用率文本"]


def build_excel_export_dataframe(result_df: pd.DataFrame) -> pd.DataFrame:
    """Return the user-facing four-column Excel table sorted by usage rate."""
    if result_df.empty:
        return pd.DataFrame(columns=EXPORT_COLUMNS)

    working = result_df.copy()
    if "使用率数字" not in working.columns:
        working["使用率数字"] = pd.NA
    if "使用率文本" not in working.columns:
        working["使用率文本"] = ""
    if "原图顺序" not in working.columns:
        working["原图顺序"] = range(1, len(working) + 1)

    working["_usage_sort"] = pd.to_numeric(working["使用率数字"], errors="coerce")
    working["_missing_usage"] = working["_usage_sort"].isna()
    working = working.sort_values(
        by=["_missing_usage", "_usage_sort", "原图顺序"],
        ascending=[True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    exported = pd.DataFrame(
        {
            "排名": range(1, len(working) + 1),
            "角色中文名": working["角色名"].fillna("").astype(str),
            "使用率": working["_usage_sort"],
            "使用率文本": working["使用率文本"].fillna("").astype(str),
        }
    )
    exported.loc[exported["使用率"].isna(), "使用率"] = pd.NA
    return exported[EXPORT_COLUMNS]
