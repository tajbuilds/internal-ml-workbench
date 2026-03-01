from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix

from app.core.config import APP_DATA_DIR
from app.core.ml import detect_task_type


def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded


def _section_html(title: str, body: str) -> str:
    return (
        f"<section><h2>{title}</h2>{body}</section>"
        "<hr style='border:none;border-top:1px solid #d0d7de;margin:20px 0;'/>"
    )


def _unavailable(message: str) -> str:
    return f"<p><i>{message}</i></p>"


def _dataset_report_dir(dataset_id: str) -> Path:
    path = APP_DATA_DIR / "reports" / dataset_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def generate_selected_eda_report(
    df: pd.DataFrame,
    dataset_id: str,
    target_col: str,
    selected_modules: list[str],
    options: dict[str, Any],
) -> Path:
    report_dir = _dataset_report_dir(dataset_id)
    report_file = report_dir / "selected_analyses_report.html"

    numeric_cols = df.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols]

    blocks: list[str] = []

    header = (
        "<h1>Selected EDA Analyses Report</h1>"
        f"<p><b>Dataset ID:</b> {dataset_id}</p>"
        f"<p><b>Rows x Cols:</b> {len(df)} x {len(df.columns)}</p>"
        f"<p><b>Target:</b> {target_col}</p>"
        f"<p><b>Modules:</b> {', '.join(selected_modules)}</p>"
        "<hr style='border:none;border-top:1px solid #d0d7de;margin:20px 0;'/>"
    )
    blocks.append(header)

    if "Data Quality Summary" in selected_modules:
        schema_df = pd.DataFrame(
            {
                "column": df.columns,
                "dtype": [str(t) for t in df.dtypes],
                "null_count": [int(df[c].isna().sum()) for c in df.columns],
                "unique": [int(df[c].nunique(dropna=True)) for c in df.columns],
            }
        )
        body = (
            f"<p>Rows: {len(df):,} | Cols: {len(df.columns):,} | "
            f"Missing Cells: {int(df.isna().sum().sum()):,} | "
            f"Duplicates: {int(df.duplicated().sum()):,}</p>"
            + schema_df.to_html(index=False)
        )
        blocks.append(_section_html("Data Quality Summary", body))

    if "Missingness" in selected_modules:
        missing = df.isna().sum().sort_values(ascending=False)
        missing = missing[missing > 0]
        if not missing.empty:
            top_n = int(options.get("top_missing", 12))
            missing = missing.head(top_n)
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.bar(missing.index.astype(str), missing.values)
            ax.set_title("Missing values by column")
            ax.set_ylabel("Missing count")
            ax.tick_params(axis="x", rotation=45)
            img = _fig_to_base64(fig)
            body = f"<img src='data:image/png;base64,{img}' style='max-width:100%;'/>"
        else:
            body = "<p>No missing values detected.</p>"
        blocks.append(_section_html("Missingness", body))

    if "Numeric Distributions" in selected_modules:
        if numeric_cols:
            cols = numeric_cols[: int(options.get("max_numeric", 6))]
            bins = int(options.get("bins", 30))
            fig, axes = plt.subplots(
                nrows=len(cols),
                ncols=1,
                figsize=(10, max(3, 2.4 * len(cols))),
            )
            if len(cols) == 1:
                axes = [axes]
            for ax, col in zip(axes, cols):
                ax.hist(df[col].dropna(), bins=bins)
                ax.set_title(f"Distribution: {col}")
            plt.tight_layout()
            img = _fig_to_base64(fig)
            body = f"<img src='data:image/png;base64,{img}' style='max-width:100%;'/>"
        else:
            body = _unavailable("No numeric columns available.")
        blocks.append(_section_html("Numeric Distributions", body))

    if "Outlier Boxplots" in selected_modules:
        if numeric_cols:
            cols = numeric_cols[: int(options.get("max_numeric", 6))]
            fig, ax = plt.subplots(figsize=(max(8, 1.6 * len(cols)), 4.5))
            data = [df[c].dropna().values for c in cols]
            ax.boxplot(data, labels=cols, showfliers=True)
            ax.set_title("Boxplots by numeric feature")
            ax.tick_params(axis="x", rotation=45)
            img = _fig_to_base64(fig)
            body = f"<img src='data:image/png;base64,{img}' style='max-width:100%;'/>"
        else:
            body = _unavailable("No numeric columns available.")
        blocks.append(_section_html("Outlier Boxplots", body))

    if "Categorical Distributions" in selected_modules:
        if categorical_cols:
            cols = categorical_cols[: max(2, int(options.get("max_numeric", 6)) // 2)]
            max_levels = int(options.get("max_categories", 15))
            parts: list[str] = []
            for col in cols[:4]:
                vc = df[col].astype(str).value_counts(dropna=False).head(max_levels)
                fig, ax = plt.subplots(figsize=(9, 3.6))
                ax.bar(vc.index, vc.values)
                ax.set_title(f"Top categories: {col}")
                ax.tick_params(axis="x", rotation=45)
                img = _fig_to_base64(fig)
                part = (
                    f"<h4>{col}</h4>"
                    f"<img src='data:image/png;base64,{img}' style='max-width:100%;'/>"
                )
                parts.append(part)
            body = "".join(parts)
        else:
            body = _unavailable("No categorical columns available.")
        blocks.append(_section_html("Categorical Distributions", body))

    if "Correlation Heatmap" in selected_modules:
        if len(numeric_cols) >= 2:
            method = str(options.get("corr_method", "pearson"))
            corr = df[numeric_cols].corr(method=method)
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
            ax.set_xticks(range(len(corr.columns)))
            ax.set_xticklabels(corr.columns, rotation=45, ha="right")
            ax.set_yticks(range(len(corr.index)))
            ax.set_yticklabels(corr.index)
            ax.set_title(f"{method.title()} correlation")
            fig.colorbar(im, ax=ax)
            img = _fig_to_base64(fig)
            body = f"<img src='data:image/png;base64,{img}' style='max-width:100%;'/>"
        else:
            body = _unavailable("Need at least 2 numeric columns for correlation.")
        blocks.append(_section_html("Correlation Heatmap", body))

    if "Pairwise Relationships" in selected_modules:
        if len(numeric_cols) >= 2:
            cols = numeric_cols[: min(int(options.get("max_numeric", 6)), 6)]
            sample_rows = int(options.get("sample_rows", 1000))
            sample_df = df[cols].dropna().head(sample_rows)
            if not sample_df.empty:
                size = min(14, 2.2 * len(cols))
                fig = plt.figure(figsize=(size, size))
                scatter_matrix(
                    sample_df,
                    alpha=0.55,
                    diagonal="hist",
                    figsize=fig.get_size_inches(),
                )
                img = _fig_to_base64(fig)
                body = f"<img src='data:image/png;base64,{img}' style='max-width:100%;'/>"
            else:
                body = "<p>Not enough non-null rows for pairwise plot.</p>"
        else:
            body = _unavailable("Need at least 2 numeric columns for pairwise relationships.")
        blocks.append(_section_html("Pairwise Relationships", body))

    if "Target Deep Dive" in selected_modules:
        if target_col in df.columns:
            task = detect_task_type(df, target_col)
            parts = [f"<p>Detected task: <b>{task}</b></p>"]
            if task == "classification":
                vc = df[target_col].astype(str).value_counts(dropna=False)
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.bar(vc.index, vc.values)
                ax.set_title(f"Target class distribution: {target_col}")
                ax.tick_params(axis="x", rotation=45)
                img = _fig_to_base64(fig)
                parts.append(f"<img src='data:image/png;base64,{img}' style='max-width:100%;'/>")
            else:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.hist(df[target_col].dropna(), bins=25)
                ax.set_title(f"Target distribution: {target_col}")
                img = _fig_to_base64(fig)
                parts.append(f"<img src='data:image/png;base64,{img}' style='max-width:100%;'/>")
            body = "".join(parts)
        else:
            body = _unavailable("Selected target is not present in dataset columns.")
        blocks.append(_section_html("Target Deep Dive", body))

    if "Custom Feature Plot" in selected_modules:
        cols = list(df.columns)
        if not cols:
            body = _unavailable("Dataset has no columns available for custom plotting.")
            blocks.append(_section_html("Custom Feature Plot", body))
        else:
            x_col = options.get("viz_x")
            y_col = options.get("viz_y")
            plot_type = str(options.get("viz_plot_type", "scatter"))
            if not x_col or x_col not in cols:
                x_col = cols[0]

            fig, ax = plt.subplots(figsize=(8.5, 4))
            if not y_col or y_col == "<none>":
                vc = df[x_col].astype(str).value_counts().head(20)
                ax.bar(vc.index, vc.values)
                ax.set_title(f"Top values for {x_col}")
                ax.tick_params(axis="x", rotation=45)
                y_label = "<none>"
            elif y_col not in cols:
                body = _unavailable("Selected Y column is not available in the dataset.")
                blocks.append(_section_html("Custom Feature Plot", body))
                plt.close(fig)
                y_label = None
            else:
                temp = df[[x_col, y_col]].dropna().head(2000)
                if plot_type == "line":
                    ax.plot(temp[x_col], temp[y_col])
                elif plot_type == "bar":
                    grouped = temp.groupby(x_col, dropna=False)[y_col].mean().head(30)
                    ax.bar(grouped.index.astype(str), grouped.values)
                    ax.tick_params(axis="x", rotation=45)
                else:
                    ax.scatter(temp[x_col], temp[y_col], alpha=0.5)
                ax.set_xlabel(x_col)
                ax.set_ylabel(y_col)
                ax.set_title(f"{plot_type.title()} plot: {x_col} vs {y_col}")
                y_label = y_col

            if y_label is not None:
                img = _fig_to_base64(fig)
                body = (
                    f"<p>X: <b>{x_col}</b> | Y: <b>{y_label}</b> | "
                    f"Plot: <b>{plot_type}</b></p>"
                    f"<img src='data:image/png;base64,{img}' style='max-width:100%;'/>"
                )
                blocks.append(_section_html("Custom Feature Plot", body))

    html = (
        "<html><head><meta charset='utf-8'><title>Selected EDA Report</title>"
        "<style>"
        "body{font-family:Arial,sans-serif;margin:24px;background:#ffffff;color:#0f172a;}"
        "table{border-collapse:collapse;width:100%;}"
        "td,th{border:1px solid #d0d7de;padding:6px;text-align:left;}"
        "h1,h2,h3,h4{color:#0f172a;}"
        "img{border:1px solid #e2e8f0;border-radius:8px;padding:2px;background:#fff;}"
        "@media (prefers-color-scheme: dark){"
        "body{background:#0b1220;color:#e5e7eb;}"
        "td,th{border:1px solid #334155;}"
        "h1,h2,h3,h4{color:#f8fafc;}"
        "img{border:1px solid #334155;background:#0f172a;}"
        "hr{border-top:1px solid #334155 !important;}"
        "}"
        "</style></head><body>"
        + "".join(blocks)
        + "</body></html>"
    )

    report_file.write_text(html, encoding="utf-8")
    return report_file
