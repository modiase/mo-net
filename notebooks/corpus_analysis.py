"""Marimo notebook for inspecting the English-sentences corpus.

Reads the JSON output of ``mo_net.scripts.corpus_stats`` and renders
totals, top-K bar charts per n-gram order, the distribution of the top
1000 unigrams, and keyword-window samples.

Default ``--stats-dir`` is ``./data/corpus_stats``; override by setting
the ``MO_NET_CORPUS_STATS`` environment variable before opening the
notebook.

Run::

    marimo edit notebooks/corpus_analysis.py
"""

import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import os
    from pathlib import Path

    import marimo as mo
    import pandas as pd
    import plotly.express as px

    stats_dir = Path(os.environ.get("MO_NET_CORPUS_STATS", "data/corpus_stats"))
    return json, mo, pd, px, stats_dir


@app.cell
def _(mo, stats_dir):
    mo.md(f"""
    # English sentences corpus

    Stats loaded from `{stats_dir}`. Re-run
    `python -m mo_net.scripts.corpus_stats --output-dir {stats_dir}` on
    herakles to refresh.
    """)
    return


@app.cell
def _(json, stats_dir):
    totals = json.loads((stats_dir / "totals.json").read_text())
    return (totals,)


@app.cell
def _(mo, totals):
    rows = "\n".join(f"| {k} | {v:,} |" for k, v in totals.items())
    mo.md(
        f"""
        ## Totals

        | metric | value |
        | --- | --- |
        {rows}
        """
    )
    return


@app.cell
def _(json, pd, stats_dir):
    def _load_ngrams(n: int) -> pd.DataFrame:
        path = stats_dir / f"{n}grams.json"
        return pd.DataFrame(json.loads(path.read_text()))

    grams = {n: _load_ngrams(n) for n in range(1, 6)}
    return (grams,)


@app.cell
def _(mo):
    n_picker = mo.ui.dropdown(
        options=["1", "2", "3", "4", "5"],
        value="1",
        label="n-gram size",
    )
    top_slider = mo.ui.slider(
        start=10, stop=200, step=10, value=50, label="top N to chart"
    )
    mo.hstack([n_picker, top_slider])
    return n_picker, top_slider


@app.cell
def _(grams, mo, n_picker, px, top_slider):
    topk_n = int(n_picker.value)
    topk_df = grams[topk_n].head(top_slider.value)
    topk_fig = px.bar(
        topk_df,
        x="gram",
        y="count",
        title=f"Top {top_slider.value} {topk_n}-grams",
    )
    topk_fig.update_layout(
        xaxis_tickangle=-45, height=420, margin=dict(l=20, r=20, t=40, b=120)
    )
    mo.ui.plotly(topk_fig)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Top-1000 unigram distribution (log-log)
    """)
    return


@app.cell
def _(grams, mo, px):
    loglog_df = grams[1].copy()
    loglog_df["rank"] = loglog_df.index + 1
    loglog_fig = px.line(
        loglog_df,
        x="rank",
        y="count",
        log_x=True,
        log_y=True,
        title="Top 1000 unigrams: rank vs frequency",
    )
    loglog_fig.update_layout(height=420, margin=dict(l=20, r=20, t=40, b=40))
    mo.ui.plotly(loglog_fig)
    return


@app.cell
def _(grams, mo, px):
    cumshare_df = grams[1].head(50).copy()
    cumshare_df["cum_share"] = cumshare_df["count"].cumsum() / grams[1]["count"].sum()
    cumshare_fig = px.line(
        cumshare_df,
        x="gram",
        y="cum_share",
        title="Cumulative share of total tokens (top 50 words)",
    )
    cumshare_fig.update_layout(
        xaxis_tickangle=-45, height=420, margin=dict(l=20, r=20, t=40, b=120)
    )
    mo.ui.plotly(cumshare_fig)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Keyword 5-gram samples
    """)
    return


@app.cell
def _(json, stats_dir):
    keyword_samples = json.loads((stats_dir / "keyword_samples.json").read_text())
    return (keyword_samples,)


@app.cell
def _(keyword_samples, mo):
    keyword_picker = mo.ui.radio(
        options=sorted(keyword_samples),
        value=sorted(keyword_samples)[0],
        label="keyword",
        inline=True,
    )
    keyword_picker
    return (keyword_picker,)


@app.cell
def _(keyword_picker, keyword_samples, mo):
    sel_word = keyword_picker.value
    sel_windows = keyword_samples[sel_word]
    sel_bullets = "\n".join(f"- `{w}`" for w in sel_windows[:20])
    mo.md(f"### `{sel_word}` ({len(sel_windows)} kept)\n\n{sel_bullets}")
    return


if __name__ == "__main__":
    app.run()
