"""Marimo notebook for inspecting trained word2vec embeddings.

Loads a saved ``.zip`` model produced by ``mo_net.samples.word2vec train``,
projects the embedding table into 2D (PCA by default), and highlights a
configurable list of common words so you can eyeball whether the cloud
shows real structure or has collapsed.

Open with::

    just nb embedding_explorer
"""

import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import hashlib
    import json
    import subprocess
    import zipfile
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go

    return Path, go, hashlib, json, mo, np, subprocess, zipfile


@app.cell
def _(mo):
    mo.md("""
    # Embedding explorer

    Point at a `.zip` produced by `mo_net.samples.word2vec train`. Accepts
    a local path or an ssh path (`herakles:/data/.../run.zip` or
    `ssh://herakles/data/.../run.zip`); ssh paths are scp-fetched into
    `~/.cache/mo-net/embedding_explorer/` (content-addressed by the source
    string, so re-pasting the same path won't refetch unless you tick
    re-fetch). Architecture is auto-detected from `metadata.json`; the
    picker is an override.
    """)
    return


@app.cell
def _(mo):
    path_input = mo.ui.text(
        placeholder="herakles:/data/mo-net/sweeps/w2v/.../run.zip",
        label="model zip path (local or ssh)",
        full_width=True,
    )
    force_refresh = mo.ui.checkbox(value=False, label="force re-fetch (ssh only)")
    mo.vstack([path_input, force_refresh])
    return force_refresh, path_input


@app.cell
def _(Path, force_refresh, hashlib, mo, path_input, subprocess):
    def _looks_ssh(s: str) -> bool:
        if not s or s.startswith(("/", "~", ".")):
            return False
        if "://" in s:
            return s.startswith("ssh://")
        head, sep, _ = s.partition(":")
        return bool(sep) and "/" not in head and bool(head)

    local_path = None
    resolve_msg = ""
    if path_input.value:
        raw = path_input.value.strip()
        if _looks_ssh(raw):
            cache_dir = Path.home() / ".cache" / "mo-net" / "embedding_explorer"
            cache_dir.mkdir(parents=True, exist_ok=True)
            if raw.startswith("ssh://"):
                stripped = raw[len("ssh://") :]
                host, _, rpath = stripped.partition("/")
                scp_src = f"{host}:/{rpath}"
            else:
                scp_src = raw
                _, _, rpath = raw.partition(":")
            suffix = Path(rpath).suffix or ".zip"
            key = hashlib.sha256(raw.encode()).hexdigest()[:16]
            cached = cache_dir / f"{key}{suffix}"
            if cached.exists() and not force_refresh.value:
                resolve_msg = f"cached: `{cached}` _(tick re-fetch to refresh)_"
                local_path = cached
            else:
                try:
                    subprocess.run(
                        ["scp", "-q", scp_src, str(cached)],
                        check=True,
                        capture_output=True,
                        text=True,
                        timeout=120,
                    )
                    resolve_msg = f"fetched → `{cached}`"
                    local_path = cached
                except subprocess.CalledProcessError as exc:
                    resolve_msg = f":x: scp failed: {(exc.stderr or '').strip() or exc}"
                except subprocess.TimeoutExpired:
                    resolve_msg = ":x: scp timed out after 120s"
        else:
            candidate = Path(raw).expanduser()
            if candidate.exists():
                local_path = candidate
            else:
                resolve_msg = f":warning: local path does not exist: `{candidate}`"
    mo.md(resolve_msg or "_paste a path above to begin_")
    return (local_path,)


@app.cell
def _(json, local_path, mo, zipfile):
    def _read_metadata(path):
        if path is None:
            return None, None
        try:
            with zipfile.ZipFile(path, "r") as zf:
                with zf.open("metadata.json") as md:
                    return json.loads(md.read().decode("utf-8")).get("type"), None
        except Exception as exc:
            return None, f"could not read metadata: {exc}"

    detected_type, load_error = _read_metadata(local_path)
    if load_error:
        mo.md(f":warning: **{load_error}**")
    elif detected_type:
        mo.md(f"Detected architecture: **{detected_type}**")
    return (detected_type,)


@app.cell
def _(detected_type, mo):
    arch_picker = mo.ui.radio(
        options=["cbow", "skipgram"],
        value=detected_type or "cbow",
        label="architecture (override)",
        inline=True,
    )
    arch_picker
    return (arch_picker,)


@app.cell
def _(arch_picker, local_path, mo):
    from mo_net.samples.word2vec.__main__ import (  # noqa: I001
        CBOWModel,
        SkipGramModel,
        MODEL_ZIP_INTERNAL_PATH,
        VOCAB_ZIP_INTERNAL_PATH,
    )
    from mo_net.samples.word2vec.vocab import Vocab
    import jax
    import zipfile as _zipfile

    def _load(path, model_kind):
        if path is None:
            return None, None, ""
        try:
            with _zipfile.ZipFile(path, "r") as zf:
                vocab = Vocab.from_bytes(zf.read(VOCAB_ZIP_INTERNAL_PATH))
                with zf.open(MODEL_ZIP_INTERNAL_PATH) as mf:
                    if model_kind == "skipgram":
                        model = SkipGramModel.load(
                            mf, training=False, key=jax.random.PRNGKey(0)
                        )
                    else:
                        model = CBOWModel.load(mf, training=False)
            emb = jax.device_get(model.embeddings)
            msg = (
                f"loaded **{model_kind}** with embeddings shape "
                f"`{emb.shape}` (vocab `{len(vocab)}`)"
            )
            return emb, vocab, msg
        except Exception as exc:
            return None, None, f":x: failed to load: {exc}"

    embeddings, vocab_obj, _summary = _load(local_path, arch_picker.value)
    mo.md(_summary or "_no model loaded yet_")
    return embeddings, vocab_obj


@app.cell
def _(mo):
    DEFAULT_HIGHLIGHTS = (
        "king queen man woman boy girl father mother son daughter "
        "city country river mountain forest tree house car road "
        "water fire earth wind sun moon star sea ocean rain snow "
        "book paper letter word language music song dance art science "
        "love peace war power money food bread wine animal dog cat"
    )
    highlight_input = mo.ui.text_area(
        value=DEFAULT_HIGHLIGHTS,
        label="highlight words (whitespace or comma separated)",
        full_width=True,
        rows=4,
    )
    highlight_input
    return (highlight_input,)


@app.cell
def _(highlight_input, mo, vocab_obj):
    import re as _re

    raw_words = _re.split(r"[\s,]+", highlight_input.value.strip())
    requested = [w.lower() for w in raw_words if w]
    matched = []
    missing = []
    if vocab_obj is not None:
        in_vocab = set(vocab_obj.vocab)
        for w in requested:
            (matched if w in in_vocab else missing).append(w)
    mo.md(
        f"{len(matched)}/{len(requested)} matched the vocab. "
        + (
            f"_missing: {', '.join(missing[:15])}{'…' if len(missing) > 15 else ''}_"
            if missing
            else ""
        )
    )
    return (matched,)


@app.cell
def _(mo):
    method_picker = mo.ui.radio(
        options=["pca", "pca-3d"],
        value="pca",
        label="projection",
        inline=True,
    )
    cap_slider = mo.ui.slider(
        start=200,
        stop=3000,
        step=100,
        value=1000,
        label="background points (top-N by frequency)",
    )
    mo.hstack([method_picker, cap_slider])
    return cap_slider, method_picker


@app.cell
def _(cap_slider, embeddings, matched, method_picker, np, vocab_obj):
    projection = None
    if embeddings is not None and vocab_obj is not None:
        counts = vocab_obj.word_counts or {}
        ranked = sorted(vocab_obj.vocab, key=lambda w: -counts.get(w, 0))
        bg_words = ranked[: cap_slider.value]
        bg_idx = np.array([vocab_obj[w] for w in bg_words], dtype=int)

        hl_idx = np.array(
            [vocab_obj[w] for w in matched if w in vocab_obj.vocab], dtype=int
        )
        idx_union = (
            np.unique(np.concatenate([bg_idx, hl_idx])) if len(hl_idx) else bg_idx
        )

        X = np.asarray(embeddings[idx_union])
        X_centered = X - X.mean(axis=0, keepdims=True)
        n_components = 3 if method_picker.value == "pca-3d" else 2
        u, s, vt = np.linalg.svd(X_centered, full_matrices=False)
        projected = X_centered @ vt[:n_components].T

        var_explained = (s[:n_components] ** 2 / (s**2).sum()).round(4)
        projection = {
            "idx_union": idx_union,
            "coords": projected,
            "var_explained": var_explained,
        }
    return (projection,)


@app.cell
def _(go, matched, mo, projection, vocab_obj):
    def _render():
        if projection is None:
            return mo.md("_load a model and pick highlight words to see the plot_")
        coords = projection["coords"]
        idx_union = projection["idx_union"]
        labels_for_idx = {idx: vocab_obj.id_to_token[int(idx)] for idx in idx_union}
        hl_set = set(matched)
        is_highlight = [labels_for_idx[i] in hl_set for i in idx_union]
        bg_mask = [not h for h in is_highlight]
        hl_mask = is_highlight
        bg_text = [labels_for_idx[idx_union[i]] for i, m in enumerate(bg_mask) if m]
        hl_text = [labels_for_idx[idx_union[i]] for i, m in enumerate(hl_mask) if m]

        if coords.shape[1] == 2:
            bg = go.Scatter(
                x=coords[bg_mask, 0],
                y=coords[bg_mask, 1],
                mode="markers",
                marker=dict(size=4, color="#444", opacity=0.4),
                name="background",
                text=bg_text,
                hoverinfo="text",
            )
            hl = go.Scatter(
                x=coords[hl_mask, 0],
                y=coords[hl_mask, 1],
                mode="markers+text",
                marker=dict(size=10, color="#ff9f3a"),
                text=hl_text,
                textposition="top center",
                textfont=dict(color="#ffd28a", size=11),
                name="highlighted",
            )
        else:
            bg = go.Scatter3d(
                x=coords[bg_mask, 0],
                y=coords[bg_mask, 1],
                z=coords[bg_mask, 2],
                mode="markers",
                marker=dict(size=2, color="#444", opacity=0.4),
                name="background",
                text=bg_text,
                hoverinfo="text",
            )
            hl = go.Scatter3d(
                x=coords[hl_mask, 0],
                y=coords[hl_mask, 1],
                z=coords[hl_mask, 2],
                mode="markers+text",
                marker=dict(size=5, color="#ff9f3a"),
                text=hl_text,
                textfont=dict(color="#ffd28a", size=10),
                name="highlighted",
            )
        fig = go.Figure([bg, hl])
        var_str = ", ".join(
            f"PC{i + 1}={v:.3f}" for i, v in enumerate(projection["var_explained"])
        )
        fig.update_layout(
            title=f"explained variance: {var_str}",
            paper_bgcolor="#0e0e10",
            plot_bgcolor="#0e0e10",
            font=dict(color="#eee"),
            height=620,
            margin=dict(l=20, r=20, t=50, b=20),
        )
        return mo.ui.plotly(fig)

    _render()
    return


if __name__ == "__main__":
    app.run()
