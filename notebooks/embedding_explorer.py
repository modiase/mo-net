"""Marimo notebook for inspecting trained word2vec embeddings.

Loads a saved ``.mar`` archive (or a legacy ``.zip``) produced by
``mo_net.samples.word2vec train``, projects the embedding table into
2D (PCA by default), and highlights a configurable list of common words
so you can eyeball whether the cloud shows real structure or has
collapsed.

Open with::

    just nb embedding_explorer
"""

import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import hashlib
    import subprocess
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go

    return Path, go, hashlib, mo, np, subprocess


@app.cell
def _(mo):
    mo.md("""
    # Embedding explorer

    Point at a `.mar` (or legacy `.zip`) produced by
    `mo_net.samples.word2vec train`. Accepts a local path or an ssh path
    (`herakles:/data/.../run.mar` or `ssh://herakles/data/.../run.mar`);
    ssh paths are scp-fetched into
    `~/.cache/mo-net/embedding_explorer/` (content-addressed by the source
    string, so re-pasting the same path won't refetch unless you tick
    re-fetch). Architecture is auto-detected from `metadata.json`; the
    picker is an override.
    """)
    return


@app.cell
def _(mo):
    path_input = mo.ui.text(
        placeholder="herakles:/data/mo-net/sweeps/w2v/.../run.mar",
        label="model archive path (local or ssh)",
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
            suffix = Path(rpath).suffix or ".mar"
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
def _(local_path, mo):
    from mo_net.samples.word2vec.archive import peek_manifest as _peek_manifest

    def _read_metadata(path):
        if path is None:
            return None, None
        try:
            return _peek_manifest(path).metadata.type, None
        except Exception as exc:
            return None, f"could not read manifest: {exc}"

    detected_type, load_error = _read_metadata(local_path)
    if load_error:
        mo.md(f":warning: **{load_error}**")
    elif detected_type:
        mo.md(f"Detected architecture: **{detected_type}**")
    return


@app.cell
def _(local_path, mo):
    import jax

    from mo_net.samples.word2vec.archive import load_word2vec_archive

    def _load(path):
        if path is None:
            return None, None, ""
        try:
            model, vocab, manifest = load_word2vec_archive(
                path, training=False, key=jax.random.PRNGKey(0)
            )
            emb = jax.device_get(model.embeddings)
            msg = (
                f"loaded **{manifest.metadata.type}** with embeddings shape "
                f"`{emb.shape}` (vocab `{len(vocab)}`)"
            )
            return emb, vocab, msg
        except Exception as exc:
            return None, None, f":x: failed to load: {exc}"

    embeddings, vocab_obj, _summary = _load(local_path)
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
def _(embeddings, matched, mo, np, vocab_obj):
    def _health():
        if embeddings is None or vocab_obj is None:
            return mo.md("_load a model first to see health metrics_")
        E = np.asarray(embeddings, dtype=np.float64)
        norms = np.linalg.norm(E, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        En = E / norms

        rng = np.random.default_rng(42)
        N = len(En)
        n_pairs = min(5000, N * (N - 1) // 2)
        i_idx = rng.integers(0, N, n_pairs * 2)
        j_idx = rng.integers(0, N, n_pairs * 2)
        keep = i_idx != j_idx
        i_idx, j_idx = i_idx[keep][:n_pairs], j_idx[keep][:n_pairs]
        random_cosines = np.sum(En[i_idx] * En[j_idx], axis=1)
        anisotropy = float(random_cosines.mean())
        sq_dists = 2.0 - 2.0 * random_cosines
        uniformity = float(np.log(np.exp(-2.0 * sq_dists).mean()))

        E_centered = E - E.mean(axis=0, keepdims=True)
        _, s, _ = np.linalg.svd(E_centered, full_matrices=False)
        var_ratios = (s**2) / (s**2).sum()
        top1 = float(var_ratios[0])
        top3 = float(var_ratios[:3].sum())

        within_block = ""
        if matched:
            hl_ids = np.array(
                [vocab_obj[w] for w in matched if w in vocab_obj.vocab],
                dtype=int,
            )
            if len(hl_ids) >= 2:
                hl_E = En[hl_ids]
                hl_sim = hl_E @ hl_E.T
                np.fill_diagonal(hl_sim, np.nan)
                within = float(np.nanmean(hl_sim))
                hl_set = set(int(i) for i in hl_ids)
                non_hl = np.array([i for i in range(N) if i not in hl_set], dtype=int)
                bg_sample = rng.choice(
                    non_hl, size=min(500, len(non_hl)), replace=False
                )
                between = float((hl_E @ En[bg_sample].T).mean())
                ratio = f"{within / between:.3f}" if abs(between) > 1e-12 else "n/a"
                within_block = (
                    f"| within-highlight cosine | {within:.4f} | "
                    f"mean pairwise among highlight words |\n"
                    f"| highlight↔random cosine | {between:.4f} | "
                    f"highlight vs random vocab |\n"
                    f"| within/between ratio | {ratio} | "
                    f">1 = highlights cluster together |\n"
                )

        body = (
            f"| anisotropy (mean random-pair cos) | {anisotropy:.4f} | "
            f"0 = isotropic, ~1 = collapsed; healthy w2v 0.05–0.2 |\n"
            f"| Wang-Isola uniformity | {uniformity:.4f} | "
            f"more negative = better sphere coverage; healthy ≈ −3 to −4 |\n"
            f"| top-1 PC variance | {top1:.4f} | "
            f">0.5 = effectively 1D model |\n"
            f"| top-3 PC variance | {top3:.4f} | "
            f"share captured by first 3 directions |\n" + within_block
        )
        return mo.md(
            f"### Model health\n\n"
            f"| metric | value | interpretation |\n|---|---|---|\n{body}"
        )

    _health()
    return


@app.cell
def _(mo):
    abtt_k = mo.ui.slider(
        start=0,
        stop=5,
        value=1,
        label="ABTT: top-K PCs to remove",
        show_value=True,
    )
    abtt_k
    return (abtt_k,)


@app.cell
def _(abtt_k, embeddings, mo, np):
    def _abtt_compare():
        if embeddings is None:
            return mo.md("_load a model first to see ABTT comparison_")

        _E = np.asarray(embeddings, dtype=np.float64)
        _mean = _E.mean(axis=0, keepdims=True)
        _centered = _E - _mean
        _U, _s, _Vt = np.linalg.svd(_centered, full_matrices=False)

        _K = int(abtt_k.value)
        if _K == 0:
            _cleaned = _centered.copy()
        else:
            _components = _Vt[:_K]
            _projections = _centered @ _components.T
            _cleaned = _centered - _projections @ _components

        def _anisotropy(_M: np.ndarray, _seed: int = 42) -> float:
            _norms = np.linalg.norm(_M, axis=1, keepdims=True)
            _norms = np.where(_norms < 1e-12, 1.0, _norms)
            _Mn = _M / _norms
            _rng = np.random.default_rng(_seed)
            _N = len(_Mn)
            _n_pairs = min(5000, _N * (_N - 1) // 2)
            _i = _rng.integers(0, _N, _n_pairs * 2)
            _j = _rng.integers(0, _N, _n_pairs * 2)
            _keep = _i != _j
            _i, _j = _i[_keep][:_n_pairs], _j[_keep][:_n_pairs]
            return float(np.sum(_Mn[_i] * _Mn[_j], axis=1).mean())

        _ratios = (_s**2) / (_s**2).sum()
        _top1_raw = float(_ratios[0])
        _top3_raw = float(_ratios[:3].sum())
        _aniso_raw = _anisotropy(_E)
        _aniso_clean = _anisotropy(_cleaned)

        _, _s_clean, _ = np.linalg.svd(
            _cleaned - _cleaned.mean(axis=0), full_matrices=False
        )
        _ratios_clean = (_s_clean**2) / (_s_clean**2).sum()
        _top1_clean = float(_ratios_clean[0])
        _top3_clean = float(_ratios_clean[:3].sum())

        return mo.md(
            f"### ABTT comparison (top-{_K} PCs removed)\n\n"
            f"| metric | raw | cleaned | delta |\n|---|---|---|---|\n"
            f"| anisotropy | {_aniso_raw:.4f} | {_aniso_clean:.4f} | "
            f"{_aniso_clean - _aniso_raw:+.4f} |\n"
            f"| top-1 PC variance | {_top1_raw:.4f} | {_top1_clean:.4f} | "
            f"{_top1_clean - _top1_raw:+.4f} |\n"
            f"| top-3 PC variance | {_top3_raw:.4f} | {_top3_clean:.4f} | "
            f"{_top3_clean - _top3_raw:+.4f} |\n\n"
            f"_If anisotropy drops near 0 after removing 1-2 PCs, the "
            f"collapse was cosmetic and the structure underneath is "
            f"recoverable — bake ABTT into your eval/inference path. "
            f"If it stays high, the geometry itself is degenerate._"
        )

    _abtt_compare()
    return


@app.cell
def _(mo):
    method_picker = mo.ui.radio(
        options=["pca-cosine", "pca-raw"],
        value="pca-cosine",
        label="projection",
        inline=True,
    )
    dim_picker = mo.ui.radio(
        options=["2D", "3D"],
        value="2D",
        label="dimensions",
        inline=True,
    )
    axis_choices = [str(i) for i in range(1, 21)]
    axis_x = mo.ui.dropdown(options=axis_choices, value="1", label="X axis (PC#)")
    axis_y = mo.ui.dropdown(options=axis_choices, value="2", label="Y axis (PC#)")
    axis_z = mo.ui.dropdown(options=axis_choices, value="3", label="Z axis (PC#)")
    cap_slider = mo.ui.slider(
        start=200,
        stop=3000,
        step=100,
        value=1000,
        label="background points (top-N by frequency)",
    )
    mo.vstack(
        [
            mo.hstack([method_picker, dim_picker]),
            mo.hstack([axis_x, axis_y, axis_z]),
            cap_slider,
        ]
    )
    return axis_x, axis_y, axis_z, cap_slider, dim_picker, method_picker


@app.cell
def _(
    axis_x,
    axis_y,
    axis_z,
    cap_slider,
    dim_picker,
    embeddings,
    matched,
    method_picker,
    np,
    vocab_obj,
):
    def _project():
        if embeddings is None or vocab_obj is None:
            return None
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

        X = np.asarray(embeddings[idx_union], dtype=np.float64)
        if method_picker.value == "pca-cosine":
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms = np.where(norms < 1e-12, 1.0, norms)
            X = X / norms
        X_centered = X - X.mean(axis=0, keepdims=True)
        _, s, vt = np.linalg.svd(X_centered, full_matrices=False)
        all_components = X_centered @ vt.T
        var_ratio_all = (s**2) / (s**2).sum()

        n_pcs = all_components.shape[1]
        ax_x = min(int(axis_x.value), n_pcs) - 1
        ax_y = min(int(axis_y.value), n_pcs) - 1
        if dim_picker.value == "3D":
            ax_z = min(int(axis_z.value), n_pcs) - 1
            indices = [ax_x, ax_y, ax_z]
        else:
            indices = [ax_x, ax_y]

        return {
            "idx_union": idx_union,
            "coords": all_components[:, indices],
            "component_labels": [i + 1 for i in indices],
            "var_explained": var_ratio_all[indices].round(4),
        }

    projection = _project()
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
            f"PC{pc}={v:.3f}"
            for pc, v in zip(
                projection["component_labels"], projection["var_explained"]
            )
        )
        axis_labels = [f"PC{pc}" for pc in projection["component_labels"]]
        layout_kwargs = dict(
            title=f"explained variance: {var_str}",
            paper_bgcolor="#0e0e10",
            plot_bgcolor="#0e0e10",
            font=dict(color="#eee"),
            height=620,
            margin=dict(l=20, r=20, t=50, b=20),
        )
        if coords.shape[1] == 2:
            layout_kwargs["xaxis_title"] = axis_labels[0]
            layout_kwargs["yaxis_title"] = axis_labels[1]
        else:
            layout_kwargs["scene"] = dict(
                xaxis_title=axis_labels[0],
                yaxis_title=axis_labels[1],
                zaxis_title=axis_labels[2],
            )
        fig.update_layout(**layout_kwargs)
        return mo.ui.plotly(fig)

    _render()
    return


@app.cell
def _(mo):
    analogy_a = mo.ui.text(value="king", label="A")
    analogy_b = mo.ui.text(value="man", label="− B")
    analogy_c = mo.ui.text(value="woman", label="+ C")
    analogy_lens = mo.ui.radio(
        options=["cosine-centred", "cosine-raw", "cosine-debiased"],
        value="cosine-centred",
        label="lens",
        inline=True,
    )
    analogy_k = mo.ui.slider(start=5, stop=30, step=1, value=10, label="top K results")
    mo.vstack(
        [
            mo.md(
                "### Analogy: `A − B + C ≈ ?`\n"
                "_lens choices: `cosine-raw` (no preprocessing — bias toward "
                "frequent words), `cosine-centred` (subtract embedding-table "
                "mean — strips common component, default), `cosine-debiased` "
                "(centre + remove top-K PCs — Mu et al. all-but-the-top; "
                "uses the K from the ABTT slider above)._"
            ),
            mo.hstack([analogy_a, analogy_b, analogy_c]),
            mo.hstack([analogy_lens, analogy_k]),
        ]
    )
    return analogy_a, analogy_b, analogy_c, analogy_k, analogy_lens


@app.cell
def _(
    abtt_k,
    analogy_a,
    analogy_b,
    analogy_c,
    analogy_k,
    analogy_lens,
    embeddings,
    mo,
    np,
    vocab_obj,
):
    def _analogy():
        if embeddings is None or vocab_obj is None:
            return mo.md("_load a model first_")
        in_vocab_set = set(vocab_obj.vocab)
        a = analogy_a.value.strip().lower()
        b = analogy_b.value.strip().lower()
        c = analogy_c.value.strip().lower()
        missing = [w for w in (a, b, c) if w and w not in in_vocab_set]
        if missing:
            return mo.md(
                f":warning: not in vocab: {', '.join(f'`{w}`' for w in missing)}"
            )
        if not (a and b and c):
            return mo.md("_fill in A, B, C_")

        E = np.asarray(embeddings, dtype=np.float64)
        lens = analogy_lens.value
        K_used = 0
        if lens in ("cosine-centred", "cosine-debiased"):
            E = E - E.mean(axis=0, keepdims=True)
        if lens == "cosine-debiased":
            K_used = int(abtt_k.value)
            if K_used > 0:
                _, _, vt = np.linalg.svd(E, full_matrices=False)
                top = vt[:K_used]
                E = E - (E @ top.T) @ top
        fingerprint = float(np.linalg.norm(E[:1])) if len(E) else 0.0
        norms = np.linalg.norm(E, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        E_n = E / norms
        target = E[vocab_obj[a]] - E[vocab_obj[b]] + E[vocab_obj[c]]
        target_n = target / (np.linalg.norm(target) or 1.0)
        sims = E_n @ target_n
        for w in (a, b, c):
            sims[vocab_obj[w]] = -np.inf
        sims[vocab_obj.unknown_token_id] = -np.inf

        k = analogy_k.value
        top_idx = np.argpartition(-sims, k)[:k]
        top_idx = top_idx[np.argsort(-sims[top_idx])]
        rows = "\n".join(
            f"| {rank} | `{vocab_obj.id_to_token[int(idx)]}` | {float(sims[idx]):.4f} |"
            for rank, idx in enumerate(top_idx, 1)
        )
        return mo.md(
            f"**`{a} − {b} + {c}` →** _(`{lens}`, K={K_used}, "
            f"A/B/C excluded; ||E[0]|| = {fingerprint:.4f})_\n\n"
            f"| rank | word | cos sim |\n|---|---|---|\n{rows}"
        )

    _analogy()
    return


if __name__ == "__main__":
    app.run()
