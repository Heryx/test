from __future__ import annotations

from typing import Any, Callable

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy.interpolate import griddata
from scipy.ndimage import median_filter
from scipy.spatial import cKDTree

from radar_filters import (
    apply_linear_interpolation_3d_cube,
    apply_normalize3d,
    apply_reduce_number_of_samples,
    apply_semblance_smoothing,
    make_amplitude_spectrum,
    normalize_for_display,
)
from radar_io import OgprProfile


def _normalize_with_fixed_bounds(data: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float64)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or np.isclose(vmin, vmax):
        return normalize_for_display(arr)
    clipped = np.clip(arr, vmin, vmax)
    return (clipped - vmin) / (vmax - vmin)


def _robust_bounds(data: np.ndarray) -> tuple[float, float]:
    arr = np.asarray(data, dtype=np.float64)
    finite = np.isfinite(arr)
    if not finite.any():
        return 0.0, 1.0
    vals = arr[finite]
    vmin = float(np.nanpercentile(vals, 2))
    vmax = float(np.nanpercentile(vals, 98))
    if np.isclose(vmin, vmax):
        vmin = float(np.nanpercentile(vals, 0.1))
        vmax = float(np.nanpercentile(vals, 99.9))
    if np.isclose(vmin, vmax):
        vmin = float(np.nanmin(vals))
        vmax = float(np.nanmax(vals))
    if np.isclose(vmin, vmax):
        return 0.0, 1.0
    return vmin, vmax


def _corr_flat(a: np.ndarray, b: np.ndarray) -> float:
    av = np.asarray(a, dtype=np.float64).ravel()
    bv = np.asarray(b, dtype=np.float64).ravel()
    mask = np.isfinite(av) & np.isfinite(bv)
    if np.count_nonzero(mask) < 4:
        return np.nan
    av = av[mask] - np.mean(av[mask])
    bv = bv[mask] - np.mean(bv[mask])
    den = float(np.linalg.norm(av) * np.linalg.norm(bv))
    if den <= 0:
        return np.nan
    return float(np.dot(av, bv) / den)


def _get_profile_axes(profile: OgprProfile, y_axis_mode: str) -> dict[str, Any]:
    """Calcola i valori e i titoli per gli assi X e Y del radargramma."""
    # --- Asse X (Distanza o Traccia) ---
    x_axis_vals = np.arange(profile.data.shape[0])
    x_axis_title = "Traccia"
    if profile.x is not None and profile.y is not None and profile.x.size > 1 and profile.y.size > 1:
        x_coords = np.asarray(profile.x, dtype=np.float64)
        y_coords = np.asarray(profile.y, dtype=np.float64)

        finite_mask = np.isfinite(x_coords) & np.isfinite(y_coords)
        if np.count_nonzero(finite_mask) > 1:
            x_finite = x_coords[finite_mask]
            y_finite = y_coords[finite_mask]

            distances = np.sqrt(np.diff(x_finite) ** 2 + np.diff(y_finite) ** 2)
            cumulative_dist_finite = np.concatenate(([0], np.cumsum(distances)))

            original_indices = np.arange(profile.data.shape[0])
            finite_indices = original_indices[finite_mask]

            if len(finite_indices) > 1:
                x_axis_vals = np.interp(original_indices, finite_indices, cumulative_dist_finite)
                x_axis_title = "Distanza (m)"

    # --- Asse Y (Campioni, Tempo o Profondità) ---
    samples_count = profile.data.shape[1]
    y_axis_vals = np.arange(samples_count)
    y_axis_title = "Campioni (samples)"

    radar_params = profile.metadata.get("radar_parameters", {})
    sampling_time_ns = radar_params.get("samplingTime_ns")

    if y_axis_mode == "Tempo (ns)" and sampling_time_ns and float(sampling_time_ns) > 0:
        y_axis_vals = np.arange(samples_count) * float(sampling_time_ns)
        y_axis_title = "Tempo (ns)"
    elif y_axis_mode == "Profondità (m)":
        velocity = radar_params.get("propagationVelocity_mPerSec")
        if sampling_time_ns and velocity and float(sampling_time_ns) > 0 and float(velocity) > 0:
            time_s = (np.arange(samples_count) * float(sampling_time_ns)) * 1e-9
            depth = time_s * float(velocity) / 2.0
            y_axis_vals = depth
            y_axis_title = "Profondità (m)"

    return {"x_vals": x_axis_vals, "x_title": x_axis_title, "y_vals": y_axis_vals, "y_title": y_axis_title}


def _srs_label(profile: OgprProfile) -> str:
    if not profile.srs:
        return "n/d"
    srs_type = str(profile.srs.get("type", "n/d"))
    srs_value = str(profile.srs.get("value", ""))
    if srs_value:
        return f"{srs_type}:{srs_value}"
    return srs_type


def _condition_coordinates(x: np.ndarray, y: np.ndarray, cfg: Any) -> tuple[np.ndarray, np.ndarray]:
    xc = np.asarray(x, dtype=np.float64).copy()
    yc = np.asarray(y, dtype=np.float64).copy()
    if xc.size != yc.size:
        return xc, yc

    if bool(cfg.filter_duplicates) and xc.size > 2:
        dx = np.diff(xc)
        dy = np.diff(yc)
        rate = np.sqrt(dx * dx + dy * dy)
        if np.any(np.isfinite(rate)):
            mean_rate = float(np.nanmean(rate))
            if mean_rate > 0:
                bad = rate < (0.5 * mean_rate)
                bad_idx = np.where(bad)[0] + 1
                if bad_idx.size > 0 and bad_idx.size < (0.5 * xc.size):
                    xc[bad_idx] = np.nan
                    yc[bad_idx] = np.nan

    if int(cfg.smooth_window) > 1 and xc.size >= int(cfg.smooth_window):
        win = int(cfg.smooth_window)
        if win % 2 == 0:
            win += 1
        finite_x = np.isfinite(xc)
        finite_y = np.isfinite(yc)
        if np.any(finite_x):
            fill_x = np.interp(
                np.arange(xc.size, dtype=np.float64),
                np.where(finite_x)[0].astype(np.float64),
                xc[finite_x],
            )
            xc = median_filter(fill_x, size=win, mode="nearest")
        if np.any(finite_y):
            fill_y = np.interp(
                np.arange(yc.size, dtype=np.float64),
                np.where(finite_y)[0].astype(np.float64),
                yc[finite_y],
            )
            yc = median_filter(fill_y, size=win, mode="nearest")

    return xc, yc


def _coordinates_for_profile(
    profile: OgprProfile,
    fallback_line: int,
    coord_cfg: Any,
) -> tuple[np.ndarray, np.ndarray, bool]:
    traces_count = profile.data.shape[0]
    if (
        profile.x is not None
        and profile.y is not None
        and len(profile.x) == traces_count
        and len(profile.y) == traces_count
    ):
        x = np.asarray(profile.x, dtype=np.float64)
        y = np.asarray(profile.y, dtype=np.float64)
        x, y = _condition_coordinates(x, y, coord_cfg)
        return (x, y, True)

    x = np.arange(traces_count, dtype=np.float64)
    y = np.full(traces_count, float(fallback_line), dtype=np.float64)
    return x, y, False


def render_profile_tab(
    profiles: list[OgprProfile],
    cfg: Any,
    viz: Any,
    auto_preset_library: list[dict[str, Any]],
    ordered_processing_steps: Callable[[Any], list[tuple[str, int]]],
    predict_ml_preset_cb: Callable[[np.ndarray, float], tuple[int, float]],
    apply_preset_cb: Callable[[dict[str, Any]], None],
    run_filter_pipeline_cb: Callable[[np.ndarray, Any], list[tuple[str, np.ndarray]]],
) -> None:
    mode_label = "Base consigliato" if cfg.workflow_mode == "base" else "Manuale"
    st.caption(f"Modalita workflow: {mode_label}")
    ordered_steps = ordered_processing_steps(cfg)
    if ordered_steps:
        st.caption("Workflow processing attivo: " + " -> ".join([f"{name}({order})" for name, order in ordered_steps]))
    else:
        st.caption("Workflow processing attivo: nessuno (dati raw).")

    y_axis_mode = st.radio(
        "Asse Y Radargramma",
        ["Campioni (samples)", "Tempo (ns)", "Profondità (m)"],
        index=1,
        horizontal=True,
        help="Cambia l'unità di misura per l'asse verticale dei radargrammi. 'Tempo' e 'Profondità' richiedono metadati nel file OGPR.",
    )

    y_flip_debug = st.checkbox(
        "Inverti asse Y (debug orientamento)",
        value=False,
        help="Usalo solo per capire se la geometria appare capovolta.",
    )

    profile_stds = [float(np.nanstd(profile.data)) for profile in profiles]
    rows = []
    for profile, pstd in zip(profiles, profile_stds):
        rows.append(
            {
                "profilo": profile.label,
                "tracce": int(profile.data.shape[0]),
                "campioni": int(profile.data.shape[1]),
                "std": float(pstd),
                "gps": "SI" if (profile.x is not None and profile.y is not None) else "NO",
                "srs": _srs_label(profile),
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)
    best_idx = int(np.nanargmax(np.asarray(profile_stds, dtype=np.float64)))

    selected_idx = st.selectbox(
        "Seleziona profilo",
        options=list(range(len(profiles))),
        index=best_idx,
        format_func=lambda idx: profiles[idx].label,
    )
    selected = profiles[selected_idx]

    ml_state_key = f"ml_preset::{selected.file_name}::{selected.channel_index}"
    with st.expander("Preset ML automatico"):
        st.caption(
            "Il preset ML suggerisce soprattutto il workflow dei filtri. "
            "I parametri tecnici (sample rate/band-pass) restano vincolati ai metadati OGPR quando disponibili."
        )
        col_calc, col_reset = st.columns(2)
        if col_calc.button("Calcola suggerimento ML", key=f"calc_ml_{selected_idx}"):
            with st.spinner("Calcolo preset ML..."):
                ml_idx, ml_conf = predict_ml_preset_cb(selected.data, sample_rate=cfg.sample_rate)
            st.session_state[ml_state_key] = {"idx": int(ml_idx), "conf": float(ml_conf)}
        if col_reset.button("Reset suggerimento", key=f"reset_ml_{selected_idx}"):
            st.session_state.pop(ml_state_key, None)

        ml_state = st.session_state.get(ml_state_key)
        if isinstance(ml_state, dict):
            ml_idx = int(ml_state.get("idx", 0))
            ml_conf = float(ml_state.get("conf", 0.0))
            ml_preset = auto_preset_library[ml_idx]
            st.write(
                {
                    "preset_ml": ml_preset["name"],
                    "confidence": round(ml_conf, 3),
                }
            )
            if st.button("Applica preset ML", key=f"apply_ml_preset_{selected_idx}"):
                apply_preset_cb(ml_preset)
                st.rerun()
        else:
            st.caption("Premi `Calcola suggerimento ML` solo quando vuoi il preset automatico.")

    axes_info = _get_profile_axes(selected, y_axis_mode)

    pipeline = run_filter_pipeline_cb(selected.data, cfg)
    stage_labels = [stage_name for stage_name, _stage_data in pipeline]
    default_stage_idx = len(stage_labels) - 1
    for idx in range(len(stage_labels) - 1, -1, -1):
        label = str(stage_labels[idx]).lower()
        if "hilbert" not in label:
            default_stage_idx = idx
            break
    stage_index = st.selectbox(
        "Stadio da visualizzare nel radargramma filtrato",
        options=list(range(len(stage_labels))),
        index=default_stage_idx,
        format_func=lambda idx: stage_labels[idx],
    )
    filtered = pipeline[stage_index][1]

    raw_std = float(np.nanstd(selected.data))
    raw_min = float(np.nanmin(selected.data))
    raw_max = float(np.nanmax(selected.data))
    filtered_std = float(np.nanstd(filtered))
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Min", f"{raw_min:.3f}")
    col_b.metric("Max", f"{raw_max:.3f}")
    col_c.metric("Std", f"{raw_std:.3f}")
    if raw_std < 1.0e-6:
        volume_std = selected.metadata.get("radar_std")
        if isinstance(volume_std, (float, int)) and float(volume_std) > 1.0:
            st.warning(
                "Questo canale sembra quasi vuoto, ma il volume contiene segnale. "
                "Prova a selezionare un altro canale/profilo."
            )
        else:
            st.warning(
                "Questo profilo risulta quasi costante (Std ~ 0). "
                "Potrebbe essere un problema di decoding del blocco radar nel file OGPR."
            )
    elif raw_std < 0.1:
        st.warning(
            "Profilo con dinamica molto bassa. "
            "Puoi provare a disattivare i filtri o selezionare un altro canale."
        )
    if raw_std > 0 and filtered_std < (0.02 * raw_std):
        st.info(
            "Lo stadio filtrato selezionato ha una dinamica molto ridotta rispetto al raw. "
            "Controlla i passaggi nel pannello diagnostica filtri."
        )
    with st.expander("Diagnostica orientamento (tempo)"):
        ns = selected.data.shape[1]
        nwin = max(8, int(round(0.10 * ns)))
        early_raw = float(np.nanmean(np.abs(selected.data[:, :nwin])))
        late_raw = float(np.nanmean(np.abs(selected.data[:, -nwin:])))
        ratio_raw = late_raw / (early_raw + 1.0e-12)
        early_f = float(np.nanmean(np.abs(filtered[:, :nwin])))
        late_f = float(np.nanmean(np.abs(filtered[:, -nwin:])))
        ratio_f = late_f / (early_f + 1.0e-12)
        st.write(
            {
                "early_raw_abs_mean": early_raw,
                "late_raw_abs_mean": late_raw,
                "late_over_early_raw": ratio_raw,
                "early_filtered_abs_mean": early_f,
                "late_filtered_abs_mean": late_f,
                "late_over_early_filtered": ratio_f,
            }
        )
        if ratio_raw > 3.0:
            st.warning(
                "Nel RAW l'energia e concentrata nel fondo traccia (late >> early). "
                "Questo indica piu spesso t0/layout/artefatto filtri che semplice inversione asse."
            )
        elif ratio_raw < 0.33:
            st.info("Nel RAW l'energia e maggiore all'inizio traccia (comportamento spesso atteso vicino a t0).")
        else:
            st.info("Distribuzione energetica RAW abbastanza bilanciata lungo il tempo.")

    reverse_default = ("Profondit" in axes_info["y_title"]) or ("Tempo" in axes_info["y_title"])
    reverse_y = (not reverse_default) if bool(y_flip_debug) else reverse_default
    with st.expander("Diagnostica filtri (step-by-step)"):
        stats_rows = []
        for step_name, step_data in pipeline:
            stats_rows.append(
                {
                    "step": step_name,
                    "min": float(np.nanmin(step_data)),
                    "max": float(np.nanmax(step_data)),
                    "mean": float(np.nanmean(step_data)),
                    "std": float(np.nanstd(step_data)),
                }
            )
        st.dataframe(stats_rows, use_container_width=True, hide_index=True)
    with st.expander("Spettro ampiezza (makeAmpspec)"):
        f_raw, a_raw = make_amplitude_spectrum(selected.data, sample_rate=float(cfg.sample_rate))
        f_flt, a_flt = make_amplitude_spectrum(filtered, sample_rate=float(cfg.sample_rate))
        if f_raw.size > 0 and a_raw.size > 0 and f_flt.size > 0 and a_flt.size > 0:
            fig_spec = go.Figure()
            fig_spec.add_trace(
                go.Scatter(
                    x=f_raw,
                    y=np.nanmean(a_raw, axis=0),
                    mode="lines",
                    name="Raw",
                )
            )
            fig_spec.add_trace(
                go.Scatter(
                    x=f_flt,
                    y=np.nanmean(a_flt, axis=0),
                    mode="lines",
                    name="Filtrato",
                )
            )
            fig_spec.update_layout(
                title="Spettro di ampiezza medio",
                xaxis_title="Frequenza (MHz)",
                yaxis_title="Ampiezza",
            )
            st.plotly_chart(fig_spec, use_container_width=True)
        else:
            st.caption("Spettro non disponibile per questo profilo.")
    with st.expander("Dettagli parser profilo"):
        st.json(selected.metadata)

    traces_count, samples_count = selected.data.shape
    sample_idx = st.slider(
        "Indice sample (time)",
        min_value=0,
        max_value=samples_count - 1,
        value=(samples_count - 1) // 2,
        key=f"sample_profile_{selected_idx}",
    )
    trace_idx = st.slider(
        "Indice traccia",
        min_value=0,
        max_value=traces_count - 1,
        value=(traces_count - 1) // 2,
        key=f"trace_profile_{selected_idx}",
    )
    view_mode = st.radio(
        "Visualizzazione radargrammi",
        options=["Solo filtrato", "Grezzo + filtrato", "Solo grezzo"],
        index=1,
        horizontal=True,
    )
    shared_scale = st.checkbox(
        "Scala colore condivisa raw/filtrato (debug filtri)",
        value=True,
        help="Usa la stessa scala sul grezzo e sul filtrato per rendere visibili le differenze reali.",
    )

    if shared_scale:
        vmin, vmax = _robust_bounds(selected.data)
        raw_disp = _normalize_with_fixed_bounds(selected.data, vmin, vmax)
        filtered_disp = _normalize_with_fixed_bounds(filtered, vmin, vmax)
    else:
        raw_disp = normalize_for_display(selected.data)
        filtered_disp = normalize_for_display(filtered)

    diff = np.asarray(filtered, dtype=np.float64) - np.asarray(selected.data, dtype=np.float64)
    with st.expander("Verifica applicazione filtri"):
        st.write(
            {
                "std_raw": float(np.nanstd(selected.data)),
                "std_filtered": float(np.nanstd(filtered)),
                "mean_abs_diff": float(np.nanmean(np.abs(diff))),
                "std_diff": float(np.nanstd(diff)),
                "corr_raw_filtered": _corr_flat(selected.data, filtered),
            }
        )

    if view_mode == "Solo filtrato":
        hovertemplate = f"x: %{{x:.2f}}<br>y: %{{y:.2f}}<br>Amp: %{{z:.3f}}<extra></extra>"
        fig_filtered = go.Figure(
            data=go.Heatmap(
                z=filtered_disp.T,
                x=axes_info["x_vals"],
                y=axes_info["y_vals"],
                colorscale=viz.profile_filtered_scale,
                reversescale=viz.profile_reverse,
                hovertemplate=hovertemplate,
            )
        )
        fig_filtered.update_layout(
            title=f"Profilo filtrato - {selected.label}",
            xaxis_title=axes_info["x_title"],
            yaxis_title=axes_info["y_title"],
        )
        if reverse_y:
            fig_filtered.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_filtered, use_container_width=True)
    elif view_mode == "Solo grezzo":
        hovertemplate = f"x: %{{x:.2f}}<br>y: %{{y:.2f}}<br>Amp: %{{z:.3f}}<extra></extra>"
        fig_raw = go.Figure(
            data=go.Heatmap(
                z=raw_disp.T,
                x=axes_info["x_vals"],
                y=axes_info["y_vals"],
                colorscale=viz.profile_raw_scale,
                reversescale=viz.profile_reverse,
                hovertemplate=hovertemplate,
            )
        )
        fig_raw.update_layout(
            title=f"Profilo originale - {selected.label}",
            xaxis_title=axes_info["x_title"],
            yaxis_title=axes_info["y_title"],
        )
        if reverse_y:
            fig_raw.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_raw, use_container_width=True)
    else:
        col_left, col_right = st.columns(2)
        hovertemplate = f"x: %{{x:.2f}}<br>y: %{{y:.2f}}<br>Amp: %{{z:.3f}}<extra></extra>"
        with col_left:
            fig_raw = go.Figure(
                data=go.Heatmap(
                    z=raw_disp.T,
                    x=axes_info["x_vals"],
                    y=axes_info["y_vals"],
                    colorscale=viz.profile_raw_scale,
                    reversescale=viz.profile_reverse,
                    hovertemplate=hovertemplate,
                )
            )
            fig_raw.update_layout(
                title=f"Profilo originale - {selected.label}",
                xaxis_title=axes_info["x_title"],
                yaxis_title=axes_info["y_title"],
            )
            if reverse_y:
                fig_raw.update_yaxes(autorange="reversed")
            st.plotly_chart(fig_raw, use_container_width=True)

        with col_right:
            fig_filtered = go.Figure(
                data=go.Heatmap(
                    z=filtered_disp.T,
                    x=axes_info["x_vals"],
                    y=axes_info["y_vals"],
                    colorscale=viz.profile_filtered_scale,
                    reversescale=viz.profile_reverse,
                    hovertemplate=hovertemplate,
                )
            )
            fig_filtered.update_layout(
                title=f"Profilo filtrato - {selected.label}",
                xaxis_title=axes_info["x_title"],
                yaxis_title=axes_info["y_title"],
            )
            if reverse_y:
                fig_filtered.update_yaxes(autorange="reversed")
            st.plotly_chart(fig_filtered, use_container_width=True)

    timeslice_fig = go.Figure()
    if view_mode in {"Solo grezzo", "Grezzo + filtrato"}:
        timeslice_fig.add_trace(go.Scatter(x=axes_info["x_vals"], y=selected.data[:, sample_idx], mode="lines", name="Originale"))
    if view_mode in {"Solo filtrato", "Grezzo + filtrato"}:
        timeslice_fig.add_trace(go.Scatter(x=axes_info["x_vals"], y=filtered[:, sample_idx], mode="lines", name="Filtrato"))
    timeslice_fig.update_layout(
        title=f"Time-line del profilo al sample {sample_idx}",
        xaxis_title=axes_info["x_title"],
        yaxis_title="Ampiezza",
    )
    st.plotly_chart(timeslice_fig, use_container_width=True)

    ascan_fig = go.Figure()
    if view_mode in {"Solo grezzo", "Grezzo + filtrato"}:
        ascan_fig.add_trace(go.Scatter(x=axes_info["y_vals"], y=selected.data[trace_idx, :], mode="lines", name="Originale"))
    if view_mode in {"Solo filtrato", "Grezzo + filtrato"}:
        ascan_fig.add_trace(go.Scatter(x=axes_info["y_vals"], y=filtered[trace_idx, :], mode="lines", name="Filtrato"))
    ascan_fig.update_layout(
        title=f"A-scan della traccia {trace_idx}",
        xaxis_title=axes_info["y_title"],
        yaxis_title="Ampiezza",
    )
    if reverse_y:
        ascan_fig.update_xaxes(autorange="reversed")
    st.plotly_chart(ascan_fig, use_container_width=True)


def _collect_timeslice_points(
    profiles: list[OgprProfile],
    arrays: list[np.ndarray],
    sample_idx: int,
    coord_cfg: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    vals: list[np.ndarray] = []
    owners: list[np.ndarray] = []
    gps_profiles = 0

    for line_idx, (profile, data) in enumerate(zip(profiles, arrays)):
        if sample_idx >= data.shape[1]:
            continue
        x, y, has_gps = _coordinates_for_profile(profile, fallback_line=line_idx, coord_cfg=coord_cfg)
        if has_gps:
            gps_profiles += 1
        values = np.asarray(data[:, sample_idx], dtype=np.float64)
        mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(values)
        if not np.any(mask):
            continue

        xs.append(x[mask])
        ys.append(y[mask])
        vals.append(values[mask])
        owners.append(np.full(np.count_nonzero(mask), line_idx, dtype=np.int32))

    if not xs:
        return (
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.int32),
            gps_profiles,
        )

    return (
        np.concatenate(xs),
        np.concatenate(ys),
        np.concatenate(vals),
        np.concatenate(owners),
        gps_profiles,
    )


def _render_coverage_map(profiles: list[OgprProfile], coord_cfg: Any) -> None:
    fig = go.Figure()
    for line_idx, profile in enumerate(profiles):
        x, y, _has_gps = _coordinates_for_profile(profile, fallback_line=line_idx, coord_cfg=coord_cfg)
        fig.add_trace(
            go.Scattergl(
                x=x,
                y=y,
                mode="lines",
                name=profile.label,
                hovertemplate=f"{profile.label}<extra></extra>",
                showlegend=len(profiles) <= 12,
            )
        )
    fig.update_layout(
        title="Copertura profili (coordinate file ogpr)",
        xaxis_title="X / Longitudine",
        yaxis_title="Y / Latitudine",
        legend_title="Profili",
    )
    st.plotly_chart(fig, use_container_width=True)


def _corr_1d(a: np.ndarray, b: np.ndarray) -> float:
    av = np.asarray(a, dtype=np.float64).ravel()
    bv = np.asarray(b, dtype=np.float64).ravel()
    mask = np.isfinite(av) & np.isfinite(bv)
    if np.count_nonzero(mask) < 3:
        return np.nan
    av = av[mask]
    bv = bv[mask]
    av = av - np.mean(av)
    bv = bv - np.mean(bv)
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
    if denom <= 0:
        return np.nan
    return float(np.dot(av, bv) / denom)


def _idw_interpolate_grid(
    points: np.ndarray,
    values: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    radius: float,
    power: float,
) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    vals = np.asarray(values, dtype=np.float64).ravel()
    valid = np.isfinite(pts[:, 0]) & np.isfinite(pts[:, 1]) & np.isfinite(vals)
    pts = pts[valid]
    vals = vals[valid]
    if pts.shape[0] == 0:
        return np.full_like(grid_x, np.nan, dtype=np.float64)

    q = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    tree = cKDTree(pts)
    r = max(float(radius), 0.0)
    p = max(float(power), 1.0e-6)

    if r > 0:
        neighbors = tree.query_ball_point(q, r=r)
    else:
        # Fallback nearest if radius non valido
        dist, idx = tree.query(q, k=1)
        out = vals[idx]
        out[~np.isfinite(dist)] = np.nan
        return out.reshape(grid_x.shape)

    out = np.full(q.shape[0], np.nan, dtype=np.float64)
    for i, ids in enumerate(neighbors):
        if not ids:
            continue
        loc = pts[ids]
        d = np.sqrt((loc[:, 0] - q[i, 0]) ** 2 + (loc[:, 1] - q[i, 1]) ** 2)
        v = vals[ids]
        zero = d < 1.0e-12
        if np.any(zero):
            out[i] = float(np.nanmean(v[zero]))
            continue
        w = 1.0 / np.maximum(d, 1.0e-12) ** p
        out[i] = float(np.nansum(w * v) / (np.nansum(w) + 1.0e-12))

    if np.any(~np.isfinite(out)):
        dist, idx = tree.query(q, k=1)
        nn = vals[idx]
        out = np.where(np.isfinite(out), out, nn)
        out[~np.isfinite(dist)] = np.nan
    return out.reshape(grid_x.shape)


def _interpolate_grid(
    points: np.ndarray,
    values: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    method: str,
    idw_radius: float,
    idw_power: float,
) -> np.ndarray:
    method_norm = str(method).lower()
    if method_norm == "idw":
        return _idw_interpolate_grid(points, values, grid_x, grid_y, radius=idw_radius, power=idw_power)

    try:
        grid = griddata(points, values, (grid_x, grid_y), method=method_norm)
        if method_norm != "nearest":
            nearest_grid = griddata(points, values, (grid_x, grid_y), method="nearest")
            grid = np.where(np.isfinite(grid), grid, nearest_grid)
        return np.asarray(grid, dtype=np.float64)
    except Exception:
        return np.asarray(griddata(points, values, (grid_x, grid_y), method="nearest"), dtype=np.float64)


def _coherence_map_for_sample(
    profiles: list[OgprProfile],
    arrays: list[np.ndarray],
    sample_idx: int,
    coord_cfg: Any,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    method: str,
    idw_radius: float,
    idw_power: float,
    sample_rate_hz: float,
    nwavelength: float,
    antenna_mhz: float,
) -> np.ndarray:
    min_samples = min(arr.shape[1] for arr in arrays)
    dt_ns = 1.0e9 / max(float(sample_rate_hz), 1.0)
    wave_ns = 1000.0 / max(float(antenna_mhz), 1.0)
    nsamp = int(max(3, round(float(nwavelength) * wave_ns / max(dt_ns, 1.0e-9))))
    half = nsamp // 2
    start = max(0, int(sample_idx) - half)
    stop = min(min_samples - 1, int(sample_idx) + half)
    idxs = np.arange(start, stop + 1, dtype=np.int32)
    if idxs.size < 3:
        return np.full_like(grid_x, np.nan, dtype=np.float64)

    slices = []
    for idx in idxs:
        xk, yk, vk, _owners, _gps = _collect_timeslice_points(profiles, arrays, int(idx), coord_cfg=coord_cfg)
        if xk.size < 4:
            slices.append(np.full_like(grid_x, np.nan, dtype=np.float64))
            continue
        pts = np.column_stack((xk, yk))
        slices.append(_interpolate_grid(pts, vk, grid_x, grid_y, method, idw_radius, idw_power))
    cube = np.stack(slices, axis=0)  # [t, y, x]

    ny, nx = grid_x.shape
    coh = np.full((ny, nx), np.nan, dtype=np.float64)
    for iy in range(1, ny - 1):
        for ix in range(1, nx - 1):
            ref = cube[:, iy, ix]
            if not np.all(np.isfinite(ref)):
                continue
            neighbors = [
                cube[:, iy - 1, ix],
                cube[:, iy + 1, ix],
                cube[:, iy, ix - 1],
                cube[:, iy, ix + 1],
            ]
            vals = []
            for nb in neighbors:
                if not np.all(np.isfinite(nb)):
                    continue
                corr = _corr_1d(ref, nb)
                if np.isfinite(corr):
                    vals.append(1.0 - corr)
            if vals:
                coh[iy, ix] = float(np.mean(vals))
    return coh


def _build_interpolated_cube(
    profiles: list[OgprProfile],
    arrays: list[np.ndarray],
    sample_indices: np.ndarray,
    coord_cfg: Any,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    method: str,
    idw_radius: float,
    idw_power: float,
) -> np.ndarray:
    slices: list[np.ndarray] = []
    for idx in np.asarray(sample_indices, dtype=np.int32).tolist():
        xk, yk, vk, _owners, _gps = _collect_timeslice_points(profiles, arrays, int(idx), coord_cfg=coord_cfg)
        if xk.size < 4:
            slices.append(np.full_like(grid_x, np.nan, dtype=np.float64))
            continue
        pts = np.column_stack((xk, yk))
        grid = _interpolate_grid(
            pts,
            vk,
            grid_x,
            grid_y,
            method=method,
            idw_radius=float(idw_radius),
            idw_power=float(idw_power),
        )
        slices.append(np.asarray(grid, dtype=np.float64))
    if not slices:
        return np.full((0,) + grid_x.shape, np.nan, dtype=np.float64)
    return np.stack(slices, axis=0)


def render_timeslice_tab(
    profiles: list[OgprProfile],
    cfg: Any,
    viz: Any,
    coord_cfg: Any,
    profiles_cache_key: tuple[Any, ...],
    ordered_processing_steps: Callable[[Any], list[tuple[str, int]]],
    filter_cfg_signature_cb: Callable[[Any], tuple[Any, ...]],
    apply_filters_cb: Callable[[np.ndarray, Any], np.ndarray],
) -> None:
    mode_label = "Base consigliato" if cfg.workflow_mode == "base" else "Manuale"
    st.caption(f"Modalita workflow: {mode_label}")
    ordered_steps = ordered_processing_steps(cfg)
    if ordered_steps:
        st.caption("Ordine processing: " + " -> ".join([f"{name}({order})" for name, order in ordered_steps]))

    use_filtered = st.checkbox("Usa dati filtrati per la time-slice", value=True)
    map_mode = st.selectbox("Mappa", options=["ampiezza", "coherence"], index=0)
    method = st.selectbox("Interpolazione", options=["linear", "nearest", "cubic", "idw"], index=0)
    idw_radius = st.number_input("IDW raggio (m/unita XY)", min_value=0.0, value=0.0, step=0.1, disabled=method != "idw")
    idw_power = st.slider("IDW power", min_value=1.0, max_value=5.0, value=2.0, step=0.1, disabled=method != "idw")
    grid_size = st.slider("Risoluzione griglia", min_value=50, max_value=400, value=180, step=10)
    show_points = st.checkbox("Mostra punti campionati", value=True)
    coherence_nwavelength = st.slider(
        "Coherence: numero lunghezze d'onda",
        min_value=1.0,
        max_value=8.0,
        value=2.0,
        step=0.5,
        disabled=map_mode != "coherence",
    )
    coherence_freq_mhz = st.number_input(
        "Coherence: frequenza antenna (MHz)",
        min_value=10.0,
        max_value=3000.0,
        value=400.0,
        step=10.0,
        disabled=map_mode != "coherence",
    )
    with st.expander("Processing 3D cube (MATLAB-like)", expanded=False):
        use_cube_processing = st.checkbox(
            "Abilita processing 3D cube su time-slice",
            value=False,
            help="Applica linearInterpolation_3Dcube / semblanceSmoothing / normalize3d su un cubo interpolato.",
        )
        cube_sample_step = st.number_input(
            "reduceNumberOfSamples n (solo riduzione, >=1)",
            min_value=1,
            max_value=32,
            value=2,
            step=1,
            disabled=not use_cube_processing,
        )
        cube_resample_factor = st.number_input(
            "reduceNumberOfSamples factor post-cube (0.25..4)",
            min_value=0.25,
            max_value=4.0,
            value=1.0,
            step=0.25,
            disabled=not use_cube_processing,
            help="1=off, >1 riduce ancora i sample, <1 li incrementa via interpolazione spline.",
        )
        cube_linear_radius = st.number_input(
            "linearInterpolation_3Dcube raggio XY",
            min_value=0.0,
            value=0.0,
            step=0.1,
            disabled=not use_cube_processing,
            help="0 = off. Interpola celle NaN entro questo raggio nel piano XY.",
        )
        cube_apply_semblance = st.checkbox(
            "semblanceSmoothing",
            value=False,
            disabled=not use_cube_processing,
        )
        cube_semblance_window = st.slider(
            "Semblance window",
            min_value=3,
            max_value=21,
            value=7,
            step=2,
            disabled=(not use_cube_processing) or (not cube_apply_semblance),
        )
        cube_semblance_exp = st.slider(
            "Semblance exponent",
            min_value=0.0,
            max_value=4.0,
            value=1.0,
            step=0.1,
            disabled=(not use_cube_processing) or (not cube_apply_semblance),
        )
        cube_apply_normalize3d = st.checkbox(
            "normalize3d",
            value=False,
            disabled=not use_cube_processing,
        )
        cube_norm_qclip = st.slider(
            "normalize3d qclip",
            min_value=0.5,
            max_value=1.0,
            value=0.98,
            step=0.01,
            disabled=(not use_cube_processing) or (not cube_apply_normalize3d),
        )

    if use_filtered:
        cache_key = (profiles_cache_key, filter_cfg_signature_cb(cfg))
        if st.session_state.get("_timeslice_filtered_key") != cache_key:
            with st.spinner("Preparo i dati filtrati per la time-slice..."):
                st.session_state["_timeslice_filtered_arrays"] = [apply_filters_cb(profile.data, cfg) for profile in profiles]
                st.session_state["_timeslice_filtered_key"] = cache_key
        arrays = st.session_state.get("_timeslice_filtered_arrays")
        if not isinstance(arrays, list) or len(arrays) != len(profiles):
            arrays = [apply_filters_cb(profile.data, cfg) for profile in profiles]
    else:
        arrays = [profile.data for profile in profiles]

    min_samples = min(arr.shape[1] for arr in arrays)
    sample_idx = st.slider(
        "Sample per time-slice",
        min_value=0,
        max_value=min_samples - 1,
        value=(min_samples - 1) // 2,
        key="timeslice_sample_idx",
    )

    x, y, values, owners, gps_profiles = _collect_timeslice_points(
        profiles,
        arrays,
        sample_idx,
        coord_cfg=coord_cfg,
    )
    if x.size < 4:
        st.error("Punti insufficienti per creare una time-slice interpolata.")
        return

    if gps_profiles == 0:
        st.warning("Nessuna coordinata GPS trovata: time-slice su coordinate locali sintetiche.")
    elif gps_profiles < len(profiles):
        st.warning(
            f"Coordinate GPS trovate in {gps_profiles}/{len(profiles)} profili. "
            "Per i restanti sono usate coordinate locali."
        )

    unique_points = np.unique(np.column_stack((x, y)), axis=0).shape[0]
    if method == "cubic" and unique_points < 16:
        st.warning("Punti unici insufficienti per interpolazione cubic. Uso linear.")
        method = "linear"
    if method == "linear" and unique_points < 3:
        st.warning("Punti unici insufficienti per linear. Uso nearest.")
        method = "nearest"

    xi = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), grid_size)
    yi = np.linspace(float(np.nanmin(y)), float(np.nanmax(y)), grid_size)
    grid_x, grid_y = np.meshgrid(xi, yi)

    display_sample_idx = int(sample_idx)
    cube_grid: np.ndarray | None = None
    if use_cube_processing and map_mode != "ampiezza":
        st.info("Processing 3D cube disponibile solo per mappa ampiezza.")
    if use_cube_processing and map_mode == "ampiezza":
        step = max(int(cube_sample_step), 1)
        sample_indices = np.arange(0, min_samples, step, dtype=np.int32)
        if sample_indices.size == 0 or int(sample_indices[-1]) != (min_samples - 1):
            sample_indices = np.append(sample_indices, np.int32(min_samples - 1))

        cube_cache_key = (
            profiles_cache_key,
            bool(use_filtered),
            filter_cfg_signature_cb(cfg) if use_filtered else "raw",
            int(grid_size),
            str(method),
            round(float(idw_radius), 6),
            round(float(idw_power), 6),
            tuple(sample_indices.tolist()),
            round(float(cube_resample_factor), 6),
            round(float(cube_linear_radius), 6),
            bool(cube_apply_semblance),
            int(cube_semblance_window),
            round(float(cube_semblance_exp), 6),
            bool(cube_apply_normalize3d),
            round(float(cube_norm_qclip), 6),
        )

        if st.session_state.get("_timeslice_cube_key") != cube_cache_key:
            with st.spinner("Costruzione cubo 3D interpolato..."):
                cube = _build_interpolated_cube(
                    profiles=profiles,
                    arrays=arrays,
                    sample_indices=sample_indices,
                    coord_cfg=coord_cfg,
                    grid_x=grid_x,
                    grid_y=grid_y,
                    method=method,
                    idw_radius=float(idw_radius),
                    idw_power=float(idw_power),
                )
                if cube.size > 0 and not np.isclose(float(cube_resample_factor), 1.0):
                    cube = apply_reduce_number_of_samples(
                        cube,
                        factor=float(cube_resample_factor),
                        time_axis=0,
                    )
                    # aggiorna gli indici sample reali lungo asse tempo
                    old_idx = sample_indices.astype(np.float64)
                    nt_new = cube.shape[0]
                    if float(cube_resample_factor) > 1.0:
                        step_r = max(int(round(float(cube_resample_factor))), 1)
                        new_pos = np.arange(0.0, float(len(old_idx) - 1) + 1.0e-9, float(step_r), dtype=np.float64)
                    else:
                        step_r = max(float(cube_resample_factor), 1.0e-6)
                        new_pos = np.arange(0.0, float(len(old_idx) - 1), step_r, dtype=np.float64)
                        if new_pos.size == 0 or new_pos[-1] < (len(old_idx) - 1):
                            new_pos = np.append(new_pos, float(len(old_idx) - 1))
                    new_pos = np.clip(new_pos, 0.0, float(len(old_idx) - 1))
                    sample_indices = np.interp(new_pos, np.arange(len(old_idx), dtype=np.float64), old_idx).astype(np.int32)
                    if sample_indices.size != nt_new:
                        sample_indices = np.linspace(
                            int(old_idx[0]),
                            int(old_idx[-1]),
                            nt_new,
                            dtype=np.int32,
                        )
                if cube.size > 0 and float(cube_linear_radius) > 0:
                    dx = float(np.abs(xi[1] - xi[0])) if xi.size > 1 else 1.0
                    dy = float(np.abs(yi[1] - yi[0])) if yi.size > 1 else dx
                    cube = apply_linear_interpolation_3d_cube(
                        cube,
                        radius_xy=float(cube_linear_radius),
                        dx=dx,
                        dy=dy,
                        time_axis=0,
                    )
                if cube.size > 0 and bool(cube_apply_semblance):
                    cube = apply_semblance_smoothing(
                        cube,
                        window=int(cube_semblance_window),
                        exponent=float(cube_semblance_exp),
                        normalize_after=False,
                        time_axis=0,
                    )
                if cube.size > 0 and bool(cube_apply_normalize3d):
                    cube = apply_normalize3d(
                        cube,
                        qclip=float(cube_norm_qclip),
                        time_axis=0,
                    )
                st.session_state["_timeslice_cube_key"] = cube_cache_key
                st.session_state["_timeslice_cube"] = np.asarray(cube, dtype=np.float64)
                st.session_state["_timeslice_cube_samples"] = np.asarray(sample_indices, dtype=np.int32)

        cube = st.session_state.get("_timeslice_cube")
        cube_samples = st.session_state.get("_timeslice_cube_samples")
        if isinstance(cube, np.ndarray) and cube.ndim == 3 and isinstance(cube_samples, np.ndarray) and cube_samples.size > 0:
            idx_pos = int(np.argmin(np.abs(cube_samples.astype(np.int64) - int(sample_idx))))
            display_sample_idx = int(cube_samples[idx_pos])
            cube_slice = np.asarray(cube[idx_pos, :, :], dtype=np.float64)
            if np.isfinite(cube_slice).any():
                cube_grid = cube_slice
                if display_sample_idx != int(sample_idx):
                    st.caption(
                        f"Time-slice visualizzata al sample {display_sample_idx} (piu vicino a {sample_idx} con riduzione n={step})."
                    )
                    x2, y2, v2, o2, _gps2 = _collect_timeslice_points(
                        profiles,
                        arrays,
                        display_sample_idx,
                        coord_cfg=coord_cfg,
                    )
                    if x2.size >= 4:
                        x, y, values, owners = x2, y2, v2, o2
            else:
                st.warning("Cubo 3D non valido con i parametri correnti: fallback a interpolazione 2D.")
        else:
            st.warning("Cubo 3D non disponibile: fallback a interpolazione 2D.")

    points = np.column_stack((x, y))
    if map_mode == "coherence":
        grid = _coherence_map_for_sample(
            profiles=profiles,
            arrays=arrays,
            sample_idx=sample_idx,
            coord_cfg=coord_cfg,
            grid_x=grid_x,
            grid_y=grid_y,
            method=method,
            idw_radius=float(idw_radius),
            idw_power=float(idw_power),
            sample_rate_hz=float(cfg.sample_rate),
            nwavelength=float(coherence_nwavelength),
            antenna_mhz=float(coherence_freq_mhz),
        )
        if not np.isfinite(grid).any():
            st.warning("Coherence non calcolabile con i parametri correnti; provo a mostrare l'ampiezza.")
            grid = _interpolate_grid(
                points,
                values,
                grid_x,
                grid_y,
                method=method,
                idw_radius=float(idw_radius),
                idw_power=float(idw_power),
            )
            map_mode = "ampiezza"
    else:
        if cube_grid is not None:
            grid = cube_grid
        else:
            grid = _interpolate_grid(
                points,
                values,
                grid_x,
                grid_y,
                method=method,
                idw_radius=float(idw_radius),
                idw_power=float(idw_power),
            )

    slice_fig = go.Figure()
    slice_fig.add_trace(
        go.Heatmap(
            x=xi,
            y=yi,
            z=normalize_for_display(grid),
            colorscale=viz.timeslice_scale,
            reversescale=viz.timeslice_reverse,
            colorbar={"title": "Coherence norm" if map_mode == "coherence" else "Amp. norm"},
            opacity=0.92,
        )
    )
    if show_points:
        slice_fig.add_trace(
            go.Scattergl(
                x=x,
                y=y,
                mode="markers",
                marker={
                    "size": 4,
                    "color": values,
                    "colorscale": viz.points_scale,
                    "reversescale": viz.points_reverse,
                    "opacity": 0.7,
                },
                text=[profiles[int(owner)].label for owner in owners],
                hovertemplate="x=%{x:.3f}<br>y=%{y:.3f}<br>amp=%{marker.color:.3f}<br>%{text}<extra></extra>",
                name="Punti campionati",
            )
        )
    slice_fig.update_layout(
        title=f"Time-slice {map_mode} al sample {display_sample_idx}",
        xaxis_title="X / Longitudine",
        yaxis_title="Y / Latitudine",
    )
    st.plotly_chart(slice_fig, use_container_width=True)

    _render_coverage_map(profiles, coord_cfg=coord_cfg)
