"""OGPR Browse App – entry point Streamlit."""
from __future__ import annotations

from io import BytesIO
from typing import Any

import numpy as np
import streamlit as st

from gpr_app.pipeline import (
    apply_filters,
    filter_cfg_signature,
    has_duplicate_processing_orders,
    ordered_processing_steps,
    preset_to_filter_config,
    run_filter_pipeline,
)
from gpr_app.ui.sidebar import (
    apply_preset_to_session_state,
    build_coordinate_config,
    build_decode_config,
    build_filter_config,
    build_visual_config,
)
from gpr_app.utils import sample_rate_from_metadata
from ml_presets import AUTO_PRESET_LIBRARY, predict_ml_preset, train_ml_preset_model
from radar_io import OgprProfile, load_ogpr_profiles
from ui_views import render_profile_tab, render_timeslice_tab


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _load_profiles_cached(
    file_name: str,
    raw_bytes: bytes,
    offset_mode: str,
    layout_mode: str,
) -> list[OgprProfile]:
    offset_override = None if offset_mode == "auto" else offset_mode
    layout_override = None if layout_mode == "auto" else layout_mode
    return load_ogpr_profiles(
        BytesIO(raw_bytes),
        file_name=file_name,
        offset_mode_override=offset_override,
        layout_mode_override=layout_override,
    )


@st.cache_resource(show_spinner=False)
def _train_ml_preset_model() -> dict[str, Any]:
    presets = [dict(preset) for preset in AUTO_PRESET_LIBRARY]
    return train_ml_preset_model(
        presets=presets,
        preset_to_filter_config=preset_to_filter_config,
        apply_filters=apply_filters,
        sample_rate=1.0e9,
        n_samples=28,
        seed=42,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _uploaded_files_signature(uploaded_files: list[Any], decode_cfg: Any) -> tuple[Any, ...]:
    entries = [
        (f.name.lower(), int(getattr(f, "size", 0)), str(getattr(f, "file_id", "")))
        for f in sorted(uploaded_files, key=lambda f: f.name.lower())
    ]
    return (decode_cfg.offset_mode, decode_cfg.layout_mode, tuple(entries))


def _predict_ml_preset(profile_data: np.ndarray, sample_rate: float) -> tuple[int, float]:
    model = _train_ml_preset_model()
    return predict_ml_preset(
        profile_data=profile_data,
        sample_rate=sample_rate,
        model=model,
        preset_to_filter_config=preset_to_filter_config,
        apply_filters=apply_filters,
    )


def _auto_configure_from_profiles(profiles: list[OgprProfile]) -> None:
    """Auto-imposta sample rate e frequenze di taglio dal mediano dei metadati."""
    sample_rates = [
        sr for profile in profiles
        if (sr := sample_rate_from_metadata(profile.metadata)) is not None and sr > 1.0
    ]
    if not sample_rates:
        return
    sample_rate = float(np.nanmedian(np.asarray(sample_rates, dtype=np.float64)))
    if not np.isfinite(sample_rate) or sample_rate <= 1.0:
        return
    sample_rate = float(np.clip(sample_rate, 1.0e6, 2.0e10))
    nyquist = sample_rate * 0.5
    low_cut = max(1.0e6, 0.04 * nyquist)
    high_cut = min(0.70 * nyquist, nyquist * 0.999)
    if high_cut <= low_cut:
        high_cut = min(nyquist * 0.90, low_cut * 1.5)
    st.session_state["f_sample_rate"] = sample_rate
    st.session_state["f_low_cut"] = float(low_cut)
    st.session_state["f_high_cut"] = float(high_cut)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="OGPR Browse App", layout="wide")
    st.title("OGPR Browse App")
    st.caption("Import multiplo file ogpr, visualizzazione profili e creazione time-slice con coordinate GPS")

    decode_cfg = build_decode_config()
    uploaded_files = st.file_uploader(
        "Carica i file .ogpr (uno o piu file, anche uno per canale)",
        type=["ogpr"],
        accept_multiple_files=True,
    )
    if not uploaded_files:
        st.info("Carica uno o piu file .ogpr per iniziare.")
        return

    profiles_key = _uploaded_files_signature(uploaded_files, decode_cfg)
    if st.session_state.get("_profiles_cache_key") != profiles_key:
        with st.spinner("Caricamento e decoding file OGPR..."):
            profiles: list[OgprProfile] = []
            errors: list[str] = []
            for uploaded in sorted(uploaded_files, key=lambda f: f.name.lower()):
                try:
                    profiles.extend(
                        _load_profiles_cached(
                            uploaded.name,
                            bytes(uploaded.getbuffer()),
                            decode_cfg.offset_mode,
                            decode_cfg.layout_mode,
                        )
                    )
                except Exception as exc:
                    errors.append(f"{uploaded.name}: {exc}")
        st.session_state.update({
            "_profiles_cache_key": profiles_key,
            "_profiles_cache_profiles": profiles,
            "_profiles_cache_errors": errors,
        })
        st.session_state.pop("_timeslice_filtered_key", None)
        st.session_state.pop("_timeslice_filtered_arrays", None)
        _auto_configure_from_profiles(profiles)
    else:
        profiles = st.session_state.get("_profiles_cache_profiles") or []
        errors = st.session_state.get("_profiles_cache_errors") or []

    if errors:
        st.error("Alcuni file non sono stati caricati:")
        for msg in errors:
            st.write(f"- {msg}")

    if not profiles:
        st.error("Nessun profilo valido caricato.")
        return

    profiles.sort(key=lambda p: (p.file_name.lower(), p.channel_index))
    st.success(f"Profili caricati: {len(profiles)} da {len(uploaded_files)} file.")

    cfg = build_filter_config()
    if cfg.workflow_mode == "manual" and has_duplicate_processing_orders(cfg):
        st.sidebar.info(
            "Ordini filtro duplicati rilevati: esecuzione comunque attiva con priorita fissa "
            "(gain -> bandpass -> background -> hilbert -> dewow -> smoothing) a parita di ordine."
        )
    viz = build_visual_config()
    coord_cfg = build_coordinate_config()

    section = st.radio("Sezione", options=["Profili", "Time-slice"], index=0, horizontal=True)
    if section == "Profili":
        render_profile_tab(
            profiles=profiles,
            cfg=cfg,
            viz=viz,
            auto_preset_library=AUTO_PRESET_LIBRARY,
            ordered_processing_steps=ordered_processing_steps,
            predict_ml_preset_cb=_predict_ml_preset,
            apply_preset_cb=apply_preset_to_session_state,
            run_filter_pipeline_cb=run_filter_pipeline,
        )
    else:
        render_timeslice_tab(
            profiles=profiles,
            cfg=cfg,
            viz=viz,
            coord_cfg=coord_cfg,
            profiles_cache_key=profiles_key,
            ordered_processing_steps=ordered_processing_steps,
            filter_cfg_signature_cb=filter_cfg_signature,
            apply_filters_cb=apply_filters,
        )


if __name__ == "__main__":
    main()
