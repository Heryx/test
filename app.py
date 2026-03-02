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
from gpr_app.metadata_validator import validate_metadata, get_ids_preset_hints
from ml_presets import AUTO_PRESET_LIBRARY, predict_ml_preset, train_ml_preset_model
from radar_io import OgprProfile, load_ogpr_profiles
from ui_views import render_profile_tab, render_timeslice_tab, render_multi_profile_tab


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


def _auto_configure_from_profiles_intelligent(profiles: list[OgprProfile]) -> None:
    """Auto-config intelligente: attivo solo se metadati validi."""
    if not profiles:
        return
    
    # Valida metadati del primo profilo con segnale
    profile_stds = [float(np.nanstd(p.data)) for p in profiles]
    best_idx = int(np.nanargmax(np.asarray(profile_stds, dtype=np.float64)))
    validation = validate_metadata(profiles[best_idx])
    
    if not validation.is_valid:
        st.sidebar.warning(
            f"⚠️ Metadati incompleti (score: {validation.completeness_score:.0%})\n\n"
            f"Auto-config disabilitato. Problemi:\n" + 
            "\n".join(f"- {w}" for w in validation.warnings)
        )
        return
    
    # Metadati validi: procedi con auto-config
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
    
    # Use IDS preset hints if antenna detected
    if validation.antenna_type and validation.detected_frequency_mhz:
        hints = get_ids_preset_hints(validation.antenna_type, validation.detected_frequency_mhz)
        low_cut = hints["bandpass_low_mhz"] * 1e6
        high_cut = hints["bandpass_high_mhz"] * 1e6
        
        st.sidebar.success(
            f"✅ Antenna rilevata: **{validation.antenna_type}**\n\n"
            f"Frequenza: {validation.detected_frequency_mhz:.0f} MHz\n\n"
            f"{hints['description']}\n\n"
            f"Auto-config attivo con preset IDS ottimizzato."
        )
    else:
        # Generic bandpass
        nyquist = sample_rate * 0.5
        low_cut = max(1.0e6, 0.04 * nyquist)
        high_cut = min(0.70 * nyquist, nyquist * 0.999)
        if high_cut <= low_cut:
            high_cut = min(nyquist * 0.90, low_cut * 1.5)
        
        st.sidebar.info(
            f"✅ Metadati validi (score: {validation.completeness_score:.0%})\n\n"
            "Auto-config attivo con preset generico."
        )
    
    st.session_state["f_sample_rate"] = sample_rate
    st.session_state["f_low_cut"] = float(low_cut)
    st.session_state["f_high_cut"] = float(high_cut)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="OGPR Browse App", layout="wide")
    st.title("📡 OGPR Browse App")
    st.caption(
        "Visualizzazione avanzata multi-profilo GPR con supporto IDS Stream DP/UP, "
        "auto-config intelligente e processing batch"
    )
    
    decode_cfg = build_decode_config()
    uploaded_files = st.file_uploader(
        "📁 Carica file .ogpr (multi-file e multi-canale supportati)",
        type=["ogpr"],
        accept_multiple_files=True,
    )
    
    if not uploaded_files:
        st.info("👆 Carica uno o più file .ogpr per iniziare. Supporto multi-canale per array IDS.")
        return
    
    profiles_key = _uploaded_files_signature(uploaded_files, decode_cfg)
    if st.session_state.get("_profiles_cache_key") != profiles_key:
        with st.spinner("⏳ Caricamento e decoding file OGPR..."):
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
        
        # Intelligent auto-config
        _auto_configure_from_profiles_intelligent(profiles)
    else:
        profiles = st.session_state.get("_profiles_cache_profiles") or []
        errors = st.session_state.get("_profiles_cache_errors") or []
    
    if errors:
        st.error("❌ Alcuni file non sono stati caricati:")
        for msg in errors:
            st.write(f"- {msg}")
    
    if not profiles:
        st.error("❌ Nessun profilo valido caricato.")
        return
    
    profiles.sort(key=lambda p: (p.file_name.lower(), p.channel_index))
    st.success(f"✅ **{len(profiles)} profili** caricati da **{len(uploaded_files)} file**")
    
    cfg = build_filter_config()
    if cfg.workflow_mode == "manual" and has_duplicate_processing_orders(cfg):
        st.sidebar.info(
            "ℹ️ Ordini filtro duplicati rilevati: esecuzione con priorità fissa."
        )
    
    viz = build_visual_config()
    coord_cfg = build_coordinate_config()
    
    section = st.radio(
        "📊 Sezione",
        options=["Profilo Singolo", "Multi-Profilo", "Time-slice"],
        index=0,
        horizontal=True,
    )
    
    if section == "Profilo Singolo":
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
    elif section == "Multi-Profilo":
        render_multi_profile_tab(
            profiles=profiles,
            cfg=cfg,
            viz=viz,
            ordered_processing_steps=ordered_processing_steps,
            run_filter_pipeline_cb=run_filter_pipeline,
            apply_filters_cb=apply_filters,
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
