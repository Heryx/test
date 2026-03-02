from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Any

import numpy as np
import streamlit as st

from ml_presets import AUTO_PRESET_LIBRARY, predict_ml_preset, train_ml_preset_model
from radar_filters import (
    apply_background_removal,
    apply_bandpass_filter,
    apply_dewow_filter,
    apply_gain_curve_db,
    apply_gain_db,
    apply_gaussian_smoothing,
    apply_hilbert_transform,
)
from radar_io import OgprProfile, load_ogpr_profiles
from ui_views import render_profile_tab, render_timeslice_tab


COLOR_SCALES = [
    "Greys",
    "RdBu",
    "Viridis",
    "Cividis",
    "Plasma",
    "Magma",
    "Inferno",
    "Turbo",
    "Jet",
    "Blues",
    "Picnic",
    "Portland",
    "Electric",
]


@dataclass
class FilterConfig:
    workflow_mode: str
    dewow_order: int
    dewow_window: int
    background_removal_order: int
    bandpass_order: int
    low_cut: float
    high_cut: float
    sample_rate: float
    order: int
    gain_order: int
    gain_db_exponent: float
    gain_db: float
    gain_mode: str
    gain_curve_db: str
    hilbert_order: int
    hilbert_mode: str
    smoothing_order: int
    smoothing_sigma: float


@dataclass
class VisualConfig:
    profile_raw_scale: str
    profile_filtered_scale: str
    profile_reverse: bool
    timeslice_scale: str
    timeslice_reverse: bool
    points_scale: str
    points_reverse: bool


@dataclass
class DecodeConfig:
    offset_mode: str
    layout_mode: str


@dataclass
class CoordinateConfig:
    filter_duplicates: bool
    smooth_window: int


LAYOUT_OPTIONS = [
    "auto",
    "layout:slices,channels,samples",
    "layout:channels,slices,samples",
    "layout:samples,channels,slices",
    "layout:slices,samples,channels",
    "layout:channels,samples,slices",
    "layout:samples,slices,channels",
]

FILTER_DEFINITIONS = [
    {
        "name": "dewow", "label": "Dewow", "key_order": "f_dewow_order",
        "params": [
            {"key": "f_dewow_window", "widget": st.sidebar.slider, "label": "Finestra Dewow", "args": (3, 301), "kwargs": {"step": 2}},
        ]
    },
    {
        "name": "background", "label": "Rimozione background", "key_order": "f_background_order",
        "params": []
    },
    {
        "name": "gain", "label": "Gain", "key_order": "f_gain_order",
        "params": [
            {
                "key": "f_gain_mode",
                "widget": st.sidebar.selectbox,
                "label": "Modalita Gain",
                "args": (["exponential", "curve"],),
                "kwargs": {"help": "curve = logica applygain.m con punti dB interpolati lungo i sample"},
            },
            {
                "key": "f_gain_curve_db",
                "widget": st.sidebar.text_input,
                "label": "Punti curva dB (Gain)",
                "args": (),
                "kwargs": {"help": "Esempio: -20,0,15,25,30"},
            },
            {"key": "f_gain_db_exponent", "widget": st.sidebar.slider, "label": "Curvatura Gain", "args": (0.1, 3.0), "kwargs": {"step": 0.1}},
            {"key": "f_gain_db", "widget": st.sidebar.slider, "label": "Gain dB finale", "args": (0.0, 60.0), "kwargs": {"step": 1.0}},
        ]
    },
    {
        "name": "bandpass", "label": "Band-pass", "key_order": "f_bandpass_order",
        "params": [
            {"key": "f_sample_rate", "widget": st.sidebar.number_input, "label": "Sample rate (Hz)", "args": (), "kwargs": {"min_value": 1.0, "step": 1.0e6, "format": "%.0f"}},
            {"key": "f_low_cut", "widget": st.sidebar.number_input, "label": "Frequenza bassa (Hz)", "args": (), "kwargs": {"min_value": 0.0, "step": 1.0e6, "format": "%.0f"}},
            {"key": "f_high_cut", "widget": st.sidebar.number_input, "label": "Frequenza alta (Hz)", "args": (), "kwargs": {"min_value": 0.0, "step": 1.0e6, "format": "%.0f"}},
            {"key": "f_bandpass_filter_order", "widget": st.sidebar.slider, "label": "Ordine filtro Butterworth", "args": (1, 8), "kwargs": {}},
        ]
    },
    {
        "name": "hilbert", "label": "Hilbert", "key_order": "f_hilbert_order",
        "params": [
            {"key": "f_hilbert_mode", "widget": st.sidebar.selectbox, "label": "Modalita Hilbert", "args": (["envelope", "real", "imag", "phase"],), "kwargs": {}},
        ]
    },
    {
        "name": "smoothing", "label": "Smooth Gaussiano", "key_order": "f_smoothing_order",
        "params": [
            {"key": "f_smoothing_sigma", "widget": st.sidebar.slider, "label": "Sigma smoothing", "args": (0.1, 6.0), "kwargs": {"step": 0.1}},
        ]
    }
]


FILTER_STATE_DEFAULTS: dict[str, Any] = {
    "f_workflow_mode": "base",
    "f_enable_gain": True,
    "f_enable_bandpass": True,
    "f_enable_background": True,
    "f_enable_hilbert": True,
    "f_enable_dewow": False,
    "f_enable_smoothing": False,
    "f_dewow_order": 0,
    "f_dewow_window": 41,
    "f_background_order": 3,
    "f_bandpass_order": 2,
    "f_sample_rate": 1.0e9,
    "f_low_cut": 20.0e6,
    "f_high_cut": 320.0e6,
    "f_bandpass_filter_order": 4,
    "f_gain_order": 1,
    "f_gain_mode": "exponential",
    "f_gain_curve_db": "-20,0,15,25,30",
    "f_gain_db_exponent": 1.0,
    "f_gain_db": 24.0,
    "f_hilbert_order": 4,
    "f_hilbert_mode": "envelope",
    "f_smoothing_order": 0,
    "f_smoothing_sigma": 1.0,
}


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


def _uploaded_files_signature(uploaded_files: list[Any], decode_cfg: DecodeConfig) -> tuple[Any, ...]:
    entries: list[tuple[str, int, str]] = []
    for uploaded in sorted(uploaded_files, key=lambda item: item.name.lower()):
        size = int(getattr(uploaded, "size", 0))
        file_id = str(getattr(uploaded, "file_id", ""))
        entries.append((uploaded.name.lower(), size, file_id))
    return (decode_cfg.offset_mode, decode_cfg.layout_mode, tuple(entries))


def _filter_cfg_signature(cfg: FilterConfig) -> tuple[Any, ...]:
    return (
        cfg.workflow_mode,
        int(cfg.dewow_order),
        int(cfg.dewow_window),
        int(cfg.background_removal_order),
        int(cfg.bandpass_order),
        float(cfg.low_cut),
        float(cfg.high_cut),
        float(cfg.sample_rate),
        int(cfg.order),
        int(cfg.gain_order),
        str(cfg.gain_mode),
        str(cfg.gain_curve_db),
        float(cfg.gain_db_exponent),
        float(cfg.gain_db),
        int(cfg.hilbert_order),
        str(cfg.hilbert_mode),
        int(cfg.smoothing_order),
        float(cfg.smoothing_sigma),
    )


def _ensure_filter_state_defaults() -> None:
    if "f_enable_gain" not in st.session_state and "f_enable_gaind" in st.session_state:
        st.session_state["f_enable_gain"] = bool(st.session_state["f_enable_gaind"])
    for key, value in FILTER_STATE_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value
    if st.session_state.get("f_workflow_mode") not in {"base", "manual"}:
        st.session_state["f_workflow_mode"] = "base"


def _preset_to_filter_config(preset: dict[str, Any], sample_rate: float) -> FilterConfig:
    nyquist = max(sample_rate * 0.5, 1.0)
    low_cut = float(preset["low_cut_ratio"]) * nyquist
    high_cut = min(float(preset["high_cut_ratio"]) * nyquist, 0.99 * nyquist)
    return FilterConfig(
        workflow_mode="manual",
        dewow_order=int(preset["dewow_order"]),
        dewow_window=int(preset["dewow_window"]),
        background_removal_order=int(preset["background_order"]),
        bandpass_order=int(preset["bandpass_order"]),
        low_cut=max(low_cut, 0.0),
        high_cut=max(high_cut, 0.0),
        sample_rate=float(sample_rate),
        order=int(preset["filter_order"]),
        gain_order=int(preset["gain_order"]),
        gain_mode=str(preset.get("gain_mode", "exponential")),
        gain_curve_db=str(preset.get("gain_curve_db", FILTER_STATE_DEFAULTS["f_gain_curve_db"])),
        gain_db_exponent=float(preset.get("gain_db_exponent", preset.get("gain_power", 1.0))),
        gain_db=float(preset.get("gain_db", preset.get("gain_scale", 24.0))),
        hilbert_order=int(preset.get("hilbert_order", 0)),
        hilbert_mode=str(preset.get("hilbert_mode", "envelope")),
        smoothing_order=int(preset["smoothing_order"]),
        smoothing_sigma=float(preset["smoothing_sigma"]),
    )


def _apply_preset_to_session_state(preset: dict[str, Any]) -> None:
    sample_rate = float(st.session_state.get("f_sample_rate", FILTER_STATE_DEFAULTS["f_sample_rate"]))
    cfg = _preset_to_filter_config(preset, sample_rate=sample_rate)
    st.session_state["f_workflow_mode"] = "manual"
    st.session_state["f_enable_dewow"] = int(cfg.dewow_order) > 0
    st.session_state["f_enable_background"] = int(cfg.background_removal_order) > 0
    st.session_state["f_enable_bandpass"] = int(cfg.bandpass_order) > 0
    st.session_state["f_enable_gain"] = int(cfg.gain_order) > 0
    st.session_state["f_enable_hilbert"] = int(cfg.hilbert_order) > 0
    st.session_state["f_enable_smoothing"] = int(cfg.smoothing_order) > 0
    st.session_state["f_dewow_order"] = int(cfg.dewow_order)
    st.session_state["f_dewow_window"] = int(cfg.dewow_window)
    st.session_state["f_background_order"] = int(cfg.background_removal_order)
    st.session_state["f_bandpass_order"] = int(cfg.bandpass_order)
    st.session_state["f_low_cut"] = float(cfg.low_cut)
    st.session_state["f_high_cut"] = float(cfg.high_cut)
    st.session_state["f_bandpass_filter_order"] = int(cfg.order)
    st.session_state["f_gain_order"] = int(cfg.gain_order)
    st.session_state["f_gain_mode"] = str(cfg.gain_mode)
    st.session_state["f_gain_curve_db"] = str(cfg.gain_curve_db)
    st.session_state["f_gain_db_exponent"] = float(cfg.gain_db_exponent)
    st.session_state["f_gain_db"] = float(cfg.gain_db)
    st.session_state["f_hilbert_order"] = int(cfg.hilbert_order)
    st.session_state["f_hilbert_mode"] = str(cfg.hilbert_mode)
    st.session_state["f_smoothing_order"] = int(cfg.smoothing_order)
    st.session_state["f_smoothing_sigma"] = float(cfg.smoothing_sigma)


def _build_filter_config() -> FilterConfig:
    _ensure_filter_state_defaults()
    st.sidebar.header("Filtri")
    workflow_mode = st.sidebar.radio(
        "Workflow filtri",
        options=["base", "manual"],
        format_func=lambda x: "Base consigliato" if x == "base" else "Manuale (ordini personalizzati)",
        key="f_workflow_mode",
        horizontal=True,
    )
    manual_mode = workflow_mode == "manual"
    st.sidebar.caption(
        "Workflow base: Gain -> Bandpass -> Background -> Hilbert -> Dewow -> Smoothing. "
        "Con Manuale puoi definire ordini custom."
    )
    with st.sidebar.expander("Come funziona l'applicazione filtri", expanded=False):
        st.markdown(
            "1. Ogni filtro ha uno switch ON/OFF.\n"
            "2. In `Base consigliato`, l'ordine e fisso e automatico.\n"
            "3. In `Manuale`, imposti tu l'ordine numerico.\n"
            "4. Un filtro OFF equivale a ordine `0` (non applicato).\n"
            "5. Il pannello `Diagnostica filtri` mostra min/max/std dopo ogni step."
        )

    orders_config = {}
    for f_def in FILTER_DEFINITIONS:
        name = f_def["name"]
        key_enable = f"f_enable_{name}"
        enabled = st.sidebar.checkbox(f"Abilita {f_def['label']}", key=key_enable)
        order_manual = st.sidebar.number_input(
            f"Ordine {f_def['label']} (manuale)",
            min_value=0, max_value=20, step=1,
            key=f_def["key_order"],
            disabled=(not manual_mode) or (not enabled),
        )
        orders_config[name] = {"manual": order_manual, "enabled": enabled}
        for param in f_def["params"]:
            param["widget"](param["label"], *param["args"], key=param["key"], **param["kwargs"])

    final_orders = {}
    if manual_mode:
        for name, cfg in orders_config.items():
            final_orders[name] = int(cfg["manual"]) if cfg["enabled"] else 0
    else:
        base_order_names = [f_def["name"] for f_def in FILTER_DEFINITIONS]
        rank = 1
        for name in base_order_names:
            if orders_config[name]["enabled"]:
                final_orders[name] = rank
                rank += 1
            else:
                final_orders[name] = 0

    return FilterConfig(
        workflow_mode=workflow_mode,
        dewow_order=final_orders["dewow"],
        dewow_window=st.session_state["f_dewow_window"],
        background_removal_order=final_orders["background"],
        bandpass_order=final_orders["bandpass"],
        low_cut=st.session_state["f_low_cut"],
        high_cut=st.session_state["f_high_cut"],
        sample_rate=st.session_state["f_sample_rate"],
        order=st.session_state["f_bandpass_filter_order"],
        gain_order=final_orders["gain"],
        gain_mode=st.session_state["f_gain_mode"],
        gain_curve_db=st.session_state["f_gain_curve_db"],
        gain_db_exponent=st.session_state["f_gain_db_exponent"],
        gain_db=st.session_state["f_gain_db"],
        hilbert_order=final_orders["hilbert"],
        hilbert_mode=st.session_state["f_hilbert_mode"],
        smoothing_order=final_orders["smoothing"],
        smoothing_sigma=st.session_state["f_smoothing_sigma"],
    )


def _build_visual_config() -> VisualConfig:
    st.sidebar.header("Colori")

    profile_raw_scale = st.sidebar.selectbox(
        "Scala colore radargramma originale",
        options=COLOR_SCALES,
        index=0,
    )
    profile_filtered_scale = st.sidebar.selectbox(
        "Scala colore radargramma filtrato",
        options=COLOR_SCALES,
        index=1,
    )
    profile_reverse = st.sidebar.checkbox("Inverti colori radargrammi", value=False)

    timeslice_scale = st.sidebar.selectbox(
        "Scala colore time-slice",
        options=COLOR_SCALES,
        index=1,
    )
    timeslice_reverse = st.sidebar.checkbox("Inverti colori time-slice", value=False)

    points_scale = st.sidebar.selectbox(
        "Scala colore punti campionati",
        options=COLOR_SCALES,
        index=0,
    )
    points_reverse = st.sidebar.checkbox("Inverti colori punti campionati", value=False)

    return VisualConfig(
        profile_raw_scale=profile_raw_scale,
        profile_filtered_scale=profile_filtered_scale,
        profile_reverse=profile_reverse,
        timeslice_scale=timeslice_scale,
        timeslice_reverse=timeslice_reverse,
        points_scale=points_scale,
        points_reverse=points_reverse,
    )


def _build_decode_config() -> DecodeConfig:
    st.sidebar.header("Decoding OGPR")
    offset_mode = st.sidebar.selectbox(
        "Offset mode",
        options=["auto", "absolute", "relative"],
        index=0,
        help="`auto` prova entrambe le varianti e sceglie la migliore.",
    )
    layout_mode = st.sidebar.selectbox(
        "Layout volume radar",
        options=LAYOUT_OPTIONS,
        index=0,
        help="Imposta un layout fisso solo se l'auto-detect non visualizza il segnale.",
    )
    return DecodeConfig(offset_mode=offset_mode, layout_mode=layout_mode)


def _build_coordinate_config() -> CoordinateConfig:
    st.sidebar.header("Coordinate")
    filter_duplicates = st.sidebar.checkbox(
        "Filtra punti quasi fermi",
        value=True,
        help="Replica la logica `filter_coords` del plugin: rimuove punti con avanzamento troppo basso.",
    )
    smooth_window = st.sidebar.slider(
        "Smoothing coordinate (mediana)",
        min_value=0,
        max_value=51,
        value=7,
        step=2,
        help="Finestra mediana (0 = off), simile a `smooth_coords` del plugin.",
    )
    return CoordinateConfig(filter_duplicates=filter_duplicates, smooth_window=smooth_window)


def _ordered_processing_steps(cfg: FilterConfig) -> list[tuple[str, int]]:
    steps: list[tuple[str, int, int]] = [
        ("gain", cfg.gain_order, 0),
        ("bandpass", cfg.bandpass_order, 1),
        ("background", cfg.background_removal_order, 2),
        ("hilbert", cfg.hilbert_order, 3),
        ("dewow", cfg.dewow_order, 4),
        ("smoothing", cfg.smoothing_order, 5),
    ]
    active = [item for item in steps if item[1] > 0]
    active.sort(key=lambda item: (item[1], item[2]))
    return [(name, order) for name, order, _priority in active]


def _has_duplicate_processing_orders(cfg: FilterConfig) -> bool:
    orders = [order for _name, order in _ordered_processing_steps(cfg)]
    return len(orders) != len(set(orders))


def _parse_gain_curve_points(text: str) -> np.ndarray | None:
    raw = str(text).strip()
    if not raw:
        return None
    normalized = raw.replace(";", ",").replace("\n", ",")
    parts = [item.strip() for item in normalized.split(",") if item.strip()]
    if not parts:
        return None
    values: list[float] = []
    for part in parts:
        try:
            values.append(float(part))
        except Exception:
            return None
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        return None
    return arr


def _run_filter_pipeline(data: np.ndarray, cfg: FilterConfig) -> list[tuple[str, np.ndarray]]:
    stages: list[tuple[str, np.ndarray]] = [("raw", data.astype(np.float64, copy=True))]
    filtered = stages[0][1]
    steps = _ordered_processing_steps(cfg)

    for step_name, order in steps:
        if step_name == "gain":
            use_curve_mode = str(cfg.gain_mode).lower() == "curve"
            curve_points = None
            if use_curve_mode:
                curve_points = _parse_gain_curve_points(cfg.gain_curve_db)

            if use_curve_mode and curve_points is not None:
                filtered = apply_gain_curve_db(filtered, curve_points)
            else:
                filtered = apply_gain_db(filtered, db_gain=cfg.gain_db, exponent=cfg.gain_db_exponent)
        elif step_name == "bandpass":
            filtered = apply_bandpass_filter(
                filtered,
                low_cut=cfg.low_cut,
                high_cut=cfg.high_cut,
                sample_rate=cfg.sample_rate,
                order=cfg.order,
            )
        elif step_name == "background":
            filtered = apply_background_removal(filtered)
        elif step_name == "hilbert":
            filtered = apply_hilbert_transform(filtered, mode=cfg.hilbert_mode)
        elif step_name == "dewow":
            filtered = apply_dewow_filter(filtered, window=cfg.dewow_window)
        elif step_name == "smoothing":
            filtered = apply_gaussian_smoothing(filtered, sigma=cfg.smoothing_sigma)
        stages.append((f"{order}:{step_name}", np.asarray(filtered, dtype=np.float64)))

    return stages


@st.cache_resource(show_spinner=False)
def _train_ml_preset_model() -> dict[str, Any]:
    presets = [dict(preset) for preset in AUTO_PRESET_LIBRARY]
    return train_ml_preset_model(
        presets=presets,
        preset_to_filter_config=_preset_to_filter_config,
        apply_filters=_apply_filters,
        sample_rate=1.0e9,
        n_samples=28,
        seed=42,
    )


def _predict_ml_preset(profile_data: np.ndarray, sample_rate: float) -> tuple[int, float]:
    model = _train_ml_preset_model()
    return predict_ml_preset(
        profile_data=profile_data,
        sample_rate=sample_rate,
        model=model,
        preset_to_filter_config=_preset_to_filter_config,
        apply_filters=_apply_filters,
    )


def _apply_filters(data: np.ndarray, cfg: FilterConfig) -> np.ndarray:
    return _run_filter_pipeline(data, cfg)[-1][1]


def _to_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except Exception:
        return None
    if not np.isfinite(parsed):
        return None
    return float(parsed)


def _sample_rate_from_metadata(meta: dict[str, Any]) -> float | None:
    radar_params = meta.get("radar_parameters")
    if not isinstance(radar_params, dict):
        return None

    direct_keys = [
        "sampleRate_Hz",
        "sampleRateHz",
        "sampleRate",
        "samplingRate_Hz",
        "samplingRateHz",
        "samplingFrequency_Hz",
        "samplingFrequencyHz",
        "samplingFrequency_hz",
    ]
    for key in direct_keys:
        value = _to_float(radar_params.get(key))
        if value is not None and value > 0:
            return value

    sampling_time_ns = _to_float(radar_params.get("samplingTime_ns"))
    if sampling_time_ns is not None and sampling_time_ns > 0:
        return 1.0e9 / sampling_time_ns
    return None


def _auto_configure_from_profiles(profiles: list[OgprProfile]) -> None:
    if not profiles:
        return

    sample_rates = []
    for profile in profiles:
        sr = _sample_rate_from_metadata(profile.metadata)
        if sr is not None and sr > 1.0:
            sample_rates.append(sr)

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




def main() -> None:
    st.set_page_config(page_title="OGPR Browse App", layout="wide")
    st.title("OGPR Browse App")
    st.caption("Import multiplo file ogpr, visualizzazione profili e creazione time-slice con coordinate GPS")
    decode_cfg = _build_decode_config()

    uploaded_files = st.file_uploader(
        "Carica i file .ogpr (uno o piu file, anche uno per canale)",
        type=["ogpr"],
        accept_multiple_files=True,
    )
    if not uploaded_files:
        st.info("Carica uno o piu file .ogpr per iniziare.")
        return

    profiles_key = _uploaded_files_signature(uploaded_files, decode_cfg)
    cache_key_prev = st.session_state.get("_profiles_cache_key")
    if cache_key_prev != profiles_key:
        with st.spinner("Caricamento e decoding file OGPR..."):
            profiles: list[OgprProfile] = []
            errors: list[str] = []
            for uploaded in sorted(uploaded_files, key=lambda item: item.name.lower()):
                try:
                    raw_bytes = bytes(uploaded.getbuffer())
                    profiles.extend(
                        _load_profiles_cached(
                            uploaded.name,
                            raw_bytes,
                            decode_cfg.offset_mode,
                            decode_cfg.layout_mode,
                        )
                    )
                except Exception as exc:
                    errors.append(f"{uploaded.name}: {exc}")
        st.session_state["_profiles_cache_key"] = profiles_key
        st.session_state["_profiles_cache_profiles"] = profiles
        st.session_state["_profiles_cache_errors"] = errors
        st.session_state.pop("_timeslice_filtered_key", None)
        st.session_state.pop("_timeslice_filtered_arrays", None)
        _auto_configure_from_profiles(profiles)
    else:
        profiles = st.session_state.get("_profiles_cache_profiles", [])
        errors = st.session_state.get("_profiles_cache_errors", [])
        if not isinstance(profiles, list):
            profiles = []
        if not isinstance(errors, list):
            errors = []

    if errors:
        st.error("Alcuni file non sono stati caricati:")
        for message in errors:
            st.write(f"- {message}")

    if not profiles:
        st.error("Nessun profilo valido caricato.")
        return

    profiles.sort(key=lambda profile: (profile.file_name.lower(), profile.channel_index))
    st.success(f"Profili caricati: {len(profiles)} da {len(uploaded_files)} file.")

    cfg = _build_filter_config()
    if cfg.workflow_mode == "manual" and _has_duplicate_processing_orders(cfg):
        st.sidebar.info(
            "Ordini filtro duplicati rilevati: esecuzione comunque attiva con priorita fissa "
            "(gain -> bandpass -> background -> hilbert -> dewow -> smoothing) a parita di ordine."
        )
    viz = _build_visual_config()
    coord_cfg = _build_coordinate_config()
    section = st.radio(
        "Sezione",
        options=["Profili", "Time-slice"],
        index=0,
        horizontal=True,
    )
    if section == "Profili":
        render_profile_tab(
            profiles=profiles,
            cfg=cfg,
            viz=viz,
            auto_preset_library=AUTO_PRESET_LIBRARY,
            ordered_processing_steps=_ordered_processing_steps,
            predict_ml_preset_cb=_predict_ml_preset,
            apply_preset_cb=_apply_preset_to_session_state,
            run_filter_pipeline_cb=_run_filter_pipeline,
        )
    else:
        render_timeslice_tab(
            profiles=profiles,
            cfg=cfg,
            viz=viz,
            coord_cfg=coord_cfg,
            profiles_cache_key=profiles_key,
            ordered_processing_steps=_ordered_processing_steps,
            filter_cfg_signature_cb=_filter_cfg_signature,
            apply_filters_cb=_apply_filters,
        )


if __name__ == "__main__":
    main()
