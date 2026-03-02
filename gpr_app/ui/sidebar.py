from __future__ import annotations

from typing import Any

import streamlit as st

from gpr_app.models import FilterConfig, VisualConfig, DecodeConfig, CoordinateConfig
from gpr_app.constants import COLOR_SCALES, LAYOUT_OPTIONS, FILTER_STATE_DEFAULTS
from gpr_app.pipeline import preset_to_filter_config


# Definizione dichiarativa dei widget sidebar per ogni filtro.
# I riferimenti a st.sidebar.* vengono risolti a runtime.
FILTER_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "dewow", "label": "Dewow", "key_order": "f_dewow_order",
        "params": [
            {"key": "f_dewow_window", "widget": lambda *a, **kw: st.sidebar.slider(*a, **kw),
             "label": "Finestra Dewow", "args": (3, 301), "kwargs": {"step": 2}},
        ],
    },
    {
        "name": "background", "label": "Rimozione background", "key_order": "f_background_order",
        "params": [],
    },
    {
        "name": "gain", "label": "Gain", "key_order": "f_gain_order",
        "params": [
            {"key": "f_gain_mode", "widget": lambda *a, **kw: st.sidebar.selectbox(*a, **kw),
             "label": "Modalita Gain", "args": (["exponential", "curve"],),
             "kwargs": {"help": "curve = logica applygain.m con punti dB interpolati lungo i sample"}},
            {"key": "f_gain_curve_db", "widget": lambda *a, **kw: st.sidebar.text_input(*a, **kw),
             "label": "Punti curva dB (Gain)", "args": (),
             "kwargs": {"help": "Esempio: -20,0,15,25,30"}},
            {"key": "f_gain_db_exponent", "widget": lambda *a, **kw: st.sidebar.slider(*a, **kw),
             "label": "Curvatura Gain", "args": (0.1, 3.0), "kwargs": {"step": 0.1}},
            {"key": "f_gain_db", "widget": lambda *a, **kw: st.sidebar.slider(*a, **kw),
             "label": "Gain dB finale", "args": (0.0, 60.0), "kwargs": {"step": 1.0}},
        ],
    },
    {
        "name": "bandpass", "label": "Band-pass", "key_order": "f_bandpass_order",
        "params": [
            {"key": "f_sample_rate", "widget": lambda *a, **kw: st.sidebar.number_input(*a, **kw),
             "label": "Sample rate (Hz)", "args": (),
             "kwargs": {"min_value": 1.0, "step": 1.0e6, "format": "%.0f"}},
            {"key": "f_low_cut", "widget": lambda *a, **kw: st.sidebar.number_input(*a, **kw),
             "label": "Frequenza bassa (Hz)", "args": (),
             "kwargs": {"min_value": 0.0, "step": 1.0e6, "format": "%.0f"}},
            {"key": "f_high_cut", "widget": lambda *a, **kw: st.sidebar.number_input(*a, **kw),
             "label": "Frequenza alta (Hz)", "args": (),
             "kwargs": {"min_value": 0.0, "step": 1.0e6, "format": "%.0f"}},
            {"key": "f_bandpass_filter_order", "widget": lambda *a, **kw: st.sidebar.slider(*a, **kw),
             "label": "Ordine filtro Butterworth", "args": (1, 8), "kwargs": {}},
        ],
    },
    {
        "name": "hilbert", "label": "Hilbert", "key_order": "f_hilbert_order",
        "params": [
            {"key": "f_hilbert_mode", "widget": lambda *a, **kw: st.sidebar.selectbox(*a, **kw),
             "label": "Modalita Hilbert", "args": (["envelope", "real", "imag", "phase"],), "kwargs": {}},
        ],
    },
    {
        "name": "smoothing", "label": "Smooth Gaussiano", "key_order": "f_smoothing_order",
        "params": [
            {"key": "f_smoothing_sigma", "widget": lambda *a, **kw: st.sidebar.slider(*a, **kw),
             "label": "Sigma smoothing", "args": (0.1, 6.0), "kwargs": {"step": 0.1}},
        ],
    },
]


def ensure_filter_state_defaults() -> None:
    """Inizializza i valori di default nel session_state se non presenti."""
    if "f_enable_gain" not in st.session_state and "f_enable_gaind" in st.session_state:
        st.session_state["f_enable_gain"] = bool(st.session_state["f_enable_gaind"])
    for key, value in FILTER_STATE_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value
    if st.session_state.get("f_workflow_mode") not in {"base", "manual"}:
        st.session_state["f_workflow_mode"] = "base"


def apply_preset_to_session_state(preset: dict[str, Any]) -> None:
    """Applica un preset ML al session_state di Streamlit."""
    sample_rate = float(st.session_state.get("f_sample_rate", FILTER_STATE_DEFAULTS["f_sample_rate"]))
    cfg = preset_to_filter_config(preset, sample_rate=sample_rate)
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


def build_filter_config() -> FilterConfig:
    """Costruisce la sidebar dei filtri e restituisce il FilterConfig corrente."""
    ensure_filter_state_defaults()
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

    orders_config: dict[str, dict[str, Any]] = {}
    for f_def in FILTER_DEFINITIONS:
        name = f_def["name"]
        enabled = st.sidebar.checkbox(f"Abilita {f_def['label']}", key=f"f_enable_{name}")
        order_manual = st.sidebar.number_input(
            f"Ordine {f_def['label']} (manuale)",
            min_value=0, max_value=20, step=1,
            key=f_def["key_order"],
            disabled=(not manual_mode) or (not enabled),
        )
        orders_config[name] = {"manual": order_manual, "enabled": enabled}
        for param in f_def["params"]:
            param["widget"](param["label"], *param["args"], key=param["key"], **param["kwargs"])

    if manual_mode:
        final_orders = {
            name: int(cfg["manual"]) if cfg["enabled"] else 0
            for name, cfg in orders_config.items()
        }
    else:
        rank = 1
        final_orders = {}
        for f_def in FILTER_DEFINITIONS:
            name = f_def["name"]
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


def build_visual_config() -> VisualConfig:
    """Costruisce la sezione colori della sidebar."""
    st.sidebar.header("Colori")
    profile_raw_scale = st.sidebar.selectbox("Scala colore radargramma originale", options=COLOR_SCALES, index=0)
    profile_filtered_scale = st.sidebar.selectbox("Scala colore radargramma filtrato", options=COLOR_SCALES, index=1)
    profile_reverse = st.sidebar.checkbox("Inverti colori radargrammi", value=False)
    timeslice_scale = st.sidebar.selectbox("Scala colore time-slice", options=COLOR_SCALES, index=1)
    timeslice_reverse = st.sidebar.checkbox("Inverti colori time-slice", value=False)
    points_scale = st.sidebar.selectbox("Scala colore punti campionati", options=COLOR_SCALES, index=0)
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


def build_decode_config() -> DecodeConfig:
    """Costruisce la sezione decoding OGPR della sidebar."""
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


def build_coordinate_config() -> CoordinateConfig:
    """Costruisce la sezione coordinate della sidebar."""
    st.sidebar.header("Coordinate")
    filter_duplicates = st.sidebar.checkbox(
        "Filtra punti quasi fermi",
        value=True,
        help="Replica la logica `filter_coords` del plugin: rimuove punti con avanzamento troppo basso.",
    )
    smooth_window = st.sidebar.slider(
        "Smoothing coordinate (mediana)",
        min_value=0, max_value=51, value=7, step=2,
        help="Finestra mediana (0 = off), simile a `smooth_coords` del plugin.",
    )
    return CoordinateConfig(filter_duplicates=filter_duplicates, smooth_window=smooth_window)
