from __future__ import annotations

from typing import Any

import numpy as np

from gpr_app.models import FilterConfig
from gpr_app.constants import FILTER_STATE_DEFAULTS
from radar_filters import (
    apply_background_removal,
    apply_bandpass_filter,
    apply_dewow_filter,
    apply_gain_curve_db,
    apply_gain_db,
    apply_gaussian_smoothing,
    apply_hilbert_transform,
)


def ordered_processing_steps(cfg: FilterConfig) -> list[tuple[str, int]]:
    """Restituisce i passi di filtraggio attivi, ordinati per priorita."""
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


def has_duplicate_processing_orders(cfg: FilterConfig) -> bool:
    """Controlla se ci sono ordini di filtraggio duplicati."""
    orders = [order for _name, order in ordered_processing_steps(cfg)]
    return len(orders) != len(set(orders))


def parse_gain_curve_points(text: str) -> np.ndarray | None:
    """Parsa una stringa di valori dB separati da virgola per la curva di gain."""
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


def run_filter_pipeline(
    data: np.ndarray,
    cfg: FilterConfig,
) -> list[tuple[str, np.ndarray]]:
    """Applica la pipeline di filtri e restituisce tutti gli stage intermedi."""
    stages: list[tuple[str, np.ndarray]] = [("raw", data.astype(np.float64, copy=True))]
    filtered = stages[0][1]

    for step_name, order in ordered_processing_steps(cfg):
        if step_name == "gain":
            use_curve = str(cfg.gain_mode).lower() == "curve"
            curve_points = parse_gain_curve_points(cfg.gain_curve_db) if use_curve else None
            if use_curve and curve_points is not None:
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


def apply_filters(data: np.ndarray, cfg: FilterConfig) -> np.ndarray:
    """Applica la pipeline e restituisce solo l'output finale."""
    return run_filter_pipeline(data, cfg)[-1][1]


def filter_cfg_signature(cfg: FilterConfig) -> tuple[Any, ...]:
    """Restituisce una tupla hashable che identifica univocamente la configurazione filtri."""
    return (
        cfg.workflow_mode,
        int(cfg.dewow_order), int(cfg.dewow_window),
        int(cfg.background_removal_order),
        int(cfg.bandpass_order),
        float(cfg.low_cut), float(cfg.high_cut), float(cfg.sample_rate), int(cfg.order),
        int(cfg.gain_order), str(cfg.gain_mode), str(cfg.gain_curve_db),
        float(cfg.gain_db_exponent), float(cfg.gain_db),
        int(cfg.hilbert_order), str(cfg.hilbert_mode),
        int(cfg.smoothing_order), float(cfg.smoothing_sigma),
    )


def preset_to_filter_config(preset: dict[str, Any], sample_rate: float) -> FilterConfig:
    """Converte un dizionario preset ML in un FilterConfig."""
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
