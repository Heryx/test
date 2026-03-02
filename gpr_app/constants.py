from __future__ import annotations

from typing import Any


COLOR_SCALES: list[str] = [
    "Greys", "RdBu", "Viridis", "Cividis", "Plasma",
    "Magma", "Inferno", "Turbo", "Jet", "Blues",
    "Picnic", "Portland", "Electric",
]

LAYOUT_OPTIONS: list[str] = [
    "auto",
    "layout:slices,channels,samples",
    "layout:channels,slices,samples",
    "layout:samples,channels,slices",
    "layout:slices,samples,channels",
    "layout:channels,samples,slices",
    "layout:samples,slices,channels",
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
