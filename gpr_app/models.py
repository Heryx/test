from __future__ import annotations

from dataclasses import dataclass


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
