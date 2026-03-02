from __future__ import annotations

from typing import Any

import numpy as np


def to_float(value: Any) -> float | None:
    """Converte un valore in float; restituisce None se non valido o non finito."""
    try:
        parsed = float(value)
    except Exception:
        return None
    if not np.isfinite(parsed):
        return None
    return float(parsed)


def sample_rate_from_metadata(meta: dict[str, Any]) -> float | None:
    """Estrae il sample rate (Hz) dai metadati OGPR, se disponibile."""
    radar_params = meta.get("radar_parameters")
    if not isinstance(radar_params, dict):
        return None

    direct_keys = [
        "sampleRate_Hz", "sampleRateHz", "sampleRate",
        "samplingRate_Hz", "samplingRateHz",
        "samplingFrequency_Hz", "samplingFrequencyHz", "samplingFrequency_hz",
    ]
    for key in direct_keys:
        value = to_float(radar_params.get(key))
        if value is not None and value > 0:
            return value

    sampling_time_ns = to_float(radar_params.get("samplingTime_ns"))
    if sampling_time_ns is not None and sampling_time_ns > 0:
        return 1.0e9 / sampling_time_ns
    return None
