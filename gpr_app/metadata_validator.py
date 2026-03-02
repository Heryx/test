"""Metadata validation and antenna detection for IDS GPR systems."""
from __future__ import annotations

from typing import Any
import numpy as np

from radar_io import OgprProfile
from gpr_app.utils import to_float


class MetadataValidationResult:
    """Result of metadata validation."""
    
    def __init__(
        self,
        is_valid: bool,
        completeness_score: float,
        has_sampling_time: bool,
        has_antenna_freq: bool,
        has_velocity: bool,
        antenna_type: str | None = None,
        detected_frequency_mhz: float | None = None,
        warnings: list[str] | None = None,
    ):
        self.is_valid = is_valid
        self.completeness_score = completeness_score
        self.has_sampling_time = has_sampling_time
        self.has_antenna_freq = has_antenna_freq
        self.has_velocity = has_velocity
        self.antenna_type = antenna_type
        self.detected_frequency_mhz = detected_frequency_mhz
        self.warnings = warnings or []


def detect_ids_antenna(freq_mhz: float | None) -> str | None:
    """Detect IDS antenna type from frequency.
    
    Returns:
        'Stream-DP' for 200-1000 MHz range
        'Stream-UP' for 200-600 MHz range (more restrictive)
        None if no match or invalid frequency
    """
    if freq_mhz is None or not np.isfinite(freq_mhz):
        return None
    
    freq = float(freq_mhz)
    
    # Stream UP: 200-600 MHz (39 channels)
    if 200 <= freq <= 600:
        return "Stream-UP"
    
    # Stream DP: 200-1000 MHz (30 channels: 19VV + 11HH)
    if 600 < freq <= 1000:
        return "Stream-DP"
    
    return None


def extract_antenna_frequency(metadata: dict[str, Any]) -> float | None:
    """Extract antenna frequency from metadata in MHz."""
    radar_params = metadata.get("radar_parameters")
    if not isinstance(radar_params, dict):
        return None
    
    # Try various key names
    freq_keys = [
        "antennaFrequency", "antenna_frequency", "antennaFrequency_MHz",
        "frequency", "frequency_MHz", "centerFrequency", "center_frequency",
    ]
    
    for key in freq_keys:
        value = to_float(radar_params.get(key))
        if value is not None and 10 <= value <= 5000:  # Reasonable MHz range
            return value
    
    return None


def validate_metadata(profile: OgprProfile) -> MetadataValidationResult:
    """Validate metadata completeness and detect antenna type.
    
    A profile is considered valid for auto-config if:
    - Has samplingTime_ns (required for time axis and bandpass)
    - Has antenna frequency (for antenna detection and bandpass hints)
    - Has propagationVelocity (for depth conversion)
    """
    metadata = profile.metadata
    radar_params = metadata.get("radar_parameters", {})
    
    # Check sampling time
    sampling_time_ns = to_float(radar_params.get("samplingTime_ns"))
    has_sampling = sampling_time_ns is not None and sampling_time_ns > 0
    
    # Check antenna frequency
    antenna_freq_mhz = extract_antenna_frequency(metadata)
    has_freq = antenna_freq_mhz is not None
    
    # Check propagation velocity
    velocity = to_float(radar_params.get("propagationVelocity_mPerSec"))
    has_velocity = velocity is not None and 0.05e9 <= velocity <= 0.20e9  # 0.05-0.20 m/ns in m/s
    
    # Completeness score (0-1)
    score = sum([has_sampling, has_freq, has_velocity]) / 3.0
    
    # Detect antenna type
    antenna_type = detect_ids_antenna(antenna_freq_mhz) if has_freq else None
    
    # Validation decision
    is_valid = has_sampling and has_freq  # Minimum requirements
    
    # Warnings
    warnings = []
    if not has_sampling:
        warnings.append("samplingTime_ns mancante o non valido")
    if not has_freq:
        warnings.append("Frequenza antenna non rilevata")
    if not has_velocity:
        warnings.append("propagationVelocity non disponibile (conversione profondità disabilitata)")
    if antenna_type is None and has_freq:
        warnings.append(f"Frequenza {antenna_freq_mhz:.0f} MHz non corrisponde ad antenna IDS nota")
    
    return MetadataValidationResult(
        is_valid=is_valid,
        completeness_score=score,
        has_sampling_time=has_sampling,
        has_antenna_freq=has_freq,
        has_velocity=has_velocity,
        antenna_type=antenna_type,
        detected_frequency_mhz=antenna_freq_mhz,
        warnings=warnings,
    )


def get_ids_preset_hints(antenna_type: str, freq_mhz: float) -> dict[str, Any]:
    """Get recommended processing hints for IDS antennas.
    
    Returns suggested bandpass frequencies based on antenna type.
    """
    if antenna_type == "Stream-DP":
        # Stream DP: 200-1000 MHz, wider bandwidth
        return {
            "bandpass_low_mhz": max(150, freq_mhz * 0.3),
            "bandpass_high_mhz": min(1100, freq_mhz * 1.5),
            "gain_recommended": True,
            "gain_db": 30.0,
            "description": "Stream DP (30 canali, 19VV+11HH)",
        }
    elif antenna_type == "Stream-UP":
        # Stream UP: 200-600 MHz, more constrained
        return {
            "bandpass_low_mhz": max(150, freq_mhz * 0.4),
            "bandpass_high_mhz": min(650, freq_mhz * 1.4),
            "gain_recommended": True,
            "gain_db": 28.0,
            "description": "Stream UP (39 canali)",
        }
    else:
        # Generic GPR
        return {
            "bandpass_low_mhz": max(50, freq_mhz * 0.3),
            "bandpass_high_mhz": min(freq_mhz * 2.0, 3000),
            "gain_recommended": True,
            "gain_db": 25.0,
            "description": "Antenna GPR generica",
        }
