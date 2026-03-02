from __future__ import annotations

from typing import Any, Callable

import numpy as np

try:
    from sklearn.neighbors import KNeighborsClassifier
except ImportError:
    KNeighborsClassifier = None


AUTO_PRESET_LIBRARY: list[dict[str, Any]] = [
    {
        "name": "Raw (no filter)",
        "dewow_order": 0,
        "background_order": 0,
        "bandpass_order": 0,
        "gain_order": 0,
        "hilbert_order": 0,
        "smoothing_order": 0,
        "dewow_window": 41,
        "low_cut_ratio": 0.03,
        "high_cut_ratio": 0.70,
        "filter_order": 4,
        "gain_db_exponent": 1.0,
        "gain_db": 0.0,
        "hilbert_mode": "envelope",
        "smoothing_sigma": 1.0,
    },
    {
        "name": "Base GPR-Slice (Gain + Bandpass + Background + Hilbert)",
        "dewow_order": 0,
        "background_order": 3,
        "bandpass_order": 2,
        "gain_order": 1,
        "hilbert_order": 4,
        "smoothing_order": 0,
        "dewow_window": 41,
        "low_cut_ratio": 0.03,
        "high_cut_ratio": 0.65,
        "filter_order": 4,
        "gain_db_exponent": 1.2,
        "gain_db": 24.0,
        "hilbert_mode": "envelope",
        "smoothing_sigma": 1.0,
    },
    {
        "name": "Gain + Bandpass + Background",
        "dewow_order": 0,
        "background_order": 3,
        "bandpass_order": 2,
        "gain_order": 1,
        "hilbert_order": 0,
        "smoothing_order": 0,
        "dewow_window": 41,
        "low_cut_ratio": 0.03,
        "high_cut_ratio": 0.70,
        "filter_order": 4,
        "gain_db_exponent": 1.1,
        "gain_db": 22.0,
        "hilbert_mode": "envelope",
        "smoothing_sigma": 1.0,
    },
    {
        "name": "Gain + Bandpass + Hilbert",
        "dewow_order": 0,
        "background_order": 0,
        "bandpass_order": 2,
        "gain_order": 1,
        "hilbert_order": 3,
        "smoothing_order": 0,
        "dewow_window": 41,
        "low_cut_ratio": 0.04,
        "high_cut_ratio": 0.70,
        "filter_order": 4,
        "gain_db_exponent": 1.0,
        "gain_db": 24.0,
        "hilbert_mode": "envelope",
        "smoothing_sigma": 1.0,
    },
    {
        "name": "Dewow + Base GPR-Slice",
        "dewow_order": 1,
        "background_order": 4,
        "bandpass_order": 3,
        "gain_order": 2,
        "hilbert_order": 5,
        "smoothing_order": 0,
        "dewow_window": 51,
        "low_cut_ratio": 0.03,
        "high_cut_ratio": 0.70,
        "filter_order": 4,
        "gain_db_exponent": 1.3,
        "gain_db": 28.0,
        "hilbert_mode": "envelope",
        "smoothing_sigma": 1.0,
    },
    {
        "name": "Base + Smooth",
        "dewow_order": 0,
        "background_order": 3,
        "bandpass_order": 2,
        "gain_order": 1,
        "hilbert_order": 4,
        "smoothing_order": 5,
        "dewow_window": 51,
        "low_cut_ratio": 0.05,
        "high_cut_ratio": 0.72,
        "filter_order": 4,
        "gain_db_exponent": 1.3,
        "gain_db": 24.0,
        "hilbert_mode": "envelope",
        "smoothing_sigma": 0.8,
    },
]


def extract_signal_features(data: np.ndarray) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim != 2:
        arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError("Atteso array 2D per feature extraction.")

    traces, samples = arr.shape
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros(10, dtype=np.float64)
    vals = arr[finite]
    std = float(np.nanstd(vals))
    p95 = float(np.nanpercentile(vals, 95))
    p05 = float(np.nanpercentile(vals, 5))
    dynamic = p95 - p05

    mean_trace = np.nanmean(arr, axis=0)
    mean_trace_std = float(np.nanstd(mean_trace))
    trace_rms = np.sqrt(np.nanmean(arr * arr, axis=1))
    rms_cv = float(np.nanstd(trace_rms) / (np.nanmean(trace_rms) + 1.0e-9))

    subset = arr[: min(traces, 96), :]
    spectrum = np.abs(np.fft.rfft(subset, axis=1)) ** 2
    spec_mean = np.nanmean(spectrum, axis=0)
    if spec_mean.size < 8 or np.nansum(spec_mean) <= 0:
        low_ratio = 0.0
        mid_ratio = 0.0
        high_ratio = 0.0
    else:
        n = spec_mean.size
        low = spec_mean[1 : max(2, int(0.10 * n))]
        mid = spec_mean[max(2, int(0.10 * n)) : max(3, int(0.45 * n))]
        high = spec_mean[max(3, int(0.45 * n)) :]
        total = float(np.nansum(spec_mean) + 1.0e-9)
        low_ratio = float(np.nansum(low) / total)
        mid_ratio = float(np.nansum(mid) / total)
        high_ratio = float(np.nansum(high) / total)

    diff_t = np.diff(arr, axis=1)
    roughness = float(np.nanmedian(np.abs(diff_t)) / (std + 1.0e-9))

    corr_vals = []
    if traces > 1:
        step = max((traces - 1) // 48, 1)
        idxs = np.arange(0, traces - 1, step, dtype=np.int32)
        for idx in idxs:
            a = arr[idx] - np.nanmean(arr[idx])
            b = arr[idx + 1] - np.nanmean(arr[idx + 1])
            denom = float(np.linalg.norm(a) * np.linalg.norm(b))
            if denom > 0:
                corr = float(np.dot(a, b) / denom)
                if np.isfinite(corr):
                    corr_vals.append(corr)
    adjacent_corr = float(np.nanmedian(corr_vals)) if corr_vals else 0.0

    return np.asarray(
        [
            std,
            dynamic,
            mean_trace_std,
            rms_cv,
            low_ratio,
            mid_ratio,
            high_ratio,
            roughness,
            adjacent_corr,
            float(traces / (samples + 1.0e-9)),
        ],
        dtype=np.float64,
    )


def generate_synthetic_radargram(
    rng: np.random.Generator,
    traces: int = 56,
    samples: int = 224,
) -> tuple[np.ndarray, np.ndarray]:
    clean = np.zeros((traces, samples), dtype=np.float64)
    t = np.arange(samples, dtype=np.float64)

    n_reflectors = int(rng.integers(4, 10))
    for _ in range(n_reflectors):
        t0 = float(rng.uniform(25.0, samples - 35.0))
        dip = float(rng.uniform(-0.22, 0.22))
        width = float(rng.uniform(2.0, 8.0))
        amp = float(rng.uniform(0.6, 1.8)) * float(rng.choice([-1.0, 1.0]))
        jitter_scale = float(rng.uniform(0.5, 2.0))
        for tr in range(traces):
            center = t0 + dip * (tr - traces * 0.5) + rng.normal(0.0, jitter_scale)
            pulse = amp * np.exp(-0.5 * ((t - center) / width) ** 2)
            clean[tr] += pulse

    attenuation = np.exp(-float(rng.uniform(0.8, 2.2)) * np.linspace(0.0, 1.0, samples))
    clean *= attenuation[None, :]

    clean_std = float(np.std(clean))
    if clean_std > 0:
        clean *= 1000.0 / clean_std

    noisy = clean.copy()
    drift_amp = float(rng.uniform(80.0, 500.0))
    drift_t = np.linspace(0.0, 1.0, samples)
    drift_base = drift_amp * (drift_t - 0.5) ** 2
    noisy += drift_base[None, :]

    stripe_amp = float(rng.uniform(0.0, 220.0))
    stripes = rng.normal(0.0, stripe_amp, size=(traces, 1))
    noisy += stripes

    noise_sigma = float(rng.uniform(60.0, 420.0))
    noisy += rng.normal(0.0, noise_sigma, size=noisy.shape)

    burst_count = int(rng.integers(0, 4))
    for _ in range(burst_count):
        tr = int(rng.integers(0, traces))
        pos = int(rng.integers(0, samples))
        noisy[tr, max(0, pos - 2) : min(samples, pos + 3)] += float(rng.uniform(-1200.0, 1200.0))

    return clean, noisy


def auto_preset_score(clean: np.ndarray, noisy: np.ndarray, filtered: np.ndarray) -> float:
    c = np.asarray(clean, dtype=np.float64)
    n = np.asarray(noisy, dtype=np.float64)
    f = np.asarray(filtered, dtype=np.float64)
    c = c - np.nanmean(c)
    n = n - np.nanmean(n)
    f = f - np.nanmean(f)
    c /= float(np.nanstd(c) + 1.0e-9)
    n /= float(np.nanstd(n) + 1.0e-9)
    f /= float(np.nanstd(f) + 1.0e-9)

    clean_corr = float(np.dot(c.ravel(), f.ravel()) / (np.linalg.norm(c.ravel()) * np.linalg.norm(f.ravel()) + 1.0e-9))
    noisy_corr = float(np.dot(n.ravel(), f.ravel()) / (np.linalg.norm(n.ravel()) * np.linalg.norm(f.ravel()) + 1.0e-9))

    feat_noisy = extract_signal_features(noisy)
    feat_filtered = extract_signal_features(filtered)
    noisy_std = feat_noisy[0]
    filtered_std = feat_filtered[0]
    flat_penalty = max(0.0, 0.06 * noisy_std - filtered_std) / (noisy_std + 1.0e-9)

    low_reduction = feat_noisy[4] - feat_filtered[4]
    corr_gain = feat_filtered[8] - feat_noisy[8]
    stripe_reduction = feat_noisy[3] - feat_filtered[3]
    roughness_penalty = max(0.0, feat_filtered[7] - 1.15 * feat_noisy[7])

    return (
        0.35 * clean_corr
        + 0.20 * noisy_corr
        + 1.6 * low_reduction
        + 1.2 * corr_gain
        + 0.8 * stripe_reduction
        - 1.6 * flat_penalty
        - 0.6 * roughness_penalty
    )


def unsupervised_filter_quality(raw: np.ndarray, filtered: np.ndarray) -> float:
    feat_raw = extract_signal_features(raw)
    feat_f = extract_signal_features(filtered)
    raw_std = feat_raw[0]
    filt_std = feat_f[0]
    flat_penalty = max(0.0, 0.05 * raw_std - filt_std) / (raw_std + 1.0e-9)
    low_reduction = feat_raw[4] - feat_f[4]
    corr_gain = feat_f[8] - feat_raw[8]
    stripe_reduction = feat_raw[3] - feat_f[3]
    roughness_penalty = max(0.0, feat_f[7] - 1.10 * feat_raw[7])
    return (
        1.7 * low_reduction
        + 1.3 * corr_gain
        + 0.7 * stripe_reduction
        - 1.7 * flat_penalty
        - 0.6 * roughness_penalty
    )


def train_ml_preset_model(
    presets: list[dict[str, Any]],
    preset_to_filter_config: Callable[[dict[str, Any], float], Any],
    apply_filters: Callable[[np.ndarray, Any], np.ndarray],
    sample_rate: float = 1.0e9,
    n_samples: int = 36,
    seed: int = 42,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    feature_rows: list[np.ndarray] = []
    labels: list[int] = []

    for _ in range(n_samples):
        clean, noisy = generate_synthetic_radargram(rng)
        features = extract_signal_features(noisy)
        best_idx = 0
        best_score = -1.0e12
        for idx, preset in enumerate(presets):
            cfg = preset_to_filter_config(preset, sample_rate=sample_rate)
            filtered = apply_filters(noisy, cfg)
            score = auto_preset_score(clean, noisy, filtered)
            if score > best_score:
                best_score = score
                best_idx = idx
        feature_rows.append(features)
        labels.append(best_idx)

    x = np.vstack(feature_rows).astype(np.float64)
    y = np.asarray(labels, dtype=np.int32)
    feature_scale = np.nanstd(x, axis=0)
    feature_scale[feature_scale <= 1.0e-9] = 1.0

    knn_model = None
    if KNeighborsClassifier is not None:
        # Usa scikit-learn se disponibile per una classificazione piu robusta
        try:
            knn_model = KNeighborsClassifier(n_neighbors=min(9, n_samples), weights="distance")
            knn_model.fit(x / feature_scale, y)
        except Exception:
            knn_model = None

    return {
        "x": x,
        "y": y,
        "scale": feature_scale,
        "presets": presets,
        "knn_model": knn_model,
    }


def predict_ml_preset(
    profile_data: np.ndarray,
    sample_rate: float,
    model: dict[str, Any],
    preset_to_filter_config: Callable[[dict[str, Any], float], Any],
    apply_filters: Callable[[np.ndarray, Any], np.ndarray],
) -> tuple[int, float]:
    x_train = model["x"]
    y_train = model["y"]
    scale = model["scale"]
    knn_model = model.get("knn_model")
    presets = model["presets"]

    x_query = extract_signal_features(profile_data)
    n_classes = len(presets)
    class_weights = np.zeros(n_classes, dtype=np.float64)

    if knn_model is not None:
        # Predizione con scikit-learn
        x_query_norm = (x_query / scale).reshape(1, -1)
        probs = knn_model.predict_proba(x_query_norm)[0]
        # Mappa le probabilita sui pesi delle classi
        classes = knn_model.classes_
        for cls, prob in zip(classes, probs):
            class_weights[int(cls)] = float(prob)
    else:
        # Fallback manuale (numpy)
        d = np.linalg.norm((x_train - x_query[None, :]) / scale[None, :], axis=1)
        k = min(9, len(d))
        idx = np.argsort(d)[:k]
        for i in idx:
            w = 1.0 / (d[i] + 1.0e-6)
            class_weights[int(y_train[i])] += float(w)

    top_classes = np.argsort(class_weights)[::-1][: min(3, n_classes)]
    pred = int(top_classes[0]) if top_classes.size > 0 else 0
    best_quality = -1.0e12
    for cls in top_classes:
        preset = presets[int(cls)]
        cfg = preset_to_filter_config(preset, sample_rate=sample_rate)
        filtered = apply_filters(profile_data, cfg)
        quality = unsupervised_filter_quality(profile_data, filtered)
        if quality > best_quality:
            best_quality = quality
            pred = int(cls)
    confidence = float(class_weights[pred] / (np.sum(class_weights) + 1.0e-9))
    return pred, confidence
