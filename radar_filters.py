from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d, gaussian_filter, median_filter
from scipy.signal import butter, filtfilt, hilbert
from scipy.interpolate import interp1d, griddata
from scipy.spatial import cKDTree


def _time_axis_for(data: np.ndarray) -> int:
    if data.ndim == 2:
        return 1
    if data.ndim == 3:
        return 0
    raise ValueError(f"Numero dimensioni non supportato: {data.ndim}")


def _trace_axis_for(data: np.ndarray) -> int:
    if data.ndim == 2:
        return 0
    if data.ndim == 3:
        return 1
    raise ValueError(f"Numero dimensioni non supportato: {data.ndim}")


def _to_2d_traces_samples(data: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim == 2:
        return arr, arr.shape
    raise ValueError("Funzione disponibile solo per radargrammi 2D (tracce x campioni).")


def _odd(value: int, minimum: int = 3) -> int:
    n = int(max(value, minimum))
    if n % 2 == 0:
        n += 1
    return n


def _find_true_chunks(mask: np.ndarray) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    if mask.size == 0:
        return out
    in_chunk = False
    start = 0
    for i, val in enumerate(mask.tolist()):
        if val and not in_chunk:
            start = i
            in_chunk = True
        elif (not val) and in_chunk:
            out.append((start, i - 1))
            in_chunk = False
    if in_chunk:
        out.append((start, mask.size - 1))
    return out


def _shift_trace_samples(trace: np.ndarray, shift: int) -> np.ndarray:
    t = np.asarray(trace, dtype=np.float64).ravel()
    n = t.size
    if n == 0 or shift == 0:
        return t.copy()
    out = np.zeros_like(t)
    if shift > 0:
        k = min(shift, n)
        out[: n - k] = t[k:]
    else:
        k = min(-shift, n)
        out[k:] = t[: n - k]
    return out


def _corr_coeff_1d(a: np.ndarray, b: np.ndarray) -> float:
    av = np.asarray(a, dtype=np.float64).ravel()
    bv = np.asarray(b, dtype=np.float64).ravel()
    mask = np.isfinite(av) & np.isfinite(bv)
    if np.count_nonzero(mask) < 4:
        return -1.0
    av = av[mask] - np.nanmean(av[mask])
    bv = bv[mask] - np.nanmean(bv[mask])
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
    if denom <= 0:
        return -1.0
    return float(np.dot(av, bv) / denom)


def _best_shift(reference: np.ndarray, target: np.ndarray, max_shift: int) -> int:
    best_shift = 0
    best_score = -1.0e12
    for shift in range(-int(max_shift), int(max_shift) + 1):
        s = _shift_trace_samples(target, shift)
        score = _corr_coeff_1d(reference, s)
        if score > best_score:
            best_score = score
            best_shift = shift
    return int(best_shift)


def _percentile_abs_by_trace(data_2d: np.ndarray, qclip: float) -> np.ndarray:
    q = float(np.clip(qclip, 0.5, 1.0))
    ref = np.nanquantile(np.abs(data_2d), q, axis=1, keepdims=True)
    ref[~np.isfinite(ref)] = 1.0
    ref[np.abs(ref) < 1.0e-12] = 1.0
    return ref


def _time_axis_3d(cube: np.ndarray, time_axis: int = 0) -> int:
    arr = np.asarray(cube)
    if arr.ndim != 3:
        raise ValueError("Atteso cubo 3D.")
    axis = int(time_axis)
    if axis < 0:
        axis += 3
    if axis not in (0, 1, 2):
        raise ValueError("time_axis non valido per cubo 3D.")
    return axis


def normalize_for_display(data: np.ndarray, strategy: str = "percentile_2_98") -> np.ndarray:
    """Normalizza dati per visualizzazione con diverse strategie.
    
    Strategie disponibili:
    - 'percentile_2_98': Percentili 2-98% (default, conservativo)
    - 'percentile_1_99': Percentili 1-99% (piu robusto)
    - 'percentile_0.5_99.5': Percentili 0.5-99.5% (molto robusto)
    - 'log': Normalizzazione logaritmica (per segnali con gain)
    - 'histogram_equalization': Equalizzazione istogramma (massimo contrasto)
    """
    arr = np.asarray(data, dtype=np.float64)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr)

    vals = arr[finite]
    strategy_norm = str(strategy).lower()

    if strategy_norm == "percentile_1_99":
        vmin = np.nanpercentile(vals, 1)
        vmax = np.nanpercentile(vals, 99)
    elif strategy_norm == "percentile_0.5_99.5":
        vmin = np.nanpercentile(vals, 0.5)
        vmax = np.nanpercentile(vals, 99.5)
    elif strategy_norm == "log":
        # Logaritmica: utile per segnali con gain esponenziale
        abs_vals = np.abs(vals)
        abs_vals_nz = abs_vals[abs_vals > 0]
        if abs_vals_nz.size == 0:
            return np.zeros_like(arr)
        vmin_abs = np.nanpercentile(abs_vals_nz, 1)
        vmax_abs = np.nanpercentile(abs_vals_nz, 99)
        if vmin_abs <= 0:
            vmin_abs = np.nanmin(abs_vals_nz[abs_vals_nz > 0])
        arr_log = np.sign(arr) * np.log10(np.maximum(np.abs(arr), vmin_abs))
        vmin = np.log10(vmin_abs)
        vmax = np.log10(vmax_abs)
        if np.isclose(vmin, vmax):
            return np.zeros_like(arr)
        clipped = np.clip(arr_log, vmin, vmax)
        return (clipped - vmin) / (vmax - vmin)
    elif strategy_norm == "histogram_equalization":
        # Equalizzazione istogramma: massimizza il contrasto
        n_bins = 2048
        hist, bin_edges = np.histogram(vals, bins=n_bins)
        cdf = hist.cumsum()
        cdf_normalized = cdf / cdf[-1]
        # Interpolazione per mappare i valori originali sulla CDF normalizzata
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        equalized = np.interp(arr.ravel(), bin_centers, cdf_normalized)
        equalized = equalized.reshape(arr.shape)
        equalized[~finite] = 0.0
        return equalized
    else:  # default: percentile_2_98
        vmin = np.nanpercentile(vals, 2)
        vmax = np.nanpercentile(vals, 98)

    # Fallback se vmin == vmax
    if np.isclose(vmin, vmax):
        vmin = np.nanpercentile(vals, 0.1)
        vmax = np.nanpercentile(vals, 99.9)
    if np.isclose(vmin, vmax):
        vmin = float(np.nanmin(vals))
        vmax = float(np.nanmax(vals))
    if np.isclose(vmin, vmax):
        return np.zeros_like(arr)
    
    clipped = np.clip(arr, vmin, vmax)
    return (clipped - vmin) / (vmax - vmin)


def apply_dewow_filter(data: np.ndarray, window: int = 41) -> np.ndarray:
    if window < 3:
        return data
    if window % 2 == 0:
        window += 1
    axis = _time_axis_for(data)
    trend = gaussian_filter1d(data, sigma=max(window // 6, 1), axis=axis, mode="nearest")
    return data - trend


def apply_background_removal(data: np.ndarray) -> np.ndarray:
    if data.ndim == 2:
        bg = data.mean(axis=0, keepdims=True)
        return data - bg
    if data.ndim == 3:
        bg = data.mean(axis=(1, 2), keepdims=True)
        return data - bg
    raise ValueError(f"Numero dimensioni non supportato: {data.ndim}")


def apply_bandpass_filter(
    data: np.ndarray,
    low_cut: float,
    high_cut: float,
    sample_rate: float,
    order: int = 4,
) -> np.ndarray:
    axis = _time_axis_for(data)
    nyquist = sample_rate * 0.5

    if high_cut <= 0:
        return data
    high = min(high_cut / nyquist, 0.999)
    low = max(low_cut / nyquist, 0.0)

    if low == 0 and high <= 0:
        return data
    if low >= high and low > 0:
        raise ValueError("Band-pass non valido: low_cut deve essere minore di high_cut.")

    if low <= 0:
        b, a = butter(order, high, btype="lowpass")
    elif high >= 0.999:
        b, a = butter(order, low, btype="highpass")
    else:
        b, a = butter(order, [low, high], btype="bandpass")
    return filtfilt(b, a, data, axis=axis, method="gust")


def apply_bandpass_gpr(
    data: np.ndarray,
    sample_rate: float,
    f_start_hz: float,
    f_end_hz: float,
) -> np.ndarray:
    axis = _time_axis_for(data)
    n = data.shape[axis]
    if n <= 2:
        return data

    f2 = max(float(f_start_hz), 1.0)
    f3 = max(float(f_end_hz), f2 + 1.0)
    width = max(f3 - f2, 1.0)
    f1 = max(1.0, f2 - width / 4.0, f2 - width / 8.0)
    f4 = f3 + width / 4.0
    nyquist = 0.5 * float(sample_rate)
    f4 = min(f4, nyquist * 0.999)
    if f4 <= f1:
        return data

    n_pad = 2 * n
    pad_width = [(0, 0)] * data.ndim
    pad_width[axis] = (0, n)
    padded = np.pad(data, pad_width=pad_width, mode="constant")

    freqs = np.fft.rfftfreq(n_pad, d=1.0 / float(sample_rate))
    h = np.zeros_like(freqs, dtype=np.float64)

    ramp_up = (freqs >= f1) & (freqs < f2)
    if np.any(ramp_up):
        h[ramp_up] = (freqs[ramp_up] - f1) / max(f2 - f1, 1.0e-9)
    passband = (freqs >= f2) & (freqs <= f3)
    h[passband] = 1.0
    ramp_down = (freqs > f3) & (freqs <= f4)
    if np.any(ramp_down):
        h[ramp_down] = (f4 - freqs[ramp_down]) / max(f4 - f3, 1.0e-9)

    spec = np.fft.rfft(padded, axis=axis)
    shape = [1] * spec.ndim
    shape[axis] = h.size
    filtered = np.fft.irfft(spec * h.reshape(shape), n=n_pad, axis=axis)

    slices = [slice(None)] * filtered.ndim
    slices[axis] = slice(0, n)
    return filtered[tuple(slices)]


def apply_time_gain(data: np.ndarray, power: float = 2.0, scale: float = 1.0) -> np.ndarray:
    axis = _time_axis_for(data)
    n = data.shape[axis]
    t = np.linspace(0.0, 1.0, n) ** power
    gain = 1.0 + scale * t

    shape = [1] * data.ndim
    shape[axis] = n
    return data * gain.reshape(shape)


def apply_gain_db(data: np.ndarray, db_gain: float = 24.0, exponent: float = 1.0) -> np.ndarray:
    axis = _time_axis_for(data)
    n = data.shape[axis]
    if n <= 1:
        return data
    t = np.linspace(0.0, 1.0, n) ** max(exponent, 0.0)
    gain = 10.0 ** ((db_gain * t) / 20.0)
    shape = [1] * data.ndim
    shape[axis] = n
    return data * gain.reshape(shape)


def apply_gain_curve_db(data: np.ndarray, gain_points_db: np.ndarray) -> np.ndarray:
    axis = _time_axis_for(data)
    n = data.shape[axis]
    if n <= 1:
        return data

    gp = np.asarray(gain_points_db, dtype=np.float64).ravel()
    if gp.size == 0:
        return data
    if gp.size == 1:
        g = np.full(n, gp[0], dtype=np.float64)
    else:
        src_idx = np.linspace(0.0, n - 1, gp.size, dtype=np.float64)
        dst_idx = np.arange(n, dtype=np.float64)
        g = np.interp(dst_idx, src_idx, gp)

    gain = 10.0 ** (g / 20.0)
    shape = [1] * data.ndim
    shape[axis] = n
    return data * gain.reshape(shape)


def apply_missing_trace_interpolation(data: np.ndarray, gap: int = 3) -> np.ndarray:
    arr, _shape = _to_2d_traces_samples(data)
    if arr.shape[0] < 2:
        return arr.copy()
    first_row = arr[:, 0]
    valid = np.isfinite(first_row)
    missing = ~valid
    if not np.any(missing) or np.count_nonzero(valid) < 2:
        return arr.copy()

    x_all = np.arange(arr.shape[0], dtype=np.float64)
    x_valid = x_all[valid]
    out = arr.copy()
    for sample_idx in range(arr.shape[1]):
        vals = arr[:, sample_idx]
        vals_valid = np.isfinite(vals) & valid
        if np.count_nonzero(vals_valid) >= 2:
            out[:, sample_idx] = np.interp(x_all, x_all[vals_valid], vals[vals_valid])
        elif np.count_nonzero(vals_valid) == 1:
            out[:, sample_idx] = vals[vals_valid][0]

    gap_eff = max(int(gap), 1)
    mask = np.full(arr.shape[0], np.nan, dtype=np.float64)
    mask[valid] = 1.0
    missing_idx = np.where(missing)[0]
    for idx in missing_idx:
        if np.min(np.abs(idx - x_valid)) < (gap_eff / 2.0):
            mask[idx] = 1.0
    out *= mask[:, None]
    return out


def apply_trace_interpolation(data: np.ndarray, minfactor: float = 2.0, maxfactor: float = 2.0) -> np.ndarray:
    arr, _shape = _to_2d_traces_samples(data)
    out = arr.copy()
    if out.shape[0] < 2:
        return out

    mean_abs = np.nanmean(np.abs(out), axis=1)
    baseline = float(np.nanmean(mean_abs))
    if (not np.isfinite(baseline)) or baseline <= 0:
        return out

    minfactor_eff = max(float(minfactor), 1.0e-6)
    maxfactor_eff = max(float(maxfactor), 1.0e-6)
    bad = (mean_abs >= baseline * maxfactor_eff) | (mean_abs <= baseline / minfactor_eff) | (~np.isfinite(mean_abs))
    chunks = _find_true_chunks(bad)

    n_traces = out.shape[0]
    for start, end in chunks:
        if start == end:
            i = start
            if 0 < i < (n_traces - 1):
                out[i, :] = np.nanmean(out[[i - 1, i + 1], :], axis=0)
            elif i == 0 and n_traces > 1:
                out[i, :] = out[i + 1, :]
            elif i == (n_traces - 1) and n_traces > 1:
                out[i, :] = out[i - 1, :]
            continue

        if start > 0 and end < (n_traces - 1):
            xl = float(start - 1)
            xr = float(end + 1)
            xq = np.arange(start, end + 1, dtype=np.float64)
            for sample_idx in range(out.shape[1]):
                yl = out[start - 1, sample_idx]
                yr = out[end + 1, sample_idx]
                out[start : end + 1, sample_idx] = np.interp(xq, [xl, xr], [yl, yr])
        elif start == 0 and end < (n_traces - 1):
            out[start : end + 1, :] = out[end + 1, :][None, :]
        elif end == (n_traces - 1) and start > 0:
            out[start : end + 1, :] = out[start - 1, :][None, :]
    return out


def apply_remove_horizontal_lines(
    data: np.ndarray,
    mode: str = "mean",
    window_traces: int = 0,
) -> np.ndarray:
    arr, _shape = _to_2d_traces_samples(data)
    out = arr.copy()
    mode_norm = str(mode).strip().lower()
    reducer = np.nanmean if mode_norm == "mean" else np.nanmedian

    if int(window_traces) <= 0:
        ref = reducer(out, axis=0, keepdims=True)
        return out - ref

    w = _odd(int(window_traces), minimum=3)
    hw = (w - 1) // 2
    out2 = np.empty_like(out)
    n_traces = out.shape[0]
    for i in range(n_traces):
        i0 = max(0, i - hw)
        i1 = min(n_traces, i + hw + 1)
        ref = reducer(out[i0:i1, :], axis=0)
        out2[i, :] = out[i, :] - ref
    return out2


def apply_k_highpass(data: np.ndarray, dx_m: float, kcutoff: float) -> np.ndarray:
    arr, _shape = _to_2d_traces_samples(data)
    dx = float(dx_m)
    kc = float(kcutoff)
    if arr.shape[0] < 8 or dx <= 0 or kc <= 0:
        return arr.copy()

    out = arr.copy()
    nan_traces = ~np.isfinite(out[:, 0])
    out[nan_traces, :] = 0.0

    n_traces = out.shape[0]
    k = np.fft.fftfreq(n_traces, d=dx)
    ka = np.abs(k)
    k1 = 0.5 * kc
    k2 = kc
    h = np.zeros_like(ka)
    h[ka >= k2] = 1.0
    ramp = (ka >= k1) & (ka < k2)
    if np.any(ramp):
        h[ramp] = (ka[ramp] - k1) / max(k2 - k1, 1.0e-12)

    spec = np.fft.fft(out, axis=0)
    filt = np.fft.ifft(spec * h[:, None], axis=0).real
    filt[nan_traces, :] = np.nan
    return filt


def apply_medfilt_time(data: np.ndarray, window_samples: int = 5) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float64)
    w = _odd(int(window_samples), minimum=3)
    if arr.ndim == 2:
        return median_filter(arr, size=(1, w), mode="nearest")
    if arr.ndim == 3:
        return median_filter(arr, size=(1, 1, w), mode="nearest")
    raise ValueError(f"Numero dimensioni non supportato: {arr.ndim}")


def apply_medfilt_x(
    data: np.ndarray,
    window_traces: int = 5,
    sample_rate: float = 1.0e9,
    tstart_ns: float = 0.0,
) -> np.ndarray:
    arr, _shape = _to_2d_traces_samples(data)
    if arr.shape[0] < 2:
        return arr.copy()
    w = _odd(int(window_traces), minimum=3)
    dt_ns = 1.0e9 / max(float(sample_rate), 1.0)
    start_idx = int(np.clip(np.floor(float(tstart_ns) / dt_ns), 0, arr.shape[1]))
    if start_idx >= arr.shape[1]:
        return arr.copy()
    out = arr.copy()
    part = out[:, start_idx:]
    out[:, start_idx:] = median_filter(part, size=(w, 1), mode="nearest")
    return out


def apply_normalize2d(data: np.ndarray, qclip: float = 0.98) -> np.ndarray:
    arr, _shape = _to_2d_traces_samples(data)
    ref = _percentile_abs_by_trace(arr, qclip=qclip)
    out = arr / ref
    out[~np.isfinite(out)] = 0.0
    return out


def apply_normalize3d(cube: np.ndarray, qclip: float = 0.98, time_axis: int = 0) -> np.ndarray:
    arr = np.asarray(cube, dtype=np.float64)
    axis = _time_axis_3d(arr, time_axis=time_axis)
    moved = np.moveaxis(arr, axis, 0)  # [t, ...]
    shp = moved.shape
    traces = moved.reshape(shp[0], -1).T  # [ntraces, t]
    ref = _percentile_abs_by_trace(traces, qclip=qclip)
    out = traces / ref
    out[~np.isfinite(out)] = 0.0
    out3 = out.T.reshape(shp)
    return np.moveaxis(out3, 0, axis)


def apply_reduce_number_of_samples(
    cube: np.ndarray,
    factor: float = 1.0,
    time_axis: int = 0,
) -> np.ndarray:
    arr = np.asarray(cube, dtype=np.float64)
    axis = _time_axis_3d(arr, time_axis=time_axis)
    moved = np.moveaxis(arr, axis, 0)  # [t, y, x]
    nt = moved.shape[0]
    f = float(factor)
    if not np.isfinite(f) or f <= 0:
        return arr.copy()
    if np.isclose(f, 1.0):
        return arr.copy()

    old_idx = np.arange(nt, dtype=np.float64)
    if f > 1.0:
        # MATLAB: take every n-th sample.
        step = max(int(round(f)), 1)
        new_idx = np.arange(0.0, float(nt - 1) + 1.0e-9, float(step), dtype=np.float64)
    else:
        # MATLAB: n<1 increases samples.
        step = max(float(f), 1.0e-6)
        new_idx = np.arange(0.0, float(nt - 1), step, dtype=np.float64)
        if new_idx.size == 0 or new_idx[-1] < (nt - 1):
            new_idx = np.append(new_idx, float(nt - 1))

    flat = moved.reshape(nt, -1)
    interp = interp1d(old_idx, flat, axis=0, kind="cubic" if f < 1.0 else "linear", bounds_error=False, fill_value="extrapolate")
    out_flat = interp(new_idx)
    out = out_flat.reshape((new_idx.size,) + moved.shape[1:])
    return np.moveaxis(out, 0, axis)


def apply_linear_interpolation_3d_cube(
    cube: np.ndarray,
    radius_xy: float = 0.0,
    dx: float = 1.0,
    dy: float | None = None,
    time_axis: int = 0,
) -> np.ndarray:
    arr = np.asarray(cube, dtype=np.float64)
    axis = _time_axis_3d(arr, time_axis=time_axis)
    moved = np.moveaxis(arr, axis, 0)  # [t, y, x]
    out = moved.copy()
    if radius_xy <= 0:
        return arr.copy()
    dy_eff = float(dx if dy is None else dy)
    rad = float(radius_xy)
    if not np.isfinite(rad) or rad <= 0:
        return arr.copy()

    ny, nx = moved.shape[1], moved.shape[2]
    gx, gy = np.meshgrid(np.arange(nx, dtype=np.float64), np.arange(ny, dtype=np.float64))

    for it in range(moved.shape[0]):
        sl = moved[it, :, :]
        valid = np.isfinite(sl)
        if np.count_nonzero(valid) < 4:
            continue

        pts = np.column_stack((gx[valid], gy[valid]))
        vals = sl[valid]

        # distance in metric units to nearest valid cell
        miss = ~valid
        if not np.any(miss):
            continue
        miss_pts = np.column_stack((gx[miss], gy[miss]))
        # approximate elliptical metric if dx != dy
        if not np.isclose(dx, dy_eff):
            pts_m = np.column_stack((pts[:, 0] * float(dx), pts[:, 1] * float(dy_eff)))
            miss_pts_m = np.column_stack((miss_pts[:, 0] * float(dx), miss_pts[:, 1] * float(dy_eff)))
        else:
            pts_m = pts
            miss_pts_m = miss_pts

        tree = cKDTree(pts_m)
        nearest_dist, _nn = tree.query(miss_pts_m, k=1)
        within = nearest_dist <= rad
        if not np.any(within):
            continue

        candidates = miss_pts[within]
        interp_vals = griddata(pts, vals, candidates, method="linear")
        nan_interp = ~np.isfinite(interp_vals)
        if np.any(nan_interp):
            interp_vals[nan_interp] = griddata(pts, vals, candidates[nan_interp], method="nearest")

        cy = candidates[:, 1].astype(np.int64)
        cx = candidates[:, 0].astype(np.int64)
        sl2 = out[it, :, :]
        sl2[cy, cx] = interp_vals
        out[it, :, :] = sl2

    return np.moveaxis(out, 0, axis)


def apply_semblance_smoothing(
    cube: np.ndarray,
    window: int = 5,
    exponent: float = 1.0,
    normalize_after: bool = False,
    qclip: float = 0.98,
    time_axis: int = 0,
) -> np.ndarray:
    arr = np.asarray(cube, dtype=np.float64)
    axis = _time_axis_3d(arr, time_axis=time_axis)
    moved = np.moveaxis(arr, axis, 0)  # [t, y, x]

    w = _odd(int(window), minimum=3)
    # local mean in a cubic window
    local_mean = gaussian_filter(moved, sigma=max((w - 1) / 6.0, 1.0), mode="nearest")
    local_mean_sq = gaussian_filter(moved * moved, sigma=max((w - 1) / 6.0, 1.0), mode="nearest")
    semb = (local_mean * local_mean) / np.maximum(local_mean_sq, 1.0e-12)
    semb = np.clip(semb, 0.0, 1.0)
    out = local_mean * (semb ** max(float(exponent), 0.0))

    if normalize_after:
        out = apply_normalize3d(out, qclip=qclip, time_axis=0)
    return np.moveaxis(out, 0, axis)


def make_amplitude_spectrum(data: np.ndarray, sample_rate: float) -> tuple[np.ndarray, np.ndarray]:
    arr, _shape = _to_2d_traces_samples(data)
    n = arr.shape[1]
    if n < 2:
        return np.array([], dtype=np.float64), np.zeros((arr.shape[0], 0), dtype=np.float64)
    nfft = int(2 ** np.ceil(np.log2(max(n, 2))))
    y = np.fft.rfft(arr, n=nfft, axis=1) / float(n)
    f_mhz = np.fft.rfftfreq(nfft, d=1.0 / max(float(sample_rate), 1.0)) / 1.0e6
    abs_amps = 2.0 * np.abs(y)
    return f_mhz, abs_amps


def apply_spectral_whitening(
    data: np.ndarray,
    sample_rate: float,
    fmin_hz: float,
    fmax_hz: float,
    alpha: float = 0.01,
) -> np.ndarray:
    arr, _shape = _to_2d_traces_samples(data)
    n_traces, n_samples = arr.shape
    if n_samples < 8 or n_traces < 1:
        return arr.copy()

    a = float(alpha)
    if (not np.isfinite(a)) or a <= 0:
        a = 0.01

    nfft = int(2 ** np.ceil(np.log2(max(n_samples, 2))))
    aw = np.zeros(nfft, dtype=np.float64)
    for i in range(n_traces):
        trace = np.nan_to_num(arr[i, :], nan=0.0)
        xc = np.correlate(trace, trace, mode="full")
        if xc.size >= nfft:
            xc_use = xc[:nfft]
        else:
            xc_use = np.pad(xc, (0, nfft - xc.size), mode="constant")
        aw += np.abs(np.fft.fft(xc_use, n=nfft) / float(n_samples))
    aw /= float(max(n_traces, 1))
    aw = np.maximum(aw, 1.0e-12)

    sw = np.fft.fft(np.nan_to_num(arr, nan=0.0), n=nfft, axis=1) / float(n_samples)
    sw_new = sw * (aw[None, :] ** (a - 1.0))
    out = np.fft.ifft(sw_new, n=nfft, axis=1).real / np.sqrt(a)
    out = out[:, :n_samples]

    return apply_bandpass_gpr(
        out,
        sample_rate=float(sample_rate),
        f_start_hz=float(max(fmin_hz, 0.0)),
        f_end_hz=float(max(fmax_hz, max(fmin_hz, 0.0) + 1.0)),
    )


def apply_spherical_divergence(data: np.ndarray, sample_rate: float) -> np.ndarray:
    axis = _time_axis_for(data)
    n = data.shape[axis]
    if n <= 1:
        return np.asarray(data, dtype=np.float64).copy()
    dt_ns = 1.0e9 / max(float(sample_rate), 1.0)
    t_ns = np.arange(n, dtype=np.float64) * dt_ns
    shape = [1] * data.ndim
    shape[axis] = n
    return np.asarray(data, dtype=np.float64) * t_ns.reshape(shape)


def apply_t0shift(data: np.ndarray, sample_rate: float, t0_ns: float) -> np.ndarray:
    arr, _shape = _to_2d_traces_samples(data)
    dt_ns = 1.0e9 / max(float(sample_rate), 1.0)
    shift = int(max(round(float(t0_ns) / dt_ns), 0))
    if shift <= 0:
        return arr.copy()
    n = arr.shape[1]
    out = np.zeros_like(arr)
    if shift < n:
        out[:, : n - shift] = arr[:, shift:]
    return out


def apply_t0corr_thresh(data: np.ndarray, threshold: float) -> np.ndarray:
    arr, _shape = _to_2d_traces_samples(data)
    out = arr.copy()
    thr = float(threshold)
    n = out.shape[1]
    for i in range(out.shape[0]):
        tr = out[i, :]
        if thr < 0:
            idxs = np.where(tr <= thr)[0]
        else:
            idxs = np.where(tr >= thr)[0]
        if idxs.size == 0:
            continue
        k = int(idxs[0])
        shifted = np.zeros(n, dtype=np.float64)
        if k < n:
            shifted[: n - k] = tr[k:]
        out[i, :] = shifted
    return out


def apply_t0correction_xcorr(
    data: np.ndarray,
    sample_rate: float,
    t0_ns: float,
    max_shift_samples: int = 80,
) -> np.ndarray:
    arr, _shape = _to_2d_traces_samples(data)
    if arr.shape[0] < 2 or arr.shape[1] < 8:
        return apply_t0shift(arr, sample_rate=sample_rate, t0_ns=t0_ns)
    ref = np.nanmean(arr, axis=0)
    out = np.zeros_like(arr)
    max_shift = max(int(max_shift_samples), 0)
    for i in range(arr.shape[0]):
        tr = np.asarray(arr[i, :], dtype=np.float64)
        shift = _best_shift(ref, tr, max_shift=max_shift)
        out[i, :] = _shift_trace_samples(tr, shift)
    return apply_t0shift(out, sample_rate=sample_rate, t0_ns=t0_ns)


def apply_hilbert_transform(data: np.ndarray, mode: str = "envelope") -> np.ndarray:
    axis = _time_axis_for(data)
    analytic = hilbert(data, axis=axis)
    mode_norm = str(mode).strip().lower()
    if mode_norm == "envelope":
        return np.abs(analytic)
    if mode_norm == "real":
        return np.real(analytic)
    if mode_norm == "imag":
        return np.imag(analytic)
    if mode_norm == "phase":
        return np.unwrap(np.angle(analytic), axis=axis)
    raise ValueError(f"Modalita Hilbert non supportata: {mode}")


def apply_gaussian_smoothing(data: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    if sigma <= 0:
        return data
    return gaussian_filter(data, sigma=sigma)


def apply_dc_removal(
    data: np.ndarray,
    sample_rate: float,
    start_ns: float = 0.0,
    end_ns: float | None = None,
) -> np.ndarray:
    axis = _time_axis_for(data)
    n = data.shape[axis]
    if n <= 1:
        return data

    dt_ns = 1.0e9 / float(sample_rate)
    t = np.arange(n, dtype=np.float64) * dt_ns
    end_eff = float(np.nanmax(t)) if end_ns is None else float(end_ns)
    start_eff = float(start_ns)
    if end_eff < start_eff:
        start_eff, end_eff = end_eff, start_eff
    mask = (t >= start_eff) & (t <= end_eff)
    if not np.any(mask):
        mask = np.ones(n, dtype=bool)

    slices = [slice(None)] * data.ndim
    slices[axis] = mask
    mean_val = np.nanmean(data[tuple(slices)], axis=axis, keepdims=True)
    return data - mean_val


def apply_cut_twt(
    data: np.ndarray,
    sample_rate: float,
    tmax_ns: float,
    fill_value: float = 0.0,
) -> np.ndarray:
    axis = _time_axis_for(data)
    n = data.shape[axis]
    if n <= 1:
        return data
    dt_ns = 1.0e9 / float(sample_rate)
    t = np.arange(n, dtype=np.float64) * dt_ns
    keep = t <= float(tmax_ns)

    out = np.array(data, copy=True)
    slices = [slice(None)] * out.ndim
    slices[axis] = ~keep
    out[tuple(slices)] = float(fill_value)
    return out


def apply_attenuation_correction(
    data: np.ndarray,
    sample_rate: float,
    sigma_s_per_m: float,
    eps_r: float,
) -> np.ndarray:
    axis = _time_axis_for(data)
    n = data.shape[axis]
    if n <= 1:
        return data

    sigma = max(float(sigma_s_per_m), 0.0)
    eps_rel = max(float(eps_r), 1.0e-6)
    eps0 = 8.854e-12
    mu0 = 4.0 * np.pi * 1.0e-7
    alpha = sigma / 2.0 * np.sqrt(mu0 / (eps_rel * eps0))
    v = 1.0 / np.sqrt(eps0 * eps_rel * mu0)

    dt_ns = 1.0e9 / float(sample_rate)
    t_ns = np.arange(n, dtype=np.float64) * dt_ns
    z = (t_ns * 1.0e-9) * 0.5 * v
    gain = np.exp(alpha * z)

    shape = [1] * data.ndim
    shape[axis] = n
    return data * gain.reshape(shape)
