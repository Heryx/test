from __future__ import annotations

import json
import itertools
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.io import loadmat


HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"


@dataclass
class OgprProfile:
    file_name: str
    channel_index: int
    data: np.ndarray  # shape: (traces, samples)
    x: np.ndarray | None
    y: np.ndarray | None
    z: np.ndarray | None
    srs: dict[str, Any] | None
    metadata: dict[str, Any]

    @property
    def label(self) -> str:
        channel_number = self.channel_index + 1
        return f"{Path(self.file_name).name} | CH{channel_number}"


def _pick_numeric_array(candidates: list[np.ndarray]) -> np.ndarray:
    valid = [arr for arr in candidates if np.issubdtype(np.asarray(arr).dtype, np.number)]
    if not valid:
        raise ValueError("Nessun array numerico trovato nel file.")
    valid.sort(key=lambda x: np.asarray(x).size, reverse=True)
    return np.asarray(valid[0], dtype=np.float64)


def _sanitize_metadata_value(value: Any, depth: int = 0, max_depth: int = 5) -> Any:
    if depth > max_depth:
        return None
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, subvalue in value.items():
            key_text = str(key)
            if key_text.lower() in {"data", "buffer", "raw"}:
                continue
            out[key_text] = _sanitize_metadata_value(subvalue, depth=depth + 1, max_depth=max_depth)
        return out
    if isinstance(value, (list, tuple)):
        return [
            _sanitize_metadata_value(subvalue, depth=depth + 1, max_depth=max_depth)
            for subvalue in list(value)[:32]
        ]
    if isinstance(value, np.ndarray):
        arr = np.asarray(value).ravel()
        if arr.size == 0:
            return []
        return [float(arr[0])]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _load_from_hdf5(buffer: BytesIO) -> np.ndarray:
    buffer.seek(0)
    arrays: list[np.ndarray] = []
    with h5py.File(buffer, "r") as handle:
        def _walker(_name: str, obj: Any) -> None:
            if isinstance(obj, h5py.Dataset):
                data = obj[()]
                if isinstance(data, np.ndarray):
                    arrays.append(data)
                elif np.isscalar(data):
                    arrays.append(np.asarray([data]))

        handle.visititems(_walker)
    return _pick_numeric_array(arrays)


def _load_from_npz(buffer: BytesIO) -> np.ndarray:
    buffer.seek(0)
    with np.load(buffer, allow_pickle=False) as archive:
        arrays = [np.asarray(archive[key]) for key in archive.files]
    return _pick_numeric_array(arrays)


def _load_from_mat(buffer: BytesIO) -> np.ndarray:
    buffer.seek(0)
    data = loadmat(buffer)
    arrays = [np.asarray(value) for key, value in data.items() if not key.startswith("__")]
    return _pick_numeric_array(arrays)


def _load_from_text(buffer: BytesIO) -> np.ndarray:
    for delimiter in [",", ";", "\t", None]:
        try:
            buffer.seek(0)
            arr = np.loadtxt(buffer, delimiter=delimiter)
            if arr.size > 0:
                return np.asarray(arr, dtype=np.float64)
        except Exception:
            continue
    raise ValueError("Impossibile interpretare il file come tabella testuale numerica.")


def _consume_line(raw: bytes, start: int) -> tuple[str, int]:
    end = raw.find(b"\n", start)
    if end < 0:
        raise ValueError("Formato OpenGPR non valido: riga non terminata.")
    line_bytes = raw[start:end]
    if line_bytes.endswith(b"\r"):
        line_bytes = line_bytes[:-1]
    try:
        return line_bytes.decode("utf-8"), end + 1
    except UnicodeDecodeError as exc:
        raise ValueError("Formato OpenGPR non valido: riga non UTF-8.") from exc


def _safe_int(value: Any, field_name: str) -> int:
    try:
        parsed = int(value)
    except Exception as exc:
        raise ValueError(f"Campo OpenGPR non valido: {field_name}.") from exc
    if parsed < 0:
        raise ValueError(f"Campo OpenGPR negativo: {field_name}.")
    return parsed


def _normalize_value_type(value_type: Any) -> str:
    if value_type is None:
        return "int16"
    text = str(value_type).strip().lower().replace("_", "").replace("-", "").replace(" ", "")
    aliases = {
        "int16": "int16",
        "short": "int16",
        "signed16": "int16",
        "int16t": "int16",
        "uint16": "uint16",
        "unsigned16": "uint16",
        "uint16t": "uint16",
        "float": "float32",
        "float32": "float32",
        "single": "float32",
        "double": "float64",
        "float64": "float64",
    }
    if text in aliases:
        return aliases[text]
    raise ValueError(
        "valueType OpenGPR non supportato: "
        f"{value_type}. Tipi supportati: int16, uint16, float32, float64."
    )


def _dtype_for_value_type(value_type: str) -> np.dtype:
    if value_type == "int16":
        return np.dtype("<i2")
    if value_type == "uint16":
        return np.dtype("<u2")
    if value_type == "float32":
        return np.dtype("<f4")
    if value_type == "float64":
        return np.dtype("<f8")
    raise ValueError(f"valueType OpenGPR non riconosciuto: {value_type}")


def _extract_block_candidates(
    raw: bytes,
    descriptor: dict[str, Any],
    data_start: int,
    preferred_mode: str | None = None,
) -> list[tuple[str, bytes]]:
    byte_offset = _safe_int(descriptor.get("byteOffset", 0), "byteOffset")
    byte_size = _safe_int(descriptor.get("byteSize"), "byteSize")
    candidates: list[tuple[str, int, int]] = [
        ("relative", data_start + byte_offset, data_start + byte_offset + byte_size),
        ("absolute", byte_offset, byte_offset + byte_size),
    ]
    if preferred_mode in {"relative", "absolute"}:
        candidates.sort(key=lambda item: 0 if item[0] == preferred_mode else 1)
    blocks: list[tuple[str, bytes]] = []
    seen_ranges: set[tuple[int, int]] = set()
    for mode, start, end in candidates:
        if (start, end) in seen_ranges:
            continue
        seen_ranges.add((start, end))
        if 0 <= start <= end <= len(raw):
            blocks.append((mode, raw[start:end]))
    if not blocks:
        raise ValueError("Offset/size OpenGPR fuori dai limiti del file.")
    return blocks


def _infer_offset_mode(descriptors: list[dict[str, Any]], data_start: int, raw_len: int) -> str:
    offsets: list[int] = []
    sizes: list[int] = []
    for descriptor in descriptors:
        try:
            offsets.append(_safe_int(descriptor.get("byteOffset", 0), "byteOffset"))
            sizes.append(_safe_int(descriptor.get("byteSize", 0), "byteSize"))
        except Exception:
            continue

    if not offsets:
        return "relative"

    absolute_valid = all(0 <= off <= off + size <= raw_len for off, size in zip(offsets, sizes))
    relative_valid = all(
        0 <= data_start + off <= data_start + off + size <= raw_len
        for off, size in zip(offsets, sizes)
    )

    if absolute_valid and not relative_valid:
        return "absolute"
    if relative_valid and not absolute_valid:
        return "relative"

    min_offset = min(offsets)
    if min_offset == 0:
        return "relative"
    if min_offset > data_start:
        return "absolute"

    return "relative"


def _select_radar_descriptor(descriptors: list[dict[str, Any]]) -> dict[str, Any]:
    radar_blocks = []
    for descriptor in descriptors:
        block_type = str(descriptor.get("type", "")).strip().lower()
        if "radar volume" in block_type:
            radar_blocks.append(descriptor)
    if radar_blocks:
        return max(radar_blocks, key=lambda item: _safe_int(item.get("byteSize", 0), "byteSize"))

    numeric_blocks = [item for item in descriptors if isinstance(item, dict) and item.get("byteSize") is not None]
    if not numeric_blocks:
        raise ValueError("Nessun data block disponibile nell'header OpenGPR.")
    return max(numeric_blocks, key=lambda item: _safe_int(item.get("byteSize"), "byteSize"))


def _adjacent_trace_correlation_score(volume: np.ndarray) -> float:
    # volume expected shape: (slices, channels, samples)
    arr = np.asarray(volume, dtype=np.float64)
    slices_count, channels_count, samples_count = arr.shape
    if slices_count < 2 or samples_count < 8:
        return 0.0

    max_pairs = 64
    step = max((slices_count - 1) // max_pairs, 1)
    pair_indices = np.arange(0, slices_count - 1, step, dtype=np.int32)
    if pair_indices.size == 0:
        return 0.0

    correlations: list[float] = []
    for channel_idx in range(channels_count):
        traces = arr[:, channel_idx, :]
        for idx in pair_indices:
            a = traces[idx]
            b = traces[idx + 1]
            if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
                continue
            a0 = a - np.mean(a)
            b0 = b - np.mean(b)
            denom = float(np.linalg.norm(a0) * np.linalg.norm(b0))
            if denom <= 0:
                continue
            corr = float(np.dot(a0, b0) / denom)
            if np.isfinite(corr):
                correlations.append(corr)

    if not correlations:
        return 0.0
    return float(np.median(correlations))


def _volume_quality_score(volume: np.ndarray) -> float:
    arr = np.asarray(volume, dtype=np.float64)
    finite = np.isfinite(arr)
    if not finite.any():
        return -1.0e12

    vals = arr[finite]
    channel_std = np.nanstd(arr, axis=(0, 2))
    median_channel_std = float(np.nanmedian(channel_std))
    nonflat_channels = float(np.mean(channel_std > 1.0e-6))
    dynamic = float(np.nanpercentile(vals, 95) - np.nanpercentile(vals, 5))
    corr = _adjacent_trace_correlation_score(arr)
    diff = np.diff(arr, axis=2)
    roughness = float(np.nanmedian(np.abs(diff)) / (np.nanstd(arr) + 1.0e-12))

    trace_rms = np.sqrt(np.nanmean(arr * arr, axis=2))
    median_rms = float(np.nanmedian(trace_rms))
    edge_zero_penalty = 0.0
    if np.isfinite(median_rms) and median_rms > 0:
        edge_n = max(4, int(0.05 * arr.shape[0]))
        edge = np.concatenate((trace_rms[:edge_n, :].ravel(), trace_rms[-edge_n:, :].ravel()))
        edge_zero_fraction = float(np.mean(edge < (0.05 * median_rms)))
        edge_zero_penalty = 3.0 * edge_zero_fraction

    max_abs = float(np.nanmax(np.abs(vals)))
    penalty = 0.0
    if not np.isfinite(max_abs) or max_abs > 1.0e9:
        penalty -= 10.0

    score = (
        2.0 * nonflat_channels
        + 6.0 * corr
        + 0.6 * np.log1p(abs(median_channel_std))
        + 0.15 * np.log1p(abs(dynamic))
        - 1.2 * roughness
        - 2.5 * edge_zero_penalty
        + penalty
    )
    return float(score)


def _geolocation_quality_score(x: np.ndarray, y: np.ndarray) -> float:
    finite = np.isfinite(x) & np.isfinite(y)
    if not finite.any():
        return -1.0e12
    xs = x[finite]
    ys = y[finite]
    max_abs = float(np.nanmax(np.abs(np.concatenate((xs, ys)))))
    if not np.isfinite(max_abs):
        return -1.0e12
    if max_abs > 1.0e9:
        return -1.0e11
    spread = float(np.nanstd(xs) + np.nanstd(ys))
    unique = float(np.unique(np.column_stack((xs, ys)), axis=0).shape[0])
    return spread + 0.001 * unique


def _descriptor_score(descriptor: dict[str, Any]) -> tuple[int, int]:
    block_type = str(descriptor.get("type", "")).strip().lower()
    score = 0
    if "sample geolocations" in block_type:
        score += 100

    srs = descriptor.get("srs")
    if isinstance(srs, dict):
        srs_type = str(srs.get("type", "")).strip().lower()
        srs_value = srs.get("value")
        if "epsg" in srs_type:
            score += 40
            if str(srs_value) == "4326":
                score += 20
        elif "mercator" in srs_type:
            score += 30
        elif "local" in srs_type:
            score += 10
    size = _safe_int(descriptor.get("byteSize", 0), "byteSize")
    return score, size


def _select_geolocation_descriptor(descriptors: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = []
    for descriptor in descriptors:
        block_type = str(descriptor.get("type", "")).strip().lower()
        if "sample geolocations" in block_type:
            candidates.append(descriptor)
    if not candidates:
        return None
    return max(candidates, key=_descriptor_score)


def _decode_open_gpr_radar_volume_candidates(
    block: bytes,
    samples_count: int,
    channels_count: int,
    slices_count: int,
    value_type: str,
) -> list[tuple[str, np.ndarray]]:
    dtype = _dtype_for_value_type(value_type)
    if len(block) % dtype.itemsize != 0:
        raise ValueError(
            f"Data block OpenGPR non allineato al valueType {value_type} "
            f"(itemsize={dtype.itemsize})."
        )

    values = np.frombuffer(block, dtype=dtype)
    expected = samples_count * channels_count * slices_count
    if values.size < expected:
        raise ValueError(
            "Data block OpenGPR incompleto: "
            f"attesi {expected} campioni, trovati {values.size}."
        )
    if values.size > expected:
        values = values[:expected]

    dim_sizes = {
        "slices": slices_count,
        "channels": channels_count,
        "samples": samples_count,
    }
    target_order = ("slices", "channels", "samples")

    candidates: list[tuple[str, np.ndarray]] = []
    seen: set[tuple[str, tuple[int, ...], tuple[int, ...]]] = set()
    for perm in itertools.permutations(target_order):
        shape = tuple(dim_sizes[name] for name in perm)
        reshaped = values.reshape(shape)
        axes = tuple(perm.index(name) for name in target_order)
        volume = np.transpose(reshaped, axes)
        key = (",".join(perm), volume.shape, volume.strides)
        if key in seen:
            continue
        seen.add(key)
        candidates.append((f"layout:{','.join(perm)}", volume))

    if not candidates:
        raise ValueError("Nessun layout candidato per Radar Volume OpenGPR.")
    return candidates


def _decode_coordinate_pair(raw: bytes, offset: int, coord_dims: int, coord_block_bytes: int) -> np.ndarray:
    if offset + 2 * coord_block_bytes > len(raw):
        raise ValueError("Coordinate OpenGPR oltre i limiti del data block.")
    start = np.frombuffer(raw, dtype="<f8", count=coord_dims, offset=offset)
    end = np.frombuffer(raw, dtype="<f8", count=coord_dims, offset=offset + coord_block_bytes)
    return (start + end) * 0.5


def _decode_open_gpr_geolocations(
    block: bytes,
    slices_count: int,
    channels_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if slices_count <= 0 or channels_count <= 0:
        raise ValueError("Dimensioni OpenGPR non valide per geolocations.")
    if len(block) == 0:
        raise ValueError("Data block geolocations OpenGPR vuoto.")

    if len(block) % slices_count != 0:
        raise ValueError("Data block geolocations non divisibile per slicesCount.")
    bytes_per_slice = len(block) // slices_count
    if bytes_per_slice <= 8:
        raise ValueError("Slice block geolocations troppo corto.")

    payload_per_slice = bytes_per_slice - 8
    per_channel = payload_per_slice % channels_count == 0
    if per_channel:
        bytes_per_sweep = payload_per_slice // channels_count
    else:
        bytes_per_sweep = payload_per_slice

    if bytes_per_sweep <= 0 or bytes_per_sweep % 16 != 0:
        raise ValueError("Sweep block geolocations non valido.")
    coord_block_bytes = bytes_per_sweep // 2
    if coord_block_bytes % 8 != 0:
        raise ValueError("Coord block geolocations non allineato su float64.")
    coord_dims = coord_block_bytes // 8
    if coord_dims < 2:
        raise ValueError("Geolocations OpenGPR senza almeno 2 coordinate (x,y).")

    x = np.full((slices_count, channels_count), np.nan, dtype=np.float64)
    y = np.full((slices_count, channels_count), np.nan, dtype=np.float64)
    z = np.full((slices_count, channels_count), np.nan, dtype=np.float64)

    for slice_idx in range(slices_count):
        slice_offset = slice_idx * bytes_per_slice
        _slice_id = np.frombuffer(block, dtype="<i8", count=1, offset=slice_offset)
        cursor = slice_offset + 8

        if per_channel:
            for channel_idx in range(channels_count):
                midpoint = _decode_coordinate_pair(block, cursor, coord_dims, coord_block_bytes)
                x[slice_idx, channel_idx] = midpoint[0]
                y[slice_idx, channel_idx] = midpoint[1]
                if coord_dims >= 3:
                    z[slice_idx, channel_idx] = midpoint[2]
                cursor += bytes_per_sweep
        else:
            midpoint = _decode_coordinate_pair(block, cursor, coord_dims, coord_block_bytes)
            x[slice_idx, :] = midpoint[0]
            y[slice_idx, :] = midpoint[1]
            if coord_dims >= 3:
                z[slice_idx, :] = midpoint[2]

    return x, y, z


def _parse_open_gpr_header(raw: bytes) -> tuple[dict[str, Any], int]:
    pos = 0
    magic, pos = _consume_line(raw, pos)
    if magic.strip().lstrip("\ufeff").lower() != "ogpr":
        raise ValueError("Magic OpenGPR non trovato.")

    _md5, pos = _consume_line(raw, pos)
    json_size_line, pos = _consume_line(raw, pos)
    json_size = _safe_int(json_size_line.strip(), "jsonHeaderSize")
    if pos + json_size > len(raw):
        raise ValueError("Header JSON OpenGPR oltre la dimensione del file.")

    header_raw = raw[pos:pos + json_size]
    pos += json_size

    try:
        header = json.loads(header_raw.decode("utf-8"))
    except Exception as exc:
        raise ValueError("Header JSON OpenGPR non valido.") from exc

    if not isinstance(header, dict):
        raise ValueError("Header OpenGPR non valido: atteso oggetto JSON.")
    return header, pos


def _build_profiles_from_volume(
    file_name: str,
    volume: np.ndarray,
    x: np.ndarray | None,
    y: np.ndarray | None,
    z: np.ndarray | None,
    srs: dict[str, Any] | None,
    metadata: dict[str, Any],
) -> list[OgprProfile]:
    if volume.ndim != 3:
        raise ValueError("Volume radar atteso 3D: (slices, channels, samples).")

    slices_count, channels_count, _samples_count = volume.shape
    profiles: list[OgprProfile] = []
    for channel_idx in range(channels_count):
        profile_x = None if x is None else np.asarray(x[:, channel_idx], dtype=np.float64)
        profile_y = None if y is None else np.asarray(y[:, channel_idx], dtype=np.float64)
        profile_z = None if z is None else np.asarray(z[:, channel_idx], dtype=np.float64)
        profile_data = np.asarray(volume[:, channel_idx, :], dtype=np.float64)
        if profile_data.shape[0] != slices_count:
            raise ValueError("Dimensioni profilo incoerenti con slicesCount.")
        profile_meta = dict(metadata)
        profile_meta["channel_index"] = channel_idx
        profile_meta["channel_std"] = float(np.nanstd(profile_data))
        profile_meta["channel_min"] = float(np.nanmin(profile_data))
        profile_meta["channel_max"] = float(np.nanmax(profile_data))
        profiles.append(
            OgprProfile(
                file_name=file_name,
                channel_index=channel_idx,
                data=profile_data,
                x=profile_x,
                y=profile_y,
                z=profile_z,
                srs=srs,
                metadata=profile_meta,
            )
        )
    return profiles


def _load_ogpr_open_gpr(
    raw: bytes,
    file_name: str,
    offset_mode_override: str | None = None,
    layout_mode_override: str | None = None,
) -> list[OgprProfile]:
    header, data_start = _parse_open_gpr_header(raw)

    main_descriptor = header.get("mainDescriptor")
    if not isinstance(main_descriptor, dict):
        raise ValueError("mainDescriptor mancante in OpenGPR.")

    samples_count = _safe_int(main_descriptor.get("samplesCount"), "samplesCount")
    channels_count = _safe_int(main_descriptor.get("channelsCount", 1), "channelsCount")
    slices_count = _safe_int(main_descriptor.get("slicesCount", 1), "slicesCount")
    if samples_count == 0 or channels_count == 0 or slices_count == 0:
        raise ValueError("Dimensioni OpenGPR non valide (valore zero).")

    raw_descriptors = header.get("dataBlockDescriptors")
    if not isinstance(raw_descriptors, list) or not raw_descriptors:
        raise ValueError("dataBlockDescriptors mancanti in OpenGPR.")
    descriptors = [item for item in raw_descriptors if isinstance(item, dict)]
    if not descriptors:
        raise ValueError("dataBlockDescriptors non validi in OpenGPR.")
    offset_mode_hint = _infer_offset_mode(descriptors, data_start=data_start, raw_len=len(raw))
    preferred_mode = offset_mode_hint
    if offset_mode_override in {"absolute", "relative"}:
        preferred_mode = offset_mode_override

    radar_descriptor = _select_radar_descriptor(descriptors)
    radar_meta = radar_descriptor.get("radar")
    radar_value_type = radar_meta.get("valueType") if isinstance(radar_meta, dict) else None
    value_type = _normalize_value_type(
        radar_descriptor.get(
            "valueType",
            main_descriptor.get("valueType", radar_value_type if radar_value_type is not None else "int16"),
        )
    )

    volume_candidates: list[tuple[str, str, np.ndarray, float]] = []
    for mode, radar_block in _extract_block_candidates(
        raw,
        radar_descriptor,
        data_start=data_start,
        preferred_mode=preferred_mode,
    ):
        try:
            for layout_name, candidate_volume in _decode_open_gpr_radar_volume_candidates(
                radar_block,
                samples_count=samples_count,
                channels_count=channels_count,
                slices_count=slices_count,
                value_type=value_type,
            ):
                if layout_mode_override and layout_mode_override != "auto" and layout_name != layout_mode_override:
                    continue
                score = _volume_quality_score(candidate_volume)
                volume_candidates.append((mode, layout_name, candidate_volume, score))
        except Exception:
            continue
    if not volume_candidates and layout_mode_override and layout_mode_override != "auto":
        for mode, radar_block in _extract_block_candidates(
            raw,
            radar_descriptor,
            data_start=data_start,
            preferred_mode=preferred_mode,
        ):
            try:
                for layout_name, candidate_volume in _decode_open_gpr_radar_volume_candidates(
                    radar_block,
                    samples_count=samples_count,
                    channels_count=channels_count,
                    slices_count=slices_count,
                    value_type=value_type,
                ):
                    score = _volume_quality_score(candidate_volume)
                    volume_candidates.append((mode, layout_name, candidate_volume, score))
            except Exception:
                continue
    if not volume_candidates:
        raise ValueError("Impossibile decodificare il Radar Volume OpenGPR.")
    candidate_pool = volume_candidates
    if offset_mode_override in {"absolute", "relative"}:
        forced_mode_candidates = [item for item in candidate_pool if item[0] == offset_mode_override]
        if forced_mode_candidates:
            candidate_pool = forced_mode_candidates
        volume_mode, volume_layout, volume, volume_score = max(candidate_pool, key=lambda item: item[3])
    else:
        best_by_mode: dict[str, tuple[str, str, np.ndarray, float]] = {}
        for candidate in candidate_pool:
            mode = candidate[0]
            current = best_by_mode.get(mode)
            if current is None or candidate[3] > current[3]:
                best_by_mode[mode] = candidate

        global_best = max(candidate_pool, key=lambda item: item[3])
        hint_best = best_by_mode.get(offset_mode_hint)
        if hint_best is not None:
            hint_arr = np.asarray(hint_best[2], dtype=np.float64)
            hint_std = float(np.nanstd(hint_arr))
            hint_dynamic = float(np.nanmax(hint_arr) - np.nanmin(hint_arr))
            hint_is_flat = hint_std < 1.0e-6 or hint_dynamic < 1.0e-6
            if hint_is_flat:
                volume_mode, volume_layout, volume, volume_score = global_best
            else:
                volume_mode, volume_layout, volume, volume_score = hint_best
        else:
            volume_mode, volume_layout, volume, volume_score = global_best
    volume = np.asarray(volume, dtype=np.float64)

    geolocation_descriptor = _select_geolocation_descriptor(descriptors)
    x = y = z = None
    srs = None
    geo_mode: str | None = None
    if geolocation_descriptor is not None:
        best_geo: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        best_geo_score = -1.0e12
        geo_candidates: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, float]] = []
        for mode, geolocation_block in _extract_block_candidates(
            raw,
            geolocation_descriptor,
            data_start=data_start,
            preferred_mode=preferred_mode,
        ):
            try:
                gx, gy, gz = _decode_open_gpr_geolocations(
                    geolocation_block,
                    slices_count=slices_count,
                    channels_count=channels_count,
                )
                score = _geolocation_quality_score(gx, gy)
                geo_candidates.append((mode, gx, gy, gz, score))
            except Exception:
                continue
        if geo_candidates:
            geo_pool = geo_candidates
            if offset_mode_override in {"absolute", "relative"}:
                forced_geo = [item for item in geo_pool if item[0] == offset_mode_override]
                if forced_geo:
                    geo_pool = forced_geo
                geo_mode, gx, gy, gz, best_geo_score = max(geo_pool, key=lambda item: item[4])
                best_geo = (gx, gy, gz)
            else:
                best_by_geo_mode: dict[str, tuple[str, np.ndarray, np.ndarray, np.ndarray, float]] = {}
                for candidate in geo_pool:
                    mode = candidate[0]
                    current = best_by_geo_mode.get(mode)
                    if current is None or candidate[4] > current[4]:
                        best_by_geo_mode[mode] = candidate
                global_geo = max(geo_pool, key=lambda item: item[4])
                hint_geo = best_by_geo_mode.get(offset_mode_hint)
                if hint_geo is not None:
                    finite_ratio = float(np.mean(np.isfinite(hint_geo[1]) & np.isfinite(hint_geo[2])))
                    if finite_ratio < 0.5:
                        geo_mode, gx, gy, gz, best_geo_score = global_geo
                    else:
                        geo_mode, gx, gy, gz, best_geo_score = hint_geo
                else:
                    geo_mode, gx, gy, gz, best_geo_score = global_geo
                best_geo = (gx, gy, gz)
        if best_geo is not None:
            x, y, z = best_geo
            srs = geolocation_descriptor.get("srs") if isinstance(geolocation_descriptor.get("srs"), dict) else None

    radar_candidate_scores = sorted(
        [
            {
                "offset_mode": item[0],
                "layout_mode": item[1],
                "score": float(item[3]),
            }
            for item in volume_candidates
        ],
        key=lambda row: row["score"],
        reverse=True,
    )
    geo_candidate_scores = []
    if geolocation_descriptor is not None:
        geo_candidate_scores = sorted(
            [
                {
                    "offset_mode": item[0],
                    "score": float(item[4]),
                }
                for item in geo_candidates
            ],
            key=lambda row: row["score"],
            reverse=True,
        )

    metadata = {
        "samples_count": samples_count,
        "channels_count": channels_count,
        "slices_count": slices_count,
        "value_type": value_type,
        "radar_block_name": str(radar_descriptor.get("name", "Radar Volume")),
        "offset_mode_hint": offset_mode_hint,
        "offset_mode_override": offset_mode_override if offset_mode_override else "auto",
        "layout_mode_override": layout_mode_override if layout_mode_override else "auto",
        "radar_offset_mode": volume_mode,
        "radar_layout_mode": volume_layout,
        "radar_candidate_score": float(volume_score),
        "radar_candidate_scores_top": radar_candidate_scores[:8],
        "radar_std": float(np.nanstd(volume)),
        "radar_min": float(np.nanmin(volume)),
        "radar_max": float(np.nanmax(volume)),
        "geo_offset_mode": geo_mode,
        "geo_candidate_scores_top": geo_candidate_scores[:8],
        "main_descriptor": _sanitize_metadata_value(main_descriptor),
        "radar_descriptor": _sanitize_metadata_value(radar_descriptor),
    }
    if isinstance(radar_meta, dict):
        metadata["radar_parameters"] = radar_meta
    if isinstance(geolocation_descriptor, dict):
        metadata["geolocation_descriptor"] = _sanitize_metadata_value(geolocation_descriptor)
    for key in ("radar", "acquisition", "instrument", "survey"):
        section = header.get(key)
        if isinstance(section, dict):
            metadata[f"header_{key}"] = _sanitize_metadata_value(section)
    return _build_profiles_from_volume(
        file_name=file_name,
        volume=volume,
        x=x,
        y=y,
        z=z,
        srs=srs,
        metadata=metadata,
    )


def _profiles_from_generic_array(array: np.ndarray, file_name: str) -> list[OgprProfile]:
    arr = np.asarray(array, dtype=np.float64)
    if arr.ndim == 2:
        # Heuristic: usually samples are fewer than traces, keep samples on axis 1.
        data = arr if arr.shape[1] <= arr.shape[0] else arr.T
        return [
            OgprProfile(
                file_name=file_name,
                channel_index=0,
                data=np.asarray(data, dtype=np.float64),
                x=None,
                y=None,
                z=None,
                srs=None,
                metadata={"source": "generic-2d"},
            )
        ]

    if arr.ndim == 3:
        # Heuristic layout target: (slices, channels, samples).
        sample_axis = int(np.argmax(arr.shape))
        normalized = np.moveaxis(arr, sample_axis, -1)
        slices_count, channels_count, samples_count = normalized.shape
        if channels_count > slices_count and slices_count <= 32:
            normalized = np.swapaxes(normalized, 0, 1)
            slices_count, channels_count, samples_count = normalized.shape
        metadata = {
            "source": "generic-3d",
            "samples_count": samples_count,
            "channels_count": channels_count,
            "slices_count": slices_count,
        }
        return _build_profiles_from_volume(
            file_name=file_name,
            volume=np.asarray(normalized, dtype=np.float64),
            x=None,
            y=None,
            z=None,
            srs=None,
            metadata=metadata,
        )

    raise ValueError(f"Array non supportato: atteso 2D/3D, trovato {arr.ndim}D.")


def load_ogpr_profiles(
    stream: BytesIO,
    file_name: str,
    offset_mode_override: str | None = None,
    layout_mode_override: str | None = None,
) -> list[OgprProfile]:
    raw = stream.getvalue()
    if raw.startswith(HDF5_SIGNATURE):
        return _profiles_from_generic_array(_load_from_hdf5(BytesIO(raw)), file_name=file_name)

    try:
        return _load_ogpr_open_gpr(
            raw,
            file_name=file_name,
            offset_mode_override=offset_mode_override,
            layout_mode_override=layout_mode_override,
        )
    except Exception:
        try:
            return _profiles_from_generic_array(_load_from_text(BytesIO(raw)), file_name=file_name)
        except Exception as exc:
            raise ValueError(
                "Formato .ogpr non riconosciuto. Supportati OpenGPR nativo e varianti HDF5/tabellari."
            ) from exc


def load_radar_array(stream: BytesIO, suffix: str) -> np.ndarray:
    suffix = suffix.lower()

    if suffix == ".npy":
        stream.seek(0)
        return np.asarray(np.load(stream, allow_pickle=False), dtype=np.float64)
    if suffix == ".npz":
        return _load_from_npz(stream)
    if suffix in {".csv", ".txt"}:
        return _load_from_text(stream)
    if suffix in {".mat"}:
        return _load_from_mat(stream)
    if suffix in {".h5", ".hdf5"}:
        return _load_from_hdf5(stream)
    if suffix == ".ogpr":
        profiles = load_ogpr_profiles(stream, file_name="dataset.ogpr")
        if len(profiles) == 1:
            return np.asarray(profiles[0].data, dtype=np.float64)
        stacks = [profile.data for profile in profiles]
        return np.stack(stacks, axis=0)

    raise ValueError(f"Estensione non supportata: {suffix}")
