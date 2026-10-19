"""Efficient HDF5 storage and queries for py4dgeo 4D-OBC analyses.

This module converts one ``py4dgeo.SpatiotemporalAnalysis`` ZIP into a portable,
netCDF4-compatible HDF5 file. The output remains directly readable with xarray
through the ``h5netcdf`` engine.

The converter is designed for long-running jobs:
- dense arrays are written in point blocks instead of materializing xarray datasets;
- progress is checkpointed in ``<name>.partial.h5``;
- interrupted conversions can resume safely;
- OBC objects are loaded once;
- optional/derived data are omitted by default to reduce file size.

Default storage keeps:
- raw M3C2 distances;
- smoothed distances when present;
- only ``lodetection`` from the structured uncertainty array;
- compact OBC metadata, object -> point membership, and spatial bounds.

Dependencies
------------
Reading/querying: numpy, pandas, xarray, h5netcdf
Polygon queries: shapely >= 2
Conversion/validation: py4dgeo, h5netcdf

Authors
------------
- Jiapan Wang
- GPT-5.6 Sol
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence
import argparse
import hashlib
import json
import logging
import re
import secrets
import time

import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)

SCHEMA_NAME = "4D-OBC HDF5"
SCHEMA_VERSION = "0.1.0"
_CROCKFORD_BASE32 = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
_ULID_RE = re.compile(r"^[0-7][0-9A-HJKMNP-TV-Z]{25}$")


def _encode_ulid(value: int) -> str:
    """Encode a 128-bit integer as a 26-character Crockford Base32 ULID."""
    if not 0 <= value < (1 << 128):
        raise ValueError("ULID value must fit in 128 bits.")
    chars = ["0"] * 26
    for i in range(25, -1, -1):
        chars[i] = _CROCKFORD_BASE32[value & 31]
        value >>= 5
    return "".join(chars)


def new_ulid(timestamp_ms: int | None = None) -> str:
    """Create a new globally unique ULID.

    Args:
        timestamp_ms: Optional Unix timestamp in milliseconds. Current time is
            used when omitted.

    Returns:
        Canonical 26-character ULID string.
    """
    timestamp_ms = time.time_ns() // 1_000_000 if timestamp_ms is None else int(timestamp_ms)
    if not 0 <= timestamp_ms < (1 << 48):
        raise ValueError("timestamp_ms must fit in 48 bits.")
    return _encode_ulid((timestamp_ms << 80) | secrets.randbits(80))


def is_ulid(value: str) -> bool:
    """Return whether ``value`` is a canonical ULID string."""
    return bool(_ULID_RE.fullmatch(str(value)))


def _new_ulids(count: int) -> list[str]:
    """Generate ``count`` unique ULIDs using one shared timestamp."""
    if count < 0:
        raise ValueError("count must be non-negative.")
    timestamp_ms = time.time_ns() // 1_000_000
    result: list[str] = []
    seen: set[str] = set()
    while len(result) < count:
        value = new_ulid(timestamp_ms)
        if value not in seen:
            result.append(value)
            seen.add(value)
    return result


@dataclass(frozen=True)
class ConversionConfig:
    """Storage options for ``convert_py4dgeo_zip``.

    Args:
        dataset_id: Stable dataset/site identifier used in metadata and display IDs.
        crs: Optional CRS description, for example ``"EPSG:25832"``.
        distance_units: Units used by distance-like arrays.
        point_chunk: HDF5 chunk size along the point axis.
        time_chunk: HDF5 chunk size along time. ``None`` uses the full time axis.
        write_block_points: Number of points copied per write operation. This may
            span multiple HDF5 chunks and mainly controls conversion throughput.
        compression_level: gzip compression level. Level 1 is the default because
            it is usually much faster than level 4 for a modest size penalty.
        include_smoothed_distances: Store py4dgeo smoothed distances when present.
        uncertainty_fields: Structured py4dgeo uncertainty fields to store.
            The default stores only ``lodetection``. Use ``None`` for all fields or
            ``()`` for none.
        include_obc_distances: Store per-member 4D-OBC region-growing/DTW distances.
            Disabled by default because the source ZIP already preserves them and
            they are not needed for most ML/labeling queries.
        include_reverse_index: Store an additional point -> OBC membership index.
            Disabled by default because it duplicates membership information.
        checkpoint_every_obcs: Flush progress after this many OBCs.
    """

    dataset_id: str
    crs: str | None = None
    distance_units: str = "m"
    point_chunk: int = 512
    time_chunk: int | None = None
    write_block_points: int = 8192
    compression_level: int = 1
    include_smoothed_distances: bool = True
    uncertainty_fields: tuple[str, ...] | None = ("lodetection",)
    include_obc_distances: bool = False
    include_reverse_index: bool = False
    checkpoint_every_obcs: int = 25


def sha256_file(path: str | Path, block_size: int = 8 * 1024 * 1024) -> str:
    """Return the SHA-256 digest of a file.

    Args:
        path: File to hash.
        block_size: Read size in bytes.

    Returns:
        Lower-case hexadecimal SHA-256 digest.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        while block := f.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _check_dataset_id(dataset_id: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9._-]+", dataset_id):
        raise ValueError("dataset_id may contain only letters, digits, '.', '_' and '-'.")


def _py4dgeo_version(py4dgeo: Any) -> str:
    return str(getattr(py4dgeo, "__version__", "unknown"))


def _timestamps_from_analysis(analysis: Any) -> np.ndarray:
    """Return py4dgeo timestamps as ``datetime64[ns]`` values."""
    reference = analysis.reference_epoch.timestamp
    return np.asarray([reference + dt for dt in analysis.timedeltas], dtype="datetime64[ns]")


def _partial_path(output_h5: Path) -> Path:
    return output_h5.with_name(f"{output_h5.stem}.partial{output_h5.suffix}")


def _config_json(config: ConversionConfig) -> str:
    payload = asdict(config)
    if payload["uncertainty_fields"] is not None:
        payload["uncertainty_fields"] = list(payload["uncertainty_fields"])
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _text(value: Any) -> str:
    """Decode HDF5 byte strings while leaving normal strings unchanged."""
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8")
    return str(value)


def _storage_kwargs(config: ConversionConfig, chunks: tuple[int, ...]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"chunks": chunks, "shuffle": True}
    if config.compression_level > 0:
        kwargs.update(compression="gzip", compression_opts=int(config.compression_level))
    return kwargs


def _ensure_dimension(group: Any, name: str, size: int) -> None:
    if name in group.dimensions:
        existing = int(group.dimensions[name].size)
        if existing != int(size):
            raise ValueError(f"Dimension {name!r} has size {existing}, expected {size}.")
    else:
        group.dimensions[name] = int(size)


def _normalize_hdf5_dtype(dtype):
    """Convert Python/NumPy Unicode strings to HDF5 UTF-8 strings."""
    if dtype is str:
        import h5py
        return h5py.string_dtype(encoding="utf-8")

    try:
        np_dtype = np.dtype(dtype)
    except (TypeError, ValueError):
        return dtype

    if np_dtype.kind == "U":
        import h5py
        return h5py.string_dtype(encoding="utf-8")

    return dtype

def _ensure_variable(
    group: Any,
    name: str,
    dimensions: tuple[str, ...],
    dtype: Any,
    *,
    chunks: tuple[int, ...] | None = None,
    config: ConversionConfig | None = None,
    compressed: bool = True,
    fillvalue: Any = None,
) -> Any:
    if name in group.variables:
        return group.variables[name]
    kwargs: dict[str, Any] = {}
    if chunks is not None:
        if compressed and config is not None:
            kwargs.update(_storage_kwargs(config, chunks))
        else:
            kwargs["chunks"] = chunks
    if fillvalue is not None:
        kwargs["fillvalue"] = fillvalue
    return group.create_variable(
        name, 
        dimensions, 
        dtype=_normalize_hdf5_dtype(dtype), 
        **kwargs
        )


def _point_index_dtype(n_point: int) -> np.dtype:
    return np.dtype(np.int32 if n_point <= np.iinfo(np.int32).max else np.int64)


def _obc_mapping(obc: Any) -> Any | None:
    """Return py4dgeo's internal point->distance mapping when available."""
    data = getattr(obc, "_data", None)
    mapping = getattr(data, "indices_distances", None)
    return mapping if mapping is not None else None


def _obc_indices(obc: Any, dtype: np.dtype) -> np.ndarray:
    """Extract member point indices with a fast path for current py4dgeo."""
    mapping = _obc_mapping(obc)
    if mapping is not None:
        try:
            return np.fromiter(mapping.keys(), dtype=dtype, count=len(mapping))
        except (AttributeError, TypeError):
            pass
    return np.asarray(obc.indices, dtype=dtype)


def _obc_indices_distances(obc: Any, dtype: np.dtype) -> tuple[np.ndarray, np.ndarray]:
    """Extract member indices and stored region-growing distances efficiently."""
    mapping = _obc_mapping(obc)
    if mapping is not None:
        try:
            n = len(mapping)
            indices = np.fromiter(mapping.keys(), dtype=dtype, count=n)
            distances = np.fromiter(mapping.values(), dtype=np.float64, count=n)
            return indices, distances
        except (AttributeError, TypeError):
            pass
    indices = np.asarray(obc.indices, dtype=dtype)
    distances = np.fromiter(
        (obc.distance(int(point)) for point in indices),
        dtype=np.float64,
        count=len(indices),
    )
    return indices, distances


def _load_reusable_object_ids(path: str | Path) -> list[str]:
    """Load OBC ULIDs from an older HDF5 file for schema migration/re-export."""
    with open_obcs(path) as ds:
        values = [str(v) for v in ds["object_id"].values]
    if len(values) != len(set(values)) or any(not is_ulid(v) for v in values):
        raise ValueError("Reusable object IDs must be unique canonical ULIDs.")
    return values


def _open_h5netcdf(path: Path, mode: str):
    try:
        import h5netcdf
    except ImportError as exc:
        raise ImportError("Conversion requires h5netcdf. Install with: pip install h5netcdf") from exc
    return h5netcdf.File(path, mode=mode, format="NETCDF4")


def _initialize_root(
    f: Any,
    *,
    source_zip: Path,
    source_sha256: str,
    source_size: int,
    config: ConversionConfig,
    py4dgeo_version: str,
) -> None:
    attrs = f.attrs
    if "schema_version" in attrs:
        expected = {
            "schema_name": SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "dataset_id": config.dataset_id,
            "source_py4dgeo_sha256": source_sha256,
            "conversion_config_json": _config_json(config),
        }
        for key, value in expected.items():
            if _text(attrs.get(key)) != _text(value):
                raise ValueError(f"Partial file does not match current conversion: {key} differs.")
        return

    attrs["schema_name"] = SCHEMA_NAME
    attrs["schema_version"] = SCHEMA_VERSION
    attrs["dataset_id"] = config.dataset_id
    attrs["source_py4dgeo_archive"] = source_zip.name
    attrs["source_py4dgeo_sha256"] = source_sha256
    attrs["source_py4dgeo_size_bytes"] = int(source_size)
    attrs["py4dgeo_version"] = py4dgeo_version
    attrs["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    attrs["conversion_state"] = "in_progress"
    attrs["conversion_config_json"] = _config_json(config)
    attrs["object_id_scheme"] = "ULID"
    if config.crs is not None:
        attrs["crs"] = config.crs


def _initialize_analysis_group(
    f: Any,
    analysis: Any,
    config: ConversionConfig,
) -> tuple[Any, np.ndarray, np.ndarray]:
    cp = np.asarray(analysis.corepoints.cloud)
    timestamps = _timestamps_from_analysis(analysis)
    if cp.ndim != 2 or cp.shape[1] != 3:
        raise ValueError(f"Unexpected core-point shape: {cp.shape}")

    group = f.groups["analysis"] if "analysis" in f.groups else f.create_group("analysis")
    n_point = int(cp.shape[0])
    n_time = int(len(timestamps))
    _ensure_dimension(group, "point", n_point)
    _ensure_dimension(group, "time", n_time)

    point_chunk = max(1, min(config.point_chunk, n_point))
    point = _ensure_variable(group, "point", ("point",), np.int64, chunks=(point_chunk,), config=config)
    x = _ensure_variable(group, "x", ("point",), cp.dtype, chunks=(point_chunk,), config=config)
    y = _ensure_variable(group, "y", ("point",), cp.dtype, chunks=(point_chunk,), config=config)
    z = _ensure_variable(group, "z", ("point",), cp.dtype, chunks=(point_chunk,), config=config)
    t = _ensure_variable(group, "time", ("time",), np.int64, compressed=False)

    if str(group.attrs.get("coordinates_complete", "false")) != "true":
        point[:] = np.arange(n_point, dtype=np.int64)
        x[:] = cp[:, 0]
        y[:] = cp[:, 1]
        z[:] = cp[:, 2]
        t[:] = timestamps.astype("datetime64[ns]").astype(np.int64)
        t.attrs["units"] = "nanoseconds since 1970-01-01 00:00:00"
        t.attrs["calendar"] = "proleptic_gregorian"
        x.attrs["axis"] = "X"
        y.attrs["axis"] = "Y"
        z.attrs["axis"] = "Z"
        group.attrs["coordinates_complete"] = "true"

    group.attrs["schema_name"] = f"{SCHEMA_NAME} analysis"
    group.attrs["schema_version"] = SCHEMA_VERSION
    group.attrs["dataset_id"] = config.dataset_id
    group.attrs["point_index_semantics"] = "Original py4dgeo core-point index"
    group.attrs["time_index_semantics"] = "Original py4dgeo time index"
    group.attrs["obc_end_epoch_semantics"] = "Inclusive"
    if config.crs is not None:
        group.attrs["crs"] = config.crs
    return group, cp, timestamps


def _write_array_blocked(
    f: Any,
    group: Any,
    *,
    name: str,
    source: np.ndarray,
    config: ConversionConfig,
    attrs: dict[str, Any] | None = None,
) -> None:
    """Write one ``(point, time)`` array with resumable point-block checkpoints."""
    if source.ndim != 2:
        raise ValueError(f"{name}: expected a 2-D point x time array, got {source.shape}.")
    n_point, n_time = map(int, source.shape)
    point_chunk = max(1, min(config.point_chunk, n_point))
    time_chunk = n_time if config.time_chunk is None else max(1, min(config.time_chunk, n_time))
    var = _ensure_variable(
        group,
        name,
        ("point", "time"),
        source.dtype,
        chunks=(point_chunk, time_chunk),
        config=config,
        fillvalue=None,
    )
    if attrs:
        for key, value in attrs.items():
            var.attrs[key] = value
    if str(var.attrs.get("conversion_complete", "false")) == "true":
        return

    start = int(var.attrs.get("points_written", 0))
    block = max(point_chunk, int(config.write_block_points))
    logger.info("Writing %s from point %d/%d", name, start, n_point)
    started = time.perf_counter()
    for p0 in range(start, n_point, block):
        p1 = min(p0 + block, n_point)
        var[p0:p1, :] = source[p0:p1, :]
        var.attrs["points_written"] = int(p1)
        f.flush()
    var.attrs["conversion_complete"] = "true"
    logger.info("Finished %s in %.1f s", name, time.perf_counter() - started)
    f.flush()


def _write_analysis_group(f: Any, analysis: Any, config: ConversionConfig) -> None:
    group, cp, timestamps = _initialize_analysis_group(f, analysis, config)
    if str(group.attrs.get("conversion_complete", "false")) == "true":
        return

    n_point, n_time = cp.shape[0], len(timestamps)

    raw_var = group.variables.get("m3c2_distance")
    if raw_var is None or str(raw_var.attrs.get("conversion_complete", "false")) != "true":
        raw = np.asarray(analysis.distances)
        if raw.shape != (n_point, n_time):
            raise ValueError(f"Distance shape {raw.shape} does not match ({n_point}, {n_time}).")
        _write_array_blocked(
            f, group, name="m3c2_distance", source=raw, config=config,
            attrs={"long_name": "Raw M3C2 distance", "units": config.distance_units},
        )

    distance_for_compute = "m3c2_distance"
    if config.include_smoothed_distances:
        smooth_var = group.variables.get("smoothed_distance")
        smooth_complete = smooth_var is not None and str(smooth_var.attrs.get("conversion_complete", "false")) == "true"
        smooth_absent = str(group.attrs.get("smoothed_distance_available", "unknown")) == "false"
        if smooth_complete:
            distance_for_compute = "smoothed_distance"
        elif not smooth_absent:
            smoothed = analysis.smoothed_distances
            if smoothed is None:
                group.attrs["smoothed_distance_available"] = "false"
            else:
                smoothed = np.asarray(smoothed)
                if smoothed.shape != (n_point, n_time):
                    raise ValueError("Smoothed distance shape differs from point x time dimensions.")
                _write_array_blocked(
                    f, group, name="smoothed_distance", source=smoothed, config=config,
                    attrs={
                        "long_name": "Temporally smoothed M3C2 distance",
                        "units": config.distance_units,
                    },
                )
                group.attrs["smoothed_distance_available"] = "true"
                distance_for_compute = "smoothed_distance"
    group.attrs["distance_for_compute_variable"] = distance_for_compute

    fields = config.uncertainty_fields
    if fields != () and str(group.attrs.get("uncertainties_complete", "false")) != "true":
        uncertainty = np.asarray(analysis.uncertainties)
        if uncertainty.shape == (n_point, n_time) and uncertainty.dtype.names:
            available = tuple(uncertainty.dtype.names)
            selected = available if fields is None else tuple(fields)
            unknown = [field for field in selected if field not in available]
            if unknown:
                raise ValueError(f"Unknown uncertainty field(s): {unknown}; available={available}")
            for field in selected:
                var_name = f"uncertainty_{field}"
                existing = group.variables.get(var_name)
                if existing is not None and str(existing.attrs.get("conversion_complete", "false")) == "true":
                    continue
                values = uncertainty[field]
                attrs: dict[str, Any] = {"long_name": f"M3C2 uncertainty field: {field}"}
                if field in {"lodetection", "spread1", "spread2"}:
                    attrs["units"] = config.distance_units
                _write_array_blocked(
                    f, group, name=var_name, source=values, config=config, attrs=attrs
                )
            group.attrs["uncertainties_complete"] = "true"
        elif uncertainty.ndim == 2 and uncertainty.shape[1] == 0:
            group.attrs["uncertainties_complete"] = "true"
            group.attrs["uncertainties_available"] = "false"
        else:
            raise ValueError(f"Unexpected uncertainty shape/dtype: {uncertainty.shape}, {uncertainty.dtype}")
    elif fields == ():
        group.attrs["uncertainties_complete"] = "true"
        group.attrs["uncertainties_stored"] = "none"

    group.attrs["conversion_complete"] = "true"
    f.flush()


def _prepare_object_ids(
    group: Any,
    *,
    n_object: int,
    dataset_id: str,
    reuse_object_ids_from: str | Path | None,
) -> tuple[list[str], list[str]]:
    if "object_id" in group.variables and "display_id" in group.variables:
        ids = [_text(v) for v in group.variables["object_id"][:]]
        displays = [_text(v) for v in group.variables["display_id"][:]]
        if len(ids) == n_object and len(set(ids)) == n_object and all(is_ulid(v) for v in ids):
            return ids, displays

    if reuse_object_ids_from is not None:
        ids = _load_reusable_object_ids(reuse_object_ids_from)
        if len(ids) != n_object:
            raise ValueError("Reused object ID count does not match source OBC count.")
    else:
        ids = _new_ulids(n_object)
    displays = [f"{dataset_id}-obc-{i:06d}" for i in range(n_object)]

    object_id = _ensure_variable(group, "object_id", ("object",), str)
    display_id = _ensure_variable(group, "display_id", ("object",), str)
    object_id[:] = np.asarray(ids, dtype=object)
    display_id[:] = np.asarray(displays, dtype=object)
    return ids, displays


def _object_sizes(objects: Sequence[Any]) -> np.ndarray:
    sizes = np.empty(len(objects), dtype=np.int64)
    for i, obc in enumerate(objects):
        mapping = _obc_mapping(obc)
        sizes[i] = len(mapping) if mapping is not None else len(obc.indices)
    return sizes


def _initialize_obcs_group(
    f: Any,
    *,
    objects: Sequence[Any],
    cp: np.ndarray,
    config: ConversionConfig,
    reuse_object_ids_from: str | Path | None,
) -> tuple[Any, np.ndarray, np.dtype]:
    group = f.groups["obcs"] if "obcs" in f.groups else f.create_group("obcs")
    n_object = len(objects)
    n_point = int(cp.shape[0])
    sizes = _object_sizes(objects)
    offsets = np.empty(n_object + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(sizes, out=offsets[1:])
    n_membership = int(offsets[-1])
    point_dtype = _point_index_dtype(n_point)

    _ensure_dimension(group, "object", n_object)
    _ensure_dimension(group, "object_offset", n_object + 1)
    _ensure_dimension(group, "membership", n_membership)

    object_chunk = max(1, min(4096, max(n_object, 1)))
    membership_chunk = max(1, min(1_000_000, max(n_membership, 1)))

    _prepare_object_ids(
        group,
        n_object=n_object,
        dataset_id=config.dataset_id,
        reuse_object_ids_from=reuse_object_ids_from,
    )

    n_points_var = _ensure_variable(group, "n_points", ("object",), np.int32, compressed=False)
    n_points_var[:] = sizes.astype(np.int32, copy=False)

    offsets_var = _ensure_variable(group, "object_point_offsets", ("object_offset",), np.int64, compressed=False)
    offsets_var[:] = offsets

    for name, dtype in (
        ("start_epoch", np.int32),
        ("end_epoch", np.int32),
        ("seed_point_index", point_dtype),
        ("seed_start_epoch", np.int32),
        ("seed_end_epoch", np.int32),
        ("threshold", np.float64),
        ("centroid_x", np.float64),
        ("centroid_y", np.float64),
        ("centroid_z", np.float64),
        ("xmin", np.float64),
        ("ymin", np.float64),
        ("zmin", np.float64),
        ("xmax", np.float64),
        ("ymax", np.float64),
        ("zmax", np.float64),
    ):
        _ensure_variable(group, name, ("object",), dtype, chunks=(object_chunk,), config=config)

    _ensure_variable(
        group,
        "object_point_indices",
        ("membership",),
        point_dtype,
        chunks=(membership_chunk,),
        config=config,
    )
    if config.include_obc_distances:
        _ensure_variable(
            group,
            "object_distance",
            ("membership",),
            np.float64,
            chunks=(membership_chunk,),
            config=config,
        )

    group.attrs["schema_name"] = f"{SCHEMA_NAME} OBC table"
    group.attrs["schema_version"] = SCHEMA_VERSION
    group.attrs["dataset_id"] = config.dataset_id
    group.attrs["membership_representation"] = "CSR-like object-to-point index"
    group.attrs["obc_end_epoch_semantics"] = "Inclusive"
    group.attrs["object_id_scheme"] = "ULID"
    group.attrs["display_id_semantics"] = "dataset-local, not a global identifier"
    group.attrs["object_distance_stored"] = "true" if config.include_obc_distances else "false"
    group.attrs["reverse_index_stored"] = "false"
    if "obcs_written" not in group.attrs:
        group.attrs["obcs_written"] = 0
    f.flush()
    return group, offsets, point_dtype


def _write_obcs_group(
    f: Any,
    *,
    objects: Sequence[Any],
    cp: np.ndarray,
    config: ConversionConfig,
    reuse_object_ids_from: str | Path | None,
) -> None:
    group, offsets, point_dtype = _initialize_obcs_group(
        f,
        objects=objects,
        cp=cp,
        config=config,
        reuse_object_ids_from=reuse_object_ids_from,
    )
    if str(group.attrs.get("conversion_complete", "false")) == "true":
        return

    start_object = int(group.attrs.get("obcs_written", 0))
    membership_var = group.variables["object_point_indices"]
    distance_var = group.variables.get("object_distance")

    scalar_vars = {name: group.variables[name] for name in (
        "start_epoch", "end_epoch", "seed_point_index", "seed_start_epoch", "seed_end_epoch",
        "threshold", "centroid_x", "centroid_y", "centroid_z",
        "xmin", "ymin", "zmin", "xmax", "ymax", "zmax",
    )}

    checkpoint_every = max(1, int(config.checkpoint_every_obcs))
    n_point = int(cp.shape[0])
    logger.info("Writing OBCs from object %d/%d", start_object, len(objects))
    started = time.perf_counter()

    for i in range(start_object, len(objects)):
        obc = objects[i]
        if distance_var is not None:
            indices, distances = _obc_indices_distances(obc, point_dtype)
        else:
            indices = _obc_indices(obc, point_dtype)
            distances = None

        if np.any(indices < 0) or np.any(indices >= n_point):
            raise ValueError(f"OBC {i}: point index out of range.")
        a, b = int(offsets[i]), int(offsets[i + 1])
        if len(indices) != b - a:
            raise ValueError(f"OBC {i}: membership size changed between passes.")

        membership_var[a:b] = indices
        if distance_var is not None and distances is not None:
            distance_var[a:b] = distances

        xyz = cp[indices.astype(np.int64, copy=False)]
        scalar_vars["start_epoch"][i] = int(obc.start_epoch)
        scalar_vars["end_epoch"][i] = int(obc.end_epoch)
        scalar_vars["seed_point_index"][i] = int(obc.seed.index)
        scalar_vars["seed_start_epoch"][i] = int(obc.seed.start_epoch)
        scalar_vars["seed_end_epoch"][i] = int(obc.seed.end_epoch)
        scalar_vars["threshold"][i] = float(obc.threshold)

        if len(indices):
            centroid = xyz.mean(axis=0)
            mins = xyz.min(axis=0)
            maxs = xyz.max(axis=0)
        else:
            centroid = np.full(3, np.nan)
            mins = np.full(3, np.nan)
            maxs = np.full(3, np.nan)

        scalar_vars["centroid_x"][i], scalar_vars["centroid_y"][i], scalar_vars["centroid_z"][i] = centroid
        scalar_vars["xmin"][i], scalar_vars["ymin"][i], scalar_vars["zmin"][i] = mins
        scalar_vars["xmax"][i], scalar_vars["ymax"][i], scalar_vars["zmax"][i] = maxs

        if (i + 1) % checkpoint_every == 0 or i + 1 == len(objects):
            group.attrs["obcs_written"] = int(i + 1)
            f.flush()
            logger.info("OBC checkpoint %d/%d", i + 1, len(objects))

    group.attrs["conversion_complete"] = "true"
    logger.info("Finished OBC writing in %.1f s", time.perf_counter() - started)
    f.flush()


def _build_reverse_index_in_group(f: Any, group: Any, n_point: int, config: ConversionConfig) -> None:
    """Build the optional point -> OBC index in O(total memberships) time."""
    if str(group.attrs.get("reverse_index_stored", "false")) == "true":
        return

    offsets = np.asarray(group.variables["object_point_offsets"][:], dtype=np.int64)
    members_var = group.variables["object_point_indices"]
    n_object = int(group.dimensions["object"].size)
    n_membership = int(group.dimensions["membership"].size)

    counts = np.zeros(n_point, dtype=np.int64)
    for i in range(n_object):
        a, b = int(offsets[i]), int(offsets[i + 1])
        points = np.asarray(members_var[a:b], dtype=np.int64)
        if len(points):
            np.add.at(counts, points, 1)

    point_offsets = np.empty(n_point + 1, dtype=np.int64)
    point_offsets[0] = 0
    np.cumsum(counts, out=point_offsets[1:])

    reverse = np.empty(n_membership, dtype=np.int32)
    cursor = point_offsets[:-1].copy()
    for i in range(n_object):
        a, b = int(offsets[i]), int(offsets[i + 1])
        points = np.asarray(members_var[a:b], dtype=np.int64)
        if not len(points):
            continue
        positions = cursor[points]
        reverse[positions] = i
        cursor[points] += 1

    _ensure_dimension(group, "point_offset", n_point + 1)
    _ensure_dimension(group, "point_membership", n_membership)
    po = _ensure_variable(group, "point_object_offsets", ("point_offset",), np.int64, compressed=False)
    pi = _ensure_variable(
        group,
        "point_object_indices",
        ("point_membership",),
        np.int32,
        chunks=(max(1, min(1_000_000, max(n_membership, 1))),),
        config=config,
    )
    po[:] = point_offsets
    pi[:] = reverse
    group.attrs["reverse_index_stored"] = "true"
    f.flush()


def convert_py4dgeo_zip(
    source_zip: str | Path,
    output_h5: str | Path,
    *,
    config: ConversionConfig,
    resume: bool = True,
    overwrite: bool = False,
    validate: bool = True,
    n_random_checks: int = 100,
    n_obc_checks: int = 100,
    reuse_object_ids_from: str | Path | None = None,
) -> Path:
    """Convert a py4dgeo ``SpatiotemporalAnalysis`` ZIP into managed HDF5.

    The conversion writes to ``<output>.partial.h5`` and checkpoints progress.
    If the process is interrupted, calling the function again with ``resume=True``
    continues from the last flushed point block/OBC checkpoint. The partial file is
    atomically renamed to ``output_h5`` only after conversion and optional validation.

    Args:
        source_zip: Input py4dgeo ``SpatiotemporalAnalysis`` ZIP.
        output_h5: Final HDF5 path.
        config: Storage and dataset configuration.
        resume: Resume a compatible ``.partial.h5`` file when present.
        overwrite: Delete an existing final/partial output and start from scratch.
        validate: Run spot-check validation against the source before finalizing.
        n_random_checks: Number of random M3C2 cells validated.
        n_obc_checks: Number of randomly selected OBC memberships validated.
        reuse_object_ids_from: Optional older HDF5 whose OBC ULIDs should be reused.

    Returns:
        Path to the completed HDF5 file.
    """
    _check_dataset_id(config.dataset_id)
    if config.point_chunk <= 0 or config.write_block_points <= 0:
        raise ValueError("point_chunk and write_block_points must be positive.")
    if not 0 <= config.compression_level <= 9:
        raise ValueError("compression_level must be between 0 and 9.")

    source_zip = Path(source_zip).resolve()
    output_h5 = Path(output_h5).resolve()
    partial_h5 = _partial_path(output_h5)
    if not source_zip.exists():
        raise FileNotFoundError(source_zip)

    output_h5.parent.mkdir(parents=True, exist_ok=True)
    if overwrite:
        if output_h5.exists():
            output_h5.unlink()
        if partial_h5.exists():
            partial_h5.unlink()
    elif output_h5.exists():
        raise FileExistsError(f"{output_h5} already exists; use overwrite=True.")
    elif partial_h5.exists() and not resume:
        raise FileExistsError(f"{partial_h5} exists; use resume=True or overwrite=True.")

    try:
        import py4dgeo
    except ImportError as exc:
        raise ImportError("Conversion requires py4dgeo.") from exc

    logger.info("Hashing source archive: %s", source_zip)
    source_sha = sha256_file(source_zip)
    source_size = source_zip.stat().st_size
    analysis = py4dgeo.SpatiotemporalAnalysis(str(source_zip), force=False)
    version = _py4dgeo_version(py4dgeo)

    with _open_h5netcdf(partial_h5, "a" if partial_h5.exists() else "w") as f:
        _initialize_root(
            f,
            source_zip=source_zip,
            source_sha256=source_sha,
            source_size=source_size,
            config=config,
            py4dgeo_version=version,
        )
        logger.info("Writing/resuming analysis group")
        _write_analysis_group(f, analysis, config)

        # Release large py4dgeo array caches before loading the object pickle.
        try:
            del analysis.distances
            del analysis.uncertainties
            del analysis.smoothed_distances
        except Exception:
            pass

        cp = np.asarray(analysis.corepoints.cloud)
        existing_obcs = f.groups.get("obcs")
        obcs_complete = (
            existing_obcs is not None
            and str(existing_obcs.attrs.get("conversion_complete", "false")) == "true"
        )
        objects: list[Any] | None = None
        if not obcs_complete:
            logger.info("Loading py4dgeo OBC objects")
            objects = list(analysis.objects or [])
            logger.info("Loaded %d OBCs", len(objects))
            _write_obcs_group(
                f,
                objects=objects,
                cp=cp,
                config=config,
                reuse_object_ids_from=reuse_object_ids_from,
            )
        if config.include_reverse_index:
            logger.info("Building optional point -> OBC reverse index")
            _build_reverse_index_in_group(f, f.groups["obcs"], cp.shape[0], config)

        f.attrs["conversion_state"] = "written"
        f.flush()

    if validate:
        validate_hdf5_export(
            analysis,
            partial_h5,
            objects=objects,
            n_random_checks=n_random_checks,
            n_obc_checks=n_obc_checks,
        )

    with _open_h5netcdf(partial_h5, "a") as f:
        f.attrs["conversion_state"] = "complete"
        f.attrs["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        f.flush()

    partial_h5.replace(output_h5)
    logger.info("Conversion complete: %s", output_h5)
    return output_h5


def build_reverse_index(h5_path: str | Path, *, compression_level: int = 1) -> None:
    """Add the optional point -> OBC lookup index to an existing HDF5 file.

    This is useful when repeated ``obcs_for_point`` queries become important. It
    duplicates membership information, so it is intentionally not required by the
    base schema.

    Args:
        h5_path: Completed OBC HDF5 file to update in place.
        compression_level: gzip compression level for the reverse index.
    """
    path = Path(h5_path).resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    with open_analysis(path) as analysis_ds:
        n_point = int(analysis_ds.sizes["point"])
        dataset_id = str(analysis_ds.attrs["dataset_id"])
    config = ConversionConfig(
        dataset_id=dataset_id,
        compression_level=compression_level,
        uncertainty_fields=(),
        include_smoothed_distances=False,
        include_reverse_index=True,
    )
    with _open_h5netcdf(path, "a") as f:
        _build_reverse_index_in_group(f, f.groups["obcs"], n_point, config)


def validate_hdf5_export(
    analysis: Any,
    h5_path: str | Path,
    *,
    objects: Sequence[Any] | None = None,
    n_random_checks: int = 100,
    n_obc_checks: int = 100,
    random_seed: int = 42,
) -> None:
    """Spot-check a managed HDF5 export against its py4dgeo source.

    Args:
        analysis: Loaded source ``SpatiotemporalAnalysis``.
        h5_path: HDF5 file to validate.
        objects: Optional already-loaded OBC list. Passing it avoids re-unpickling.
        n_random_checks: Number of randomly sampled M3C2 cells to compare exactly.
        n_obc_checks: Number of randomly sampled OBC memberships to compare.
        random_seed: Seed for reproducible sampling.
    """
    rng = np.random.default_rng(random_seed)
    objects = list(analysis.objects or []) if objects is None else list(objects)

    with OBCH5(h5_path) as data:
        if data.analysis.sizes["point"] != analysis.corepoints.cloud.shape[0]:
            raise AssertionError("Point count differs from source.")
        source_distance = np.asarray(analysis.distances)
        if data.analysis.sizes["time"] != source_distance.shape[1]:
            raise AssertionError("Time count differs from source.")
        if data.obcs.sizes["object"] != len(objects):
            raise AssertionError("OBC count differs from source.")

        ids = [_text(v) for v in data.obcs["object_id"].values]
        if len(ids) != len(set(ids)) or any(not is_ulid(v) for v in ids):
            raise AssertionError("Invalid or duplicate OBC ULIDs.")

        n = min(int(n_random_checks), source_distance.size)
        if n:
            p = rng.integers(0, source_distance.shape[0], size=n)
            t = rng.integers(0, source_distance.shape[1], size=n)
            managed = np.asarray([
                data.analysis["m3c2_distance"].isel(point=int(pi), time=int(ti)).item()
                for pi, ti in zip(p, t)
            ])
            np.testing.assert_allclose(source_distance[p, t], managed, equal_nan=True, rtol=0, atol=0)

        if objects:
            checks = min(int(n_obc_checks), len(objects))
            chosen = rng.choice(len(objects), size=checks, replace=False)
            for i in chosen:
                np.testing.assert_array_equal(
                    data.obc_point_indices(int(i)),
                    np.asarray(objects[int(i)].indices, dtype=np.int64),
                )
                if int(data.obcs["start_epoch"].values[i]) != int(objects[int(i)].start_epoch):
                    raise AssertionError(f"OBC {i}: start epoch differs.")
                if int(data.obcs["end_epoch"].values[i]) != int(objects[int(i)].end_epoch):
                    raise AssertionError(f"OBC {i}: end epoch differs.")


class OBCH5:
    """Convenience reader for managed 4D-OBC HDF5 files.

    The ``analysis`` and ``obcs`` groups are opened as xarray datasets. Spatial,
    object, and ML helpers return xarray/pandas/NumPy objects without requiring
    py4dgeo.
    """

    def __init__(self, path: str | Path, *, chunks: Any = None):
        """Open an HDF5 dataset.

        Args:
            path: Managed HDF5 file.
            chunks: Optional xarray/Dask chunk specification.
        """
        self.path = Path(path)
        self.analysis = xr.open_dataset(self.path, engine="h5netcdf", group="analysis", chunks=chunks)
        self.obcs = xr.open_dataset(self.path, engine="h5netcdf", group="obcs", chunks=chunks)
        self._xyz: np.ndarray | None = None
        self._object_table: pd.DataFrame | None = None

    def close(self) -> None:
        """Close xarray datasets and underlying HDF5 handles."""
        self.analysis.close()
        self.obcs.close()

    def __enter__(self) -> "OBCH5":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    @property
    def dataset_id(self) -> str:
        """Return the stored dataset identifier."""
        return str(self.analysis.attrs["dataset_id"])

    @property
    def distance_variable(self) -> str:
        """Return the preferred M3C2 variable for analysis/ML."""
        return str(self.analysis.attrs.get("distance_for_compute_variable", "m3c2_distance"))

    @property
    def xyz(self) -> np.ndarray:
        """Return and cache core-point coordinates as ``(point, 3)``."""
        if self._xyz is None:
            self._xyz = np.column_stack([
                self.analysis["x"].values,
                self.analysis["y"].values,
                self.analysis["z"].values,
            ])
        return self._xyz

    @property
    def has_reverse_index(self) -> bool:
        """Return whether the optional point -> OBC index is stored."""
        return "point_object_offsets" in self.obcs and "point_object_indices" in self.obcs

    def object_table(self) -> pd.DataFrame:
        """Return one row per OBC as a pandas DataFrame.

        Stored metadata are supplemented with derived ``start_time``, ``end_time``
        and seed XYZ values, so these useful columns do not need to be duplicated in
        the HDF5 file.
        """
        if self._object_table is None:
            names = [
                "object_id", "display_id", 
                "start_epoch", "end_epoch", "seed_point_index",
                "seed_start_epoch", "seed_end_epoch", "threshold", "n_points",
                "centroid_x", "centroid_y", "centroid_z",
                "xmin", "ymin", "zmin", "xmax", "ymax", "zmax",
            ]
            table = self.obcs[names].to_dataframe().reset_index()
            table["object_id"] = table["object_id"].astype(str)
            table["display_id"] = table["display_id"].astype(str)
            times = np.asarray(self.analysis["time"].values)
            starts = table["start_epoch"].to_numpy(dtype=np.int64)
            ends = table["end_epoch"].to_numpy(dtype=np.int64)
            table["start_time"] = times[starts]
            table["end_time"] = times[ends]
            seed = table["seed_point_index"].to_numpy(dtype=np.int64)
            table[["seed_x", "seed_y", "seed_z"]] = self.xyz[seed]
            self._object_table = table
        return self._object_table.copy()

    def point_indices_bbox(
        self,
        xmin: float,
        ymin: float,
        xmax: float,
        ymax: float,
        *,
        zmin: float | None = None,
        zmax: float | None = None,
    ) -> np.ndarray:
        """Return core-point indices inside an inclusive bounding box."""
        xyz = self.xyz
        mask = (
            (xyz[:, 0] >= xmin) & (xyz[:, 0] <= xmax)
            & (xyz[:, 1] >= ymin) & (xyz[:, 1] <= ymax)
        )
        if zmin is not None:
            mask &= xyz[:, 2] >= zmin
        if zmax is not None:
            mask &= xyz[:, 2] <= zmax
        return np.flatnonzero(mask)

    def point_indices_polygon(self, polygon: Any) -> np.ndarray:
        """Return core-point indices intersecting an XY Shapely polygon."""
        try:
            from shapely import intersects_xy
        except ImportError as exc:
            raise ImportError("Polygon queries require shapely >= 2.") from exc
        xmin, ymin, xmax, ymax = polygon.bounds
        candidates = self.point_indices_bbox(xmin, ymin, xmax, ymax)
        if len(candidates) == 0:
            return candidates
        xyz = self.xyz[candidates]
        return candidates[intersects_xy(polygon, xyz[:, 0], xyz[:, 1])]

    def query_points(
        self,
        point_indices: Sequence[int] | np.ndarray,
        *,
        variables: Sequence[str] | None = None,
        time: slice | Sequence[int] | np.ndarray | int | None = None,
        load: bool = False,
    ) -> xr.Dataset:
        """Return selected analysis variables for selected core-point indices."""
        points = np.asarray(point_indices, dtype=np.int64)
        variables = list(variables) if variables is not None else [self.distance_variable]
        result = self.analysis[variables].isel(point=points)
        if time is not None:
            result = result.isel(time=time)
        return result.load() if load else result

    def query_bbox(
        self,
        xmin: float,
        ymin: float,
        xmax: float,
        ymax: float,
        *,
        zmin: float | None = None,
        zmax: float | None = None,
        variables: Sequence[str] | None = None,
        time: slice | Sequence[int] | np.ndarray | int | None = None,
        load: bool = False,
    ) -> xr.Dataset:
        """Return all selected point time series inside a bounding box."""
        points = self.point_indices_bbox(xmin, ymin, xmax, ymax, zmin=zmin, zmax=zmax)
        return self.query_points(points, variables=variables, time=time, load=load)

    def query_polygon(
        self,
        polygon: Any,
        *,
        variables: Sequence[str] | None = None,
        time: slice | Sequence[int] | np.ndarray | int | None = None,
        load: bool = False,
    ) -> xr.Dataset:
        """Return all selected point time series inside an XY polygon."""
        points = self.point_indices_polygon(polygon)
        return self.query_points(points, variables=variables, time=time, load=load)

    def _resolve_obc_index(self, object_id_or_index: str | int) -> int:
        if isinstance(object_id_or_index, (int, np.integer)):
            i = int(object_id_or_index)
            if not 0 <= i < self.obcs.sizes["object"]:
                raise IndexError(i)
            return i
        key = str(object_id_or_index)
        for column in ("object_id", "display_id"):
            values = self.obcs[column].values.astype(str)
            match = np.flatnonzero(values == key)
            if len(match) == 1:
                return int(match[0])
        raise KeyError(object_id_or_index)

    def obc_point_indices(self, object_id_or_index: str | int) -> np.ndarray:
        """Return core-point indices belonging to one OBC."""
        i = self._resolve_obc_index(object_id_or_index)
        offsets = self.obcs["object_point_offsets"].values
        a, b = int(offsets[i]), int(offsets[i + 1])
        return self.obcs["object_point_indices"].values[a:b].astype(np.int64, copy=False)

    def obc_member_distances(self, object_id_or_index: str | int) -> np.ndarray:
        """Return stored region-growing distances for one OBC.

        Raises:
            KeyError: If ``include_obc_distances=False`` was used at conversion.
        """
        if "object_distance" not in self.obcs:
            raise KeyError("object_distance was not stored; reconvert with include_obc_distances=True if needed.")
        i = self._resolve_obc_index(object_id_or_index)
        offsets = self.obcs["object_point_offsets"].values
        a, b = int(offsets[i]), int(offsets[i + 1])
        return self.obcs["object_distance"].values[a:b]

    def _obc_indices_for_points_without_reverse(self, points: np.ndarray) -> np.ndarray:
        members = self.obcs["object_point_indices"].values
        offsets = self.obcs["object_point_offsets"].values
        mask = np.isin(members, points, assume_unique=False)
        positions = np.flatnonzero(mask)
        if not len(positions):
            return np.empty(0, dtype=np.int32)
        objects = np.searchsorted(offsets[1:], positions, side="right")
        return np.unique(objects).astype(np.int32, copy=False)

    def obcs_for_point(self, point_index: int) -> pd.DataFrame:
        """Return all OBCs containing one core point.

        The optional reverse index is used when present. Otherwise the compact
        object -> point membership array is scanned, which is slower but avoids
        duplicated storage in the default file.
        """
        point_index = int(point_index)
        if not 0 <= point_index < self.analysis.sizes["point"]:
            raise IndexError(point_index)
        if self.has_reverse_index:
            offsets = self.obcs["point_object_offsets"].values
            a, b = int(offsets[point_index]), int(offsets[point_index + 1])
            indices = self.obcs["point_object_indices"].values[a:b]
        else:
            indices = self._obc_indices_for_points_without_reverse(np.asarray([point_index], dtype=np.int64))
        table = self.object_table()
        return table.loc[table["object"].isin(indices)].reset_index(drop=True)

    def obcs_at_time(self, time_index: int) -> pd.DataFrame:
        """Return OBCs active at one time index; end epochs are inclusive."""
        t = int(time_index)
        table = self.object_table()
        return table.loc[(table["start_epoch"] <= t) & (table["end_epoch"] >= t)].reset_index(drop=True)

    def obc_indices_for_points(self, point_indices: Iterable[int]) -> np.ndarray:
        """Return unique OBC indices touching any selected core points."""
        points = np.asarray(list(point_indices), dtype=np.int64)
        if len(points) == 0:
            return np.empty(0, dtype=np.int32)
        if self.has_reverse_index:
            offsets = self.obcs["point_object_offsets"].values
            values = self.obcs["point_object_indices"].values
            parts = [values[int(offsets[p]):int(offsets[p + 1])] for p in points]
            parts = [part for part in parts if len(part)]
            return np.unique(np.concatenate(parts)).astype(np.int32, copy=False) if parts else np.empty(0, np.int32)
        return self._obc_indices_for_points_without_reverse(points)

    def obcs_intersecting_bbox(
        self,
        xmin: float,
        ymin: float,
        xmax: float,
        ymax: float,
        *,
        zmin: float | None = None,
        zmax: float | None = None,
    ) -> pd.DataFrame:
        """Return OBCs whose stored bounds intersect a bounding box."""
        table = self.object_table()
        mask = (
            (table["xmax"] >= xmin) & (table["xmin"] <= xmax)
            & (table["ymax"] >= ymin) & (table["ymin"] <= ymax)
        )
        if zmin is not None:
            mask &= table["zmax"] >= zmin
        if zmax is not None:
            mask &= table["zmin"] <= zmax
        return table.loc[mask].reset_index(drop=True)

    def obcs_intersecting_polygon(self, polygon: Any) -> pd.DataFrame:
        """Return OBCs containing at least one core point inside an XY polygon."""
        candidates = self.obcs_intersecting_bbox(*polygon.bounds)
        if candidates.empty:
            return candidates
        points = self.point_indices_polygon(polygon)
        indices = self.obc_indices_for_points(points)
        return candidates.loc[candidates["object"].isin(indices)].reset_index(drop=True)

    def query_obc(
        self,
        object_id_or_index: str | int,
        *,
        variables: Sequence[str] | None = None,
        temporal_padding: int = 0,
        load: bool = False,
    ) -> xr.Dataset:
        """Return member point time series over one OBC's active interval."""
        i = self._resolve_obc_index(object_id_or_index)
        points = self.obc_point_indices(i)
        start = max(0, int(self.obcs["start_epoch"].values[i]) - temporal_padding)
        end = min(
            self.analysis.sizes["time"] - 1,
            int(self.obcs["end_epoch"].values[i]) + temporal_padding,
        )
        result = self.query_points(points, variables=variables, time=slice(start, end + 1), load=load)
        result.attrs.update({
            "object_id": _text(self.obcs["object_id"].values[i]),
            "display_id": _text(self.obcs["display_id"].values[i]),
            "obc_index": i,
            "obc_start_epoch": int(self.obcs["start_epoch"].values[i]),
            "obc_end_epoch": int(self.obcs["end_epoch"].values[i]),
        })
        return result

    def ml_sample_for_obc(
        self,
        object_id_or_index: str | int,
        *,
        variable: str | None = None,
        temporal_padding: int = 0,
        dtype: np.dtype | str = np.float32,
    ) -> dict[str, Any]:
        """Return one OBC as an ML-ready dictionary.

        Returns:
            ``x`` in point x time order, XYZ coordinates, timestamps, member point
            indices, IDs, and the object's metadata row.
        """
        variable = variable or self.distance_variable
        i = self._resolve_obc_index(object_id_or_index)
        points = self.obc_point_indices(i)
        subset = self.query_obc(i, variables=[variable], temporal_padding=temporal_padding, load=True)
        row = self.object_table().loc[lambda df: df["object"] == i].iloc[0]
        return {
            "object_id": str(row["object_id"]),
            "display_id": str(row["display_id"]),
            "obc_index": i,
            "x": np.asarray(subset[variable].values, dtype=dtype),
            "xyz": np.asarray(self.xyz[points], dtype=dtype),
            "time": np.asarray(subset["time"].values),
            "point_indices": points,
            "metadata": row.to_dict(),
        }


def open_analysis(path: str | Path, *, chunks: Any = None) -> xr.Dataset:
    """Open the ``analysis`` HDF5 group directly as an xarray Dataset."""
    return xr.open_dataset(path, engine="h5netcdf", group="analysis", chunks=chunks)


def open_obcs(path: str | Path, *, chunks: Any = None) -> xr.Dataset:
    """Open the ``obcs`` HDF5 group directly as an xarray Dataset."""
    return xr.open_dataset(path, engine="h5netcdf", group="obcs", chunks=chunks)


def inspect_hdf5(path: str | Path) -> dict[str, Any]:
    """Return a compact summary of an OBC HDF5 file for notebooks/CLI output."""
    with OBCH5(path) as data:
        return {
            "path": str(Path(path)),
            "dataset_id": data.dataset_id,
            "points": int(data.analysis.sizes["point"]),
            "times": int(data.analysis.sizes["time"]),
            "objects": int(data.obcs.sizes["object"]),
            "memberships": int(data.obcs.sizes["membership"]),
            "analysis_variables": list(data.analysis.data_vars),
            "has_obc_distances": "object_distance" in data.obcs,
            "has_reverse_index": data.has_reverse_index,
        }


def _parse_uncertainty_fields(text: str) -> tuple[str, ...] | None:
    value = text.strip().lower()
    if value == "all":
        return None
    if value in {"none", ""}:
        return ()
    return tuple(part.strip() for part in text.split(",") if part.strip())


def _cli() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Efficient py4dgeo 4D-OBC HDF5 utilities")
    sub = parser.add_subparsers(dest="command", required=True)

    convert = sub.add_parser("convert", help="Convert/resume py4dgeo ZIP -> HDF5")
    convert.add_argument("source_zip", type=Path)
    convert.add_argument("output_h5", type=Path)
    convert.add_argument("--dataset-id", required=True)
    convert.add_argument("--crs", default=None)
    convert.add_argument("--distance-units", default="m")
    convert.add_argument("--point-chunk", type=int, default=512)
    convert.add_argument("--time-chunk", type=int, default=None)
    convert.add_argument("--write-block-points", type=int, default=8192)
    convert.add_argument("--compression-level", type=int, default=1)
    convert.add_argument(
        "--uncertainty-fields",
        default="lodetection",
        help="Comma-separated fields, 'all', or 'none'. Default: lodetection",
    )
    convert.add_argument("--no-smoothed-distances", action="store_true")
    convert.add_argument("--include-obc-distances", action="store_true")
    convert.add_argument("--include-reverse-index", action="store_true")
    convert.add_argument("--checkpoint-every-obcs", type=int, default=25)
    convert.add_argument("--no-resume", action="store_true")
    convert.add_argument("--overwrite", action="store_true")
    convert.add_argument("--no-validate", action="store_true")
    convert.add_argument("--reuse-object-ids-from", type=Path, default=None)

    reverse = sub.add_parser("build-reverse-index", help="Add point -> OBC lookup index")
    reverse.add_argument("h5_path", type=Path)
    reverse.add_argument("--compression-level", type=int, default=1)

    inspect = sub.add_parser("inspect", help="Print a short HDF5 summary")
    inspect.add_argument("h5_path", type=Path)

    args = parser.parse_args()
    if args.command == "convert":
        config = ConversionConfig(
            dataset_id=args.dataset_id,
            crs=args.crs,
            distance_units=args.distance_units,
            point_chunk=args.point_chunk,
            time_chunk=args.time_chunk,
            write_block_points=args.write_block_points,
            compression_level=args.compression_level,
            include_smoothed_distances=not args.no_smoothed_distances,
            uncertainty_fields=_parse_uncertainty_fields(args.uncertainty_fields),
            include_obc_distances=args.include_obc_distances,
            include_reverse_index=args.include_reverse_index,
            checkpoint_every_obcs=args.checkpoint_every_obcs,
        )
        path = convert_py4dgeo_zip(
            args.source_zip,
            args.output_h5,
            config=config,
            resume=not args.no_resume,
            overwrite=args.overwrite,
            validate=not args.no_validate,
            reuse_object_ids_from=args.reuse_object_ids_from,
        )
        print(path)
    elif args.command == "build-reverse-index":
        build_reverse_index(args.h5_path, compression_level=args.compression_level)
    elif args.command == "inspect":
        print(json.dumps(inspect_hdf5(args.h5_path), indent=2, default=str))


if __name__ == "__main__":
    _cli()
