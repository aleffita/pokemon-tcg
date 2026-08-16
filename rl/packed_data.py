"""Fixed-width, memory-mappable training data for the MLX BC trainer.

The packed format is deliberately boring: one C-contiguous ``.npy`` array per
logical Parquet column plus a JSON manifest.  ``numpy.load(..., mmap_mode='r')``
keeps the source representation out of Python object memory while preserving
the trainer's existing row-wise and column-wise dtypes.

The manifest is part of the data contract.  It records the source Parquet
hash, the exact episode cap/split parameters, the selected episode IDs, and a
per-column digest.  A packed store is therefore rejected rather than silently
used when it was built from a different source or selection.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
from pathlib import Path
from typing import Iterator

import numpy as np


PACKED_FORMAT_VERSION = 1


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _digest_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.tobytes(order="C")).hexdigest()


def _digest_columns(columns: dict[str, np.ndarray], names: list[str]) -> str:
    digest = hashlib.sha256()
    for name in names:
        array = np.ascontiguousarray(columns[name])
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(array.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def split_episode_ids(
    episode_ids: np.ndarray,
    *,
    max_rows: int,
    val_frac: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reproduce the trainer's stable first-appearance episode selection."""
    unique, first, counts = np.unique(
        episode_ids, return_index=True, return_counts=True
    )
    appearance_order = np.argsort(first)
    unique = unique[appearance_order]
    counts = counts[appearance_order]
    if max_rows > 0:
        cutoff = min(
            int(np.searchsorted(np.cumsum(counts), max_rows) + 1), len(unique)
        )
        selected = unique[:cutoff]
    else:
        selected = unique
    n_val = max(1, int(round(len(selected) * val_frac)))
    if n_val >= len(selected):
        n_val = len(selected) - 1
    if n_val <= 0 or len(selected) - n_val <= 0:
        raise ValueError(
            "episode split produced an empty train or val set: "
            f"selected={len(selected)} val_frac={val_frac}"
        )
    return selected, selected[n_val:], selected[:n_val]


class PackedArrayStore:
    """Read-only mmap-backed row store implementing the trainer cache API."""

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        row_start: int = 0,
        row_stop: int | None = None,
        columns: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        self.root = Path(root)
        manifest_path = self.root / "manifest.json"
        with open(manifest_path, encoding="utf-8") as handle:
            self.manifest = json.load(handle)
        if self.manifest.get("format") != "fixed-width-npy-mmap":
            raise ValueError(f"unsupported packed format in {manifest_path}")
        if int(self.manifest.get("format_version", -1)) != PACKED_FORMAT_VERSION:
            raise ValueError(f"unsupported packed format version in {manifest_path}")
        total_rows = int(self.manifest["selected_rows"])
        if row_start < 0 or row_start > total_rows:
            raise ValueError(f"invalid packed row_start={row_start}")
        if row_stop is None:
            row_stop = total_rows
        if row_stop < row_start or row_stop > total_rows:
            raise ValueError(f"invalid packed row range {row_start}:{row_stop}")
        self.row_start = row_start
        self.row_stop = row_stop
        self._arrays: dict[str, np.ndarray] = {}
        self._bytes_returned = 0
        self._read_calls = 0
        self._column_names = (
            list(self.manifest["columns"]) if columns is None else list(columns)
        )
        unknown_columns = set(self._column_names) - set(self.manifest["columns"])
        if unknown_columns:
            raise ValueError(f"packed columns are absent from manifest: {sorted(unknown_columns)}")
        for name in self._column_names:
            spec = self.manifest["column_specs"][name]
            path = self.root / spec["file"]
            if sha256_file(path) != spec["file_sha256"]:
                raise ValueError(f"packed column hash mismatch: {name}")
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            expected_shape = tuple(spec["shape"])
            if tuple(array.shape) != expected_shape or str(array.dtype) != spec["dtype"]:
                raise ValueError(
                    f"packed column shape/dtype mismatch for {name}: "
                    f"got={array.shape}/{array.dtype} expected={expected_shape}/{spec['dtype']}"
                )
            self._arrays[name] = array
        if row_start == 0 and row_stop == total_rows:
            for name in self._column_names:
                expected = self.manifest["column_digests"][name]
                actual = _digest_array(np.asarray(self._arrays[name][row_start:row_stop]))
                if actual != expected:
                    raise ValueError(f"packed column value digest mismatch: {name}")

    @property
    def columns(self) -> tuple[str, ...]:
        return tuple(self._column_names)

    @property
    def selected_rows(self) -> int:
        return self.row_stop - self.row_start

    def array(self, name: str) -> np.ndarray:
        """Return a read-only mmap view for one column in this row range."""
        if name not in self._arrays:
            raise KeyError(f"packed column is not open: {name}")
        return self._arrays[name][self.row_start : self.row_stop]

    @property
    def logical_bytes(self) -> int:
        return sum(
            int(self._arrays[name].dtype.itemsize)
            * int(np.prod(self._arrays[name].shape[1:], dtype=np.int64))
            * self.selected_rows
            for name in self._column_names
        )

    def read_rows(
        self,
        row_indices: np.ndarray,
        file_indices: np.ndarray | None = None,
        row_group_indices: np.ndarray | None = None,
        offsets: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        del file_indices, row_group_indices, offsets
        indices = np.asarray(row_indices, dtype=np.int64)
        if np.any(indices < 0) or np.any(indices >= self.selected_rows):
            raise IndexError("packed row index outside the selected split")
        global_indices = indices + self.row_start
        result = {
            name: np.asarray(self._arrays[name][global_indices])
            for name in self._column_names
        }
        self._read_calls += 1
        self._bytes_returned += sum(int(array.nbytes) for array in result.values())
        return result

    @contextlib.contextmanager
    def in_opt_step(self) -> Iterator[None]:
        yield

    def report(self) -> dict[str, int]:
        return {
            "hits": max(0, self._read_calls - 1),
            "misses": 1 if self._read_calls else 0,
            "evictions": 0,
            "promotions": 0,
            "ssd_hits": 0,
            "ssd_spills": 0,
            "bytes_loaded": self._bytes_returned,
            "resident_row_groups": 0,
            "resident_hot": 0,
            "resident_transient": 0,
            "ssd_resident": 0,
            "ssd_bytes": 0,
            "packed_mmap_bytes": self.logical_bytes,
        }


def validate_selection(
    store: PackedArrayStore,
    *,
    source_sha256: str,
    max_rows: int,
    val_frac: float,
    seed: int,
) -> None:
    selection = store.manifest["selection"]
    checks = {
        "source_sha256": (store.manifest["source_sha256"], source_sha256),
        "max_rows": (int(selection["max_rows"]), int(max_rows)),
        "val_frac": (float(selection["val_frac"]), float(val_frac)),
        "seed": (int(selection["seed"]), int(seed)),
    }
    mismatches = {
        name: (actual, expected)
        for name, (actual, expected) in checks.items()
        if actual != expected
    }
    if mismatches:
        raise ValueError(f"packed selection contract mismatch: {mismatches}")


__all__ = [
    "PACKED_FORMAT_VERSION",
    "PackedArrayStore",
    "_digest_columns",
    "_digest_array",
    "sha256_file",
    "split_episode_ids",
    "validate_selection",
]
