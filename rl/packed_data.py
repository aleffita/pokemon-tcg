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


PACKED_FORMAT_VERSION = 2
PACKED_BACKEND_NAME = "fixed-width-npy-mmap"

# The legacy checkpoint is a compatibility exception, not a filename-based
# format.  Keep the path project-relative so the policy remains portable while
# binding it to the frozen artifact bytes.
APPROVED_STAGE4_ROOT_RELATIVE_PATH = Path(
    "experiments/autoresearch/root/stage4_root.pkl"
)
APPROVED_STAGE4_ROOT_SHA256 = (
    "b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b"
)

# These are deliberately independent of the Parquet sidecar manifest.  The
# trainer's runtime contract must not shrink when a producer forgets to update
# a manifest field.  The input portion is supplied by TokenEncoder.shapes;
# the metadata portion is the complete Stage-4 payload kept by the packer.
TRAINER_ORDER_COLUMNS = (
    "episode_id",
    "side",
    "step_id",
    "decision_id",
    "substep",
)
TRAINER_METADATA_COLUMNS = (
    "y",
    "is_attack",
    "opt_group",
    "aux_ko",
    "aux_prize_delta",
    "aux_terminal",
    "aux_return",
    "aux_valid",
    "episode_id",
    "side",
    "step_id",
    "decision_id",
    "substep",
    "new_episode",
    "terminal",
    "reward",
    "outcome",
    "is_self",
    "day_id",
)

PACKED_DEDUP_CONTRACT = {
    "column": "opt_group",
    "available": True,
}
PACKED_TBPTT_CONTRACT = {
    "metadata_columns": list(TRAINER_ORDER_COLUMNS),
    "group_columns": ["episode_id", "side"],
    "decision_columns": [
        "episode_id",
        "side",
        "step_id",
        "decision_id",
        "substep",
    ],
}


def required_trainer_columns(input_columns) -> list[str]:
    """Return the independent, complete column contract for the trainer.

    ``action_mask`` and every ``*_mask`` input are explicitly required.  This
    makes a future encoder change fail loudly instead of allowing a packed
    manifest to omit the legal-action surface while still looking complete.
    """
    inputs = list(input_columns)
    required_masks = {"action_mask", *(name for name in inputs if name.endswith("_mask"))}
    missing_masks = required_masks - set(inputs)
    if missing_masks:
        raise ValueError(
            "trainer input contract is missing legal/action mask columns: "
            f"{sorted(missing_masks)}"
        )
    return sorted(set(inputs)) + list(TRAINER_METADATA_COLUMNS)


def source_digest(paths: list[str | os.PathLike[str]]) -> str:
    """Digest an ordered Parquet source list, including each file's content."""
    digest = hashlib.sha256()
    for index, raw_path in enumerate(paths):
        path = Path(raw_path)
        digest.update(str(index).encode("ascii"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def approved_stage4_root_matches(
    path: str | os.PathLike[str],
    *,
    repo_root: str | os.PathLike[str] | None = None,
) -> bool:
    """Return whether ``path`` is exactly the approved frozen root artifact."""
    project_root = (
        Path(repo_root).resolve()
        if repo_root is not None
        else Path(__file__).resolve().parents[1]
    )
    expected = project_root / APPROVED_STAGE4_ROOT_RELATIVE_PATH
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    candidate = candidate.absolute()
    return (
        candidate == expected
        and not candidate.is_symlink()
        and candidate.is_file()
        and sha256_file(candidate) == APPROVED_STAGE4_ROOT_SHA256
    )


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


def _ordered_unique(values: np.ndarray) -> np.ndarray:
    """Return first-appearance values without changing their order."""
    array = np.asarray(values)
    if not len(array):
        return array.copy()
    keep = np.empty(len(array), dtype=bool)
    seen = set()
    for index, value in enumerate(array.tolist()):
        keep[index] = value not in seen
        seen.add(value)
    return array[keep]


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
        required_columns: list[str] | tuple[str, ...] | None = None,
        strict_contract: bool = False,
    ) -> None:
        self.root = Path(root)
        manifest_path = self.root / "manifest.json"
        with open(manifest_path, encoding="utf-8") as handle:
            self.manifest = json.load(handle)
        if self.manifest.get("format") != "fixed-width-npy-mmap":
            raise ValueError(f"unsupported packed format in {manifest_path}")
        if int(self.manifest.get("format_version", -1)) != PACKED_FORMAT_VERSION:
            raise ValueError(f"unsupported packed format version in {manifest_path}")
        manifest_columns = set(self.manifest.get("columns", []))
        if required_columns is not None:
            missing = sorted(set(required_columns) - manifest_columns)
            if missing:
                raise ValueError(
                    "packed store is missing required trainer columns: "
                    f"{missing}"
                )
        total_rows = int(self.manifest["selected_rows"])
        if row_start < 0 or row_start > total_rows:
            raise ValueError(f"invalid packed row_start={row_start}")
        if row_stop is None:
            row_stop = total_rows
        if row_stop < row_start or row_stop > total_rows:
            raise ValueError(f"invalid packed row range {row_start}:{row_stop}")
        self.row_start = row_start
        self.row_stop = row_stop
        self.strict_contract = bool(strict_contract)
        self._arrays: dict[str, np.ndarray] = {}
        self._bytes_returned = 0
        self._read_calls = 0
        self._column_names = (
            list(self.manifest["columns"]) if columns is None else list(columns)
        )
        unknown_columns = set(self._column_names) - set(self.manifest["columns"])
        if unknown_columns:
            raise ValueError(f"packed columns are absent from manifest: {sorted(unknown_columns)}")
        if self.strict_contract:
            if row_start != 0 or row_stop != total_rows:
                raise ValueError(
                    "strict packed contract validation requires the complete store"
                )
            if required_columns is None:
                raise ValueError(
                    "strict packed contract validation requires required_columns"
                )
            if set(self._column_names) != set(required_columns):
                raise ValueError(
                    "strict packed validation must open every required trainer column"
                )
            self._validate_manifest_contract(required_columns)
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
            if set(TRAINER_ORDER_COLUMNS).issubset(self._arrays):
                self._validate_order_digest()
            elif self.strict_contract:
                raise ValueError(
                    "strict packed validation did not open every order column"
                )

    def _validate_manifest_contract(self, required_columns) -> None:
        """Reject incomplete manifests before any packed training can start."""
        required = set(required_columns)
        manifest_columns = set(self.manifest.get("columns", []))
        if required - manifest_columns:
            raise ValueError(
                "packed store is missing required trainer columns: "
                f"{sorted(required - manifest_columns)}"
            )
        if not self.manifest.get("source_sha256"):
            raise ValueError("packed manifest is missing source_sha256")

        contract = self.manifest.get("required_contract")
        if not isinstance(contract, dict):
            raise ValueError("packed manifest is missing required_contract metadata")
        contract_columns = contract.get("columns")
        if not isinstance(contract_columns, list) or set(contract_columns) != required:
            raise ValueError(
                "packed required_contract columns do not match the trainer contract"
            )
        if contract.get("order") != list(TRAINER_ORDER_COLUMNS):
            raise ValueError("packed required_contract is missing the order contract")
        if contract.get("dedup") != PACKED_DEDUP_CONTRACT:
            raise ValueError("packed required_contract is missing dedup metadata")
        if contract.get("tbptt") != PACKED_TBPTT_CONTRACT:
            raise ValueError("packed required_contract is missing TBPTT metadata")

        selection = self.manifest.get("selection")
        if not isinstance(selection, dict):
            raise ValueError("packed manifest is missing selection metadata")
        selection_fields = {
            "max_rows",
            "val_frac",
            "selected_episode_ids",
            "train_episode_ids",
            "val_episode_ids",
            "train_rows",
            "val_rows",
        }
        missing_selection = sorted(selection_fields - set(selection))
        if missing_selection:
            raise ValueError(
                "packed manifest is missing selection fields: "
                f"{missing_selection}"
            )

    def _validate_order_digest(self) -> None:
        order = self.manifest.get("row_order")
        if not order:
            raise ValueError("packed manifest is missing the row-level order contract")
        names = list(order.get("columns", []))
        if list(TRAINER_ORDER_COLUMNS) != names:
            raise ValueError(
                "packed row-order columns do not match trainer contract: "
                f"got={names} expected={list(TRAINER_ORDER_COLUMNS)}"
            )
        missing = [name for name in names if name not in self._arrays]
        if missing:
            raise ValueError(
                "packed row-order contract was not opened for columns: "
                f"{missing}"
            )
        required_order_fields = {"val_rows", "train_rows", "val_digest", "train_digest"}
        missing_order = sorted(required_order_fields - set(order))
        if missing_order:
            raise ValueError(
                "packed row-order contract is missing fields: " f"{missing_order}"
            )
        boundary = int(order.get("val_rows", -1))
        if boundary < 0 or boundary > self.manifest["selected_rows"]:
            raise ValueError(f"invalid packed val/train row boundary: {boundary}")
        expected_train_rows = int(self.manifest["selected_rows"]) - boundary
        if int(order.get("train_rows", -1)) != expected_train_rows:
            raise ValueError("packed row-order train_rows disagrees with val boundary")
        selection = self.manifest.get("selection", {})
        for field in (
            "selected_episode_ids",
            "val_episode_ids",
            "train_episode_ids",
            "val_rows",
            "train_rows",
        ):
            if field not in selection:
                raise ValueError(f"packed selection is missing {field}")
        if int(selection["val_rows"]) != boundary:
            raise ValueError("packed selection val_rows disagrees with val boundary")
        if int(selection["train_rows"]) != expected_train_rows:
            raise ValueError("packed selection train_rows disagrees with val boundary")
        actual_val = np.asarray(self._arrays["episode_id"][:boundary])
        actual_train = np.asarray(self._arrays["episode_id"][boundary:])
        expected_val = np.asarray(selection["val_episode_ids"], dtype=actual_val.dtype)
        expected_train = np.asarray(selection["train_episode_ids"], dtype=actual_train.dtype)
        expected_selected = np.asarray(
            selection["selected_episode_ids"], dtype=actual_val.dtype
        )
        if not np.array_equal(_ordered_unique(actual_val), expected_val):
            raise ValueError("packed val rows do not match manifest val episodes")
        if not np.array_equal(_ordered_unique(actual_train), expected_train):
            raise ValueError("packed train rows do not match manifest train episodes")
        if not np.array_equal(
            np.concatenate([expected_val, expected_train]), expected_selected
        ):
            raise ValueError("packed selected episodes do not match the train/val split")
        for split, start, stop in (
            ("val", 0, boundary),
            ("train", boundary, int(self.manifest["selected_rows"])),
        ):
            expected = order.get(f"{split}_digest")
            actual = _digest_columns(
                {name: np.asarray(self._arrays[name][start:stop]) for name in names},
                names,
            )
            if actual != expected:
                raise ValueError(f"packed {split} row-order digest mismatch")

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
) -> None:
    """Validate immutable source/split identity.

    Episode selection is stable first-appearance order and intentionally does
    not use a seed.  Trainer ``seed`` controls batch shuffling, not membership
    or val/train assignment, so it is not part of this contract.
    """
    selection = store.manifest["selection"]
    checks = {
        "source_sha256": (store.manifest["source_sha256"], source_sha256),
        "max_rows": (int(selection["max_rows"]), int(max_rows)),
        "val_frac": (float(selection["val_frac"]), float(val_frac)),
    }
    mismatches = {
        name: (actual, expected)
        for name, (actual, expected) in checks.items()
        if actual != expected
    }
    if mismatches:
        raise ValueError(f"packed selection contract mismatch: {mismatches}")


def validate_packed_tbptt_compatibility(store: PackedArrayStore, tbptt_chunk: int) -> None:
    """Reject the packed backend unless the sequential TBPTT path is active."""
    if int(tbptt_chunk) <= 0:
        raise ValueError(
            "unsupported packed-data combination: fixed-width stores require "
            "the Stage-4 TBPTT path (set --tbptt-chunk > 0); no Parquet fallback"
        )
    contract = store.manifest.get("required_contract", {}).get("tbptt")
    if contract != PACKED_TBPTT_CONTRACT:
        raise ValueError("packed store has no validated TBPTT metadata")


def build_resume_identity(
    *,
    source_sha256: str,
    selection: dict,
    split: dict,
    backend: dict,
    seed: int,
    dedup: bool,
    tbptt_chunk: int,
) -> dict:
    """Build the canonical data/backend identity stored in checkpoints."""
    return {
        "version": 1,
        "source": {"sha256": str(source_sha256)},
        "selection": selection,
        "split": split,
        "backend": backend,
        "trainer": {
            "seed": int(seed),
            "dedup": bool(dedup),
            "tbptt_chunk": int(tbptt_chunk),
        },
    }


def validate_resume_identity(
    saved: dict | None,
    current: dict,
    *,
    packed: bool,
    resume_path: str | os.PathLike[str] | None = None,
    optimizer_state: str = "reset",
    scheduler_state: str = "reset",
    repo_root: str | os.PathLike[str] | None = None,
) -> str:
    """Validate checkpoint data identity with an explicit legacy policy."""
    if saved is None:
        if (
            not packed
            and optimizer_state == "reset"
            and scheduler_state == "reset"
            and resume_path is not None
            and approved_stage4_root_matches(resume_path, repo_root=repo_root)
        ):
            return "legacy-stage4-warmstart-no-data-identity"
        if packed:
            raise ValueError(
                "checkpoint has no data_identity; refusing legacy checkpoint with "
                "--packed-data to prevent corpus mixing"
            )
        raise ValueError(
            "checkpoint has no data_identity; only an explicit Stage 4 warm-start "
            "may use a legacy checkpoint"
        )
    if not isinstance(saved, dict):
        raise ValueError("checkpoint data_identity must be an object")
    if saved != current:
        raise ValueError(
            "checkpoint data/backend identity mismatch: "
            f"saved={saved!r}, current={current!r}"
        )
    return "validated"


__all__ = [
    "PACKED_BACKEND_NAME",
    "PACKED_FORMAT_VERSION",
    "PACKED_DEDUP_CONTRACT",
    "PACKED_TBPTT_CONTRACT",
    "APPROVED_STAGE4_ROOT_RELATIVE_PATH",
    "APPROVED_STAGE4_ROOT_SHA256",
    "PackedArrayStore",
    "TRAINER_METADATA_COLUMNS",
    "TRAINER_ORDER_COLUMNS",
    "required_trainer_columns",
    "source_digest",
    "_digest_columns",
    "_digest_array",
    "sha256_file",
    "approved_stage4_root_matches",
    "split_episode_ids",
    "validate_selection",
    "validate_packed_tbptt_compatibility",
    "build_resume_identity",
    "validate_resume_identity",
]
