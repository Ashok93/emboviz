"""Processor-pipeline loading across lerobot checkpoint save formats.

lerobot >= 0.5 stores the input/output processor pipelines (normalization,
rename, tokenization) as ``policy_preprocessor.json`` /
``policy_postprocessor.json`` beside the weights. Earlier checkpoints have
no such files: the normalization statistics live as buffers inside
``model.safetensors`` under ``<module>.buffer_<feature_key>.<stat>``, where
the feature key's dots are written as underscores and ``<module>`` is one of
``normalize_inputs`` / ``normalize_targets`` / ``unnormalize_outputs``. The
current policy classes load those buffers as unexpected keys and discard
them, so they must be read back and handed to the processor factory.

:func:`load_processors` covers both layouts. Saved pipelines are loaded as
given. For the older layout, the baked buffers are read into a
``dataset_stats`` mapping and passed to the factory's create path. If a
feature requires normalization but no statistics are present, it raises:
the lerobot normalizer skips un-statted features silently, which would run
the model on un-normalized inputs.
"""

from __future__ import annotations

import os
import re
from typing import Any, Callable

_BUFFER_MODULES = ("normalize_inputs", "normalize_targets", "unnormalize_outputs")
_BUFFER_RE = re.compile(
    r"^(?:" + "|".join(_BUFFER_MODULES) + r")\.buffer_(?P<key>.+)\.(?P<stat>mean|std|min|max)$"
)
_PREPROCESSOR_FILE = "policy_preprocessor.json"
_WEIGHTS_FILE = "model.safetensors"


def _has_saved_processors(checkpoint: str) -> bool:
    """True if ``checkpoint`` carries a saved processor pipeline."""
    if os.path.isdir(checkpoint):
        return os.path.isfile(os.path.join(checkpoint, _PREPROCESSOR_FILE))
    from huggingface_hub import file_exists

    return file_exists(repo_id=checkpoint, filename=_PREPROCESSOR_FILE, repo_type="model")


def _weights_path(checkpoint: str) -> str:
    """Local path to the checkpoint weights (downloading from the Hub if
    ``checkpoint`` is a repo id; a cache hit after ``from_pretrained``)."""
    if os.path.isdir(checkpoint):
        path = os.path.join(checkpoint, _WEIGHTS_FILE)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"{checkpoint!r} has no {_PREPROCESSOR_FILE} and no "
                f"{_WEIGHTS_FILE} to read normalization statistics from."
            )
        return path
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=checkpoint, filename=_WEIGHTS_FILE)


def _read_baked_stats(checkpoint: str, cfg) -> dict[str, dict[str, Any]]:
    """Read ``dataset_stats`` from the normalization buffers baked into the
    checkpoint weights. Buffer names are matched to the policy's declared
    feature keys (dots written as underscores), so the dotted key is
    recovered unambiguously."""
    from safetensors import safe_open

    under_to_key = {
        key.replace(".", "_"): key
        for key in (*cfg.input_features, *cfg.output_features)
    }
    grouped: dict[str, dict[str, Any]] = {}
    with safe_open(_weights_path(checkpoint), framework="pt") as f:
        for name in f.keys():
            match = _BUFFER_RE.match(name)
            if match is None:
                continue
            key = under_to_key.get(match.group("key"))
            if key is None:
                continue
            grouped.setdefault(key, {})[match.group("stat")] = f.get_tensor(name)

    stats: dict[str, dict[str, Any]] = {}
    for key, by_stat in grouped.items():
        if "mean" in by_stat and "std" in by_stat:
            stats[key] = {"mean": by_stat["mean"], "std": by_stat["std"]}
        elif "min" in by_stat and "max" in by_stat:
            stats[key] = {"min": by_stat["min"], "max": by_stat["max"]}
    return stats


def _assert_stats_complete(checkpoint: str, cfg, stats: dict[str, Any]) -> None:
    """Raise if any feature with a non-identity normalization mode lacks
    statistics. lerobot's normalizer skips such features silently."""
    from lerobot.configs.types import NormalizationMode

    for key, feature in {**cfg.input_features, **cfg.output_features}.items():
        mode = cfg.normalization_mapping.get(feature.type.name, NormalizationMode.IDENTITY)
        if mode == NormalizationMode.IDENTITY:
            continue
        if key not in stats:
            raise ValueError(
                f"{checkpoint!r}: feature {key!r} uses {mode.name} normalization "
                "but the checkpoint provides no statistics for it (no saved "
                "processor pipeline and no baked buffer). Refusing to run it "
                "un-normalized."
            )


def load_processors(cfg, checkpoint: str, factory: Callable):
    """Return ``(preprocessor, postprocessor)`` for ``checkpoint``.

    ``factory`` is ``lerobot.policies.factory.make_pre_post_processors``.
    Saved pipelines (lerobot >= 0.5) are loaded directly; for older
    checkpoints the baked normalization buffers are read into
    ``dataset_stats`` and passed to the factory's create path.
    """
    if _has_saved_processors(checkpoint):
        return factory(cfg, pretrained_path=checkpoint)
    stats = _read_baked_stats(checkpoint, cfg)
    _assert_stats_complete(checkpoint, cfg, stats)
    return factory(cfg, dataset_stats=stats)
