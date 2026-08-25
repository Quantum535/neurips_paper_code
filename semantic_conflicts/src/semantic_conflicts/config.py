"""Canonical configuration loader. All scientific constants live in YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from semantic_conflicts.paths import default_config_path


class DuplicateTitleConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    min_tokens: int = 3
    jaccard_threshold: float = 0.5
    containment_min_chars: int = 15


class RevertConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pattern: str = r"\b(revert|undo|rollback)\b"


class FixInFlightConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    keywords: list[str]


class NearMissConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = "dir_prefix_depth2"
    max_dir_components: int = 2
    exclude_root_files: bool = True
    include_filename_as_component: bool = False


class FileClassesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    lockfile_basenames: list[str]
    manifest_basenames: list[str]
    docker_basenames: list[str]
    spec_config_classes: list[str]


class PrevalenceSampleConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    n: int
    min_per_stratum: int = 80
    strata: list[str]


class FrameAConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_gap_days: float = 7.0
    cap_per_repo: int = 50


class EnrichedSampleConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    census_if_n_le: int = 50
    force_census_if: list[str]
    targets: dict[str, Any]
    substrata: list[str]
    frame_a: FrameAConfig


class CalibrationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    n: int
    quotas: dict[str, int] = {}


class ValidationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    n: int


class SamplingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    seed: int
    prevalence: PrevalenceSampleConfig
    enriched: EnrichedSampleConfig
    calibration: CalibrationConfig
    validation: ValidationConfig


class AnnotationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    labels: list[str]
    confidence: list[int]
    n_annotators: int = 2
    rubric_version: str = "v1"
    hidden_fields: list[str]


class JudgeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    official_without_diff: list[str]
    categories: list[str]
    prompt_version: str = "v1"
    body_chars: int = 1200
    max_retries: int = 6
    timeout_seconds: int = 300


class StatisticsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    wilson_z: float = 1.959963984540054
    bootstrap_iterations: int = 5000
    bootstrap_seed: int = 20260824
    prevalence_estimator: str = "hajek"
    extreme_weight_ratio: float = 50.0
    min_slice_n: int = 30


class BenchmarkConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = "SemanticConflictBench"
    version: str = "v1"
    split_seed: int = 20260824
    min_gold_for_train: int = 200
    gold_dev_frac: float = 0.4
    gold_test_frac: float = 0.6


class StaticDetectorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = "static_dependency_detector"
    min_symbol_length: int = 3
    identifier_similarity_threshold: float = 0.92
    weak_fallback: str = "identifier_similarity"


class HistoricalReleased(BaseModel):
    """Frozen published counts. Audit/reconciliation only — never used as substitutes."""

    model_config = ConfigDict(extra="allow")
    purpose: str
    source: str
    n_pairs: int
    n_conflict: int
    dup_title: int
    revert: int
    fix_in_flight: int
    near_miss: int
    both_merged_shared: int
    judging_frame: int
    buggy_near_miss_path_components_including_filename: int
    near_miss_depth1_noroot: int
    near_miss_depth2_with_root: int


class DataPathsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pairs: str | None = None
    files: str | None = None
    texts: str | None = None
    repos: str | None = None


class Settings(BaseModel):
    model_config = ConfigDict(extra="allow")
    version: str
    seed: int
    expected_n_pairs: int | None = 577045
    body_truncation_chars: int = 1200
    files_list_cap: int = 80
    fixture: bool = False
    data: DataPathsConfig = Field(default_factory=DataPathsConfig)
    mega_repos: list[str]
    duplicate_title: DuplicateTitleConfig
    revert: RevertConfig
    fix_in_flight: FixInFlightConfig
    near_miss: NearMissConfig
    file_classes: FileClassesConfig
    sampling: SamplingConfig
    annotation: AnnotationConfig
    judge: JudgeConfig
    statistics: StatisticsConfig
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)
    static_detector: StaticDetectorConfig = Field(default_factory=StaticDetectorConfig)
    historical_released: HistoricalReleased | None = None
    source_path: Path | None = None


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if k == "extends":
            continue
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    extends = data.get("extends")
    if extends:
        parent = _load_yaml((path.parent / extends).resolve())
        data = _deep_merge(parent, data)
    return data


def load_settings(path: Path | str | None = None) -> Settings:
    cfg_path = Path(path) if path is not None else default_config_path()
    cfg_path = cfg_path.resolve()
    raw = _load_yaml(cfg_path)
    settings = Settings.model_validate(raw)
    settings.source_path = cfg_path
    return settings
