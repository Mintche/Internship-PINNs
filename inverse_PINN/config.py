"""Strict JSON configuration for modular inverse-PINN campaigns."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _keys(raw: Mapping[str, Any], expected: set[str], name: str) -> None:
    missing = sorted(expected - set(raw))
    unknown = sorted(set(raw) - expected)
    if missing or unknown:
        raise ValueError(f"Invalid {name} keys; missing={missing}, unknown={unknown}")


def _positive_int(value: Any, name: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{name} must be an integer")
    if isinstance(value, str) and str(result) != value.strip():
        raise ValueError(f"{name} must be an integer")
    minimum = 0 if allow_zero else 1
    if result < minimum:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {qualifier}")
    return result


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive(value: Any, name: str) -> float:
    result = _finite(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _non_negative(value: Any, name: str) -> float:
    result = _finite(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _tuple_int(raw: Any, length: int, name: str) -> tuple[int, ...]:
    values = tuple(_positive_int(item, name) for item in raw)
    if len(values) != length:
        raise ValueError(f"{name} must contain {length} positive integers")
    return values


def _tuple_float(raw: Any, length: int, name: str, *, positive: bool = True) -> tuple[float, ...]:
    converter = _positive if positive else _finite
    values = tuple(converter(item, name) for item in raw)
    if len(values) != length:
        raise ValueError(f"{name} must contain {length} values")
    return values


def _tuple_non_negative(raw: Any, length: int, name: str) -> tuple[float, ...]:
    values = tuple(_non_negative(item, name) for item in raw)
    if len(values) != length:
        raise ValueError(f"{name} must contain {length} values")
    return values


def _path(value: Any) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else REPOSITORY_ROOT / path


@dataclass(frozen=True, order=True)
class Case:
    frequency: float
    mode: int
    incidence: int

    def __post_init__(self) -> None:
        if not math.isfinite(self.frequency) or self.frequency <= 0.0:
            raise ValueError("Case frequency must be finite and positive")
        if self.mode < 0:
            raise ValueError("Case mode must be non-negative")
        if self.incidence not in {-1, 1}:
            raise ValueError("Case incidence must be -1 or 1")

    @property
    def id(self) -> str:
        frequency = f"{self.frequency:g}".replace(".", "p")
        incidence = "m1" if self.incidence == -1 else "p1"
        return f"i{incidence}_f{frequency}_m{self.mode}"

    def manifest(self) -> dict[str, float | int | str]:
        return {
            "id": self.id,
            "frequency": self.frequency,
            "mode": self.mode,
            "incidence": self.incidence,
        }


@dataclass(frozen=True)
class TruthRegion:
    shape: str
    speed_ratio: float
    center: tuple[float, float] | None = None
    radius: float | None = None
    bounds: tuple[float, float, float, float] | None = None


@dataclass(frozen=True)
class GeometryConfig:
    height: float
    half_length: float
    c0: float
    celerity_ratio_bounds: tuple[float, float]
    truth_regions: tuple[TruthRegion, ...]

    @property
    def c_min(self) -> float:
        return self.c0 * self.celerity_ratio_bounds[0]

    @property
    def c_max(self) -> float:
        return self.c0 * self.celerity_ratio_bounds[1]

    @property
    def m_min(self) -> float:
        return 1.0 / self.c_max**2

    @property
    def m_max(self) -> float:
        return 1.0 / self.c_min**2

    @property
    def m0(self) -> float:
        return 1.0 / self.c0**2


@dataclass(frozen=True)
class DataConfig:
    boundary_left: Path
    boundary_right: Path
    fem_field: Path
    mass_matrix: Path
    stiffness_matrix: Path
    mesh_nodes: Path
    mesh_triangles: Path


@dataclass(frozen=True)
class ModelConfig:
    fourier_features: int
    field_hidden_layers: tuple[int, ...]
    material_hidden_layers: tuple[int, ...]


@dataclass(frozen=True)
class FieldAdweightsConfig:
    epsilon: float
    alpha: float
    initial_lambdas: tuple[float, float, float, float]
    custom_weights: tuple[float, float, float, float]
    update_interval_adam: int


@dataclass(frozen=True)
class MaterialAdweightsConfig:
    epsilon: float
    alpha: float
    initial_lambda: float
    custom_weight: float
    update_interval_adam: int


@dataclass(frozen=True)
class TVConfig:
    weight: float
    epsilon_squared: float


@dataclass(frozen=True)
class LossConfig:
    field_weights: tuple[float, float, float, float]
    field_adweights: FieldAdweightsConfig
    material_adweights: MaterialAdweightsConfig
    tv: TVConfig


@dataclass(frozen=True)
class OptimizationConfig:
    field_learning_rate: float
    material_learning_rate: float
    sigma_learning_rate: float
    sigma_decay_fraction: float
    sigma_cosine_alpha: float
    data_initial_factor: float
    data_transition_fraction: float


@dataclass(frozen=True)
class SamplingConfig:
    adam: tuple[int, int, int]
    monitor: tuple[int, int, int]
    lbfgs: tuple[int, int, int]
    sobol_scramble: bool
    sobol_seed_offset: int


@dataclass(frozen=True)
class LoggingConfig:
    loss_interval_adam: int
    loss_interval_lbfgs: int
    pressure_gradient_interval_adam: int
    sigma_interval_adam: int
    print_interval_adam: int
    material_snapshot_fractions: tuple[float, ...]
    snapshot_grid: tuple[int, int]
    prediction_batch_size: int


@dataclass(frozen=True)
class WarmupBudget:
    adam_steps: int
    lbfgs_steps: int


@dataclass(frozen=True)
class InverseBudget:
    adam_steps: int
    lbfgs_cycles: int
    lbfgs_field_steps: int
    lbfgs_material_steps: int


@dataclass(frozen=True)
class TrainingPackage:
    cases: tuple[Case, ...]
    warmup: WarmupBudget
    inverse: InverseBudget

    @property
    def label(self) -> str:
        return "__".join(case.id for case in self.cases)


@dataclass(frozen=True)
class InverseConfig:
    geometry: GeometryConfig
    data: DataConfig
    models: ModelConfig
    loss: LossConfig
    optimization: OptimizationConfig
    sampling: SamplingConfig
    logging: LoggingConfig
    training_packages: tuple[TrainingPackage, ...]
    output_root: Path
    source: Path

    @property
    def all_cases(self) -> tuple[Case, ...]:
        return tuple(sorted({case for package in self.training_packages for case in package.cases}))

    def manifest(self) -> dict[str, Any]:
        result = asdict(self)
        result["source"] = str(self.source)
        result["output_root"] = str(self.output_root)
        result["data"] = {name: str(value) for name, value in asdict(self.data).items()}
        result["training_packages"] = [
            {
                "cases": [case.manifest() for case in package.cases],
                "warmup": asdict(package.warmup),
                "inverse": asdict(package.inverse),
            }
            for package in self.training_packages
        ]
        return result

    @classmethod
    def from_json(cls, path: str | Path) -> "InverseConfig":
        source = Path(path).resolve()
        with source.open(encoding="utf-8") as stream:
            raw = _mapping(json.load(stream), "configuration")
        _keys(
            raw,
            {
                "geometry", "data", "models", "loss", "optimization",
                "sampling", "logging", "training_packages", "output_root",
            },
            "configuration",
        )
        config = cls(
            geometry=_parse_geometry(raw["geometry"]),
            data=_parse_data(raw["data"]),
            models=_parse_models(raw["models"]),
            loss=_parse_loss(raw["loss"]),
            optimization=_parse_optimization(raw["optimization"]),
            sampling=_parse_sampling(raw["sampling"]),
            logging=_parse_logging(raw["logging"]),
            training_packages=_parse_packages(raw["training_packages"]),
            output_root=_path(raw["output_root"]),
            source=source,
        )
        config.validate()
        return config

    def validate(self) -> None:
        geometry = self.geometry
        if min(geometry.height, geometry.half_length, geometry.c0) <= 0.0:
            raise ValueError("Geometry dimensions and c0 must be positive")
        lower, upper = geometry.celerity_ratio_bounds
        if not 0.0 < lower < upper:
            raise ValueError("celerity_ratio_bounds must be ordered and positive")
        if not (lower <= 1.0 <= upper):
            raise ValueError("celerity_ratio_bounds must contain the homogeneous ratio 1")
        if not self.models.field_hidden_layers or not self.models.material_hidden_layers:
            raise ValueError("Both network hidden-layer lists must be non-empty")
        if min(*self.models.field_hidden_layers, *self.models.material_hidden_layers) <= 0:
            raise ValueError("Network widths must be positive")
        if self.models.fourier_features <= 0:
            raise ValueError("fourier_features must be positive")
        for case in self.all_cases:
            transverse = case.mode * math.pi / geometry.height
            k0 = 2.0 * math.pi * case.frequency / geometry.c0
            if k0 <= transverse:
                raise ValueError(f"Incident case {case.id} is evanescent")
        for region in geometry.truth_regions:
            _validate_region(region, geometry)
        for path in asdict(self.data).values():
            if not Path(path).is_file():
                raise FileNotFoundError(f"Missing configured data file: {path}")


def _parse_geometry(value: Any) -> GeometryConfig:
    raw = _mapping(value, "geometry")
    _keys(raw, {"height", "half_length", "c0", "celerity_ratio_bounds", "truth_regions"}, "geometry")
    regions = tuple(_parse_region(item, index) for index, item in enumerate(raw["truth_regions"]))
    return GeometryConfig(
        height=_positive(raw["height"], "geometry.height"),
        half_length=_positive(raw["half_length"], "geometry.half_length"),
        c0=_positive(raw["c0"], "geometry.c0"),
        celerity_ratio_bounds=_tuple_float(raw["celerity_ratio_bounds"], 2, "geometry.celerity_ratio_bounds"),
        truth_regions=regions,
    )


def _parse_region(value: Any, index: int) -> TruthRegion:
    raw = _mapping(value, f"truth_regions[{index}]")
    shape = str(raw.get("shape", ""))
    if shape == "circle":
        _keys(raw, {"shape", "center", "radius", "speed_ratio"}, f"truth_regions[{index}]")
        center = _tuple_float(raw["center"], 2, "region.center", positive=False)
        return TruthRegion(shape, _positive(raw["speed_ratio"], "region.speed_ratio"), center=center, radius=_positive(raw["radius"], "region.radius"))
    if shape == "rectangle":
        _keys(raw, {"shape", "bounds", "speed_ratio"}, f"truth_regions[{index}]")
        bounds = _tuple_float(raw["bounds"], 4, "region.bounds", positive=False)
        return TruthRegion(shape, _positive(raw["speed_ratio"], "region.speed_ratio"), bounds=bounds)
    raise ValueError("Truth region shape must be 'circle' or 'rectangle'")


def _validate_region(region: TruthRegion, geometry: GeometryConfig) -> None:
    if region.shape == "circle":
        assert region.center is not None and region.radius is not None
        x, y = region.center
        radius = region.radius
        if not (-geometry.half_length <= x - radius and x + radius <= geometry.half_length and 0.0 <= y - radius and y + radius <= geometry.height):
            raise ValueError(f"Truth circle {region} lies outside the waveguide")
    else:
        assert region.bounds is not None
        xmin, xmax, ymin, ymax = region.bounds
        if not (-geometry.half_length <= xmin < xmax <= geometry.half_length and 0.0 <= ymin < ymax <= geometry.height):
            raise ValueError(f"Truth rectangle {region} lies outside the waveguide")


def _parse_data(value: Any) -> DataConfig:
    raw = _mapping(value, "data")
    expected = {"boundary_left", "boundary_right", "fem_field", "mass_matrix", "stiffness_matrix", "mesh_nodes", "mesh_triangles"}
    _keys(raw, expected, "data")
    return DataConfig(**{name: _path(raw[name]) for name in expected})


def _parse_models(value: Any) -> ModelConfig:
    raw = _mapping(value, "models")
    _keys(raw, {"fourier_features", "field_hidden_layers", "material_hidden_layers"}, "models")
    return ModelConfig(
        fourier_features=_positive_int(raw["fourier_features"], "models.fourier_features"),
        field_hidden_layers=tuple(_positive_int(item, "models.field_hidden_layers") for item in raw["field_hidden_layers"]),
        material_hidden_layers=tuple(_positive_int(item, "models.material_hidden_layers") for item in raw["material_hidden_layers"]),
    )


def _adaptive_common(raw: Mapping[str, Any], name: str) -> tuple[float, float, int]:
    epsilon = _positive(raw["epsilon"], f"{name}.epsilon")
    alpha = _finite(raw["alpha"], f"{name}.alpha")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"{name}.alpha must lie in [0, 1]")
    interval = _positive_int(raw["update_interval_adam"], f"{name}.update_interval_adam")
    return epsilon, alpha, interval


def _parse_loss(value: Any) -> LossConfig:
    raw = _mapping(value, "loss")
    _keys(raw, {"field_weights", "field_adweights", "material_adweights", "tv"}, "loss")
    field_raw = _mapping(raw["field_adweights"], "loss.field_adweights")
    _keys(field_raw, {"epsilon", "alpha", "initial_lambdas", "custom_weights", "update_interval_adam"}, "loss.field_adweights")
    feps, falpha, finterval = _adaptive_common(field_raw, "loss.field_adweights")
    material_raw = _mapping(raw["material_adweights"], "loss.material_adweights")
    _keys(material_raw, {"epsilon", "alpha", "initial_lambda", "custom_weight", "update_interval_adam"}, "loss.material_adweights")
    meps, malpha, minterval = _adaptive_common(material_raw, "loss.material_adweights")
    tv_raw = _mapping(raw["tv"], "loss.tv")
    _keys(tv_raw, {"weight", "epsilon_squared"}, "loss.tv")
    field_weights = _tuple_non_negative(raw["field_weights"], 4, "loss.field_weights")
    custom_field_weights = _tuple_non_negative(
        field_raw["custom_weights"], 4, "field_adweights.custom_weights"
    )
    if not any(field_weights):
        raise ValueError("loss.field_weights must contain a positive value")
    if not any(custom_field_weights):
        raise ValueError("field_adweights.custom_weights must contain a positive value")
    return LossConfig(
        field_weights=field_weights,
        field_adweights=FieldAdweightsConfig(
            feps, falpha,
            _tuple_non_negative(field_raw["initial_lambdas"], 4, "field_adweights.initial_lambdas"),
            custom_field_weights,
            finterval,
        ),
        material_adweights=MaterialAdweightsConfig(
            meps, malpha,
            _positive(material_raw["initial_lambda"], "material_adweights.initial_lambda"),
            _positive(material_raw["custom_weight"], "material_adweights.custom_weight"),
            minterval,
        ),
        tv=TVConfig(_non_negative(tv_raw["weight"], "loss.tv.weight"), _positive(tv_raw["epsilon_squared"], "loss.tv.epsilon_squared")),
    )


def _parse_optimization(value: Any) -> OptimizationConfig:
    raw = _mapping(value, "optimization")
    expected = {"field_learning_rate", "material_learning_rate", "sigma_learning_rate", "sigma_decay_fraction", "sigma_cosine_alpha", "data_initial_factor", "data_transition_fraction"}
    _keys(raw, expected, "optimization")
    result = OptimizationConfig(
        field_learning_rate=_positive(raw["field_learning_rate"], "optimization.field_learning_rate"),
        material_learning_rate=_positive(raw["material_learning_rate"], "optimization.material_learning_rate"),
        sigma_learning_rate=_positive(raw["sigma_learning_rate"], "optimization.sigma_learning_rate"),
        sigma_decay_fraction=_non_negative(raw["sigma_decay_fraction"], "optimization.sigma_decay_fraction"),
        sigma_cosine_alpha=_finite(raw["sigma_cosine_alpha"], "optimization.sigma_cosine_alpha"),
        data_initial_factor=_non_negative(raw["data_initial_factor"], "optimization.data_initial_factor"),
        data_transition_fraction=_positive(raw["data_transition_fraction"], "optimization.data_transition_fraction"),
    )
    for name in ("sigma_decay_fraction", "data_initial_factor", "data_transition_fraction"):
        if getattr(result, name) > 1.0:
            raise ValueError(f"optimization.{name} must lie in (0, 1]")
    if not 0.0 <= result.sigma_cosine_alpha <= 1.0:
        raise ValueError("optimization.sigma_cosine_alpha must lie in [0, 1]")
    return result


def _parse_sampling(value: Any) -> SamplingConfig:
    raw = _mapping(value, "sampling")
    _keys(raw, {"adam", "monitor", "lbfgs", "sobol_scramble", "sobol_seed_offset"}, "sampling")
    lbfgs = _tuple_int(raw["lbfgs"], 3, "sampling.lbfgs")
    if any(count & (count - 1) for count in lbfgs):
        raise ValueError("All sampling.lbfgs counts must be powers of two")
    if not isinstance(raw["sobol_scramble"], bool):
        raise ValueError("sampling.sobol_scramble must be a boolean")
    seed_offset = raw["sobol_seed_offset"]
    if isinstance(seed_offset, bool) or not isinstance(seed_offset, int):
        raise ValueError("sampling.sobol_seed_offset must be an integer")
    return SamplingConfig(
        adam=_tuple_int(raw["adam"], 3, "sampling.adam"),
        monitor=_tuple_int(raw["monitor"], 3, "sampling.monitor"),
        lbfgs=lbfgs,
        sobol_scramble=raw["sobol_scramble"],
        sobol_seed_offset=seed_offset,
    )


def _parse_logging(value: Any) -> LoggingConfig:
    raw = _mapping(value, "logging")
    expected = {"loss_interval_adam", "loss_interval_lbfgs", "pressure_gradient_interval_adam", "sigma_interval_adam", "print_interval_adam", "material_snapshot_fractions", "snapshot_grid", "prediction_batch_size"}
    _keys(raw, expected, "logging")
    fractions = tuple(_finite(item, "logging.material_snapshot_fractions") for item in raw["material_snapshot_fractions"])
    if len(set(fractions)) != len(fractions) or any(not 0.0 < item <= 1.0 for item in fractions):
        raise ValueError("material_snapshot_fractions must be unique values in (0, 1]")
    return LoggingConfig(
        loss_interval_adam=_positive_int(raw["loss_interval_adam"], "logging.loss_interval_adam"),
        loss_interval_lbfgs=_positive_int(raw["loss_interval_lbfgs"], "logging.loss_interval_lbfgs"),
        pressure_gradient_interval_adam=_positive_int(raw["pressure_gradient_interval_adam"], "logging.pressure_gradient_interval_adam"),
        sigma_interval_adam=_positive_int(raw["sigma_interval_adam"], "logging.sigma_interval_adam"),
        print_interval_adam=_positive_int(raw["print_interval_adam"], "logging.print_interval_adam"),
        material_snapshot_fractions=tuple(sorted(fractions)),
        snapshot_grid=_tuple_int(raw["snapshot_grid"], 2, "logging.snapshot_grid"),
        prediction_batch_size=_positive_int(raw["prediction_batch_size"], "logging.prediction_batch_size"),
    )


def _parse_cases(value: Any, name: str) -> tuple[Case, ...]:
    raw = _mapping(value, name)
    cases: list[Case] = []
    for incidence_raw, frequency_map_raw in raw.items():
        incidence = int(incidence_raw)
        frequency_map = _mapping(frequency_map_raw, f"{name}.{incidence_raw}")
        for frequency_raw, modes_raw in frequency_map.items():
            frequency = float(frequency_raw)
            modes = tuple(_positive_int(mode, f"{name}.modes", allow_zero=True) for mode in modes_raw)
            if not modes or len(set(modes)) != len(modes):
                raise ValueError(f"{name} mode lists must be non-empty and unique")
            cases.extend(Case(frequency, mode, incidence) for mode in modes)
    result = tuple(sorted(cases))
    if not result or len(set(result)) != len(result):
        raise ValueError(f"{name} must define unique cases")
    return result


def _parse_packages(value: Any) -> tuple[TrainingPackage, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise ValueError("training_packages must be a non-empty list")
    packages = []
    for index, item in enumerate(value):
        raw = _mapping(item, f"training_packages[{index}]")
        _keys(raw, {"cases", "warmup", "inverse"}, f"training_packages[{index}]")
        warmup_raw = _mapping(raw["warmup"], "warmup")
        _keys(warmup_raw, {"adam_steps", "lbfgs_steps"}, "warmup")
        inverse_raw = _mapping(raw["inverse"], "inverse")
        _keys(inverse_raw, {"adam_steps", "lbfgs_cycles", "lbfgs_field_steps", "lbfgs_material_steps"}, "inverse")
        packages.append(
            TrainingPackage(
                cases=_parse_cases(raw["cases"], f"training_packages[{index}].cases"),
                warmup=WarmupBudget(
                    _positive_int(warmup_raw["adam_steps"], "warmup.adam_steps", allow_zero=True),
                    _positive_int(warmup_raw["lbfgs_steps"], "warmup.lbfgs_steps", allow_zero=True),
                ),
                inverse=InverseBudget(
                    _positive_int(inverse_raw["adam_steps"], "inverse.adam_steps", allow_zero=True),
                    _positive_int(inverse_raw["lbfgs_cycles"], "inverse.lbfgs_cycles", allow_zero=True),
                    _positive_int(inverse_raw["lbfgs_field_steps"], "inverse.lbfgs_field_steps", allow_zero=True),
                    _positive_int(inverse_raw["lbfgs_material_steps"], "inverse.lbfgs_material_steps", allow_zero=True),
                ),
            )
        )
        inverse = packages[-1].inverse
        if inverse.lbfgs_cycles == 0 and (
            inverse.lbfgs_field_steps or inverse.lbfgs_material_steps
        ):
            raise ValueError("L-BFGS block steps require inverse.lbfgs_cycles > 0")
        if inverse.lbfgs_cycles > 0 and not (
            inverse.lbfgs_field_steps or inverse.lbfgs_material_steps
        ):
            raise ValueError("A positive L-BFGS cycle count requires a non-empty block")
    return tuple(packages)
