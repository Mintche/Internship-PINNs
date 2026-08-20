"""Ground-truth sound-speed maps shared by checkpoint diagnostics.

The inversion scripts identify a synthetic geometry through the
``defect_name`` stored in their checkpoint metadata.  Keeping the matching
map in one registry prevents the full-field and scattered-field comparison
tools from silently using different hard-coded geometries.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


MaskBuilder = Callable[[np.ndarray, np.ndarray], np.ndarray]


def _rectangle(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> MaskBuilder:
    def mask(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return (
            (x >= x_min)
            & (x <= x_max)
            & (y >= y_min)
            & (y <= y_max)
        )

    return mask


def _circle(center_x: float, center_y: float, radius: float) -> MaskBuilder:
    def mask(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return (x - center_x) ** 2 + (y - center_y) ** 2 <= radius**2

    return mask


def _two_circles(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    left = _circle(-0.3, 0.3, 0.1)(x, y)
    right = _circle(0.3, 0.3, 0.1)(x, y)
    return left | right


def _homogeneous(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.zeros(np.broadcast_shapes(x.shape, y.shape), dtype=bool)


# These definitions mirror the parameter values in FEM/data/*.geo.  Aliases
# are explicit so a typo in checkpoint metadata is rejected instead of being
# evaluated against an unrelated map.
DEFECT_MASKS: dict[str, MaskBuilder] = {
    "homogeneous": _homogeneous,
    "bar": _rectangle(-0.2, 0.2, 0.0, 0.6),
    "barhalf": _rectangle(-0.2, 0.2, 0.0, 0.3),
    "barhalfup": _rectangle(-0.2, 0.2, 0.3, 0.6),
    "barthird": _rectangle(-0.2, 0.2, 0.05, 0.25),
    "circlecenter": _circle(0.0, 0.3, 0.1),
    "circlebottomleft": _circle(-0.2, 0.2, 0.1),
    # The suffix refers to the longer guide, not to a larger circle.
    "circlebottomleftlarge": _circle(-0.2, 0.2, 0.1),
    "circlebottomright": _circle(0.2, 0.2, 0.1),
    "2circ": _two_circles,
}


def registered_defects() -> tuple[str, ...]:
    """Return the synthetic defect names supported by the registry."""
    return tuple(sorted(DEFECT_MASKS))


def build_registered_sound_speed(
    x: np.ndarray,
    y: np.ndarray,
    *,
    defect_name: str,
    c0: float,
    contrast_ratio: float,
) -> np.ndarray:
    """Evaluate a registered single-contrast geometry on physical points.

    Unknown names are rejected deliberately.  Multi-material maps need an
    explicit extension carrying one contrast per region and must not be
    approximated by this single-ratio helper.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    try:
        x, y = np.broadcast_arrays(x, y)
    except ValueError as error:
        raise ValueError("x and y cannot be broadcast to a common grid") from error

    if not np.isfinite(c0) or c0 <= 0.0:
        raise ValueError("c0 must be finite and strictly positive")
    if not np.isfinite(contrast_ratio) or contrast_ratio <= 0.0:
        raise ValueError("contrast_ratio must be finite and strictly positive")

    normalized_name = str(defect_name)
    try:
        mask_builder = DEFECT_MASKS[normalized_name]
    except KeyError as error:
        supported = ", ".join(registered_defects())
        raise ValueError(
            f"Unknown ground-truth defect {normalized_name!r}. "
            f"Supported defects: {supported}. Add an explicit geometry to "
            "tools/ground_truth.py before computing reconstruction metrics."
        ) from error

    mask = np.asarray(mask_builder(x, y), dtype=bool)
    if mask.shape != x.shape:
        raise ValueError(
            f"Ground-truth mask for {normalized_name!r} has shape {mask.shape}, "
            f"expected {x.shape}"
        )

    sound_speed = np.full(x.shape, float(c0), dtype=np.float64)
    sound_speed[mask] = float(c0) * float(contrast_ratio)
    return sound_speed
