"""Process-wide JAX runtime configuration shared by training and inference."""

from __future__ import annotations

from pathlib import Path

import jax


DEFAULT_COMPILATION_CACHE = Path(__file__).resolve().parent / "cache"


def configure_jax_compilation_cache(
    directory: Path | str | None = None,
) -> Path:
    """Enable JAX's persistent compilation cache and return its directory.

    An explicitly configured JAX cache (for example through
    ``JAX_COMPILATION_CACHE_DIR``) has priority. Otherwise the project-local
    ``inverse_PINN/cache`` directory is used. Caching even short compilations
    makes reduced campaigns and tests benefit from subsequent processes too.
    """
    configured = jax.config.jax_compilation_cache_dir
    cache_directory = Path(configured) if configured else Path(
        directory if directory is not None else DEFAULT_COMPILATION_CACHE
    )
    cache_directory = cache_directory.expanduser().resolve()
    cache_directory.mkdir(parents=True, exist_ok=True)
    if not configured:
        jax.config.update("jax_compilation_cache_dir", str(cache_directory))
    jax.config.update("jax_enable_compilation_cache", True)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
    return cache_directory
