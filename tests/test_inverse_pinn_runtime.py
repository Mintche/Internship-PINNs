from __future__ import annotations

from pathlib import Path

import jax

from inverse_PINN.runtime import configure_jax_compilation_cache


def test_persistent_jax_compilation_cache_is_enabled():
    directory = configure_jax_compilation_cache()
    assert directory.is_dir()
    assert Path(jax.config.jax_compilation_cache_dir).resolve() == directory
    assert jax.config.jax_enable_compilation_cache
    assert jax.config.jax_persistent_cache_min_compile_time_secs == 0.0
