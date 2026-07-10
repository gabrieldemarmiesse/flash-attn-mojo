"""Pytest fixtures + beartype claw setup."""

# Runtime type checking under test. `beartype_this_package` would only
# cover *this* conftest module, so we explicitly target the package
# under test with `beartype_package`. This must happen *before*
# `flash_attn_mojo` (or any of its submodules) is imported — the claw
# installs a sys.meta_path hook that rewrites every function in the
# package at import time.
from beartype.claw import beartype_package  # noqa: I001

beartype_package("flash_attn_mojo")

import pytest  # noqa: E402
import torch  # noqa: E402

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA device"
)

_mps = getattr(torch.backends, "mps", None)
requires_mps = pytest.mark.skipif(
    _mps is None or not _mps.is_available(),
    reason="needs an Apple GPU (MPS)",
)


@pytest.fixture(autouse=True)
def _seed_rng():
    """Every test gets the same RNG state — failures near tolerance
    boundaries stay reproducible."""
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
