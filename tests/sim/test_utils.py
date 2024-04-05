import pytest
from logistics_envs.sim.utils import generate_colors


def test_generate_colors():
    n = 10
    colors = generate_colors(n)
    assert len(colors) == n

    n = 200
    colors = generate_colors(n)
    assert len(colors) == n

    n = 1
    colors = generate_colors(n)
    assert len(colors) == n

    n = 0
    with pytest.raises(ValueError):
        generate_colors(n)
