import numpy as np
import pytest

from diffusion import calculate_stable_time_step
#from diffusion import diffuse_until
#from diffusion import step_like


def test_time_step_is_float():
    time_step = calculate_stable_time_step(1, 1)
    assert isinstance(time_step, float)