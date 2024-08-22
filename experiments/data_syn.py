# NORMAL
time_normal = np.arange(10000)
mag_normal = np.random.normal(size=10000)
error_normal = np.random.normal(loc=1, scale=0.008, size=10000)

mag_normal2 = np.random.normal(size=10000)
error_normal2 = np.random.normal(loc=1, scale=0.008, size=10000)

lc_normal = {
    "time": time_normal,
    "magnitude": mag_normal,
    "error": error_normal,
    "magnitude2": mag_normal2,
    "aligned_time": time_normal,
    "aligned_magnitude": mag_normal,
    "aligned_magnitude2": mag_normal2,
    "aligned_error": error_normal,
    "aligned_error2": error_normal2,
}

# PERIODIC
import numpy as np

rand = np.random.RandomState(42)
time_periodic = 100 * rand.rand(100)
mag_periodic = np.sin(2 * np.pi * time_periodic) + 0.1 * rand.randn(100)

lc_periodic = {"time": time_periodic, "magnitude": mag_periodic}

# UNIFORM
lc_uniform = {
    "time": np.arange(10000),
    "magnitude": np.random.uniform(size=10000),
}
