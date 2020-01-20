import numpy as np
from astropy.modeling.models import Sersic1D, Sersic2D
from astropy.modeling.core import Fittable1DModel, Parameter
import matplotlib.pyplot as plt
from astropy.visualization import hist
from scipy.stats import rv_continuous


class Sersic1D(Fittable1DModel, rv_continuous):
    amplitude = Parameter(default=1)
    r_eff = Parameter(default=1)
    n = Parameter(default=4)
    _gammaincinv = None

    @classmethod
    def evaluate(cls, r, amplitude, r_eff, n):
        """One dimensional Sersic profile function."""

        if cls._gammaincinv is None:
            try:
                from scipy.special import gammaincinv
                cls._gammaincinv = gammaincinv
            except ValueError:
                raise ImportError('Sersic1D model requires scipy > 0.11.')

        return (amplitude * np.exp(
            -cls._gammaincinv(2 * n, 0.5) * ((r / r_eff) ** (1 / n) - 1)))
    
    def _pdf(self, r):
        s = Sersic1D(amplitude=self.amplitude, r_eff=self.r_eff, n=self.n)
        return s(r)

s = Sersic1D(amplitude=1, r_eff=5, n=4, name="sersic")

n = s.rvs(size=100)

print(n)



