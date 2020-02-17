"""
Aditional 1D models
"""



import numpy as np

from astropy import units as u
from astropy.units import Quantity, UnitsError
from astropy.utils.decorators import deprecated
from astropy.modeling.core import (Fittable1DModel, Fittable2DModel,
                                   ModelDefinitionError)

from astropy.modeling.parameters import Parameter, InputParameterError
from astropy.modeling.utils import ellipse_extent


TWOPI = 2 * np.pi
FLOAT_EPSILON = float(np.finfo(np.float32).tiny)
GAUSSIAN_SIGMA_TO_FWHM = 2.0 * np.sqrt(2.0 * np.log(2.0))


# ----- Models copied from astropy.modeling.functional_models
# ----- They are here for completeness and templating


class KingProjectedAnalytic1D(Fittable1DModel):
    """
    Projected (surface density) analytic King Model.


    Parameters
    ----------
    amplitude : float
        Amplitude or scaling factor.
    r_core : float
        Core radius (f(r_c) ~ 0.5 f_0)
    r_tide : float
        Tidal radius.


    Notes
    -----

    This model approximates a King model with an analytic function. The derivation of this
    equation can be found in King '62 (equation 14). This is just an approximation of the
    full model and the parameters derived from this model should be taken with caution.
    It usually works for models with a concentration (c = log10(r_t/r_c) paramter < 2.

    Model formula:

    .. math::

        f(x) = A r_c^2  \\left(\\frac{1}{\\sqrt{(x^2 + r_c^2)}} -
        \\frac{1}{\\sqrt{(r_t^2 + r_c^2)}}\\right)^2

    Examples
    --------
    .. plot::
        :include-source:

        import numpy as np
        from astropy.modeling.models import KingProjectedAnalytic1D
        import matplotlib.pyplot as plt

        plt.figure()
        rt_list = [1, 2, 5, 10, 20]
        for rt in rt_list:
            r = np.linspace(0.1, rt, 100)

            mod = KingProjectedAnalytic1D(amplitude = 1, r_core = 1., r_tide = rt)
            sig = mod(r)


            plt.loglog(r, sig/sig[0], label='c ~ {:0.2f}'.format(mod.concentration))

        plt.xlabel("r")
        plt.ylabel(r"$\\sigma/\\sigma_0$")
        plt.legend()
        plt.show()

    References
    ----------
    .. [1] http://articles.adsabs.harvard.edu/pdf/1962AJ.....67..471K
    """

    amplitude = Parameter(default=1, bounds=(FLOAT_EPSILON, None))
    r_core = Parameter(default=1, bounds=(FLOAT_EPSILON, None))
    r_tide = Parameter(default=2, bounds=(FLOAT_EPSILON, None))

    @property
    def concentration(self):
        """Concentration parameter of the king model"""
        return np.log10(np.abs(self.r_tide/self.r_core))

    @staticmethod
    def evaluate(x, amplitude, r_core, r_tide):
        """
        Analytic King model function.
        """

        result = amplitude * r_core ** 2 * (1/np.sqrt(x ** 2 + r_core ** 2) -
                                            1/np.sqrt(r_tide ** 2 + r_core ** 2)) ** 2

        # Set invalid r values to 0
        bounds = (x >= r_tide) | (x<0)
        result[bounds] = result[bounds] * 0.

        return result

    @staticmethod
    def fit_deriv(x, amplitude, r_core, r_tide):
        """
        Analytic King model function derivatives.
        """
        d_amplitude = r_core ** 2 * (1/np.sqrt(x ** 2 + r_core ** 2) -
                                     1/np.sqrt(r_tide ** 2 + r_core ** 2)) ** 2

        d_r_core = 2 * amplitude * r_core ** 2 * (r_core/(r_core ** 2 + r_tide ** 2) ** (3/2) -
                                                  r_core/(r_core ** 2 + x ** 2) ** (3/2)) * \
                   (1./np.sqrt(r_core ** 2 + x ** 2) - 1./np.sqrt(r_core ** 2 + r_tide ** 2)) + \
                   2 * amplitude * r_core * (1./np.sqrt(r_core ** 2 + x ** 2) -
                                             1./np.sqrt(r_core ** 2 + r_tide ** 2)) ** 2

        d_r_tide = (2 * amplitude * r_core ** 2 * r_tide *
                    (1./np.sqrt(r_core ** 2 + x ** 2) -
                     1./np.sqrt(r_core ** 2 + r_tide ** 2)))/(r_core ** 2 + r_tide ** 2) ** (3/2)

        # Set invalid r values to 0
        bounds = (x >= r_tide) | (x < 0)
        d_amplitude[bounds] = d_amplitude[bounds]*0
        d_r_core[bounds] = d_r_core[bounds]*0
        d_r_tide[bounds] = d_r_tide[bounds]*0

        return [d_amplitude, d_r_core, d_r_tide]


    @property
    def bounding_box(self):
        """
        Tuple defining the default ``bounding_box`` limits.

        The model is not defined for r > r_tide.

        ``(r_low, r_high)``
        """

        return (0 * self.r_tide, 1 * self.r_tide)

    @property
    def input_units(self):
        if self.r_core.unit is None:
            return None
        else:
            return {'x': self.r_core.unit}

    def _parameter_units_for_data_units(self, inputs_unit, outputs_unit):
        return {'r_core': inputs_unit['x'],
                'r_tide': inputs_unit['x'],
                'amplitude': outputs_unit['y']}



class Sersic1D(Fittable1DModel):
    r"""
    One dimensional Sersic surface brightness profile.

    Parameters
    ----------
    amplitude : float
        Surface brightness at r_eff.
    r_eff : float
        Effective (half-light) radius
    n : float
        Sersic Index.

    See Also
    --------
    Gaussian1D, Moffat1D, Lorentz1D

    Notes
    -----
    Model formula:

    .. math::

        I(r)=I_e\exp\left\{-b_n\left[\left(\frac{r}{r_{e}}\right)^{(1/n)}-1\right]\right\}

    The constant :math:`b_n` is defined such that :math:`r_e` contains half the total
    luminosity, and can be solved for numerically.

    .. math::

        \Gamma(2n) = 2\gamma (b_n,2n)

    Examples
    --------
    .. plot::
        :include-source:

        import numpy as np
        from astropy.modeling.models import Sersic1D
        import matplotlib.pyplot as plt

        plt.figure()
        plt.subplot(111, xscale='log', yscale='log')
        s1 = Sersic1D(amplitude=1, r_eff=5)
        r=np.arange(0, 100, .01)

        for n in range(1, 10):
             s1.n = n
             plt.plot(r, s1(r), color=str(float(n) / 15))

        plt.axis([1e-1, 30, 1e-2, 1e3])
        plt.xlabel('log Radius')
        plt.ylabel('log Surface Brightness')
        plt.text(.25, 1.5, 'n=1')
        plt.text(.25, 300, 'n=10')
        plt.xticks([])
        plt.yticks([])
        plt.show()

    References
    ----------
    .. [1] http://ned.ipac.caltech.edu/level5/March05/Graham/Graham2.html
    """

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


    @property
    def input_units(self):
        if self.r_eff.unit is None:
            return None
        else:
            return {'x': self.r_eff.unit}

    def _parameter_units_for_data_units(self, inputs_unit, outputs_unit):
        return {'r_eff': inputs_unit['x'],
                'amplitude': outputs_unit['y']}


# ----------------------------------- End -------------------------------------------------------


class NFW1D(Fittable1DModel):
    pass


class SchechterFunc(Fittable1DModel):
    pass


class PowerLaw1D(Fittable1DModel):
    pass



