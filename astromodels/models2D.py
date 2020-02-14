"""
Aditional 2D models
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

# ----- Models copied from astropy.modeling.functional_models  -------------
# ----- They are here for completeness and templating     ------------------

class Gaussian2D(Fittable2DModel):
    r"""
    Two dimensional Gaussian model.

    Parameters
    ----------
    amplitude : float
        Amplitude of the Gaussian.
    x_mean : float
        Mean of the Gaussian in x.
    y_mean : float
        Mean of the Gaussian in y.
    x_stddev : float or None
        Standard deviation of the Gaussian in x before rotating by theta. Must
        be None if a covariance matrix (``cov_matrix``) is provided. If no
        ``cov_matrix`` is given, ``None`` means the default value (1).
    y_stddev : float or None
        Standard deviation of the Gaussian in y before rotating by theta. Must
        be None if a covariance matrix (``cov_matrix``) is provided. If no
        ``cov_matrix`` is given, ``None`` means the default value (1).
    theta : float, optional
        Rotation angle in radians. The rotation angle increases
        counterclockwise.  Must be None if a covariance matrix (``cov_matrix``)
        is provided. If no ``cov_matrix`` is given, ``None`` means the default
        value (0).
    cov_matrix : ndarray, optional
        A 2x2 covariance matrix. If specified, overrides the ``x_stddev``,
        ``y_stddev``, and ``theta`` defaults.

    Notes
    -----
    Model formula:

        .. math::

            f(x, y) = A e^{-a\left(x - x_{0}\right)^{2}  -b\left(x - x_{0}\right)
            \left(y - y_{0}\right)  -c\left(y - y_{0}\right)^{2}}

    Using the following definitions:

        .. math::
            a = \left(\frac{\cos^{2}{\left (\theta \right )}}{2 \sigma_{x}^{2}} +
            \frac{\sin^{2}{\left (\theta \right )}}{2 \sigma_{y}^{2}}\right)

            b = \left(\frac{\sin{\left (2 \theta \right )}}{2 \sigma_{x}^{2}} -
            \frac{\sin{\left (2 \theta \right )}}{2 \sigma_{y}^{2}}\right)

            c = \left(\frac{\sin^{2}{\left (\theta \right )}}{2 \sigma_{x}^{2}} +
            \frac{\cos^{2}{\left (\theta \right )}}{2 \sigma_{y}^{2}}\right)

    If using a ``cov_matrix``, the model is of the form:
        .. math::
            f(x, y) = A e^{-0.5 \left(\vec{x} - \vec{x}_{0}\right)^{T} \Sigma^{-1} \left(\vec{x} - \vec{x}_{0}\right)}

    where :math:`\vec{x} = [x, y]`, :math:`\vec{x}_{0} = [x_{0}, y_{0}]`,
    and :math:`\Sigma` is the covariance matrix:

        .. math::
            \Sigma = \left(\begin{array}{ccc}
            \sigma_x^2               & \rho \sigma_x \sigma_y \\
            \rho \sigma_x \sigma_y   & \sigma_y^2
            \end{array}\right)

    :math:`\rho` is the correlation between ``x`` and ``y``, which should
    be between -1 and +1.  Positive correlation corresponds to a
    ``theta`` in the range 0 to 90 degrees.  Negative correlation
    corresponds to a ``theta`` in the range of 0 to -90 degrees.

    See [1]_ for more details about the 2D Gaussian function.

    See Also
    --------
    Gaussian1D, Box2D, Moffat2D

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Gaussian_function
    """

    amplitude = Parameter(default=1)
    x_mean = Parameter(default=0)
    y_mean = Parameter(default=0)
    x_stddev = Parameter(default=1)
    y_stddev = Parameter(default=1)
    theta = Parameter(default=0.0)

    def __init__(self, amplitude=amplitude.default, x_mean=x_mean.default,
                 y_mean=y_mean.default, x_stddev=None, y_stddev=None,
                 theta=None, cov_matrix=None, **kwargs):
        if cov_matrix is None:
            if x_stddev is None:
                x_stddev = self.__class__.x_stddev.default
            if y_stddev is None:
                y_stddev = self.__class__.y_stddev.default
            if theta is None:
                theta = self.__class__.theta.default
        else:
            if x_stddev is not None or y_stddev is not None or theta is not None:
                raise InputParameterError("Cannot specify both cov_matrix and "
                                          "x/y_stddev/theta")
            else:
                # Compute principle coordinate system transformation
                cov_matrix = np.array(cov_matrix)

                if cov_matrix.shape != (2, 2):
                    # TODO: Maybe it should be possible for the covariance matrix
                    # to be some (x, y, ..., z, 2, 2) array to be broadcast with
                    # other parameters of shape (x, y, ..., z)
                    # But that's maybe a special case to work out if/when needed
                    raise ValueError("Covariance matrix must be 2x2")

                eig_vals, eig_vecs = np.linalg.eig(cov_matrix)
                x_stddev, y_stddev = np.sqrt(eig_vals)
                y_vec = eig_vecs[:, 0]
                theta = np.arctan2(y_vec[1], y_vec[0])

        # Ensure stddev makes sense if its bounds are not explicitly set.
        # stddev must be non-zero and positive.
        # TODO: Investigate why setting this in Parameter above causes
        #       convolution tests to hang.
        kwargs.setdefault('bounds', {})
        kwargs['bounds'].setdefault('x_stddev', (FLOAT_EPSILON, None))
        kwargs['bounds'].setdefault('y_stddev', (FLOAT_EPSILON, None))

        super().__init__(
            amplitude=amplitude, x_mean=x_mean, y_mean=y_mean,
            x_stddev=x_stddev, y_stddev=y_stddev, theta=theta, **kwargs)

    @property
    def x_fwhm(self):
        """Gaussian full width at half maximum in X."""
        return self.x_stddev * GAUSSIAN_SIGMA_TO_FWHM

    @property
    def y_fwhm(self):
        """Gaussian full width at half maximum in Y."""
        return self.y_stddev * GAUSSIAN_SIGMA_TO_FWHM

    def bounding_box(self, factor=5.5):
        """
        Tuple defining the default ``bounding_box`` limits in each dimension,
        ``((y_low, y_high), (x_low, x_high))``

        The default offset from the mean is 5.5-sigma, corresponding
        to a relative error < 1e-7. The limits are adjusted for rotation.

        Parameters
        ----------
        factor : float, optional
            The multiple of `x_stddev` and `y_stddev` used to define the limits.
            The default is 5.5.

        Examples
        --------
        >>> from astropy.modeling.models import Gaussian2D
        >>> model = Gaussian2D(x_mean=0, y_mean=0, x_stddev=1, y_stddev=2)
        >>> model.bounding_box
        ((-11.0, 11.0), (-5.5, 5.5))

        This range can be set directly (see: `Model.bounding_box
        <astropy.modeling.Model.bounding_box>`) or by using a different factor
        like:

        >>> model.bounding_box = model.bounding_box(factor=2)
        >>> model.bounding_box
        ((-4.0, 4.0), (-2.0, 2.0))
        """

        a = factor * self.x_stddev
        b = factor * self.y_stddev
        theta = self.theta.value
        dx, dy = ellipse_extent(a, b, theta)

        return ((self.y_mean - dy, self.y_mean + dy),
                (self.x_mean - dx, self.x_mean + dx))

    @staticmethod
    def evaluate(x, y, amplitude, x_mean, y_mean, x_stddev, y_stddev, theta):
        """Two dimensional Gaussian function"""

        cost2 = np.cos(theta) ** 2
        sint2 = np.sin(theta) ** 2
        sin2t = np.sin(2. * theta)
        xstd2 = x_stddev ** 2
        ystd2 = y_stddev ** 2
        xdiff = x - x_mean
        ydiff = y - y_mean
        a = 0.5 * ((cost2 / xstd2) + (sint2 / ystd2))
        b = 0.5 * ((sin2t / xstd2) - (sin2t / ystd2))
        c = 0.5 * ((sint2 / xstd2) + (cost2 / ystd2))
        return amplitude * np.exp(-((a * xdiff ** 2) + (b * xdiff * ydiff) +
                                    (c * ydiff ** 2)))

    @staticmethod
    def fit_deriv(x, y, amplitude, x_mean, y_mean, x_stddev, y_stddev, theta):
        """Two dimensional Gaussian function derivative with respect to parameters"""

        cost = np.cos(theta)
        sint = np.sin(theta)
        cost2 = np.cos(theta) ** 2
        sint2 = np.sin(theta) ** 2
        cos2t = np.cos(2. * theta)
        sin2t = np.sin(2. * theta)
        xstd2 = x_stddev ** 2
        ystd2 = y_stddev ** 2
        xstd3 = x_stddev ** 3
        ystd3 = y_stddev ** 3
        xdiff = x - x_mean
        ydiff = y - y_mean
        xdiff2 = xdiff ** 2
        ydiff2 = ydiff ** 2
        a = 0.5 * ((cost2 / xstd2) + (sint2 / ystd2))
        b = 0.5 * ((sin2t / xstd2) - (sin2t / ystd2))
        c = 0.5 * ((sint2 / xstd2) + (cost2 / ystd2))
        g = amplitude * np.exp(-((a * xdiff2) + (b * xdiff * ydiff) +
                                 (c * ydiff2)))
        da_dtheta = (sint * cost * ((1. / ystd2) - (1. / xstd2)))
        da_dx_stddev = -cost2 / xstd3
        da_dy_stddev = -sint2 / ystd3
        db_dtheta = (cos2t / xstd2) - (cos2t / ystd2)
        db_dx_stddev = -sin2t / xstd3
        db_dy_stddev = sin2t / ystd3
        dc_dtheta = -da_dtheta
        dc_dx_stddev = -sint2 / xstd3
        dc_dy_stddev = -cost2 / ystd3
        dg_dA = g / amplitude
        dg_dx_mean = g * ((2. * a * xdiff) + (b * ydiff))
        dg_dy_mean = g * ((b * xdiff) + (2. * c * ydiff))
        dg_dx_stddev = g * (-(da_dx_stddev * xdiff2 +
                              db_dx_stddev * xdiff * ydiff +
                              dc_dx_stddev * ydiff2))
        dg_dy_stddev = g * (-(da_dy_stddev * xdiff2 +
                              db_dy_stddev * xdiff * ydiff +
                              dc_dy_stddev * ydiff2))
        dg_dtheta = g * (-(da_dtheta * xdiff2 +
                           db_dtheta * xdiff * ydiff +
                           dc_dtheta * ydiff2))
        return [dg_dA, dg_dx_mean, dg_dy_mean, dg_dx_stddev, dg_dy_stddev,
                dg_dtheta]


    @property
    def input_units(self):
        if self.x_mean.unit is None and self.y_mean.unit is None:
            return None
        else:
            return {'x': self.x_mean.unit,
                    'y': self.y_mean.unit}

    def _parameter_units_for_data_units(self, inputs_unit, outputs_unit):
        # Note that here we need to make sure that x and y are in the same
        # units otherwise this can lead to issues since rotation is not well
        # defined.
        if inputs_unit['x'] != inputs_unit['y']:
            raise UnitsError("Units of 'x' and 'y' inputs should match")
        return {'x_mean': inputs_unit['x'],
                'y_mean': inputs_unit['x'],
                'x_stddev': inputs_unit['x'],
                'y_stddev': inputs_unit['x'],
                'theta': u.rad,
                'amplitude': outputs_unit['z']}


class Sersic2D(Fittable2DModel):
    r"""
    Two dimensional Sersic surface brightness profile.

    Parameters
    ----------
    amplitude : float
        Surface brightness at r_eff.
    r_eff : float
        Effective (half-light) radius
    n : float
        Sersic Index.
    x_0 : float, optional
        x position of the center.
    y_0 : float, optional
        y position of the center.
    ellip : float, optional
        Ellipticity.
    theta : float, optional
        Rotation angle in radians, counterclockwise from
        the positive x-axis.

    See Also
    --------
    Gaussian2D, Moffat2D

    Notes
    -----
    Model formula:

    .. math::

        I(x,y) = I(r) = I_e\exp\left\{-b_n\left[\left(\frac{r}{r_{e}}\right)^{(1/n)}-1\right]\right\}

    The constant :math:`b_n` is defined such that :math:`r_e` contains half the total
    luminosity, and can be solved for numerically.

    .. math::

        \Gamma(2n) = 2\gamma (b_n,2n)

    Examples
    --------
    .. plot::
        :include-source:

        import numpy as np
        from astropy.modeling.models import Sersic2D
        import matplotlib.pyplot as plt

        x,y = np.meshgrid(np.arange(100), np.arange(100))

        mod = Sersic2D(amplitude = 1, r_eff = 25, n=4, x_0=50, y_0=50,
                       ellip=.5, theta=-1)
        img = mod(x, y)
        log_img = np.log10(img)


        plt.figure()
        plt.imshow(log_img, origin='lower', interpolation='nearest',
                   vmin=-1, vmax=2)
        plt.xlabel('x')
        plt.ylabel('y')
        cbar = plt.colorbar()
        cbar.set_label('Log Brightness', rotation=270, labelpad=25)
        cbar.set_ticks([-1, 0, 1, 2], update_ticks=True)
        plt.show()

    References
    ----------
    .. [1] http://ned.ipac.caltech.edu/level5/March05/Graham/Graham2.html
    """

    amplitude = Parameter(default=1)
    r_eff = Parameter(default=1)
    n = Parameter(default=4)
    x_0 = Parameter(default=0)
    y_0 = Parameter(default=0)
    ellip = Parameter(default=0)
    theta = Parameter(default=0)
    _gammaincinv = None

    @classmethod
    def evaluate(cls, x, y, amplitude, r_eff, n, x_0, y_0, ellip, theta):
        """Two dimensional Sersic profile function."""

        if cls._gammaincinv is None:
            try:
                from scipy.special import gammaincinv
                cls._gammaincinv = gammaincinv
            except ValueError:
                raise ImportError('Sersic2D model requires scipy > 0.11.')

        bn = cls._gammaincinv(2. * n, 0.5)
        a, b = r_eff, (1 - ellip) * r_eff
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)
        x_maj = (x - x_0) * cos_theta + (y - y_0) * sin_theta
        x_min = -(x - x_0) * sin_theta + (y - y_0) * cos_theta
        z = np.sqrt((x_maj / a) ** 2 + (x_min / b) ** 2)

        return amplitude * np.exp(-bn * (z ** (1 / n) - 1))


    @property
    def input_units(self):
        if self.x_0.unit is None:
            return None
        else:
            return {'x': self.x_0.unit,
                    'y': self.y_0.unit}

    def _parameter_units_for_data_units(self, inputs_unit, outputs_unit):
        # Note that here we need to make sure that x and y are in the same
        # units otherwise this can lead to issues since rotation is not well
        # defined.
        if inputs_unit['x'] != inputs_unit['y']:
            raise UnitsError("Units of 'x' and 'y' inputs should match")
        return {'x_0': inputs_unit['x'],
                'y_0': inputs_unit['x'],
                'r_eff': inputs_unit['x'],
                'theta': u.rad,
                'amplitude': outputs_unit['z']}

# ------------------ END --------------------------------------------------------


class Exponential2D(Sersic2D):
    r"""
    Two dimensional exponential profile. Appropiate for disk galaxies using a scale radius rd

    Parameters
    ----------
    amplitude : float
        Surface brightness at r_eff.
    r_d : float
        Effective (half-light) radius
    x_0 : float, optional
        x position of the center.
    y_0 : float, optional
        y position of the center.
    ellip : float, optional
        Ellipticity.
    theta : float, optional
        Rotation angle in radians, counterclockwise from
        the positive x-axis.

    """
    amplitude = Parameter(default=1)
    r_d = Parameter(default=1)
    x_0 = Parameter(default=0)
    y_0 = Parameter(default=0)
    ellip = Parameter(default=0)
    theta = Parameter(default=0)

    def __init__(self, amplitude=amplitude.default, r_d=r_d.default,
                 x_0=x_0.default, y_0=y_0.default, ellip=ellip.default, theta=theta.default):

        r_eff = 1.678 * r_d
        super().__init__(self, amplitude=amplitude, r_eff=r_eff, n=1, x_0=x_0, y_0=y_0,
                         ellip=ellip, theta=theta)


class Nuker2D(Fittable2DModel):
    pass


class EdgeOn2D(Fittable2DModel):
    pass


class KingProfile2d(Fittable2DModel):
    pass


class NFW2D(Fittable2DModel):
    pass


class Beta2D(Fittable2DModel):
    pass






class VelFieldCourteau(Fittable2DModel):
    r"""
    Two dimensional Velocity Field following Courteau 1997.

    Parameters
    ----------

    incl : float, u.Quantity
        Inclination inclination between the normal to the galaxy plane and the line-of-sight,

    phi : float, u.Quantity
        Position angle of the major axis wrt to north (=up) measured counterclockwise,
    vmax : float
        Constant rotation velocity for R>>rd,

    r_d : float
        scale length of galaxy (assumed to be turnover radius)

    x0 : float, optional
        x position of the center.
    y0 : float, optional
        y position of the center.
    """
    incl = Parameter(default=45)
    phi = Parameter(default=0)
    vmax = Parameter(default=100)
    r_d = Parameter(default=1)
    x0 = Parameter(default=0)
    y0 = Parameter(default=0)
    v0 = Parameter(default=0)
    alpha = Parameter(default=5)

    @classmethod
    def evaluate(cls, x, y, incl, phi, vmax, r_d, x0, y0, v0, alpha):
        """
        TODO: Be consistent with Sersic2D
        (x,y) kartesian sky coordinates,
        (x0,y0) kartesian sky coordiantes of rotation centre of galaxy,
        V0 velocity of centre wrt observer,
        incl inclination angle between the normal to the galaxy plane and the line-of-sight,
        phi position angle of the major axis wrt to north (=up) measured counterclockwise,
        Vmax constant rotation for R>>rd,
        rd scale length of galaxy (assumed to be turnover radius)
        """
        if isinstance(incl, u.Quantity) is False:
            incl = incl*u.deg
        if isinstance(phi, u.Quantity) is False:
            phi = phi*u.deg

        phi = phi.to(u.rad)
        incl = incl.to(u.rad)
        r = ((x - x0)**2 + (y - y0)**2)**0.5
        #   azimuthal angle in the plane of the galaxy = cos(theta) = cost
        cost = (-(x - x0) * np.sin(phi) + (y - y0) * np.cos(phi)) / (r + 0.00001)

        vrot = vmax*r / (r**alpha + r_d**alpha)**(1/alpha)

        return v0 + vrot * np.sin(incl) * cost

    @property
    def input_units(self):
        if self.x0.unit is None:
            return None
        else:
            return {'x': self.x0.unit,
                    'y': self.y0.unit}

    def _parameter_units_for_data_units(self, inputs_unit, outputs_unit):
        # Note that here we need to make sure that x and y are in the same
        # units otherwise this can lead to issues since rotation is not well
        # defined.
        if inputs_unit['x'] != inputs_unit['y']:
            raise UnitsError("Units of 'x' and 'y' inputs should match")
        return {'x0': inputs_unit['x'],
                'y0': inputs_unit['x'],
                'r_d': inputs_unit['x'],
                'phi': u.deg,
                'amplitude': outputs_unit['z']}


class VelFieldArctan(Fittable2DModel):
    r"""
        Two dimensional Velocity Field following arctan approximation

        Parameters
        ----------

        incl : float, u.Quantity
            Inclination inclination between the normal to the galaxy plane and the line-of-sight,

        phi : float, u.Quantity
            Position angle of the major axis wrt to north (=up) measured counterclockwise,
        vmax : float
            Constant rotation velocity for R>>rd,

        r_d : float
            scale length of galaxy (assumed to be turnover radius)

        x0 : float, optional
            x position of the center.
        y0 : float, optional
            y position of the center.
        """
    incl = Parameter(default=45)
    phi = Parameter(default=0)
    vmax = Parameter(default=100)
    r_d = Parameter(default=1)
    x0 = Parameter(default=0)
    y0 = Parameter(default=0)
    v0 = Parameter(default=0)


    @classmethod
    def evaluate(cls, x, y, incl, phi, vmax, r_d, x0, y0, v0):
        """
        TODO: Be consistent with Sersic2D
        (x,y) kartesian sky coordinates,
        (x0,y0) kartesian sky coordiantes of rotation centre of galaxy,
        V0 velocity of centre wrt observer,
        incl inclination angle between the normal to the galaxy plane and the line-of-sight,
        phi position angle of the major axis wrt to north (=up) measured counterclockwise,
        Vmax constant rotation for R>>rd,
        rd scale length of galaxy (assumed to be turnover radius)
        """
        if isinstance(incl, u.Quantity) is False:
            incl = incl * u.deg
        if isinstance(phi, u.Quantity) is False:
            phi = phi * u.deg

        phi = phi.to(u.rad)
        incl = incl.to(u.rad)
        r = ((x - x0) ** 2 + (y - y0) ** 2) ** 0.5
        #   azimuthal angle in the plane of the galaxy = cos(theta) = cost
        cost = (-(x - x0) * np.sin(phi) + (y - y0) * np.cos(phi)) / (r + 0.00001)

        vrot = vmax*2/np.pi*np.arctan(r/r_d)         #arctan model

        return v0 + vrot * np.sin(incl) * cost

    @property
    def input_units(self):
        if self.x0.unit is None:
            return None
        else:
            return {'x': self.x0.unit,
                    'y': self.y0.unit}

    def _parameter_units_for_data_units(self, inputs_unit, outputs_unit):
        # Note that here we need to make sure that x and y are in the same
        # units otherwise this can lead to issues since rotation is not well
        # defined.
        if inputs_unit['x'] != inputs_unit['y']:
            raise UnitsError("Units of 'x' and 'y' inputs should match")
        return {'x0': inputs_unit['x'],
                'y0': inputs_unit['x'],
                'r_d': inputs_unit['x'],
                'phi': u.deg,
                'amplitude': outputs_unit['z']}



