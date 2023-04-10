"""EoR profile simulation."""

from typing import Annotated, Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

EoRProfileT = Annotated[npt.NDArray[np.float_], Literal["N", 2]]


class EoRProfile:
    """
    EoR profile simulation.

    Examples
    --------
    >>> eor = EoRProfile.simulate()
    >>> EoRProfile.plot(eor)
    >>> plt.show()
    """

    t_s = 0.068
    """Spin temperature [Kelvin]."""

    t_gamma = 2.73
    """Cosmic microwave background temperature [Kelvin]."""

    omega_b = 0.05
    """Baryon density parameter."""

    omega_m = 0.31
    """Matter density parameter."""

    delta_m = 0
    """Density contrast of matter."""

    hubble_constant = 0.7
    """Hubble constant."""

    hubble_param = hubble_constant * 100
    """Hubble parameter [km/s/Mpc]."""

    @classmethod
    def simulate(
        cls,
        x_hi: float = 0.1,
        dv_r_over_dr: float = 0,
        z_range: tuple[float, float] = (0, 200),
        step_size: float = 10,
    ) -> EoRProfileT:
        """
        Calculate the approximate evolution of fluctuations in the 21cm brightness.

        Implemented per https://arxiv.org/pdf/1602.02351.pdf, equation (1)

        Parameters
        ----------
        x_hi : float, optional
            Neutral hydrogen fraction, by default 0.1
        dv_r_over_dr : float, optional
            ???. By default 0
        z_range : tuple[float, float], optional
            redshift range to plot in. by default (0, 200)
        step_size : float, optional
            Step size of the redshift value, by default 10

        Returns
        -------
        EoRProfileT
            An array of the shape ((z_end - z_start) / step_size, 2), containing the
            redshift in the first column and the EoR profile in the second.
        """
        redshift_range = np.arange(*z_range, step=step_size)

        eor_profile = (
            27
            * x_hi
            * (1 + cls.delta_m)
            * (cls.hubble_param / (dv_r_over_dr + cls.hubble_param))
            * (1 - cls.t_gamma / cls.t_s)
            * (
                ((1 + redshift_range) / 10)
                * (0.15 / (cls.omega_m * cls.hubble_constant))
            )
            ** (1 / 2)
            * ((cls.omega_b * cls.hubble_constant) / 0.023)
        )

        return np.stack((redshift_range, eor_profile), axis=-1)

    @classmethod
    def plot(cls, profile: Optional[EoRProfileT] = None) -> Figure:
        """
        Plot the fluctuation profile of the 21cm signal.

        Parameters
        ----------
        profile : Optional[EoRProfileT], optional
            An optional profile to be plotted. If not given, a default EoR profile will
            be plotted. By default None.

        Returns
        -------
        Figure
            The plotted figure of the 21cm signal.
        """
        if profile is None:
            profile = cls.simulate()

        redshift = profile[:, 0]
        delta_tb = profile[:, 1]

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 6))
        ax.set_title("Fluctuation profile")
        ax.semilogx(redshift, delta_tb / 1e3)
        ax.set_xlabel("Redshift")
        ax.set_ylabel("Brightness [mK]")
        ax.grid(axis="y")
        ax.invert_xaxis()

        ax.xaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
        ax.xaxis.set_minor_formatter(FuncFormatter(lambda y, _: f"{y:g}"))

        return fig


if __name__ == "__main__":
    eor = EoRProfile.simulate()
    EoRProfile.plot(eor)
    plt.show()
