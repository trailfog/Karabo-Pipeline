"""EoR profile simulation."""

from typing import Annotated, Literal, Optional, Union

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

    frequency_21cm = 1420.405751768e6
    """Frequency of the 21cm signal [Hz]."""

    @classmethod
    def simulate(
        cls,
        x_hi: float = 0.1,
        dv_r_over_dr: float = 0,
        f_range: tuple[int, int] = (1e6, 200e6),
        plot_points: Union[float, int] = 1e6,
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
        f_range : tuple[float, float], optional
            Frequency range to plot in [Hz]. by default (2, 200e6)
        plot_points : float | int, optional
            How many points to be plotted, by default 1e6

        Returns
        -------
        EoRProfileT
            An array of the shape (floor((f_end - f_start) / step_size), 2), containing
            the frequency in the first column and the corresponding EoR profile in the
            second.
        """
        freq_range = np.linspace(*f_range, num=int(plot_points))
        z_range = (EoRProfile.frequency_21cm / freq_range) - 1

        eor_profile = (
            27
            * x_hi
            * (1 + cls.delta_m)
            * (cls.hubble_param / (dv_r_over_dr + cls.hubble_param))
            * (1 - cls.t_gamma / cls.t_s)
            * (((1 + z_range) / 10) * (0.15 / (cls.omega_m * cls.hubble_constant)))
            ** (1 / 2)
            * ((cls.omega_b * cls.hubble_constant) / 0.023)
        )

        return np.stack((freq_range, eor_profile), axis=-1)

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
        ax.plot(redshift / 1e6, delta_tb / 1e3)
        ax.set_xlabel("Frequency [MHz]")
        ax.set_ylabel("Brightness [mK]")
        ax.grid()

        ax.xaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
        ax.xaxis.set_minor_formatter(FuncFormatter(lambda y, _: f"{y:g}"))

        return fig


if __name__ == "__main__":
    eor = EoRProfile.simulate()
    EoRProfile.plot(eor)
    plt.show()
