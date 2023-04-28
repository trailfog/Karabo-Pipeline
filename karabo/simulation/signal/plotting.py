"""Signal plotting helpers."""
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import tools21cm as t2c
from matplotlib.figure import Figure

from karabo.simulation.signal.typing import Image2D, Image3D, XFracDensFilePair


class SignalPlotting:
    """Signal plotting helpers."""

    @classmethod
    def xfrac_dens(cls, data: XFracDensFilePair) -> Figure:
        """
        Plot the xfrac and dens files.

        Parameters
        ----------
        data : XFracDensFilePair
            The xfrac and dens file pair.

        Returns
        -------
        Figure
            The figure that was plotted.
        """
        loaded = data.load()
        x, y = loaded.xy_dims()

        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))
        fig.suptitle(f"$z={loaded.z:.1f},~x_v=${loaded.x_frac.mean():.2f}", size=18)
        axs[0].set_title("Density contrast slice")
        pcm_dens = axs[0].pcolormesh(x, y, loaded.dens[0] / loaded.dens.mean() - 1)
        fig.colorbar(pcm_dens, ax=axs[0])
        axs[0].set_xlabel(r"$x$ [Mpc]")
        axs[0].set_ylabel(r"$y$ [Mpc]")

        axs[1].set_title("Ionisation fraction slice")
        pcm_ion = axs[1].pcolormesh(x, y, loaded.x_frac[0])
        fig.colorbar(pcm_ion, ax=axs[1])
        axs[1].set_xlabel(r"$x$ [Mpc]")
        axs[1].set_ylabel(r"$y$ [Mpc]")

        return fig

    @classmethod
    def brightness_temperature(
        cls, data: Union[Image2D, Image3D], z_layer: int = 0
    ) -> Figure:
        """
        Plot the brightness temperature of a 2D image.

        Parameters
        ----------
        data : Union[Image2D, Image3D]
            The image to be plotted.

        z_layer : int, optional
            The Z layer to be used, when a Image3D is used.

        Returns
        -------
        Figure
            Figure of the plotted image
        """
        image_data = data.data
        x_label = data.x_label
        y_label = data.y_label

        if isinstance(data, Image3D):
            image_data = image_data[z_layer, :, :]
            x_label = x_label[z_layer, :, :]
            y_label = y_label[z_layer, :, :]

        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        ax.set_title("21 cm signal")
        colour_bar = ax.pcolormesh(x_label, y_label, image_data)
        ax.set_xlabel(r"$x$ [Mpc]")
        ax.set_ylabel(r"$y$ [Mpc]")
        fig.colorbar(colour_bar, ax=ax, label="mK")

        return fig

    def power_spectrum(self, data: XFracDensFilePair, kbins: int = 15) -> Figure:
        """
        Plot the power spectrum the 21cm signal.

        Parameters
        ----------
        data : XFracDensFilePair
            The xfrac and dens file pair.
        kbins : int, optional
            Count of bins for the spectrum plot, by default 15

        Returns
        -------
        Figure
            The generated plot figure.
        """
        loaded = data.load()

        d_t = t2c.calc_dt(loaded.x_frac, loaded.dens, loaded.z)
        d_t_subtracted = t2c.subtract_mean_signal(d_t, 0)
        ps_1d = t2c.power_spectrum_1d(
            d_t_subtracted,
            kbins=kbins,
            box_dims=loaded.box_dims,
        )

        ps = ps_1d[0]
        ks = ps_1d[1]
        fig, ax = plt.subplots(figsize=(16, 6))
        ax.set_title("Spherically averaged power spectrum.")
        ax.loglog(ks, ps * ks**3 / 2 / np.pi**2)
        ax.set_xlabel(r"k (Mpc$^{-1}$)")
        ax.set_ylabel(r"P(k) k$^{3}$/$(2\pi^2)$")

        return fig
