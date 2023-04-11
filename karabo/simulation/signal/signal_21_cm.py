"""21cm Signal simulation."""

from pathlib import Path
from typing import Final

import numpy as np
import tools21cm as t2c

from karabo.data.external_data import DownloadObject
from karabo.error import KaraboError
from karabo.simulation.signal.base_signal import BaseSignal2D
from karabo.simulation.signal.typing import Image2D, XFracDensFilePair


class Signal21cm(BaseSignal2D):
    """
    21cm Signal simulation wrapper.

    Examples
    --------
    >>> z1 = Signal21cm.get_xfrac_dens_file(z=6.000, box_dims=244 / 0.7)
    >>> z2 = Signal21cm.get_xfrac_dens_file(z=6.028, box_dims=244 / 0.7)
    >>> z3 = Signal21cm.get_xfrac_dens_file(z=7.059, box_dims=244 / 0.7)
    >>> sig = Signal21cm([z1, z2, z3])
    >>> signal_images = sig.simulate()
    """

    def __init__(self, files: list[XFracDensFilePair]) -> None:
        """
        21cm Signal simulation.

        Parameters
        ----------
        files : list[XFracDensFilePair]
            The xfrac and dens files to be used in the
        """
        self.files: Final[list[XFracDensFilePair]] = files

    def simulate(self) -> list[Image2D]:
        """
        Simulate the 21cm signal as a 2D intensity image.

        Returns
        -------
        list[Image2D]
            A list of images, based on the `self.files` list of provided xfrac and dens
            files.

        Raises
        ------
        KaraboError
            If a pair of xfrac and dens files do not have the same redshift values.
        """
        images: list[Image2D] = []

        for file in self.files:
            loaded = file.load()

            if (z := loaded.x_file.z) != loaded.d_file.z:
                raise KaraboError(
                    "The redshift of the xfrac and dens files are not the same", file
                )

            x_frac = loaded.x_file.xi
            dens = loaded.d_file.cgs_density

            dx, dy = (
                loaded.box_dims / x_frac.shape[1],
                loaded.box_dims / x_frac.shape[2],
            )
            y, x = np.mgrid[
                slice(dy / 2, loaded.box_dims, dy), slice(dx / 2, loaded.box_dims, dx)
            ]

            d_t = t2c.calc_dt(x_frac, dens, z)
            d_t_subtracted = t2c.subtract_mean_signal(d_t, 0)
            images.append(Image2D(data=d_t_subtracted[0, :, :], x=x, y=y, redshift=z))

        return images

    @classmethod
    def randomized_lightcones(
        cls, n_cells: int, z: float, bubble_count: int = 3
    ) -> Image2D:
        """
        Generate an image with randomized lighcones.

        Parameters
        ----------
        n_cells : int
            The count of cells to produce.
        z : float
            The redshift value for this image.
        bubble_count : int, optional
            How many bubbles to produce, by default 3

        Returns
        -------
        Image2D
            The generated image with multiple lightcones.
        """
        cube = np.zeros((n_cells, n_cells, n_cells))
        xx, yy, zz = np.meshgrid(
            np.arange(n_cells), np.arange(n_cells), np.arange(n_cells), sparse=True
        )

        r = 30 * np.exp(-(z - 7.0) / 3)
        r2 = (xx - n_cells / 2) ** 2 + (yy - n_cells / 2) ** 2 + (zz - n_cells / 2) ** 2
        xx_0 = n_cells // 2
        yy_0 = n_cells // 2
        zz_0 = n_cells // 2
        cube0 = np.zeros((n_cells, n_cells, n_cells))
        cube0[r2 <= r**2] = 1
        cube0 = np.roll(
            np.roll(np.roll(cube0, -xx_0, axis=0), -yy_0, axis=1), -zz_0, axis=2
        )

        for _ in range(bubble_count):
            cube = cube + np.roll(
                np.roll(np.roll(cube0, xx_0, axis=0), yy_0, axis=1), zz_0, axis=2
            )

        return Image2D(
            data=cube[100, :, :],
            x=np.arange(0, n_cells, 1, dtype=float),
            y=np.arange(0, n_cells, 1, dtype=float),
            redshift=z,
        )

    @staticmethod
    def get_xfrac_dens_file(z: float, box_dims: float) -> XFracDensFilePair:
        """
        Get the xfrac and dens files from the server.

        They are downloaded and cached on the first access.

        Parameters
        ----------
        z : float
            Redshift value.
        box_dims : float
            Box dimensions used for these files.

        Returns
        -------
        XFracDensFilePair
            A tuple of xfrac and dens files.
        """
        xfrac_name = f"xfrac3d_{z:.3f}.bin"
        dens_name = f"{z:.3f}n_all.dat"

        xfrac_path = DownloadObject(
            xfrac_name,
            f"https://ttt.astro.su.se/~gmell/244Mpc/244Mpc_f2_0_250/{xfrac_name}",
        ).get()
        dens_path = DownloadObject(
            dens_name,
            f"https://ttt.astro.su.se/~gmell/244Mpc/densities/nc250/coarser_densities/{dens_name}",
        ).get()

        return XFracDensFilePair(
            xfrac_path=Path(xfrac_path), dens_path=Path(dens_path), box_dims=box_dims
        )
