"""Galactic Foreground signal catalogue wrapper."""

from collections.abc import Iterator
from pathlib import Path
from typing import Optional

import numpy as np
from astropy import units
from astropy.coordinates import Angle, SkyCoord
from astropy.table import Table
from scipy.interpolate import griddata

from karabo.data.external_data import GLEAMSurveyDownloadObject
from karabo.simulation.signal.base_signal import BaseSignal2D
from karabo.simulation.signal.typing import Image2DOriented


class SignalGalacticForeground(BaseSignal2D):
    """Galactic Foreground signal catalogue wrapper."""

    def __init__(
        self,
        centre: SkyCoord,
        fov: Angle,
        gleam_file_path: Optional[Path] = None,
    ) -> None:
        """
        Galactic Foreground signal catalogue wrapper.

        Parameters
        ----------
        centre : Angle
            Center point. lon = right ascention, lat = declination
        fov : Angle
            Field of view for the right ascention in degrees. Must be between 0 < fov <= 180.
            lon = right ascention fov, lat = declination fov
        gleam_file_path : Optional[Path], optional
            Path to the gleam catalogue path to use, by default None. If None, the
            default GELAM Catalogue from Karabo is used.
        """
        if gleam_file_path is None:
            gleam_file_path = Path(GLEAMSurveyDownloadObject().get())

        self.gleam_file_path = gleam_file_path
        self.centre = centre

        fov.wrap_angle = 180 * units.deg
        self.fov = fov

        self.gleam_catalogue = Table.read(gleam_file_path)

    def simulate(self) -> list[Image2DOriented]:
        """Simulate a signal to get a 2D image output."""
        # TODO: Decide on which column to use!
        pos_df = self.gleam_catalogue[self.gleam_catalogue["Fint076"] >= 0].to_pandas()

        fov_ra: float = (self.fov.degree[0] / 2) * units.deg
        fov_dec: float = (self.fov.degree[1] / 2) * units.deg

        bottom_left = Angle(
            [self.centre.ra - fov_ra, self.centre.dec - fov_dec], unit=units.deg
        )
        top_right = Angle(
            [self.centre.ra + fov_ra, self.centre.dec + fov_dec], unit=units.deg
        )

        ra_filter = (pos_df["RAJ2000"] >= bottom_left.degree[0]) & (
            pos_df["RAJ2000"] <= top_right.degree[0]
        )
        dec_filter = (pos_df["DEJ2000"] >= bottom_left.degree[1]) & (
            pos_df["DEJ2000"] <= top_right.degree[1]
        )
        pos_df = pos_df[ra_filter & dec_filter]

        ra = pos_df["RAJ2000"]
        dec = pos_df["DEJ2000"]
        flux = pos_df["Fint076"]  # TODO: Decide on which column to use!

        ra_grid, dec_grid = np.meshgrid(ra, dec)
        grid_intensity = griddata(
            (ra, dec), flux, (ra_grid, dec_grid), method="nearest"
        )

        # TODO: Remove
        # from astropy.wcs import WCS
        # from astropy.io import fits

        # hdu = fits.open(self.gleam_file_path)[0]
        # wcs = WCS(hdu.header)

        # ax = plt.axes(projection=wcs)
        import seaborn as sns

        fig, axs = plt.subplots(1, 2)
        img = axs[0].imshow(
            grid_intensity,
            extent=[
                bottom_left.degree[0],
                top_right.degree[0],
                bottom_left.degree[1],
                top_right.degree[1],
            ],
            aspect="equal",
            origin="lower",
            cmap="viridis",
        )
        plt.colorbar(img)

        sns.scatterplot(x=ra, y=dec, hue=flux, ax=axs[1], alpha=0.6)

        plt.show()

        return [grid_intensity]


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    centre = SkyCoord(ra=10.625 * units.degree, dec=15 * units.degree, frame="icrs")

    gf = SignalGalacticForeground(centre, fov=Angle([20, 10], unit=units.degree))
    images = gf.simulate()
