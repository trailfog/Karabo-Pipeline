"""Galactic Foreground signal catalogue wrapper."""

from pathlib import Path
from typing import Annotated, Final, Literal, Optional

import numpy as np
import numpy.typing as npt
import pandas as pd
from astropy import units
from astropy.coordinates import Angle, SkyCoord
from astropy.table import Table

from karabo.data.external_data import GLEAMSurveyDownloadObject
from karabo.error import KaraboError
from karabo.simulation.signal.base_signal import BaseSignal
from karabo.simulation.signal.typing import Image2D


# pylint: disable=too-few-public-methods
class SignalGalacticForeground(BaseSignal[Image2D]):
    """
    Galactic Foreground signal catalogue wrapper.

    Examples
    --------
    >>> cent = SkyCoord(ra=10 * units.degree, dec=20 * units.degree, frame="icrs")
    >>> gf = SignalGalacticForeground(
    ...    cent,
    ...    redshifts=[7.6],
    ...    grid_size=(30, 30),
    ...    fov=Angle([20, 20], unit=units.degree),
    ... )
    >>> imgs = gf.simulate()
    """

    RA_COLUMN: Final[str] = "RAJ2000"
    DEC_COLUMN: Final[str] = "DEJ2000"

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        centre: SkyCoord,
        fov: Angle,
        redshifts: list[float],
        grid_size: tuple[Annotated[int, Literal["X"]], Annotated[int, Literal["Y"]]],
        gleam_file_path: Optional[Path] = None,
    ) -> None:
        """
        Galactic Foreground signal catalogue wrapper.

        Parameters
        ----------
        centre : Angle
            Center point. lon = right ascention, lat = declination
        fov : Angle
            Field of view for the right ascention in degrees. Must be between 0 < fov <=
            180. lon = right ascention fov, lat = declination fov
        redshifts : list[float]
            At what redshifts to observe the catalogue at.
        grid_size : tuple[Annotated[int, Literal["X"]], Annotated[int, Literal["Y"]]]
            Size of the simulated output image (X, Y).
        gleam_file_path : Optional[Path], optional
            Path to the gleam catalogue path to use, by default None. If None, the
            default GELAM Catalogue from Karabo is used.
        """
        if gleam_file_path is None:
            gleam_file_path = Path(GLEAMSurveyDownloadObject().get())

        self.gleam_file_path = gleam_file_path
        self.centre = centre
        self.redshifts = redshifts
        self.grid_size = grid_size

        fov.wrap_angle = 180 * units.deg
        self.fov = fov

        self.gleam_catalogue = Table.read(gleam_file_path)

    def simulate(self) -> list[Image2D]:
        """Simulate a signal to get a 2D image output."""
        images: list[Image2D] = []

        for redshift in self.redshifts:
            flux_column = SignalGalacticForeground._flux_column(redshift)

            if flux_column not in self.gleam_catalogue.columns:
                raise KaraboError(
                    "The GLEAM catalogue does not contain the redshift value of"
                    + str(redshift)
                )

            pos_df = self.gleam_catalogue[
                self.gleam_catalogue[flux_column] >= 0
            ].to_pandas()

            fov_ra: float = (self.fov.degree[0] / 2) * units.deg
            fov_dec: float = (self.fov.degree[1] / 2) * units.deg

            bottom_left = Angle(
                [self.centre.ra - fov_ra, self.centre.dec - fov_dec], unit=units.deg
            )
            top_right = Angle(
                [self.centre.ra + fov_ra, self.centre.dec + fov_dec], unit=units.deg
            )

            ra_filter = (
                pos_df[SignalGalacticForeground.RA_COLUMN] >= bottom_left.degree[0]
            ) & (pos_df[SignalGalacticForeground.RA_COLUMN] < top_right.degree[0])
            dec_filter = (
                pos_df[SignalGalacticForeground.DEC_COLUMN] >= bottom_left.degree[1]
            ) & (pos_df[SignalGalacticForeground.DEC_COLUMN] < top_right.degree[1])
            pos_df = pos_df[ra_filter & dec_filter]
            grid_intensity = self._map_datapoints(
                pos_df,
                flux_column=flux_column,
                grid_size=self.grid_size,
            )

            x_label = np.linspace(
                bottom_left.degree[0], top_right.degree[0], num=self.grid_size[0]
            )
            y_label = np.linspace(
                bottom_left.degree[1], top_right.degree[1], num=self.grid_size[1]
            )

            images.append(
                Image2D(
                    data=grid_intensity,
                    x_label=x_label,
                    y_label=y_label,
                    redshift=redshift,
                )
            )

        return images

    @classmethod
    def _flux_column(cls, redshift: float) -> str:
        """
        Get the flux column name from the redshift value.

        Parameters
        ----------
        redshift : float
            The redshift value.

        Returns
        -------
        str
            Flux column name.
        """
        return f"Fint{(redshift*10):0>3.0f}"

    # pylint: disable=too-many-locals
    @classmethod
    def _map_datapoints(
        cls,
        data: pd.DataFrame,
        flux_column: str,
        grid_size: tuple[Annotated[int, Literal["X"]], Annotated[int, Literal["Y"]]],
    ) -> Annotated[npt.NDArray[np.float_], Literal["X", "Y"]]:
        """
        Map the given datapoints with a destination to source mapping.

        For each pixel in the destination grid, the equivalent degree range in the
        source will be summed together and set in the destination grid.

        Parameters
        ----------
        data : pd.DataFrame,
            The data that is to be plotted onto the grid.
        flux_column : str
            Name of the flux column to use for the intensities.
        grid_size : tuple[Annotated[int, Literal["X"]], Annotated[int, Literal["Y"]]]
            Size of the output grid.

        Returns
        -------
        Annotated[npt.NDArray[np.float_], Literal["X", "Y"]]
            A 2D numpy array representing an image with the dimensions of the grid_size
            parameter.
        """
        grid = np.zeros(grid_size)

        x_min, x_max = data[SignalGalacticForeground.RA_COLUMN].agg(["min", "max"])
        y_min, y_max = data[SignalGalacticForeground.DEC_COLUMN].agg(["min", "max"])

        x_delta = (x_max - x_min) / grid_size[0]
        y_delta = (y_max - y_min) / grid_size[1]

        for x in range(grid_size[0]):
            x_beg = x_min + x * x_delta
            x_end = x_beg + x_delta

            ra_filter = (data[SignalGalacticForeground.RA_COLUMN] >= x_beg) & (
                data[SignalGalacticForeground.RA_COLUMN] < x_end
            )

            for y in range(grid_size[1]):
                y_beg = y_min + y * y_delta
                y_end = y_beg + y_delta

                dec_filter = (data[SignalGalacticForeground.DEC_COLUMN] >= y_beg) & (
                    data[SignalGalacticForeground.DEC_COLUMN] < y_end
                )

                pixel_value = data.loc[ra_filter & dec_filter, flux_column].sum()
                grid[x, y] = pixel_value

        return np.flip(grid, (0, 1))
