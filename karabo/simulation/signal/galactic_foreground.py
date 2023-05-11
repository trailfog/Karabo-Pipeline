"""Galactic Foreground signal catalogue wrapper."""
from pathlib import Path
from typing import Annotated, Final, Literal, Optional

import numpy as np
from astropy import units
from astropy.coordinates import Angle, SkyCoord
from astropy.table import Table

from karabo.data.external_data import GLEAMSurveyDownloadObject
from karabo.error import KaraboError
from karabo.simulation.signal import helpers
from karabo.simulation.signal.base_signal import BaseSignal
from karabo.simulation.signal.typing import Image2D


# pylint: disable=too-few-public-methods
class SignalGalacticForeground(BaseSignal[Image2D]):
    """
    Galactic Foreground signal catalogue wrapper.

    Examples
    --------
    >>> from karabo.simulation.signal.plotting import SignalPlotting
    >>> cent = SkyCoord(ra=10 * units.degree, dec=20 * units.degree, frame="icrs")
    >>> gf = SignalGalacticForeground(
    ...    cent,
    ...    redshifts=[7.6],
    ...    grid_size=(30, 30),
    ...    fov=Angle([20, 20], unit=units.degree),
    ... )
    >>> imgs = gf.simulate()
    >>> SignalPlotting.general_img(imgs[0], "Galactic foreground")
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

            redshifts = [
                col for col in self.gleam_catalogue.columns if col.startswith("Fint")
            ]
            if flux_column not in redshifts:
                raise KaraboError(
                    "The GLEAM catalogue does not contain the redshift value of "
                    + f"{redshift}. Available: {redshifts}"
                )

            pos_df = self.gleam_catalogue[
                self.gleam_catalogue[flux_column] >= 0
            ].to_pandas()

            # RA Axis from 0 to 360, but we need -180 to 180
            pos_df[SignalGalacticForeground.RA_COLUMN] = (
                pos_df[SignalGalacticForeground.RA_COLUMN] + 180
            ) % 360 - 180

            pos_df, bottom_left, top_right = helpers.filter_dataframe_radec(
                df=pos_df,
                centre=self.centre,
                fov=self.fov,
                ra_column=SignalGalacticForeground.RA_COLUMN,
                dec_column=SignalGalacticForeground.DEC_COLUMN,
                wrap_offset=True,
            )

            grid_intensity = helpers.map_radec_datapoints_to_grid(
                pos_df,
                grid_size=self.grid_size,
                ra_column=SignalGalacticForeground.RA_COLUMN,
                dec_column=SignalGalacticForeground.DEC_COLUMN,
                intensity_column=flux_column,
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
                    box_dims=0,  # TODO
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
