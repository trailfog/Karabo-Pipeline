"""Galactic Foreground signal catalogue wrapper."""

from collections.abc import Iterator
from pathlib import Path
from typing import Optional

from astropy.table import Table

from karabo.data.external_data import GLEAMSurveyDownloadObject
from karabo.simulation.signal.base_signal import BaseSignal2D
from karabo.simulation.signal.typing import Image2DOriented


class SignalGalacticForeground(BaseSignal2D):
    """Galactic Foreground signal catalogue wrapper."""

    def __init__(self, gleam_file_path: Optional[Path] = None) -> None:
        """
        Galactic Foreground signal catalogue wrapper.

        Parameters
        ----------
        gleam_file_path : Optional[Path], optional
            Path to the gleam catalogue path to use, by default None. If None, the
            default GELAM Catalogue from Karabo is used.
        """
        if gleam_file_path is None:
            gleam_file_path = Path(GLEAMSurveyDownloadObject().get())

        self.gleam_file_path = gleam_file_path

        self.gleam_catalogue = Table.read(gleam_file_path)

    def simulate(self) -> list[Image2DOriented]:
        """Simulate a signal to get a 2D image output."""
        raj = self.gleam_catalogue["RAJ2000"]
        dej = self.gleam_catalogue["DEJ2000"]
        flux = self.gleam_catalogue["Fpwide"]

        columns: Iterator[str] = self.gleam_catalogue.columns
        intensity_cols = [col for col in columns if col.startswith("Fint")]

        for i_col in intensity_cols:
            ...

        return []
