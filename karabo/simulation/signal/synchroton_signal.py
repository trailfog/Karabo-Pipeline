"""Galactic Foreground signal catalogue wrapper."""

import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import QTable


def run(table: QTable, title: str) -> None:
    """Run a simple plot."""
    fig, ax = plt.subplots(1, 1)
    fig.suptitle(title)
    ax.imshow(table.to_pandas().to_numpy())


fitsfile = fits.open("lambda_mollweide_haslam408_dsds.fits")

table1 = QTable(fitsfile[1].data)
table2 = QTable(fitsfile[1].data)

run(table1, "Table 1")
run(table2, "Table 2")

plt.show()
