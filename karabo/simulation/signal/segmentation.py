""" TODO."""

from pathlib import Path
from typing import Callable, Final, Union

import numpy as np
import tools21cm as t2c
import tools21cm.segmentation as t2cseg
from matplotlib import pyplot as plt
from sklearn.metrics import matthews_corrcoef
from karabo.simulation.signal.plotting import SignalPlotting

from karabo.simulation.signal.signal_21_cm import Signal21cm
from karabo.simulation.signal.superimpose import Superimpose
from karabo.simulation.signal.typing import Image2D, Image2DOriented


class Segmentation:
    """
    TODO
    """

    def segment(self, image: Union[Image2DOriented, Image2D]) -> None:
        """
        TODO
        """

        # inputs
        dT_subtracted = image.data
        redshift = image.redshift
        boxsize = 244 / 0.7  # TODO change that box_s from the inputs comes

        # 35
        # dT2 = dT_subtracted + noise_cube
        dT2 = dT_subtracted

        dT_smooth = t2c.smooth_coeval(
            cube=dT2,  # Data cube that is to be smoothed
            z=redshift,  # Redshift of the coeval cube
            box_size_mpc=boxsize,  # Box size in cMpc
            max_baseline=70.0,  # Maximum baseline of the telescope
            ratio=1.0,  # Ratio of smoothing scale in frequency direction
            nu_axis=2,
        )  # frequency axis

        mask_xHI = (
            t2c.smooth_coeval(
                cube=dT_subtracted,
                z=redshift,
                box_size_mpc=boxsize,
                max_baseline=70.0,
                nu_axis=2,
            )
            < 0.5
        )

        # 42
        # tta 0=super-fast, 1=fast, 2=slow(better acc)
        seg = t2cseg.segunet21cm(tta=2, verbose=True)

        # 43
        dT_cut = dT_smooth[:128, :128, :128]
        mask_xHI2 = mask_xHI[:128, :128, :128]

        xHI_seg, xHI_seg_err = seg.prediction(x=dT_cut)
        phicoef_seg = matthews_corrcoef(mask_xHI2.flatten(), xHI_seg.flatten())

        if True:
            SignalPlotting.segmentploting(boxsize=boxsize, xHI_seg=xHI_seg, xHI_seg_err=xHI_seg_err, phicoef_seg=phicoef_seg, mask_xHI2=mask_xHI2)


if __name__ == "__main__":
    z1 = Signal21cm.get_xfrac_dens_file(z=6.000, box_dims=244 / 0.7)
    z3 = Signal21cm.get_xfrac_dens_file(z=7.059, box_dims=244 / 0.7)
    sig = Signal21cm([z1, z3])
    signal_images = sig.simulate()
    # signal_images2 = sig.simulate()

    # superimpose_images = Superimpose.impose([signal_images, signal_images2])
    # superimpose_images = Superimpose.combine([signal_images, signal_images2])
    # seg.segment(superimpose_images[0])

    seg = Segmentation()
    seg.segment(signal_images[0])
    print("✂️ "*5 + "done with segmenting 🔍 " + "✂️ "*10)
