from karabo.simulation.signal.plotting import SignalPlotting, SegmentationPlotting
from karabo.simulation.signal.signal_21_cm import Signal21cm
from karabo.simulation.signal.synchroton_signal import SignalSynchroton
from karabo.simulation.signal.galactic_foreground import SignalGalacticForeground
from karabo.simulation.signal.superimpose import Superimpose
from karabo.simulation.signal.superpixel_segmentation import SuperpixelSegmentation
from karabo.simulation.signal.seg_u_net_segmentation import SegUNetSegmentation
from karabo.simulation.signal.base_signal import BaseSignal
from karabo.simulation.signal.typing import BaseImage
import multiprocessing
import threading
import time
from karabo.simulation.signal.base_signal import BaseSignal
from karabo.simulation.signal.typing import BaseImage


from astropy.coordinates import Angle, SkyCoord
from astropy import units
from karabo.simulation.signal.typing import Image2D, Image3D

redshift_sig21 = [
    8.397,
    10.673,
    14.294,
    17.215,
    20.134,
]
redshift_gf = [
    8.4,
    10.7,
    14.3,
    17.4,
    20.4,
]


class ZPipeline:
    def __init__(
        self, redshifts_sig21: list[float], redshifts_galactic: list[float]
    ) -> None:
        self.redshifts_sig21 = redshifts_sig21
        self.redshifts_galactic = redshifts_galactic

        # prepare signal21cm
        files_sig21 = [
            Signal21cm.get_xfrac_dens_file(z=z, box_dims=244 / 0.7)
            for z in redshifts_sig21
        ]
        self.signal_21 = Signal21cm(files_sig21)
        grid_size = (250, 250)

        # prepare galactic foreground
        cent = SkyCoord(ra=10 * units.degree, dec=20 * units.degree, frame="icrs")
        fov = Angle([20, 20], unit=units.degree)
        self.gf = SignalGalacticForeground(
            cent,
            redshifts=redshifts_galactic,
            fov=fov,
            grid_size=grid_size,
        )

        # prepare synchroton
        self.sync = SignalSynchroton(
            centre=cent,
            fov=fov,
            grid_size=grid_size,
        )

    def run(self) -> list[BaseImage]:
        image_sync = self.sync.simulate()[0]
        images_sig21 = self.signal_21.simulate()
        images_galactic = self.gf.simulate()

        images = []
        for im_21, im_gal in zip(images_sig21, images_galactic):
            image = Superimpose.combine(im_21, im_gal, image_sync)
            images.append(image)

        res = []
        for image, sig_21 in zip(images, images_sig21):
            superpixel_image = SuperpixelSegmentation().segment(image)
            _ = SegmentationPlotting.superpixel_plotting(superpixel_image, sig_21)

            seg_u_net_image = SegUNetSegmentation(tta=0).segment(image)
            _ = SegmentationPlotting.seg_u_net_plotting(seg_u_net_image)

            res.append((superpixel_image, seg_u_net_image))

        return res


class ThreadWithReturnValue(threading.Thread):
    def __init__(
        self, group=None, target=None, name=None, args=(), kwargs={}, Verbose=None
    ):
        threading.Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)

    def join(self, *args):
        threading.Thread.join(self, *args)
        return self._return


class ParallelZPipeline:
    def __init__(
        self,
        redshifts_sig21: list[float],
        redshifts_galactic: list[float],
        max_baseline: float = 70.0,
        max_iter: int = 1000,  # TODO set to 5000
        tta: int = 1,  # TODO set to 2
    ) -> None:
        self.redshifts_sig21 = redshifts_sig21
        self.redshifts_galactic = redshifts_galactic

        # prepare signal21cm
        files_sig21 = [
            Signal21cm.get_xfrac_dens_file(z=z, box_dims=244 / 0.7)
            for z in redshifts_sig21
        ]
        self.signal_21 = Signal21cm(files_sig21)
        grid_size = (250, 250)

        # prepare galactic foreground
        cent = SkyCoord(ra=10 * units.degree, dec=20 * units.degree, frame="icrs")
        fov = Angle([20, 20], unit=units.degree)
        self.gf = SignalGalacticForeground(
            cent,
            redshifts=redshifts_galactic,
            fov=fov,
            grid_size=grid_size,
        )

        # prepare synchroton
        self.sync = SignalSynchroton(
            centre=cent,
            fov=fov,
            grid_size=grid_size,
        )

        self.max_baseline = max_baseline
        self.max_iter = max_iter
        self.tta = tta

        self._results = multiprocessing.Queue()

    def run(self) -> list[BaseImage]:
        sig_sync_thread = ThreadWithReturnValue(target=self._simulate_syncroton)
        sig_21cm_thread = ThreadWithReturnValue(target=self._simulate_signal_21cm)
        sig_gf_thread = ThreadWithReturnValue(target=self._simulate_galactic_foreground)

        sig_sync_thread.start()
        sig_21cm_thread.start()
        sig_gf_thread.start()

        image_sync = sig_sync_thread.join()[0]
        images_sig21 = sig_21cm_thread.join()
        images_galactic = sig_gf_thread.join()

        print("Done simulating")

        images: list[BaseImage] = []
        for im_21, im_gal in zip(images_sig21, images_galactic):
            image = Superimpose.combine(im_21, im_gal, image_sync)
            images.append(image)

        print("Done combining, on to the Segmentation")

        # procs: list[multiprocessing.Process] = []
        # for idx, (image, sig_21) in enumerate(zip(images, images_sig21)):
        #     print(f"Segmenting {idx}")

        #     su_proc = multiprocessing.Process(
        #         target=self._run_superpixel,
        #         args=(image, idx),
        #     )
        #     procs.append(su_proc)
        #     su_proc.start()

        #     segu_proc = multiprocessing.Process(
        #         target=self._run_segunet,
        #         args=(image, idx),
        #     )
        #     procs.append(segu_proc)
        #     segu_proc.start()

        # for proc in procs:
        #     idx = 0
        #     while proc.is_alive():
        #         print(f"waiting on segmentation (for {idx*20} seconds)")
        #         idx += 1
        #         time.sleep(20)
        #     proc.join()

        # print("All segmentation done")

        # return [x[1] for x in sorted(list(self._results), key=lambda x: x[0])]

        res = []
        idx = 0
        for image, sig_21 in zip(images, images_sig21):
            superpixel_image = SuperpixelSegmentation(
                # max_baseline=self.max_baseline, max_iter=self.max_iter
                n_segments=100
            ).segment(image)
            superpixel_plot = SegmentationPlotting.superpixel_plotting(
                superpixel_image, sig_21
            )
            superpixel_plot.savefig(
                f"./karabo/simulation/signal/plots/superpixel_plot_z{image.redshift}_baseline_{self.max_baseline}.png"
            )

            seg_u_net_image = SegUNetSegmentation(
                # max_baseline=self.max_baseline, tta=self.tta
                max_baseline=70,
                tta=1,
            ).segment(image)
            seg_u_net_plot = SegmentationPlotting.seg_u_net_plotting(seg_u_net_image)
            seg_u_net_plot.savefig(
                f"./karabo/simulation/signal/plots/seg_u_net_plot_z{image.redshift}_baseline_{self.max_baseline}.png"
            )

            res.append((superpixel_image, seg_u_net_image))
            idx += 1
            print(f"Segmentation {idx}/{len(images)} done")

        return res

    def _simulate_syncroton(self) -> Image2D:
        print("Running synchroton")
        return self.sync.simulate()

    def _simulate_signal_21cm(self) -> Image2D:
        print("Running 21cm")
        return self.signal_21.simulate()

    def _simulate_galactic_foreground(self) -> Image2D:
        print("Running Galactic foreground")
        return self.gf.simulate()

    # def _run_superpixel(self, image: Image3D, idx: int) -> None:
    #     print("Running Superpixel")
    #     segmented = SuperpixelSegmentation(max_iter=100).segment(image)  # TODO
    #     self._results.put((idx, segmented.image))

    # def _run_segunet(self, image: Image3D, idx: int) -> None:
    #     print("Running  SegU-net")
    #     segmented = SegUNetSegmentation(tta=0).segment(image)
    #     self._results.put((idx, segmented))  # TODO


images = ParallelZPipeline(
    redshifts_sig21=redshift_sig21,
    redshifts_galactic=redshift_gf,
    max_baseline=70.0,
    max_iter=1000,
    tta=1,
).run()

# images = ZPipeline(redshifts_sig21=redshift_sig21, redshifts_galactic=redshift_gf).run()
