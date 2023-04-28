"""Superimpose two or more signals."""
from typing import Union, cast, overload

import numpy as np

from karabo.error import KaraboError
from karabo.simulation.signal.typing import Image2D, Image2DOriented


# pylint: disable=too-few-public-methods
class Superimpose:
    """Superimpose two or more signals."""

    @overload
    @classmethod
    def combine(cls, signals: list[Image2DOriented]) -> Image2DOriented:
        ...

    @overload
    @classmethod
    def combine(cls, signals: list[Image2D]) -> Image2D:
        ...

    @overload
    @classmethod
    def combine(  # type: ignore
        cls,
        signals: list[Union[Image2D, Image2DOriented]],
    ) -> Image2DOriented:
        ...

    @classmethod
    def combine(
        cls,
        signals: Union[
            list[Image2D], list[Image2DOriented], list[Union[Image2D, Image2DOriented]]
        ],
    ) -> Union[Image2D, Image2DOriented]:
        """
        Superimpose two or more signals int a single signal.

        Superimposing is done by adding each signal to the previous one. If a simple 2D
        Image is passed with an oriented image, the simple 2D Image will be "oriented"
        in the same direction s the oriented one.

        This function does not check if all oriented signals have the same orientation.

        If only one signal is passed, it gets returned without any further processing.

        Parameters
        ----------
        signals : Union[list[Image2D], list[Union[Image2D, Image2DOriented]]]
            The signals that are to be combined.

        Returns
        -------
        Union[Image2D, Image2DOriented]
            Either a Image2D if no oriented image is passed in, otherwise an oriented
            image is returned.

        Raises
        ------
        KaraboError
            When an empty signal list is passed in.
        """
        if (sig_count := len(signals)) == 1:
            return signals[0]

        if sig_count == 0:
            raise KaraboError(
                "You need to pass at least one signals to superimpose them."
            )

        oriented_count = 0
        for obj in signals:
            if isinstance(obj, Image2DOriented):
                oriented_count += 1

        has_unoriented = (sig_count - oriented_count) > 0
        has_oriented = oriented_count > 0

        if has_unoriented and not has_oriented:
            signals_unoriented = cast(list[Image2D], signals)
            output = np.zeros(shape=signals_unoriented[0].data.shape)
            for signal in signals_unoriented:
                output += signal.data
            return Image2D(
                data=output,
                x_label=signals[0].x_label,
                y_label=signals[0].y_label,
                redshift=signals[0].redshift,
            )

        signals_comb = cast(list[Union[Image2D, Image2DOriented]], signals)
        if has_unoriented:
            # TODO: Orient the unoriented
            ...

        signals_oriented = cast(list[Image2DOriented], signals_comb)
        output = np.zeros(shape=signals_oriented[0].data.shape)
        for o_signal in signals_oriented:
            output += o_signal.data

        first_sig = signals_oriented[0]
        return Image2DOriented(
            sky_model=first_sig.sky_model,
            data=output,
            x_label=first_sig.x_label,
            y_label=first_sig.y_label,
            redshift=first_sig.redshift,
        )
