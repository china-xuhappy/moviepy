import numbers
from dataclasses import dataclass
from typing import Union

import numpy as np
from PIL import Image

from moviepy.Effect import Effect


@dataclass
class Resize(Effect):
    """返回一个调整大小后的视频剪辑的效果。

    参数
    ----------
    
    new_size：元组、浮点数或函数，可选
    可以是：
    - 以像素为单位的“(width, height)”或表示缩放因子的浮点数
    - 缩放因子，例如“0.5”。
    - 返回以下之一的时间函数。
    
    height：整数，可选
    新剪辑的高度（以像素为单位）。然后计算宽度，使
    宽高比保持不变。
    
    width：整数，可选
    新剪辑的宽度（以像素为单位）。然后计算高度，使
    宽高比保持不变。

    Examples
    --------

    .. code:: python

        clip.with_effects([vfx.Resize((460,720))]) # New resolution: (460,720)
        clip.with_effects([vfx.Resize(0.6)]) # width and height multiplied by 0.6
        clip.with_effects([vfx.Resize(width=800)]) # height computed automatically.
        clip.with_effects([vfx.Resize(lambda t : 1+0.02*t)]) # slow clip swelling
    """

    new_size: Union[tuple, float, callable] = None
    height: int = None
    width: int = None
    apply_to_mask: bool = True

    def resizer(self, pic, new_size):
        """Resize the image using PIL."""
        new_size = list(map(int, new_size))
        pil_img = Image.fromarray(pic)
        resized_pil = pil_img.resize(new_size, Image.Resampling.LANCZOS)
        return np.array(resized_pil)

    def apply(self, clip):
        """Apply the effect to the clip."""
        w, h = clip.size

        if self.new_size is not None:

            def translate_new_size(new_size_):
                """Returns a [w, h] pair from `new_size_`. If `new_size_` is a
                scalar, then work out the correct pair using the clip's size.
                Otherwise just return `new_size_`
                """
                if isinstance(new_size_, numbers.Number):
                    return [new_size_ * w, new_size_ * h]
                else:
                    return new_size_

            if hasattr(self.new_size, "__call__"):
                # The resizing is a function of time

                def get_new_size(t):
                    return translate_new_size(self.new_size(t))

                if clip.is_mask:

                    def filter(get_frame, t):
                        return (
                            self.resizer(
                                (255 * get_frame(t)).astype("uint8"), get_new_size(t)
                            )
                            / 255.0
                        )

                else:

                    def filter(get_frame, t):
                        return self.resizer(
                            get_frame(t).astype("uint8"), get_new_size(t)
                        )

                newclip = clip.transform(
                    filter,
                    keep_duration=True,
                    apply_to=(["mask"] if self.apply_to_mask else []),
                )
                if self.apply_to_mask and clip.mask is not None:
                    newclip.mask = clip.mask.with_effects(
                        [Resize(self.new_size, apply_to_mask=False)]
                    )

                return newclip

            else:
                self.new_size = translate_new_size(self.new_size)

        elif self.height is not None:
            if hasattr(self.height, "__call__"):

                def func(t):
                    return 1.0 * int(self.height(t)) / h

                return clip.with_effects([Resize(func)])

            else:
                self.new_size = [w * self.height / h, self.height]

        elif self.width is not None:
            if hasattr(self.width, "__call__"):

                def func(t):
                    return 1.0 * self.width(t) / w

                return clip.with_effects([Resize(func)])

            else:
                self.new_size = [self.width, h * self.width / w]
        else:
            raise ValueError(
                "You must provide either 'new_size' or 'height' or 'width'"
            )

        # From here, the resizing is constant (not a function of time), size=newsize

        if clip.is_mask:

            def image_filter(pic):
                return (
                    1.0
                    * self.resizer((255 * pic).astype("uint8"), self.new_size)
                    / 255.0
                )

        else:

            def image_filter(pic):
                return self.resizer(pic.astype("uint8"), self.new_size)

        new_clip = clip.image_transform(image_filter)

        if self.apply_to_mask and clip.mask is not None:
            new_clip.mask = clip.mask.with_effects(
                [Resize(self.new_size, apply_to_mask=False)]
            )

        return new_clip
