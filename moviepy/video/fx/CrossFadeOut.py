from dataclasses import dataclass

from moviepy.Clip import Clip
from moviepy.Effect import Effect
from moviepy.video.fx.FadeOut import FadeOut


@dataclass
class CrossFadeOut(Effect):
    """使剪辑在 ``duration`` 秒内逐渐消失。
    仅当剪辑包含在 CompositeVideoClip 中时才有效。
    """

    duration: float

    def apply(self, clip: Clip) -> Clip:
        """Apply the effect to the clip."""
        if clip.duration is None:
            raise ValueError("Attribute 'duration' not set")

        if clip.mask is None:
            clip = clip.with_mask()

        clip.mask.duration = clip.duration
        clip.mask = clip.mask.with_effects([FadeOut(self.duration)])

        return clip
