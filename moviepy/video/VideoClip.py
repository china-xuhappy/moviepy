"""Implements VideoClip (base class for video clips) and its main subclasses:

- Animated clips:     VideoFileClip, ImageSequenceClip, BitmapClip
- Static image clips: ImageClip, ColorClip, TextClip,
"""

import copy as _copy
import os
import threading
from numbers import Real
from typing import TYPE_CHECKING, Callable, List, Union

import numpy as np
import proglog
from imageio.v2 import imread as imread_v2
from imageio.v3 import imwrite
from PIL import Image, ImageDraw, ImageFont

from moviepy.video.io.ffplay_previewer import ffplay_preview_video

if TYPE_CHECKING:
    from moviepy.Effect import Effect

from moviepy.Clip import Clip
from moviepy.decorators import (
    add_mask_if_none,
    apply_to_mask,
    convert_masks_to_RGB,
    convert_parameter_to_seconds,
    convert_path_to_string,
    outplace,
    requires_duration,
    requires_fps,
    use_clip_fps_by_default,
)
from moviepy.tools import compute_position, extensions_dict, find_extension
from moviepy.video.fx.Crop import Crop
from moviepy.video.fx.Resize import Resize
from moviepy.video.fx.Rotate import Rotate
from moviepy.video.io.ffmpeg_writer import ffmpeg_write_video
from moviepy.video.io.gif_writers import write_gif_with_imageio


class VideoClip(Clip):
    """
        视频剪辑的基类。
        参考 `VideoFileClip`、`ImageClip` 等更易用的类。

        属性
        ----------
        size
          剪辑的尺寸（宽度，高度），单位为像素。
        w, h
          剪辑的宽度 (`w`) 和高度 (`h`)，单位为像素。
        is_mask
          如果剪辑是遮罩，则该值为 `True`。
        frame_function
          一个函数 `t -> frame at time t`，用于在时间 `t` 获取当前帧图像。
          其中 `frame` 是一个 `w*h*3` 的 RGB 数组。
        mask (默认值 `None`)
          该剪辑附带的遮罩。如果 `mask` 为 `None`，则该视频剪辑完全不透明。
        audio (默认值 `None`)
          该剪辑的音频部分，为 `AudioClip` 实例。
        pos
          一个函数 `t -> (x,y)`，用于指定该剪辑在合成多个剪辑时的位置。
          详见 `VideoClip.set_pos` 方法。
        relative_pos
          见变量 `pos`。
        layer
          当多个剪辑在 `CompositeVideoClip` 中重叠时，该值用于确定层级顺序。
          值越大，剪辑越靠上（优先渲染）。默认值为 `0`。
    """

    def __init__(
            self,
            frame_function=None,
            is_mask=False,  # 如果该剪辑用作遮罩（mask），则为 `True`。
            duration=None,  # 剪辑的时长（秒）。如果为 `None`，则表示剪辑时长无限。
            has_constant_size=True  # 指定剪辑的尺寸是否固定。如果 `True`，表示尺寸恒定；如果 `False`，表示尺寸可能随时间变化。默认值为 `True`。
    ):
        super().__init__()
        self.mask = None
        self.audio = None
        self.pos = lambda t: (0, 0)
        self.relative_pos = False
        self.layer_index = 0
        if frame_function:
            self.frame_function = frame_function
            self.size = self.get_frame(0).shape[:2][::-1]
        self.is_mask = is_mask
        self.has_constant_size = has_constant_size
        if duration is not None:
            self.duration = duration
            self.end = duration

    @property
    def w(self):
        """返回视频的宽度。"""
        return self.size[0]

    @property
    def h(self):
        """返回视频的高度。"""
        return self.size[1]

    @property
    def aspect_ratio(self):
        """返回视频的宽高比。"""
        return self.w / float(self.h)

    @property
    @requires_duration
    @requires_fps
    def n_frames(self):
        """ 返回视频的帧数。"""
        return int(self.duration * self.fps)

    def __copy__(self):
        """Mixed copy of the clip.

        Returns a shallow copy of the clip whose mask and audio will
        be shallow copies of the clip's mask and audio if they exist.

        This method is intensively used to produce new clips every time
        there is an outplace transformation of the clip (clip.resize,
        clip.subclipped, etc.)

        Acts like a deepcopy except for the fact that readers and other
        possible unpickleables objects are not copied.
        """
        cls = self.__class__
        new_clip = cls.__new__(cls)
        for attr in self.__dict__:
            value = getattr(self, attr)
            if attr in ("mask", "audio"):
                value = _copy.copy(value)
            setattr(new_clip, attr, value)
        return new_clip

    copy = __copy__

    # ===============================================================
    # EXPORT OPERATIONS 导出操作

    @convert_parameter_to_seconds(["t"])
    @convert_masks_to_RGB
    def save_frame(self, filename, t=0, with_mask=True):
        """
        功能：
        该方法将视频在指定时间点 t 的帧保存为一张图像，并保存到给定的文件 filename 中。
        如果视频有一个遮罩（mask），并且 with_mask=True，它还会将遮罩信息存储到图像的 alpha 通道（透明度通道）中，适用于 PNG 格式。
        该方法支持以秒（如 15.35），分钟-秒（如 01:05），小时-分钟-秒（如 01:03:05）或字符串（如 '01:03:05.35'）等不同方式指定时间点 t。

        参数：
        filename：
            保存图像文件的路径和文件名，图像将被保存在这个文件中。
        t：
            可选参数，指定保存的帧时间。可以是：
            float（秒），如 15.35。
            tuple 或 str，表示时间（如 (1, 5) 或 '01:03:05'）。
            默认情况下，保存的是第一帧（t=0）。
        with_mask：
            可选布尔参数。如果为 True，并且视频有遮罩，遮罩将被保存到图像的 alpha 通道中。适用于 PNG 格式的图像。如果为 False，则仅保存视频帧，不包含遮罩信息。

        工作流程：
        获取帧：首先调用 get_frame(t) 获取视频在时间 t 的帧。
        处理遮罩：如果 with_mask 为 True 且视频包含遮罩，它会将遮罩数据（通过 mask.get_frame(t) 获取）合并到图像的 alpha 通道中，创建一个带透明度的图像。
        保存图像：最后将处理过的图像保存到指定路径的文件中。
        """
        im = self.get_frame(t)
        if with_mask and self.mask is not None:
            mask = 255 * self.mask.get_frame(t)
            im = np.dstack([im, mask]).astype("uint8")
        else:
            im = im.astype("uint8")

        imwrite(filename, im)

    @requires_duration
    @use_clip_fps_by_default
    @convert_masks_to_RGB
    @convert_path_to_string(["filename", "temp_audiofile", "temp_audiofile_path"])
    def write_videofile(
            self,
            filename,
            # 要写入的视频文件的名称，作为字符串或类似路径的对象。
            # 扩展名必须与使用的“编解码器”相对应（见下文），或者简单使用 '.avi'（适用于任何编解码器）。
            fps=None, # 结果视频文件的每秒帧数。如果没有提供，并且剪辑有 fps 属性，则使用该 fps。
            codec=None,
            # 用于图像编码的编解码器。可以是任何 ffmpeg 支持的编解码器。如果文件名具有扩展名 '.mp4'、'.ogv'、'.webm'，
            # 则编解码器会自动设置，但你仍然可以根据需要进行设置。对于其他扩展名，输出文件名必须相应设置。
            # 一些编解码器示例：
            # - ``'libx264'``（默认的 '.mp4' 扩展名编解码器）
            #   制作压缩良好的视频（质量可以通过 'bitrate' 调整）。
            # - ``'mpeg4'``（另一个适用于 '.mp4' 的编解码器）可以作为 ``'libx264'`` 的替代，
            #   默认会生成较高质量的视频。
            # - ``'rawvideo'``（使用 '.avi' 扩展名）将生成完美质量的视频，文件可能非常大。
            # - ``'png'``（使用 '.avi' 扩展名）将生成完美质量的视频，文件比使用 ``rawvideo`` 的文件要小。
            # - ``'libvorbis'``（使用 '.ogv' 扩展名）是一种不错的视频格式，完全免费/开源。然而，并不是所有人都默认安装了该编解码器。
            # - ``'libvpx'``（使用 '.webm' 扩展名）是一种非常小的视频格式，适合用于网页视频（支持 HTML5）。开源。
            bitrate=None, # 视频质量控制，如 "5000k"（越高质量越大）
            audio=True, # 可以是 ``True``、``False`` 或一个文件名。
            # 如果为 ``True``，并且剪辑有音频轨道，它将被作为视频的背景音乐加入。
            # 如果 `audio` 是一个音频文件的路径，指定的音频文件将作为视频的背景音乐加入。
            audio_fps=44100, # 用于生成音频的帧率。
            preset="medium", # 设置 FFMPEG 用于优化压缩的时间。
            # 可选择的设置有：ultrafast、superfast、veryfast、faster、fast、medium、slow、slower、veryslow、placebo。
            # 请注意，这不会影响视频质量，只会影响视频文件的大小。因此，如果你急需生成文件并且文件大小无关紧要，可以选择 ultrafast。
            audio_nbytes=4, # 指定生成音频时每帧音频数据的字节数。 4 字节（32-bit）：高精度（适用于专业音频）
            audio_codec=None, # 要使用的音频编解码器。例如：'libmp3lame'（用于 '.mp3'）、'libvorbis'（用于 '.ogg'）、'libfdk_aac'（用于 '.m4a'）、
            # 'pcm_s16le'（用于 16 位 wav）和 'pcm_s32le'（用于 32 位 wav）。默认是 'libmp3lame'，除非视频扩展名是 'ogv' 或 'webm'，
            # 在这种情况下，默认是 'libvorbis'。
            audio_bitrate=None, # 音频比特率，以字符串表示，如 '50k'、'500k'、'3000k'。
            # 它将决定输出文件中音频的大小/质量。请注意，它主要是一个目标值，比特率不会一定在最终文件中达到这个值。
            audio_bufsize=2000,
            temp_audiofile=None, # 临时音频文件的名称，作为字符串或类似路径的对象，
            # 它会被创建然后用于生成完整的视频（如果有的话）。
            temp_audiofile_path="", # 临时音频文件的存储位置，作为字符串或类似路径的对象。默认是当前工作目录。
            remove_temp=True, # 是否删除中间音频临时文件
            write_logfile=False, # 如果为 `True`，将为音频和视频生成日志文件。这些文件将以 '.log' 结尾，文件名与输出文件相同。
            threads=None, # 用于 ffmpeg 的线程数。可以在多核计算机上加速视频写入过程。
            ffmpeg_params=None, # 任何你希望传递给 ffmpeg 的额外参数，以列表形式提供，例如 ['-option1', 'value1', '-option2', 'value2']。
            logger="bar", # 可以是 ``"bar"`` 显示进度条，或 ``None``，或任何 Proglog 日志记录器。
            pixel_format=None, # 输出视频文件的像素格式。
    ):
        """
            用于将剪辑（clip）保存为视频文件的方法。它会根据用户提供的参数生成一个视频文件，
            并支持配置编码、音频、压缩、帧速率等多个选项。
            示例
            --------
            from moviepy import VideoFileClip
            clip = VideoFileClip("myvideo.mp4").subclipped(100,120)
            clip.write_videofile("my_new_video.mp4")
            clip.close()
        """

        name, ext = os.path.splitext(os.path.basename(filename))
        ext = ext[1:].lower()
        logger = proglog.default_bar_logger(logger)

        if codec is None:
            try:
                codec = extensions_dict[ext]["codec"][0]
            except KeyError:
                raise ValueError(
                    "MoviePy couldn't find the codec associated "
                    "with the filename. Provide the 'codec' "
                    "parameter in write_videofile."
                )

        if audio_codec is None:
            if ext in ["ogv", "webm"]:
                audio_codec = "libvorbis"
            else:
                audio_codec = "libmp3lame"
        elif audio_codec == "raw16":
            audio_codec = "pcm_s16le"
        elif audio_codec == "raw32":
            audio_codec = "pcm_s32le"

        audiofile = audio if isinstance(audio, str) else None
        make_audio = (
                (audiofile is None) and (audio is True) and (self.audio is not None)
        )

        if make_audio and temp_audiofile:
            # 音频将是剪辑的音频
            audiofile = temp_audiofile
        elif make_audio:
            audio_ext = find_extension(audio_codec)
            audiofile = os.path.join(
                temp_audiofile_path,
                name + Clip._TEMP_FILES_PREFIX + "wvf_snd.%s" % audio_ext,
            )

        # 有足够的 CPU 进行多处理？目前无用，以后还会用
        # enough_cpu = (multiprocessing.cpu_count() > 1)
        logger(message="MoviePy - Building video %s." % filename)
        if make_audio:
            self.audio.write_audiofile(
                audiofile,
                audio_fps,
                audio_nbytes,
                audio_bufsize,
                audio_codec,
                bitrate=audio_bitrate,
                write_logfile=write_logfile,
                logger=logger,
            )
            # 音频已编码，
            # 因此在视频导出时无需对其进行编码
            audio_codec = "copy"

        ffmpeg_write_video(
            self,
            filename,
            fps,
            codec,
            bitrate=bitrate,
            preset=preset,
            write_logfile=write_logfile,
            audiofile=audiofile,
            audio_codec=audio_codec,
            threads=threads,
            ffmpeg_params=ffmpeg_params,
            logger=logger,
            pixel_format=pixel_format,
        )

        if remove_temp and make_audio:
            if os.path.exists(audiofile):
                os.remove(audiofile)
        logger(message="MoviePy - video ready %s" % filename)

    @requires_duration
    @use_clip_fps_by_default
    @convert_masks_to_RGB
    def write_images_sequence(
            self,
            name_format, # 图像文件名格式，指定输出图像的命名规则和文件扩展名。例如 "frame.png" 表示图像的文件名将按 3 位数字格式化（如 frame001.png、frame002.png），文件格式为 PNG。也可以是其他格式，例如 "some_folder/frame%04d.jpeg"，会将文件保存为 JPEG 格式。
            fps=None, # 每秒帧数，指定在保存图像时每秒提取多少帧。如果没有指定，则使用视频剪辑的 fps 属性（如果有的话）。
            with_mask=True, # 如果设置为 True，则会保存剪辑的掩膜（如果存在），并作为 PNG 文件的 alpha 通道（仅适用于 PNG 格式）。掩膜通常用于保存透明背景部分。
            logger="bar" # 进度条显示的日志记录器。可以设置为 "bar" 来显示进度条，或 None 来禁用日志，或者使用任何支持的 Proglog 日志记录器。
    ):
        """
        方法功能
        write_images_sequence 方法将视频剪辑保存为一系列图像文件。
        你可以指定输出图像文件的命名格式、每秒帧数以及是否保存视频的掩膜（透明部分）。
        生成的图像文件会按照指定的命名格式和扩展名进行保存。

        返回值
        names_list:
        返回一个包含所有生成的文件名的列表。这些文件是按顺序保存的图像文件的路径。

        备注
        图像序列读取:
        生成的图像序列可以使用 ImageSequenceClip 类读取并创建新的视频剪辑。
        该方法通常用于将视频剪辑的每一帧保存为单独的图像文件，适用于视频分析、特效制作等需要逐帧处理的任务。
        """
        logger = proglog.default_bar_logger(logger)
        # Fails on GitHub macos CI
        # logger(message="MoviePy - Writing frames %s." % name_format)

        timings = np.arange(0, self.duration, 1.0 / fps)

        filenames = []
        for i, t in logger.iter_bar(t=list(enumerate(timings))):
            name = name_format % i
            filenames.append(name)
            self.save_frame(name, t, with_mask=with_mask)
        # logger(message="MoviePy - Done writing frames %s." % name_format)

        return filenames

    @requires_duration
    @convert_masks_to_RGB
    @convert_path_to_string("filename")
    def write_gif(
            self,
            filename, # 生成的 GIF 文件的名称，可以是字符串或路径。
            fps=None, # 每秒帧数。如果没有提供，函数会尝试从视频剪辑中读取 fps 属性。
            loop=0,# 一个可选参数，决定 GIF 循环播放的次数。如果不指定，默认会无限循环。
            logger="bar", # 进度条显示的日志记录器。可以设置为 "bar" 来显示进度条，或 None 来禁用日志，或者使用任何支持的 Proglog 日志记录器。
    ):
        """
        这个方法的作用是将一个 VideoClip 转换成一个动画 GIF 文件。具体来说，它使用 imageio 库来进行 GIF 文件的创建。

        说明：
        生成的 GIF 将会按视频的真实播放速度播放。如果想让 GIF 播放得更慢，
        可以使用 multiply_speed 方法来调整播放速度，如将视频速度减慢到 50%。
        """
        # A little sketchy at the moment, maybe move all that in write_gif,
        #  refactor a little... we will see.
        write_gif_with_imageio(
            self,
            filename,
            fps=fps,
            loop=loop,
            logger=logger,
        )

    # ===============================================================
    # PREVIEW OPERATIONS 预览操作

    @convert_masks_to_RGB
    @convert_parameter_to_seconds(["t"])
    def show(
            self,
            t=0, # 可选参数，表示时间，单位为秒。可以是浮动数值、元组或字符串（例如："00:03:05"）。这个时间对应的是视频中要显示的那一帧。
            with_mask=True # 可选参数，默认值为 True。如果视频有遮罩（mask），设置为 False 会显示不带遮罩的帧。设置为 True 时，显示带遮罩的帧。
    ):
        """
        这个方法用于显示视频剪辑在给定时间 t 的某一帧。它可以用于查看视频的特定帧，方便调试或展示特定的画面。

        Examples
        --------
        .. code:: python
            from moviepy import *
            clip = VideoFileClip("media/chaplin.mp4")
            clip.show(t=4)
        """
        clip = self.copy()

        # 警告：注释以修复 compositevideoclip 预览中的错误
        # 它破坏了 compositevideoclip，并且它对带有 alpha 的正常剪辑没有任何作用
        # if with_mask and (self.mask is not None):
        # # 讨厌它，但无法找到更好的方法来解决 python 可怕的循环问题
        # # 依赖关系
        # from mpy.video.compositing.CompositeVideoClip import CompositeVideoClip
        # clip = CompositeVideoClip([self.with_position((0, 0))])

        frame = clip.get_frame(t)
        pil_img = Image.fromarray(frame.astype("uint8"))
        pil_img.show()

    @requires_duration
    @convert_masks_to_RGB
    def preview(
            self,
            fps=15, # 可选参数，指定视频预览时的帧率（每秒帧数）。默认值是 15。
            audio=True,# 可选参数，默认为 True，表示是否在预览时播放视频的音频。如果设置为 False，视频将不带音频播放。
            audio_fps=22050,# 可选参数，指定生成音频时使用的帧率。默认值依赖于音频源的帧率。
            audio_buffersize=3000,# 可选参数，指定生成音频时使用的缓冲区大小，影响音频处理的稳定性。
            audio_nbytes=2 # 可选参数，指定生成音频时每帧音频数据的字节数。
    ):
        """
        在窗口中以给定的每秒帧数显示剪辑。
        它可以避免剪辑播放速度比正常情况更快，但如果计算复杂，则无法避免剪辑播放速度比正常情况更慢。在这种情况下，请尝试降低“fps”。
        这个方法用于在窗口中显示视频剪辑，并按指定的帧率播放视频，适合用来预览视频内容。

        Examples
        --------
        .. code:: python
            from moviepy import *
            clip = VideoFileClip("media/chaplin.mp4")
            clip.preview(fps=10, audio=False)
        """
        audio = audio and (self.audio is not None)
        audio_flag = None
        video_flag = None

        if audio:
            # the sound will be played in parallel. We are not
            # parralellizing it on different CPUs because it seems that
            # ffplay use several cpus.

            # two synchro-flags to tell whether audio and video are ready
            video_flag = threading.Event()
            audio_flag = threading.Event()
            # launch the thread
            audiothread = threading.Thread(
                target=self.audio.audiopreview,
                args=(
                    audio_fps,
                    audio_buffersize,
                    audio_nbytes,
                    audio_flag,
                    video_flag,
                ),
            )
            audiothread.start()

        # passthrough to ffmpeg, passing flag for ffmpeg to set
        ffplay_preview_video(
            clip=self, fps=fps, audio_flag=audio_flag, video_flag=video_flag
        )

    # -----------------------------------------------------------------
    # FILTERING  过滤操作/滤镜处理

    def with_effects_on_subclip(
            self, effects: List["Effect"], start_time=0, end_time=None, **kwargs
    ):
        """对剪辑的一部分应用变换。
        返回一个新剪辑，其中函数“fun”（clip->clip）已在“start_time”和“end_time”（以秒为单位）之间应用于子剪辑。

        Examples
        --------
        .. code:: python
            #``clip`` 中时间 t=3s 和 t=6s 之间的场景将在 ``new_clip`` 中以两倍慢的速度播放
            new_clip = clip.with_sub_effect(MultiplySpeed(0.5), 3, 6)
        """
        left = None if (start_time == 0) else self.subclipped(0, start_time)
        center = self.subclipped(start_time, end_time).with_effects(effects, **kwargs)
        right = None if (end_time is None) else self.subclipped(start_time=end_time)

        clips = [clip for clip in [left, center, right] if clip is not None]

        # beurk, have to find other solution
        from moviepy.video.compositing.CompositeVideoClip import concatenate_videoclips

        return concatenate_videoclips(clips).with_start(self.start)

    # IMAGE FILTERS 图像过滤器

    def image_transform(self, image_func, apply_to=None):
        """通过将帧“get_frame(t)”替换为另一个帧“image_func(get_frame(t))”来修改剪辑的图像。"""
        apply_to = apply_to or []
        return self.transform(lambda get_frame, t: image_func(get_frame(t)), apply_to)

    # --------------------------------------------------------------
    # COMPOSITING  合成

    def fill_array(
            self,
            pre_array, # 需要调整形状的原始数组。通常是图像帧或视频帧，类型为 NumPy 数组。
            shape=(0, 0) # 目标数组的形状，指定期望的高度和宽度。
    ):
        """填充数组以匹配指定的形状。
        如果 `pre_array` 小于所需形状，则将缺失的行或列分别添加到底部或右侧，直到形状匹配。
        如果 `pre_array` 大于所需形状，则将多余的行或列分别从底部或右侧裁剪，直到形状匹配。
        返回具有填充形状的结果数组。
        """
        pre_shape = pre_array.shape
        dx = shape[0] - pre_shape[0]
        dy = shape[1] - pre_shape[1]
        post_array = pre_array
        if dx < 0:
            post_array = pre_array[: shape[0]]
        elif dx > 0:
            x_1 = [[[1, 1, 1]] * pre_shape[1]] * dx
            post_array = np.vstack((pre_array, x_1))
        if dy < 0:
            post_array = post_array[:, : shape[1]]
        elif dy > 0:
            x_1 = [[[1, 1, 1]] * dy] * post_array.shape[0]
            post_array = np.hstack((post_array, x_1))
        return post_array

    def compose_on(
            self,
            background: Image.Image,# 背景图像，通常是一个 PIL.Image 对象。如果背景图像是透明的，则必须将其作为 RGBA 图像提供
            t # 视频剪辑中要显示的时间，单位为秒。
    ) -> Image.Image:
        """返回剪辑在时间 `t` 时在给定 `picture` 顶部的帧的结果，剪辑的位置由剪辑的 ``pos`` 属性指定。用于合成。
        如果剪辑/背景具有透明度，则将考虑透明度。返回的是 Pillow 图像
        """
        ct = t - self.start  # clip time

        # GET IMAGE AND MASK IF ANY
        clip_frame = self.get_frame(ct).astype("uint8")
        clip_img = Image.fromarray(clip_frame)

        if self.mask is not None:
            clip_mask = (self.mask.get_frame(ct) * 255).astype("uint8")
            clip_mask_img = Image.fromarray(clip_mask).convert("L")

            # Resize clip_mask_img to match clip_img, always use top left corner
            if clip_mask_img.size != clip_img.size:
                mask_width, mask_height = clip_mask_img.size
                img_width, img_height = clip_img.size

                if mask_width > img_width or mask_height > img_height:
                    # Crop mask if it is larger
                    clip_mask_img = clip_mask_img.crop((0, 0, img_width, img_height))
                else:
                    # Fill mask with 0 if it is smaller
                    new_mask = Image.new("L", (img_width, img_height), 0)
                    new_mask.paste(clip_mask_img, (0, 0))
                    clip_mask_img = new_mask

            clip_img = clip_img.convert("RGBA")
            clip_img.putalpha(clip_mask_img)

        # SET POSITION
        pos = self.pos(ct)
        pos = compute_position(clip_img.size, background.size, pos, self.relative_pos)

        # # 如果背景和剪辑都没有 alpha 层（用 A 检查模式是否结束），我们可以使用pillow粘贴
        if clip_img.mode[-1] != "A" and background.mode[-1] != "A":
            background.paste(clip_img, pos)
            return background

        # For images with transparency we must use pillow alpha composite
        # instead of a simple paste, because pillow paste dont work nicely
        # with alpha compositing
        if background.mode[-1] != "A":
            background = background.convert("RGBA")

        if clip_img.mode[-1] != "A":
            clip_img = clip_img.convert("RGBA")

        # We need both image to do the same size for alpha compositing in pillow
        # so we must start by making a fully transparent canvas of background's
        # size and paste our clip img into it in position pos, only then can we
        # composite this canvas on top of background
        canvas = Image.new("RGBA", (background.width, background.height), (0, 0, 0, 0))
        canvas.paste(clip_img, pos)
        result = Image.alpha_composite(background, canvas)
        return result

    def compose_mask(
            self,
            background_mask: np.ndarray, # 将在其上合成剪辑蒙版的基础蒙版。
            t: float # 剪辑中提取蒙版的时间位置。
    ) -> np.ndarray:
        """返回在时间 `t` 处剪辑的蒙版与给定的 `background_mask` 合成的结果，剪辑的位置由剪辑的 ``pos`` 属性指定。用于合成。
        （警告：只能使用此功能将两个蒙版连接在一起，不能将图像连接在一起）
        """
        ct = t - self.start  # clip time
        clip_mask = self.get_frame(ct).astype("float")

        # numpy shape is H*W not W*H
        bg_h, bg_w = background_mask.shape
        clip_h, clip_w = clip_mask.shape

        # SET POSITION
        pos = self.pos(ct)
        pos = compute_position((clip_w, clip_h), (bg_w, bg_h), pos, self.relative_pos)

        # ALPHA COMPOSITING
        # Determine the base_mask region to merge size
        x_start = int(max(pos[0], 0))  # Dont go under 0 left
        x_end = int(min(pos[0] + clip_w, bg_w))  # Dont go over base_mask width
        y_start = int(max(pos[1], 0))  # Dont go under 0 top
        y_end = int(min(pos[1] + clip_h, bg_h))  # Dont go over base_mask height

        # Determine the clip_mask region to overlapp
        # Dont go under 0 for horizontal, if we have negative margin of X px start at X
        # And dont go over clip width
        clip_x_start = int(max(0, -pos[0]))
        clip_x_end = int(clip_x_start + min((x_end - x_start), (clip_w - clip_x_start)))
        # same for vertical
        clip_y_start = int(max(0, -pos[1]))
        clip_y_end = int(clip_y_start + min((y_end - y_start), (clip_h - clip_y_start)))

        # Blend the overlapping regions
        # The calculus is base_opacity + clip_opacity * (1 - base_opacity)
        # this ensure that masks are drawn in the right order and
        # the contribution of each mask is proportional to their transparency
        #
        # Note :
        # Thinking in transparency is hard, as we tend to think
        # that 50% opaque + 40% opaque = 90% opacity, when it really its 70%
        # It's a lot easier to think in terms of "passing light"
        # Consider I emit 100 photons, and my first layer is 50% opaque, meaning it
        # will "stop" 50% of the photons, I'll have 50 photons left
        # now my second layer is blocking 40% of thoses 50 photons left
        # blocking 50 * 0.4 = 20 photons, and leaving me with only 30 photons
        # So, by adding two layer of 50% and 40% opacity my finaly opacity is only
        # of (100-30)*100 = 70% opacity !
        background_mask[y_start:y_end, x_start:x_end] = background_mask[
                                                        y_start:y_end, x_start:x_end
                                                        ] + clip_mask[clip_y_start:clip_y_end,
                                                            clip_x_start:clip_x_end] * (
                                                                1 - background_mask[y_start:y_end, x_start:x_end]
                                                        )

        return background_mask

    def with_background_color(
            self,
            size=None, # 类型：tuple（宽度, 高度）描述：最终生成的视频的大小。如果未提供，默认为当前剪辑的大小。
            color=(0, 0, 0), # 类型：tuple（R, G, B）描述：背景的颜色，默认为黑色 (0, 0, 0)，即 RGB 值为 (0, 0, 0)。
            pos=None, # 类型：str 或 tuple 描述：剪辑在最终视频中的位置。默认为 "center"，即将剪辑放在背景的中央。可以传入具体的位置坐标（如 (x, y)）。
            opacity=None # 类型：float（0 到 1 之间） 描述：背景色的透明度。如果未提供，背景色将是完全不透明的。如果提供，值范围为 0 到 1，0 表示完全透明，1 表示完全不透明。
    ):
        """
        功能：
        该方法可以将当前的剪辑（可能是透明的）放置在一个指定颜色的背景上，生成一个新的剪辑。
        它可以用于给透明的剪辑添加背景色，或者调整剪辑的大小和位置。

        功能：
            将当前的剪辑放在一个背景色为指定颜色的图层上。
            可以用来将透明剪辑放置在有背景色的区域中。
            可以指定最终视频的大小和剪辑的位置。

        工作原理：
            如果 opacity 被设置了，则背景颜色的透明度将根据 opacity 参数调整。通过使用 ColorClip 创建一个背景色的剪辑，并设置透明度，然后将背景剪辑和当前剪辑合成在一起。
            如果没有设置 opacity，则背景为不透明，直接合成。
            最终返回一个合成后的 CompositeVideoClip，其中包含当前剪辑和背景。

        返回值：
            返回一个新的 CompositeVideoClip，这个剪辑由当前剪辑和背景色的剪辑合成而成。
        """
        from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip

        if size is None:
            size = self.size
        if pos is None:
            pos = "center"

        if opacity is not None:
            colorclip = ColorClip(
                size, color=color, duration=self.duration
            ).with_opacity(opacity)
            result = CompositeVideoClip([colorclip, self.with_position(pos)])
        else:
            result = CompositeVideoClip(
                [self.with_position(pos)], size=size, bg_color=color
            )

        if (
                isinstance(self, ImageClip)
                and (not hasattr(pos, "__call__"))
                and ((self.mask is None) or isinstance(self.mask, ImageClip))
        ):
            new_result = result.to_ImageClip()
            if result.mask is not None:
                new_result.mask = result.mask.to_ImageClip()
            return new_result.with_duration(result.duration)

        return result

    @outplace
    def with_updated_frame_function(
            self, frame_function: Callable[[float], np.ndarray]
    ):
        """更改剪辑的“get_frame”。
        返回 VideoClip 实例的副本，并将 frame_function
        属性设置为“mf”。
        """
        self.frame_function = frame_function
        self.size = self.get_frame(0).shape[:2][::-1]

    @outplace
    def with_audio(self, audioclip):
        """ 将 AudioClip 附加到 VideoClip。返回 VideoClip 实例的副本，并将 `audio` 属性设置为 ``audio``，该实例必须是 AudioClip 实例。"""
        self.audio = audioclip

    @outplace
    def with_mask(
            self,
            mask: Union["VideoClip", str] = "auto"
            #  Union["VideoClip", str]，可选
            #  要应用于剪辑的蒙版。
            #  如果设置为“auto”，将生成默认蒙版：
            #  - 如果剪辑具有恒定大小，则将创建值为 1.0 的实心蒙版。
            #  - 否则，将根据帧大小创建动态实心蒙版。
    ):
        """
        设置剪辑的蒙版。

        返回 VideoClip 的副本，其蒙版属性设置为
        ``mask``，必须是灰度（值在 0-1 之间）的 VideoClip。
        """
        if mask == "auto":
            if self.has_constant_size:
                mask = ColorClip(self.size, 1.0, is_mask=True)
            else:

                def frame_function(t):
                    return np.ones(self.get_frame(t).shape[:2], dtype=float)

                mask = VideoClip(is_mask=True, frame_function=frame_function)
        self.mask = mask

    @outplace
    def without_mask(self):
        """移除剪辑的mask。"""
        self.mask = None

    @add_mask_if_none
    @outplace
    def with_opacity(
            self,
            opacity # 如果 opacity = 1，表示视频完全不透明。 如果 opacity = 0，表示视频完全透明，视频不可见。
    ):
        """
        设置剪辑的不透明度/透明度级别。
        返回剪辑的一个半透明副本。其实现方式是将剪辑的遮罩（mask）
        与指定的 ``opacity`` 值（通常是 0 到 1 之间的浮点数）相乘。
        """
        self.mask = self.mask.image_transform(lambda pic: opacity * pic)

    @apply_to_mask
    @outplace
    def with_position(
            self,
            pos, # tuple (x, y): 绝对坐标，表示剪辑的左上角在画面上的位置。x 为水平位置，y 为垂直位置。
            # str: 字符串表示某个常见位置，如 "center"（中心）、"top"（顶部）等。
            # function t-> (x, y): 随时间变化的函数，允许动态地改变位置，适用于动画效果。
            relative=False # relative=True: pos 表示相对于画面大小的比例坐标，例如 (0.4, 0.7) 表示位置在画面宽度的 40% 处和画面高度的 70% 处。
            # relative=False: pos 表示绝对坐标，以像素为单位。
    ):
        """
        设置剪辑在合成视频中的位置。
        设置剪辑在包含于合成视频中时所处的位置。参数 ``pos`` 可以是一个坐标对 ``(x,y)`` 或一个函数 ``t-> (x,y)``。
        `x` 和 `y` 标记剪辑的左上角的位置，并且可以是多种类型。

        例子
        ----
            clip.with_position((45,150)) # x=45, y=150
            # 剪辑水平居中，位于画面顶部
            clip.with_position(("center","top"))
            # 剪辑位于宽度 40%，高度 70% 的位置：
            clip.with_position((0.4,0.7), relative=True)
            # 剪辑的位置水平居中，并且向上移动！
            clip.with_position(lambda t: ('center', 50+t))

            为视频剪辑提供了灵活的定位功能，允许您使用绝对坐标、相对坐标或随时间变化的坐标来精确控制剪辑在合成视频中的位置。
            这对于创建复杂的视频布局和动画效果非常有用。
        """
        self.relative_pos = relative
        if hasattr(pos, "__call__"):
            self.pos = pos
        else:
            self.pos = lambda t: pos

    @apply_to_mask
    @outplace
    def with_layer_index(self, index):
        """在合成中设置剪辑的图层。具有更大“层”的剪辑 属性将显示在其他属性之上。
        注意：只有在CompositeVideoClip中使用剪辑时才有效。
        """
        self.layer_index = index

    def resized(
            self,
            new_size=None, # 新的视频大小，可以是一个 (width, height) 的元组，用于同时设置宽度和高度。如果提供了该参数，则 height 和 width 可以忽略。
            height=None, # 新的视频高度（像素）。如果同时提供了 new_size 参数，则此参数会被忽略。
            width=None, # 新的视频宽度（像素）。如果同时提供了 new_size 参数，则此参数会被忽略。
            apply_to_mask=True # 一个布尔值，指示是否将重设大小的效果应用到视频的蒙版（mask）上。如果为 True，则蒙版会随视频一起被重设大小。默认为 True。
    ):
        """
        功能：
        该方法通过调用 with_effects 方法，将一个 Resize 效果应用到视频剪辑上，生成一个新的调整大小后的剪辑。
        Resize 类是一个视频效果类，用于调整视频的大小，具体的行为由传入的参数决定。
        该方法的关键作用是通过 Resize 将视频尺寸进行变更，并且可以选择是否将效果应用到蒙版上。

        返回值：
        返回一个经过调整尺寸（重设大小）的视频剪辑。
        有关参数的更多信息，请参见 ``vfx.Resize``。
        """
        return self.with_effects(
            [
                Resize(
                    new_size=new_size,
                    height=height,
                    width=width,
                    apply_to_mask=apply_to_mask,
                )
            ]
        )

    def rotated(
            self,
            angle: float, # 旋转角度。表示旋转的度数或弧度数。该角度是逆时针方向的。如果角度不是 90 的倍数，或者提供了 center、translate 或 bg_color，则会进行复杂的旋转操作。
            unit: str = "deg", # 角度的单位，默认为 "deg"（度）。如果设置为 "rad"，则角度将以弧度为单位进行旋转。
            resample: str = "bicubic", # 重采样方法，控制旋转后的图像质量。默认为 "bicubic"，这通常用于图像处理中的平滑重采样。
            expand: bool = False, # 布尔值，指示是否扩展图像以适应旋转后的新大小。如果为 True，图像大小会根据旋转后的内容进行自动扩展，否则图像会裁剪掉多余的部分。
            center: tuple = None, # 一个二元组 (x, y)，指定旋转中心点。如果为 None，旋转将在剪辑的中心进行。
            translate: tuple = None, # 一个二元组 (dx, dy)，表示旋转后对图像的平移（移动）。这会在旋转过程中改变剪辑的位置。
            bg_color: tuple = None, # 背景颜色，指定旋转时可能出现的空白区域的填充颜色。这个参数通常与 expand=True 配合使用。
    ):
        """
        通过 ``angle`` 角度（度数或弧度）逆时针旋转指定的剪辑。
        如果角度不是 90 的倍数，或者 ``center``、``translate`` 和 ``bg_color`` 不是 ``None``，则会进行更复杂的旋转。
        有关参数的更多信息，请参见 ``vfx.Rotate``。

        功能：
            该方法通过调用 with_effects 方法，将一个 Rotate 效果应用到视频剪辑上，实现旋转操作。
            Rotate 类是一个视频效果类，专门用于旋转视频。在旋转时，如果角度不是 90 的倍数，或者需要使用指定的旋转中心、平移或背景色时，旋转操作会变得更复杂。
            这个方法允许对旋转进行精确控制，能够选择旋转中心、旋转后图像的处理方式（是否扩展、平移等），并且可以设置背景颜色。

        返回值：
        返回一个新的视频剪辑，经过旋转后的效果。
        """
        return self.with_effects(
            [
                Rotate(
                    angle=angle,
                    unit=unit,
                    resample=resample,
                    expand=expand,
                    center=center,
                    translate=translate,
                    bg_color=bg_color,
                )
            ]
        )

    def cropped(
            self,
            x1: int = None, # 裁剪区域左上角的 x 坐标（以像素为单位）。可以是浮动值，表示裁剪区域的起始位置。
            y1: int = None, # 裁剪区域左上角的 y 坐标（以像素为单位）。可以是浮动值，表示裁剪区域的起始位置。
            x2: int = None, # 裁剪区域右下角的 x 坐标（以像素为单位）。可以是浮动值，表示裁剪区域的结束位置。
            y2: int = None, # 裁剪区域右下角的 y 坐标（以像素为单位）。可以是浮动值，表示裁剪区域的结束位置。
            width: int = None, # 如果提供了这个值，它会指定裁剪区域的宽度。x1 和 y1 定义了裁剪区域的起始位置，width 会指定裁剪区域的宽度。
            height: int = None, # 如果提供了这个值，它会指定裁剪区域的高度。y1 和 x1 定义了裁剪区域的起始位置，height 会指定裁剪区域的高度。
            x_center: int = None, # 裁剪区域的 x 轴中心。通过 x_center 和 width，可以定义裁剪区域的中心和宽度。
            y_center: int = None, # 裁剪区域的 y 轴中心。通过 y_center 和 height，可以定义裁剪区域的中心和高度。
    ):
        """
        返回一个新的剪辑，其中仅保留原始剪辑中的矩形子区域。
        x1,y1 表示裁剪区域的左上角，x2,y2 表示裁剪区域的右下角。
        所有坐标均以像素为单位。可以接受浮动数值。
        有关参数的更多信息，请参见 ``vfx.Crop``。

        功能：
            这个方法会返回一个新的剪辑，裁剪出的区域是原始剪辑的一部分。裁剪是通过调用 Crop 效果实现的，Crop 类会根据传入的参数裁剪出一个矩形区域。
            如果同时提供了 x1, y1 和 x2, y2，它们表示裁剪区域的左上角和右下角。如果提供了 width 和 height，则从左上角 x1, y1 位置开始，裁剪出指定大小的矩形区域。
            如果指定了 x_center 和 y_center，则裁剪区域将围绕这个中心点展开。

        返回值：
            返回一个新的 VideoClip，其中包含裁剪后的部分。
        """
        return self.with_effects(
            [
                Crop(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    width=width,
                    height=height,
                    x_center=x_center,
                    y_center=y_center,
                )
            ]
        )

    # --------------------------------------------------------------
    # CONVERSIONS TO OTHER TYPES 转换为其他类型

    @convert_parameter_to_seconds(["t"])
    def to_ImageClip(self, t=0, with_mask=True, duration=None):
        """
        返回由时间“t”的剪辑帧构成的 ImageClip，可以以秒（15.35）、（分，秒）
        （时，分，秒）表示，也可以用字符串表示：'01:03:05.35'。
        """
        new_clip = ImageClip(self.get_frame(t), is_mask=self.is_mask, duration=duration)
        if with_mask and self.mask is not None:
            new_clip.mask = self.mask.to_ImageClip(t)
        return new_clip

    def to_mask(self, canal=0):
        """ 返回由剪辑制作的视频剪辑的蒙版。 """
        if self.is_mask:
            return self
        else:
            new_clip = self.image_transform(lambda pic: 1.0 * pic[:, :, canal] / 255)
            new_clip.is_mask = True
            return new_clip

    def to_RGB(self):
        """ 返回由蒙版视频片段制作的非蒙版视频片段。 """
        if self.is_mask:
            new_clip = self.image_transform(
                lambda pic: np.dstack(3 * [255 * pic]).astype("uint8")
            )
            new_clip.is_mask = False
            return new_clip
        else:
            return self

    # ----------------------------------------------------------------
    # Audio 音频

    @outplace
    def without_audio(self):
        """ 删除剪辑的音频。 返回音频设置为“无”的剪辑副本。"""
        self.audio = None

    def __add__(self, other):
        if isinstance(other, VideoClip):
            from moviepy.video.compositing.CompositeVideoClip import (
                concatenate_videoclips,
            )

            method = "chain" if self.size == other.size else "compose"
            return concatenate_videoclips([self, other], method=method)
        return super(VideoClip, self).__add__(other)

    def __or__(self, other):
        """
        Implement the or (self | other) to produce a video with self and other
        placed side by side horizontally.
        """
        if isinstance(other, VideoClip):
            from moviepy.video.compositing.CompositeVideoClip import clips_array

            return clips_array([[self, other]])
        return super(VideoClip, self).__or__(other)

    def __truediv__(self, other):
        """
        Implement division (self / other) to produce a video with self
        placed on top of other.
        """
        if isinstance(other, VideoClip):
            from moviepy.video.compositing.CompositeVideoClip import clips_array

            return clips_array([[self], [other]])
        return super(VideoClip, self).__or__(other)

    def __matmul__(self, n):
        """
        Implement matrice multiplication (self @ other) to rotate a video
        by other degrees
        """
        if not isinstance(n, Real):
            return NotImplemented

        from moviepy.video.fx.Rotate import Rotate

        return self.with_effects([Rotate(n)])

    def __and__(self, mask):
        """
        Implement the and (self & other) to produce a video with other
        used as a mask for self.
        """
        return self.with_mask(mask)


class DataVideoClip(VideoClip):
    """
    Class of video clips whose successive frames are functions
    of successive datasets

    Parameters
    ----------
    data
      A list of datasets, each dataset being used for one frame of the clip

    data_to_frame
      A function d -> video frame, where d is one element of the list `data`

    fps
      Number of frames per second in the animation
    """

    def __init__(self, data, data_to_frame, fps, is_mask=False, has_constant_size=True):
        self.data = data
        self.data_to_frame = data_to_frame
        self.fps = fps

        def frame_function(t):
            return self.data_to_frame(self.data[int(self.fps * t)])

        VideoClip.__init__(
            self,
            frame_function,
            is_mask=is_mask,
            duration=1.0 * len(data) / fps,
            has_constant_size=has_constant_size,
        )


class UpdatedVideoClip(VideoClip):
    """
    Class of clips whose frame_function requires some objects to
    be updated. Particularly practical in science where some
    algorithm needs to make some steps before a new frame can
    be generated.

    UpdatedVideoClips have the following frame_function:

    .. code:: python

        def frame_function(t):
            while self.world.clip_t < t:
                world.update() # updates, and increases world.clip_t
            return world.to_frame()

    Parameters
    ----------

    world
      An object with the following attributes:
      - world.clip_t: the clip's time corresponding to the world's state.
      - world.update() : update the world's state, (including increasing
      world.clip_t of one time step).
      - world.to_frame() : renders a frame depending on the world's state.

    is_mask
      True if the clip is a WxH mask with values in 0-1

    duration
      Duration of the clip, in seconds

    """

    def __init__(self, world, is_mask=False, duration=None):
        self.world = world

        def frame_function(t):
            while self.world.clip_t < t:
                world.update()
            return world.to_frame()

        VideoClip.__init__(
            self, frame_function=frame_function, is_mask=is_mask, duration=duration
        )


"""---------------------------------------------------------------------

    ImageClip (base class for all 'static clips') and its subclasses
    ColorClip and TextClip.
    I would have liked to put these in a separate file but Python is bad
    at cyclic imports.

---------------------------------------------------------------------"""


# ImageClip 类继承自 VideoClip 类，表示它是一个视频剪辑，但它表示的是静态图像。
class ImageClip(VideoClip):
    """
    用于非移动 VideoClips 的类。
    由图片生成的视频剪辑。此剪辑将始终显示给定的图片。

    例子
    --------
    >>> clip = ImageClip("myHouse.jpeg")
    >>> clip = ImageClip( someArray ) # 一个表示图像的 Numpy 数组

    属性
    ----------
    img
      表示剪辑图像的数组。
    """

    def __init__(
            self,
            img,  # 任何图片文件（png、tiff、jpeg 等）的字符串或路径类对象， 或表示 RGB 图像的任何数组（例如来自 VideoClip 的帧）。
            is_mask=False,  # 如果剪辑是遮罩，请将此参数设置为 `True`。
            transparent=True,  # 如果要将图片的 alpha 层（如果存在）用作遮罩，请将此参数设置为 `True`（默认）。
            fromalpha=False, # 将 alpha 通道单独作为 mask 使用，一般用不上，默认 False
            duration=None # 剪辑的持续时间。
    ):
        VideoClip.__init__(self, is_mask=is_mask, duration=duration)

        # 如果 img 不是 NumPy 数组，则从文件读取图像。
        # 处理图像的 alpha 通道，根据 is_mask 和 transparent 参数创建遮罩。
        # 设置 frame_function，使其始终返回图像数据。
        # 设置 size 和 img 属性。

        if not isinstance(img, np.ndarray):
            # img 是字符串或路径类对象，因此从磁盘读取
            img = imread_v2(img)  # 我们使用 v2 imread，因为 v3 在 gif 上失败

        if len(img.shape) == 3:  # img（现在）是一个 RGB(a) numpy 数组
            if img.shape[2] == 4:
                if fromalpha:
                    img = 1.0 * img[:, :, 3] / 255
                elif is_mask:
                    img = 1.0 * img[:, :, 0] / 255
                elif transparent:
                    self.mask = ImageClip(1.0 * img[:, :, 3] / 255, is_mask=True)
                    img = img[:, :, :3]
            elif is_mask:
                img = 1.0 * img[:, :, 0] / 255

        # 如果图像只是一个 2D 遮罩，它应该在此处保持不变
        # unchanged
        self.frame_function = lambda t: img
        self.size = img.shape[:2][::-1]
        self.img = img

    def transform(self, func, apply_to=None, keep_duration=True):
        """通用转换过滤器。
        等效于 VideoClip.transform。结果不再是 ImageClip，而是 VideoClip 类（因为它可能是动画）。
        """
        if apply_to is None:
            apply_to = []
        # 当我们在图像剪辑上使用 transform 时，它可能会变成动画。
        # 因此，结果不是 ImageClip，只是 VideoClip。
        new_clip = VideoClip.transform(
            self, func, apply_to=apply_to, keep_duration=keep_duration
        )
        new_clip.__class__ = VideoClip
        return new_clip

    @outplace
    def image_transform(self, image_func, apply_to=None):
        """图像转换过滤器。

        与 VideoClip.image_transform 相同，但对于 ImageClip，
        转换后的剪辑在开始时一次性计算，而不是为每个“帧”计算。
        """
        if apply_to is None:
            apply_to = []
        arr = image_func(self.get_frame(0))
        self.size = arr.shape[:2][::-1]
        self.frame_function = lambda t: arr
        self.img = arr

        for attr in apply_to:
            a = getattr(self, attr, None)
            if a is not None:
                new_a = a.image_transform(image_func)
                setattr(self, attr, new_a)

    @outplace
    def time_transform(self, time_func, apply_to=None, keep_duration=False):
        """时间转换过滤器。

        将转换应用于剪辑的时间线（请参阅 Clip.time_transform）。
        此方法对 ImageClip 无效（但可能会影响其遮罩或音频）。
        结果仍然是 ImageClip。
        """
        if apply_to is None:
            apply_to = ["mask", "audio"]
        for attr in apply_to:
            a = getattr(self, attr, None)
            if a is not None:
                new_a = a.time_transform(time_func)
                setattr(self, attr, new_a)


# ColorClip 类继承自 ImageClip 类，表示它是一个静态图像剪辑，但它显示的是单一颜色。
# 它可以用于创建背景、遮罩或简单的颜色填充效果。
class ColorClip(ImageClip):
    """一个只显示一种颜色的 ImageClip。"""

    def __init__(
            self,
            size, # 剪辑的尺寸元组（宽度，高度），以像素为单位。
            color=None,
            #     color 剪辑的颜色
            #       如果 is_mask 为 False，则 color 是一个 RGB 元组，例如 (255, 0, 0) 表示红色。
            #       如果 is_mask 为 True，则 color 是一个 0 到 1 之间的浮点数，表示遮罩的不透明度。
            #       如果 color 为 None，则默认颜色为黑色（is_mask 为 False）或 0（is_mask 为 True）。
            is_mask=False, # 如果剪辑将用作遮罩，则设置为 true。
            duration=None # 剪辑的持续时间。
    ):
        w, h = size

        if is_mask:
            shape = (h, w)
            if color is None:
                color = 0
            elif not np.isscalar(color):
                raise Exception("当 mask 为 true 时，Color 必须是标量")
        else:
            if color is None:
                color = (0, 0, 0)
            elif not hasattr(color, "__getitem__"):
                raise Exception("Color 必须包含剪辑的 RGB")
            elif isinstance(color, str):
                raise Exception(
                    "Color 不能是字符串。Color 必须包含剪辑的 RGB"
                )
            shape = (h, w, len(color))

        super().__init__(
            np.tile(color, w * h).reshape(shape), is_mask=is_mask, duration=duration
        )


# TextClip 类继承自 ImageClip 类，表示它是一个静态图像剪辑，但它显示的是文本。
# TextClip 类用于创建由文本生成的视频剪辑。
# 它允许您指定字体、文本、大小、颜色、边距等参数。
class TextClip(ImageClip):
    """
    自动生成文本剪辑的类。
    创建一个由脚本生成的文本图像产生的 ImageClip。

    .. note::
      ** 关于最终 TextClip 尺寸 **

      最终 TextClip 尺寸将是字体和行数的绝对最大高度。
      它特别意味着最终高度可能比实际文本高度稍大，即文本的绝对底部像素 - 文本的绝对顶部像素。
      这是因为在字体中，某些字母高于标准顶线（例如，带有重音的字母），
      而某些字母低于标准基线（例如，p、y、g 等字母）。

      此概念被称为 ascent 和 descent，表示基线上方和下方的最高和最低像素。

      如果您的第一行没有“重音字符”，而最后一行没有“降序字符”，则会有一些“脂肪”围绕着文本。
    """

    @convert_path_to_string("filename")
    def __init__(
            self,
            font=None,  # 要使用的字体的路径。必须是 OpenType 字体。如果设置为 None（默认），将使用 Pillow 默认字体。
            text=None,  # 要写入的文本字符串。可以使用参数 ``filename`` 替换。
            filename=None,  # 包含要写入的文本的文件的名称，作为字符串或路径类对象。 可以代替参数 ``text`` 提供。
            font_size=None,  # 字体大小，以磅为单位。如果 method='caption' 或 method='label' 且设置了 size，则可以自动设置。
            size=(None, None),
            #       （字体区域的大小）图片的尺寸，以像素为单位。如果 method='label' 且 font_size 已设置，则可以自动设置；
            #       如果 method='caption 字幕'，则必须设置。如果定义了 font_size，则 caption 的高度可以为 None，
            #       它将自动确定。
            margin=(None, None),
            #       要添加到文本周围的边距，作为两个（对称）或四个（不对称）的元组。
            #       可以是 ``(水平, 垂直)`` 或 ``(左, 上, 右, 下)``。默认情况下，没有边距 (None, None)。
            #       这对于自动计算尺寸以给文本一些额外空间特别有用。
            color="black",  # 文本的颜色。默认为“黑色”。可以是 RGB（或 RGBA，如果 transparent = ``True``）``tuple``、颜色名称或十六进制表示法。
            bg_color=None,  # 背景颜色。如果不需要背景，则默认为 None。可以是 RGB（或 RGBA，如果 transparent = ``True``）``tuple``、颜色名称或十六进制表示法。
            stroke_color=None,  # 文本的描边（=轮廓线）颜色。如果为 ``None``，则没有描边。
            stroke_width=0,  # 描边的宽度，以像素为单位。必须是整数。
            method="label",
            #       可以是：
            #         - 'label'（默认），图片将自动调整大小以适应文本，
            #           如果提供了宽度，则自动计算字体大小；如果定义了字体大小，则自动计算宽度和高度。
            #
            #         - 'caption'，文本将在使用 ``size`` 参数提供的固定尺寸的图片中绘制。
            #           文本将自动换行，如果提供了宽度和高度，则自动计算字体大小；
            #           如果定义了字体大小，则在必要时添加换行符。
            text_align="left",  # center | left | right。类似于 CSS 的文本对齐方式。默认为 ``left``。
            horizontal_align="center",  # center | left | right。定义图像中文本块的水平对齐方式。默认为 ``center``。
            vertical_align="center",  # center | top | bottom。定义图像中文本块的垂直对齐方式。默认为 ``center``。
            interline=4,  # 行间距。默认为 ``4``。
            transparent=True,  # 如果希望考虑图像中的透明度，则为 ``True``（默认）。
            duration=None,  # 剪辑的持续时间
    ):
        """
        这段代码实现了 TextClip 类的构造函数，用于创建由文本生成的视频剪辑。它处理字体加载、文本读取、尺寸计算、图像创建和文本绘制等任务。
        它还根据 method 参数实现了 "caption" 和 "label" 两种文本布局方法。
        """
        if font is not None:
            # 如果提供了 font 参数，则尝试使用 ImageFont.truetype(font) 加载字体。
            # 如果加载失败，则抛出 ValueError，指示字体无效。
            try:
                _ = ImageFont.truetype(font)
            except Exception as e:
                raise ValueError(
                    "Invalid font {}, pillow failed to use it with error {}".format(
                        font, e
                    )
                )

        if filename:
            # 如果提供了 filename 参数，则打开文件并读取文本内容。
            # text = file.read().rstrip()：读取文件内容并删除末尾的换行符。
            with open(filename, "r") as file:
                text = file.read().rstrip()  # 删除结尾处的换行符

        if text is None:
            # 如果 text 和 filename 都没有提供，则抛出 ValueError，指示缺少文本。
            raise ValueError("未提供文本或文件名")

        if method not in ["caption", "label"]:
            # 验证 method 参数是否为 "caption" 或 "label"。
            # 如果不是，则抛出 ValueError。
            raise ValueError("方法必须是`caption`或`label`。")

        # 计算边距并应用
        # 根据 margin 参数的长度，设置左、右、上、下边距。
        if len(margin) == 2:
            left_margin = right_margin = int(margin[0] or 0)
            top_margin = bottom_margin = int(margin[1] or 0)
        elif len(margin) == 4:
            left_margin = int(margin[0] or 0)
            top_margin = int(margin[1] or 0)
            right_margin = int(margin[2] or 0)
            bottom_margin = int(margin[3] or 0)
        else:
            # 如果 margin 不是长度为 2 或 4 的元组，则抛出 ValueError。
            raise ValueError("Margin必须是2或4个元素的元组。")

        # Compute all img and text sizes if some are missing
        img_width, img_height = size

        if method == "caption":  # (标题/字幕)
            if img_width is None:
                # 如果 method 为 "caption"，则 img_width 必须提供 img_height 和 font_size 可以自动计算。
                raise ValueError("当方法是caption(标题/字幕)时，尺寸是必需的")

            if img_height is None and font_size is None:
                raise ValueError(
                    "当方法是caption(标题/字幕)且字体大小为无时，高度是必需的"
                )

            if font_size is None:
                # 使用 __find_optimum_font_size 和 __find_text_size 方法计算尺寸。
                font_size = self.__find_optimum_font_size(
                    text=text,
                    font=font,
                    stroke_width=stroke_width,
                    align=text_align,
                    spacing=interline,
                    width=img_width,
                    height=img_height,
                    allow_break=True,
                )

            # 需要时添加换行符
            # 使用__break_text进行换行。
            text = "\n".join(
                self.__break_text(
                    width=img_width,
                    text=text,
                    font=font,
                    font_size=font_size,
                    stroke_width=stroke_width,
                    align=text_align,
                    spacing=interline,
                )
            )

            if img_height is None:
                # 使用 __find_text_size 方法计算尺寸。
                img_height = self.__find_text_size(
                    text=text,
                    font=font,
                    font_size=font_size,
                    stroke_width=stroke_width,
                    align=text_align,
                    spacing=interline,
                    max_width=img_width,
                    allow_break=True,
                )[1]

        elif method == "label":
            if font_size is None and img_width is None:
                # 如果 method 为 "label"，则 font_size 或 img_width 必须提供，其余尺寸可以自动计算。
                raise ValueError(
                    "当方法为label(标签)且大小为无时，字体大小是必需的"
                )

            if font_size is None:
                font_size = self.__find_optimum_font_size(
                    text=text,
                    font=font,
                    stroke_width=stroke_width,
                    align=text_align,
                    spacing=interline,
                    width=img_width,
                    height=img_height,
                )

            if img_width is None:
                img_width = self.__find_text_size(
                    text=text,
                    font=font,
                    font_size=font_size,
                    stroke_width=stroke_width,
                    align=text_align,
                    spacing=interline,
                )[0]

            if img_height is None:
                img_height = self.__find_text_size(
                    text=text,
                    font=font,
                    font_size=font_size,
                    stroke_width=stroke_width,
                    align=text_align,
                    spacing=interline,
                    max_width=img_width,
                )[1]

        # img_width += left_margin + right_margin 和 img_height += top_margin + bottom_margin：添加边距到图像尺寸。
        img_width += left_margin + right_margin
        img_height += top_margin + bottom_margin

        # Trace the image
        # img_mode = "RGBA" if transparent else "RGB"：根据 transparent 参数设置图像模式。
        img_mode = "RGBA" if transparent else "RGB"

        if bg_color is None and transparent:
            # 如果背景颜色为 None 且 transparent 为 True，则设置背景颜色为透明黑色。
            bg_color = (0, 0, 0, 0)

        img = Image.new(img_mode, (img_width, img_height), color=bg_color)  # 创建新的图像。

        # 加载字体。
        if font:
            pil_font = ImageFont.truetype(font, font_size)
        else:
            pil_font = ImageFont.load_default(font_size)

        # 创建绘图对象。
        draw = ImageDraw.Draw(img)

        # 这里不需要break，因为我们已经在标题中break了
        # 计算文本的尺寸。
        text_width, text_height = self.__find_text_size(
            text=text,
            font=font,
            font_size=font_size,
            stroke_width=stroke_width,
            align=text_align,
            spacing=interline,
            max_width=img_width,
        )

        # 根据 horizontal_align 和 vertical_align 参数，计算文本的 x 和 y 坐标。
        x = 0
        if horizontal_align == "right":
            x = img_width - text_width - left_margin - right_margin
        elif horizontal_align == "center":
            x = (img_width - left_margin - right_margin - text_width) / 2

        y = 0
        if vertical_align == "bottom":
            y = img_height - text_height - top_margin - bottom_margin
        elif vertical_align == "center":
            y = (img_height - top_margin - bottom_margin - text_height) / 2

        # 我们使用基线作为我们的锚，因为它是可预测和可靠的
        # 这意味着我们必须始终使用左基线。否则我们会
        # 总是有一个无用的保证金（上升和顶部之间的差异）在任何
        # 文本。这意味着我们的Y实际上不是从0到顶部，而是需要从0到顶部。
        # 递增，因为我们必须从基线开始参考。
        (ascent, _) = pil_font.getmetrics()  # 获取字体的 ascent。
        y += ascent  # 调整 y 坐标，以基于基线绘制文本。

        # 向起点添加边距和描边大小
        y += top_margin
        x += left_margin
        y += stroke_width
        x += stroke_width

        # 使用 draw 对象绘制文本。
        draw.multiline_text(
            xy=(x, y),
            text=text,
            fill=color,
            font=pil_font,
            spacing=interline,
            align=text_align,
            stroke_width=stroke_width,
            stroke_fill=stroke_color,
            anchor="ls",
        )

        # 我们只需要图像作为一个numpy数组
        img_numpy = np.array(img)  # 将图像转换为 NumPy 数组。

        # 调用父类 ImageClip 的构造函数。
        ImageClip.__init__(
            self, img=img_numpy, transparent=transparent, duration=duration
        )
        # 设置 self.text、self.color 和 self.stroke_color 属性。
        self.text = text
        self.color = color
        self.stroke_color = stroke_color

    def __break_text(
            self, width, text, font, font_size, stroke_width, align, spacing
    ) -> List[str]:
        """Break text to never overflow a width"""
        img = Image.new("RGB", (1, 1))
        if font:
            font_pil = ImageFont.truetype(font, font_size)
        else:
            font_pil = ImageFont.load_default(font_size)
        draw = ImageDraw.Draw(img)

        lines = []
        current_line = ""

        # We try to break on spaces as much as possible
        # if a text dont contain spaces (ex chinese), we will break when possible
        last_space = 0
        for index, char in enumerate(text):
            if char == " ":
                last_space = index

            temp_line = current_line + char
            temp_left, temp_top, temp_right, temp_bottom = draw.multiline_textbbox(
                (0, 0),
                temp_line,
                font=font_pil,
                spacing=spacing,
                align=align,
                stroke_width=stroke_width,
            )
            temp_width = temp_right - temp_left

            if temp_width >= width:
                # If we had a space previously, add everything up to the space
                # and reset last_space and current_line else add everything up
                # to previous char
                if last_space:
                    lines.append(temp_line[0:last_space])
                    current_line = temp_line[last_space + 1: index + 1]
                    last_space = 0
                else:
                    lines.append(current_line[0:index])
                    current_line = char
                    last_space = 0
            else:
                current_line = temp_line

        if current_line:
            lines.append(current_line)

        return lines

    def __find_text_size(
            self,
            text,
            font,
            font_size,
            stroke_width,
            align,
            spacing,
            max_width=None,
            allow_break=False,
    ) -> tuple[int, int]:
        """Find *real* dimensions a text will occupy, return a tuple (width, height)

        .. note::
            Text height calculation is quite complex due to how `Pillow` works.
            When calculating line height, `Pillow` actually uses the letter ``A``
            as a reference height, adding the spacing and the stroke width.
            However, ``A`` is a simple letter and does not account for ascent and
            descent, such as in ``Ô``.

            This means each line will be considered as having a "standard"
            height instead of the real maximum font size (``ascent + descent``).

            When drawing each line, `Pillow` will offset the new line by
            ``standard height * number of previous lines``.
            This mostly works, but if the spacing is not big enough,
            lines will overlap if a letter with an ascent (e.g., ``d``) is above
            a letter with a descent (e.g., ``p``).

            For our case, we use the baseline as the text anchor. This means that,
            no matter what, we need to draw the absolute top of our first line at
            ``0 + ascent + stroke_width`` to ensure the first pixel of any possible
            letter is aligned with the top border of the image (ignoring any
            additional margins, if needed).

            Therefore, our first line height will not start at ``0`` but at
            ``ascent + stroke_width``, and we need to account for that. Each
            subsequent line will then be drawn at
            ``index * standard height`` from this point. The position of the last
            line can be calculated as:
            ``(total_lines - 1) * standard height``.

            Finally, as we use the baseline as the text anchor, we also need to
            consider that the real size of the last line is not "standard" but
            rather ``standard + descent + stroke_width``.

            To summarize, the real height of the text is:
              ``initial padding + (lines - 1) * height + end padding``
            or:
              ``(ascent + stroke_width) + (lines - 1) * height + (descent + stroke_width)``
            or:
              ``real_font_size + (stroke_width * 2) + (lines - 1) * height``
        """
        img = Image.new("RGB", (1, 1))
        if font:
            font_pil = ImageFont.truetype(font, font_size)
        else:
            font_pil = ImageFont.load_default(font_size)
        ascent, descent = font_pil.getmetrics()
        real_font_size = ascent + descent
        draw = ImageDraw.Draw(img)

        # Compute individual line height with spaces using pillow internal method
        line_height = draw._multiline_spacing(font_pil, spacing, stroke_width)

        if max_width is not None and allow_break:
            lines = self.__break_text(
                width=max_width,
                text=text,
                font=font,
                font_size=font_size,
                stroke_width=stroke_width,
                align=align,
                spacing=spacing,
            )

            text = "\n".join(lines)

        # Use multiline textbbox to get width
        left, top, right, bottom = draw.multiline_textbbox(
            (0, 0),
            text,
            font=font_pil,
            spacing=spacing,
            align=align,
            stroke_width=stroke_width,
            anchor="ls",
        )

        # For height calculate manually as textbbox is not realiable
        line_breaks = text.count("\n")
        lines_height = line_breaks * line_height
        paddings = real_font_size + stroke_width * 2

        return (int(right - left), int(lines_height + paddings))

    def __find_optimum_font_size(
            self,
            text,
            font,
            stroke_width,
            align,
            spacing,
            width,
            height=None,
            allow_break=False,
    ):
        """Find the best font size to fit as optimally as possible
        in a box of some width and optionally height
        """
        max_font_size = width
        min_font_size = 1

        # Try find best size using bisection
        while min_font_size < max_font_size:
            avg_font_size = int((max_font_size + min_font_size) // 2)
            text_width, text_height = self.__find_text_size(
                text,
                font,
                avg_font_size,
                stroke_width,
                align,
                spacing,
                max_width=width,
                allow_break=allow_break,
            )

            if text_width <= width and (height is None or text_height <= height):
                min_font_size = avg_font_size + 1
            else:
                max_font_size = avg_font_size - 1

        # Check if the last font size tested fits within the given width and height
        text_width, text_height = self.__find_text_size(
            text,
            font,
            min_font_size,
            stroke_width,
            align,
            spacing,
            max_width=width,
            allow_break=allow_break,
        )
        if text_width <= width and (height is None or text_height <= height):
            return min_font_size
        else:
            return min_font_size - 1


class BitmapClip(VideoClip):
    """ 由颜色位图组成的剪辑。主要用于测试目的。 """

    DEFAULT_COLOR_DICT = {
        "R": (255, 0, 0),
        "G": (0, 255, 0),
        "B": (0, 0, 255),
        "O": (0, 0, 0),
        "W": (255, 255, 255),
        "A": (89, 225, 62),
        "C": (113, 157, 108),
        "D": (215, 182, 143),
        "E": (57, 26, 252),
        "F": (225, 135, 33),
    }

    @convert_parameter_to_seconds(["duration"])
    def __init__(
            self, bitmap_frames, *, fps=None, duration=None, color_dict=None, is_mask=False
    ):
        """从位图表示创建 VideoClip 对象。主要用于测试套件。

        参数
        ----------

        bitmap_frames
          帧列表。每个帧都是字符串列表。每个字符串表示一行颜色。每个颜色表示一个 (r, g, b) 元组。
          示例输入（2 帧，5x3 像素尺寸）：：

              [["RRRRR",
                "RRBRR",
                "RRBRR"],
               ["RGGGR",
                "RGGGR",
                "RGGGR"]]

        fps
          显示剪辑的每秒帧数。将根据帧总数计算“duration”。如果同时设置了“fps”和“duration”，则忽略“duration”。

        duration
          剪辑的总持续时间。将根据帧总数计算“fps”。如果同时设置了“fps”和“duration”，则忽略“duration”。

        color_dict
          可用于设置与“bitmap_frames”中使用的字母相对应的特定 (r, g, b) 值的字典。
          例如“{"A": (50, 150, 150)}”。

          默认为：：

              {
                "R": (255, 0, 0),
                "G": (0, 255, 0),
                "B": (0, 0, 255),
                "O": (0, 0, 0),  # “O”表示黑色
                "W": (255, 255, 255),
                # “A”、“C”、“D”、“E”、“F”表示任意颜色
                "A": (89, 225, 62),
                "C": (113, 157, 108),
                "D": (215, 182, 143),
                "E": (57, 26, 252),
              }

        is_mask
          如果剪辑将用作遮罩，则设置为 ``True``。
        """
        assert fps is not None or duration is not None

        self.color_dict = color_dict if color_dict else self.DEFAULT_COLOR_DICT

        frame_list = []
        for input_frame in bitmap_frames:
            output_frame = []
            for row in input_frame:
                output_frame.append([self.color_dict[color] for color in row])
            frame_list.append(np.array(output_frame))

        frame_array = np.array(frame_list)
        self.total_frames = len(frame_array)

        if fps is None:
            fps = self.total_frames / duration
        else:
            duration = self.total_frames / fps

        VideoClip.__init__(
            self,
            frame_function=lambda t: frame_array[int(t * fps)],
            is_mask=is_mask,
            duration=duration,
        )
        self.fps = fps

    def to_bitmap(self, color_dict=None):
        """返回表示剪辑每一帧的有效位图列表。
        如果未指定“color_dict”，则它将使用用于创建剪辑的相同“color_dict”。
        """
        color_dict = color_dict or self.color_dict

        bitmap = []
        for frame in self.iter_frames():
            bitmap.append([])
            for line in frame:
                bitmap[-1].append("")
                for pixel in line:
                    letter = list(color_dict.keys())[
                        list(color_dict.values()).index(tuple(pixel))
                    ]
                    bitmap[-1][-1] += letter

        return bitmap
