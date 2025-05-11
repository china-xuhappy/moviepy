from moviepy import VideoFileClip, ColorClip, CompositeVideoClip, ImageClip, VideoClip
import numpy as np

# 加载视频文件
video = VideoFileClip("example2.mp4")  # 缩小视频的尺寸

# 将蒙版转换为ImageClip，并将其标记为蒙版
# mask_clip = ImageClip("mask.png", is_mask=True, duration=1, fromalpha=True)
w, h = video.size
duration = video.duration
#
# # 中间白色区域的尺寸
# mask_width = 300
# mask_height = 200
#
# # 创建一个黑色背景的 mask（值为 0）
# mask_array = np.zeros((h, w))
#
# # 计算中间白色区域的位置
# x1 = (w - mask_width) // 2
# y1 = (h - mask_height) // 2
# x2 = x1 + mask_width
# y2 = y1 + mask_height
#
# # 将中间区域设为白色（值为 1）
# mask_array[y1:y2, x1:x2] = 1.0
#
# 遮罩动画的帧函数
def make_mask_frame(t):
    """
    返回每一帧的遮罩图像：中间白色矩形逐渐放大
    """
    progress = min(t / duration, 1)  # 进度 0~1

    # 动态计算矩形宽高（从很小到全屏）
    max_width, max_height = w, h
    width = int(progress * max_width)
    height = int(progress * max_height)

    # 生成黑背景
    mask = np.zeros((h, w))

    # 计算中心位置
    x1 = (w - width) // 2
    y1 = (h - height) // 2
    x2 = x1 + width
    y2 = y1 + height

    # 设置中间区域为白色（可见）
    mask[y1:y2, x1:x2] = 1.0

    return mask

mask_clip = VideoClip(make_mask_frame, duration=duration, is_mask=True).with_duration(0.5)

# 创建 ImageClip 作为遮罩，设置 is_mask=True
# mask_clip = ImageClip(mask_array, is_mask=True).with_duration(2)

# 将蒙版应用到视频上
final_clip = CompositeVideoClip([video.with_mask(mask_clip)])

# 导出最终视频
final_clip.write_videofile("final_video.mp4", fps=24)