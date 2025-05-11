import math
import re
import shutil
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from moviepy import VideoFileClip, ImageClip, VideoClip, CompositeVideoClip, TextClip, concatenate_videoclips, vfx, afx, \
    ColorClip
import numpy as np
import multiprocessing
from functools import partial
from proglog import ProgressBarLogger
import subprocess
import tempfile
import time
import pickle
import json
import requests
from urllib.parse import urlparse
import traceback

# '''
#     1. 增加特效
#         添加过渡效果：每个子视频可以在开头、结尾或中间添加不同的过渡效果（如：FadeIn、FadeOut、CrossFadeIn、CrossFadeOut）。你还可以调整过渡效果的持续时间和位置，以适应不同场景。
#         使用不同的滤镜效果：根据视频的内容使用 BlackAndWhite、InvertColors、GammaCorrection 等效果。你还可以调整这些效果的强度，来保持不同子视频的独特性。
    
#     2. 调整尺寸与位置
#         缩放视频：你可以稍微缩小或放大视频（例如，将视频缩小到原来尺寸的90%或110%），并将视频位置稍微偏移，甚至添加背景。
#         旋转视频：轻微旋转视频（例如，旋转5到10度），这样视频看起来会有新的视觉效果。
        
#     3. 颜色调整
#         颜色调节：轻微调整视频的颜色（如改变亮度、对比度、色温等），可以让每个子视频看起来不同。
#         饱和度调整：改变视频的饱和度，增加或减少颜色的鲜艳程度，产生不同的视觉效果。
        
#     4. 裁剪与添加蒙版
#         裁剪：你可以裁剪视频的不同区域，生成多个子视频，裁剪的区域可以是左侧、右侧、上方或下方。甚至可以在裁剪后应用蒙版，使特定区域的显示或隐藏。
#         使用遮罩：利用遮罩技术，可以创建各种效果，比如从黑色遮罩中逐渐显示内容，或者仅显示中间部分，其他区域透明。
        
#     5. 添加动态元素
#         可以每个频道都有一个动态吉祥物 gif，在某个区域展示。
#         频道的logo 和 名字，常驻展示
        
#     6. 字幕或音频修改
#         改变音频的音量或速度：除了视频内容外，还可以在每个子视频中改变音频的音量，或者调节音频的速度，以增加多样性。
#         替换音轨：为不同的视频片段替换音频，以确保每个子视频的音频内容不一样。
#         改变字幕语言，例如中文/英文
    
#     7. 拍摄时使用绿幕
#         绿幕替换背景图片或视频
        
#     需求是：多个频道（随便选），每个频道都有自己的模板要求，例如颜色，播放速度，倍率，转场等等。 都是随机搭配的，确保永远不会重样。
    
# '''

import uuid
import hashlib, random


from PIL import Image, ImageDraw
def create_rounded_logo(image_path, output_path):
    """将方形 logo 裁剪为圆形，并带透明背景"""
    img = Image.open(image_path).convert("RGBA")
    size = img.size

    # 创建透明遮罩
    mask = Image.new('L', size, 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0) + size, fill=255)

    # 应用遮罩
    rounded_img = Image.new('RGBA', size)
    rounded_img.paste(img, (0, 0), mask)

    # 保存为 PNG，保持 alpha 通道
    rounded_img.save(output_path)

def uuid_to_seed(uuid_str):
    """将 UUID 转为整型种子"""
    return int(hashlib.md5(uuid_str.encode()).hexdigest(), 16)


def get_random_filter_color(seed):
    """
    生成一个随机滤镜颜色，保持自然轻微偏色
    """
    random.seed(seed)  # 保证每个 uuid 一致生成相同风格

    base = 1.0  # 原始颜色为白（1.0, 1.0, 1.0）

    # 每个通道随机偏移 -0.1 ~ +0.1
    r = round(random.uniform(0.9, 1.1), 2)
    g = round(random.uniform(0.9, 1.1), 2)
    b = round(random.uniform(0.9, 1.1), 2)
    # 每次运行可能得到：
    #
    # (1.0, 0.95, 0.92) → 微暖色
    # (0.92, 1.0, 1.1) → 清冷感
    # (1.1, 1.05, 0.95) → 偏黄暖

    return (r, g, b)


def get_warm_soft_margin_color(seed) -> tuple:
    """
    返回一个柔和、观影友好的暖色调颜色
    """
    random.seed(seed)

    """
    生成一个随机的暖色调颜色，偏浅偏亮
    """
    r = random.randint(200, 255)
    g = random.randint(180, 240)
    b = random.randint(180, 230)
    return (r, g, b)


def parse_rgba_string(rgba_str):
    if not rgba_str or not isinstance(rgba_str, str):
        return None

    match = re.match(r"rgba?\((\d+),\s*(\d+),\s*(\d+),\s*([0-9.]+)\)", rgba_str)
    if not match:
        raise ValueError(f"Invalid rgba string: {rgba_str}")

    r, g, b, a = match.groups()
    return (
        int(r),
        int(g),
        int(b),
        int(float(a) * 255)  # convert alpha from 0.0~1.0 to 0~255
    )

def get_video_params_from_uuid(uuid_str):
    seed = uuid_to_seed(uuid_str)
    random.seed(seed)  # 保证每个 uuid 一致生成相同风格
    rotate_switch = random.choice([True, False])
    return {
        "Effects": {  # 特效
            "FadeinFadeout": {  # FadeIn FadeOut  afx.AudioFadeIn(4) afx.AudioFadeOut(3)
                "switch": random.choice([True, False]),  # 开头和结束淡入淡出特效开关
                "fadein": round(random.uniform(0.95, 2), 3),  # 淡入时间
                "fadeout": round(random.uniform(0.95, 2), 3),  # 淡出时间
            },
            "Speed": {  # MultiplySpeed
                "switch": random.choice([True, False]),  # 开头和结束淡入淡出特效开关
                "timeNode": round(random.uniform(1.5, 5), 2),  # 速度的区分前半段和后半段时间节点 duration / timeNode
                "firstHalf": round(random.uniform(0.95, 1), 3),  # 前半段速度
                "secondHalf": round(random.uniform(1, 1.05), 3),  # 后半段速度
            },
            "Resize": {  # 宽度和高度乘以百分之几
                "switch": True,  # 开头和结束淡入淡出特效开关
                "new_size": round(random.uniform(1.2, 1.5), 3)
            },
            "Rotate": {  # 旋转
                "switch": rotate_switch,  # 开头和结束淡入淡出特效开关
                "angle": round(random.uniform(-1, 1), 2)
            },
            "Margin": {
                "switch": True, 
                # "switch": True if rotate_switch else random.choice([True, False]),  # 如果有旋转，边框必须有，要遮住
                "marginSize": int(random.uniform(3, 8)),
                "marginColor": get_warm_soft_margin_color(seed)
            },

            # ===============下面都是颜色的调整====================
            "LumContrast": {  # 亮度对比度调整，常用来调节视频的整体亮度和对比度。
                "switch": True,  # 开头和结束淡入淡出特效开关
                "lum": round(random.uniform(-15, 15), 3),
                "contrast": round(random.uniform(-0.3, 0.3), 3),
            },
            "GammaCorrection": {
                "switch": True,  #
                "gamma": round(random.uniform(0.8, 1.3), 3),
            },
            "MultiplyColor": {
                "switch": True,  #
                "factor": round(random.uniform(0.85, 1.15), 2),
            },
            "SuperSample": {
                "switch": True,  #
                "factor": round(random.uniform(0.85, 1.15), 2),
            },
            "InvertColors": {  # 极端反差效果，适合结尾/片头/转场"冲击感"
                # 建议场景：
                # 开头1秒或结尾1秒
                # 某段节奏转折时用作视觉冲击（建议只用几秒）
                "switch": True,
                "startTime": round(random.uniform(0.8, 2), 2),
                "endTime": round(random.uniform(0.8, 2), 2),
            }
        },
        "ChannelContent": {
            "name": {
                "content": "Happy Chinese Kitchen",
                "color": get_warm_soft_margin_color(seed),
                "fontSize": 50,
            },
            "logo": {
                "location": random.choice(["top-left", "top-right", "bottom-left", "bottom-right"]),
            }
        }
    }

def get_position(logo_location: str):
    position_map = {
        'left-top': ('left', 'top'),
        'right-top': ('right', 'top'),
        'left-bottom': ('left', 'bottom'),
        'right-bottom': ('right', 'bottom'),
        'center-top': ('center', 'top'),
        'left-center': ('left', 'center'),
        'right-center': ('right', 'center'),
        'center-bottom': ('center', 'bottom'),
        'center-center': ('center', 'center'),
    }

    # 默认位置为右下角
    return position_map.get(logo_location, ('right', 'bottom'))

def common_template(params: dict, video: VideoClip) -> VideoClip:
    """
    公共模板，logo，吉祥物gif，频道名
    
    从云端URL获取logo图片，并从utils/ttf目录获取字体文件
    :return:
    """
    import requests
    import tempfile
    from urllib.parse import urlparse
    
    try:
        channel_content = params.get('ChannelContent', {})
        if not isinstance(channel_content, dict):
            channel_content = {}

        random_effects = random.uniform(0.5, 2.0)

        # 获取Logo
        logo_image = None
        if 'logo' in channel_content and 'url' in channel_content['logo']:
            logo_url = channel_content['logo']['url']
            logo_size = channel_content['logo']['size']
            try:
                # 创建临时文件保存logo
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                    logo_path = temp_file.name
                    print(f"下载logo: {logo_url}")
                    response = requests.get(logo_url, stream=True, timeout=10)  # 添加10秒超时
                    response.raise_for_status()  # 检查是否成功
                    temp_file.write(response.content)

                # rounded_logo_path = logo_path.replace('.png', '_rounded.png')
                # create_rounded_logo(logo_path, rounded_logo_path)

                # 确定logo位置
                logo_location = channel_content['logo'].get('location', 'bottom-right')
                position = get_position(logo_location)
                
                # 创建logo图像剪辑
                logo_image = ImageClip(
                    img=logo_path,
                    duration=video.duration
                ).resized(width=logo_size['width'], height=logo_size['height']).with_position(position)
                logo_image = logo_image.with_effects([vfx.CrossFadeIn(random_effects), vfx.CrossFadeOut(random_effects)])
                print(f"Logo加载成功，位置: {position}")
            except Exception as e:
                print(f"加载logo出错，将继续而不包含logo: {str(e)}")
                logo_image = None
                
        # 获取正确的字体路径
        # 确定字体文件路径（相对路径或绝对路径）
        import os
        font_file = "Roboto_SemiCondensed-Black.ttf"
        font_paths = [
            os.path.join("utils", "ttf", font_file),  # 相对路径
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "utils", "ttf", font_file),  # 从当前文件向上导航
            os.path.join(os.getcwd(), "utils", "ttf", font_file),  # 从当前工作目录
            os.path.abspath(os.path.join("utils", "ttf", font_file))  # 绝对路径
        ]
        
        # 尝试找到存在的字体文件路径
        font_path = None
        for path in font_paths:
            if os.path.exists(path):
                font_path = path
                print(f"找到字体文件: {font_path}")
                break
        
        if not font_path:
            print("警告: 找不到字体文件，使用默认字体")
            font_path = None  # TextClip会使用默认字体
        
        # 创建频道名称文本
        channel_name = None
        if 'name' in channel_content and 'content' in channel_content['name']:
            try:
                channel_content_name = channel_content['name']
                # 这个可以搞个动画，从中间移动下去
                name_location = channel_content_name['namePosition'].get('position', 'bottom-right')
                position = get_position(name_location)

                channel_name = TextClip(
                    font=font_path,
                    text=channel_content_name['content'],
                    font_size=channel_content_name.get('nameSize', 50),
                    color=channel_content_name.get('nameStyle_color', (0, 0, 0, 255)),
                    bg_color=channel_content_name.get('nameStyle_bgColor', None),
                    text_align=channel_content_name['namePosition']['textAlign'],
                    horizontal_align=channel_content_name['namePosition']['horizontalAlign'],
                    vertical_align=channel_content_name['namePosition']['verticalAlign'],
                    duration=video.duration,
                    margin=(int(channel_content_name['namePosition']['margin_x']), int(channel_content_name['namePosition']['margin_y']))
                ).with_position(position)
                channel_name = channel_name.with_effects([vfx.CrossFadeIn(random_effects), vfx.CrossFadeOut(random_effects)])
                print(f"频道名称添加成功: {channel_content['name']['content']}")
            except Exception as e:
                print(f"创建频道名称文本出错: {str(e)}")
                channel_name = None
                raise e

        # 合成所有元素
        clips = [video]
        if channel_name is not None:
            clips.append(channel_name)
        if logo_image is not None:
            clips.append(logo_image)
        
        clip = CompositeVideoClip(clips)
        
        # 清理临时文件
        if 'logo_path' in locals() and os.path.exists(locals()['logo_path']):
            try:
                os.remove(locals()['logo_path'])
                print(f"已删除临时logo文件: {locals()['logo_path']}")
            except Exception as e:
                print(f"清理临时logo文件失败: {str(e)}")
        
        return clip
    except Exception as e:
        print(f"在common_template中出现错误，返回原始视频: {str(e)}")
        traceback.print_exc()
        return video  # 如果出现任何错误，直接返回原始视频

def get_required_scale(angle_deg: float) -> float:
    """
    根据旋转角度计算最小安全缩放倍数（避免黑边）。
    -2° ~ +5° 时，推荐缩放范围为：1.0 - 1.2 之间。
    """
    angle = abs(angle_deg)

    # 0度不需要缩放
    if angle == 0:
        return 1.0

    # 线性拟合：角度越大，放大倍数越高
    # 2度 -> 1.10， 5度 -> 1.20
    if angle <= 5:
        scale = 1.0 + (angle / 5.0) * 0.20  # 从 1.0 线性增长到 1.2
    else:
        # 更大的角度你可以自己加设定，或做更强放大
        scale = 1.25

    return round(scale, 3)

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def apply_speed_ranges(clip, speed_ranges):
    duration = clip.duration
    n = len(speed_ranges)
    segment_duration = duration / n
    clips = []

    for i, speed_config in enumerate(speed_ranges):
        start = i * segment_duration
        end = (i + 1) * segment_duration
        speed = speed_config['value']

        subclip = clip.subclipped(start, end).with_speed_scaled(speed)
        clips.append(subclip)

    return concatenate_videoclips(clips)

def apply_magic_effects_with_uuid(params: dict, video_clip: VideoClip) -> VideoClip:
    """
    应用魔法效果到视频剪辑
    
    处理各种特效并应用到视频，包括大小调整、旋转、颜色调整等
    如果特效应用过程中出现错误，将跳过该特效并继续
    
    :param params: 特效参数字典
    :param video_clip: 原始视频剪辑
    :return: 应用了特效的视频剪辑
    """
    try:
        duration = video_clip.duration
        effects_params = params.get('Effects', {})
        
        print(f"开始应用魔法效果，共 {len(effects_params)} 种特效")
        
        # 临时接收可能的修改后的视频对象
        modified_clip = video_clip

        # 先缩放放大视频，保存后再加特效
        if 'Resize' in effects_params and effects_params['Resize']['switch']:
            new_size = effects_params['Resize'].get('new_size', 1.0)
            if new_size != 1.0:
                w, h = modified_clip.size

                if new_size > 1.0: # 缩放大于1，再旋转视频。但是要确保旋转的角度和缩放的大小
                    if 'Rotate' in effects_params and effects_params['Rotate']['switch']:
                        angle = effects_params['Rotate'].get('angle', 1.0)
                        required_scale = get_required_scale(angle)
                        # 防止太小缩放
                        new_size = max(required_scale, new_size)
                        print(f"  - 旋转角度: {angle}, 防止太小缩放: {new_size}")
                        modified_clip = modified_clip.resized(new_size=float(new_size))

                        modified_clip = modified_clip.rotated(angle=float(angle), bg_color=(0, 0, 0), expand=True)
                    else:
                        modified_clip = modified_clip.resized(new_size=float(new_size))

                    # 添加随机偏移（例如在 ±30 像素范围内）
                    x_offset = random.randint(-30, 30)
                    y_offset = random.randint(-30, 30)

                    x_center = min(max((modified_clip.w / 2) + x_offset, 0), modified_clip.w)
                    y_center = min(max((modified_clip.h / 2) + y_offset, 0), modified_clip.h)
                    modified_clip = modified_clip.cropped(x_center=x_center, y_center=y_center, width=w, height=h)

                else:  # 视频缩小，需要东西填充黑边
                    # 创建一个背景颜色为黑色的画布，尺寸为原视频尺寸
                    modified_clip = modified_clip.resized(new_size=float(new_size))

                    background = ColorClip(size=(w, h), color=(100, 100, 100), duration=modified_clip.duration)
                    modified_clip = modified_clip.with_position("center")
                    modified_clip = CompositeVideoClip([background, modified_clip])

                print(f"  - 调整大小: {new_size}")

        if 'Speed' in effects_params and effects_params['Speed']['switch']:
            speed_ranges = effects_params['Speed'].get('speedRanges', [])
            modified_clip = apply_speed_ranges(modified_clip, speed_ranges)

        effects = []
        # 尝试应用所有特效
        for effect_name, effect_config in effects_params.items():
            if not isinstance(effect_config, dict) or not effect_config.get('switch', False):
                continue
                
            try:
                print(f"应用特效: {effect_name}")
                if effect_name == 'Margin' and effect_config.get('switch', False):
                    margin_size = int(effect_config.get('marginSize', 0))
                    margin_color = hex_to_rgb(effect_config.get('marginColor', '#FF0000'))
                    effects.append(
                        vfx.Margin(margin_size=margin_size, color=margin_color)
                    )
                    print(f"  - 边距: {margin_size}, 颜色: {margin_color}")

                elif effect_name == 'LumContrast' and effect_config.get('switch', False):
                    lum = effect_config.get('lum', 0)
                    contrast = effect_config.get('contrast', 0)
                    effects.append(vfx.LumContrast(lum=float(lum), contrast=float(contrast)))
                    print(f"  - 亮度: {lum}, 对比度: {contrast}")
                
                elif effect_name == 'GammaCorrection' and effect_config.get('switch', False):
                    gamma = effect_config.get('gamma', 1.0)
                    effects.append(vfx.GammaCorrection(gamma=float(gamma)))
                    print(f"  - 伽马校正: {gamma}")
                
                elif effect_name == 'MultiplyColor' and effect_config.get('switch', False):
                    factor = effect_config.get('factor', 1.0)
                    effects.append(vfx.MultiplyColor(factor=float(factor)))
                    print(f"  - 颜色乘数: {factor}")
                
                elif effect_name == 'InvertColors' and effect_config.get('switch', False):
                    # 这里需要特殊处理，因为InvertColors没有直接的持续时间参数
                    try:
                        start_time = float(effect_config.get('startTime', 0))
                        end_time = float(effect_config.get('endTime', 0))
                        
                        if start_time > 0 and start_time < duration:
                            # 只在视频开头应用反色效果
                            part1 = video_clip.subclipped(0, start_time).with_effects([vfx.InvertColors()])
                            part2 = video_clip.subclipped(start_time)
                            modified_clip = concatenate_videoclips([part1, part2])
                            print(f"  - 开头反色效果: 0-{start_time}秒")
                            
                        if end_time > 0 and end_time < duration:
                            # 只在视频结尾应用反色效果
                            end_start = max(0, duration - end_time)
                            part1 = modified_clip.subclipped(0, end_start)
                            part2 = modified_clip.subclipped(end_start).with_effects([vfx.InvertColors()])
                            modified_clip = concatenate_videoclips([part1, part2])
                            print(f"  - 结尾反色效果: {end_start}-{duration}秒")
                    except Exception as e:
                        print(f"应用反色特效时出错: {str(e)}")
                
            except Exception as e:
                print(f"应用特效 {effect_name} 时出错: {str(e)}")
        
        # 应用所有收集的特效
        print("effects ->", effects)
        if effects:
            modified_clip = modified_clip.with_effects(effects=effects)
            print(f"成功应用了 {len(effects)} 个特效")

        # 最后应用模板效果（logo和频道名称）
        result_clip = common_template(params, modified_clip)
        print("模板效果应用完成")
        
        return result_clip
        
    except Exception as e:
        print(f"应用魔法效果时出现严重错误: {str(e)}")
        traceback.print_exc()
        # 如果出现严重错误，返回原始视频
        return video_clip


# 增加高性能多进程帧保存函数
def save_frames_parallel(clip, output_dir, time_points, process_id, quality=95):
    """
    并行保存指定时间点的帧
    
    参数:
        clip: 视频剪辑
        output_dir: 输出目录
        time_points: 要保存的时间点列表 [(index, time), ...]
        process_id: 进程ID，用于打印
        quality: JPEG压缩质量
    """
    try:
        import os
        import imageio
        
        # 确保目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 为调试性能添加时间测量
        start_time = time.time()
        count = len(time_points)
        
        print(f"进程 {process_id}: 开始生成 {count} 帧")
        
        # 处理每个时间点
        for i, (frame_index, t) in enumerate(time_points):
            # 每处理10%的帧输出一次进度
            if i % max(1, count // 10) == 0:
                elapsed = time.time() - start_time
                fps = (i+1) / elapsed if elapsed > 0 else 0
                print(f"进程 {process_id}: {i+1}/{count} 帧 ({(i+1)/count*100:.1f}%), {fps:.1f} fps")
            
            # 生成帧
            frame = clip.get_frame(t)
            
            # 使用frame_index确保帧的正确顺序
            filename = os.path.join(output_dir, f"frame_{frame_index:06d}.jpg")
            
            # 保存为JPEG，效率更高
            imageio.imwrite(
                filename, 
                (frame * 255).astype('uint8'), 
                format='jpeg', 
                quality=quality
            )
            
        elapsed = time.time() - start_time
        fps = count / elapsed if elapsed > 0 else 0
        print(f"进程 {process_id}: 完成 {count} 帧, 平均速度: {fps:.2f} fps")
        
        return process_id, count
        
    except Exception as e:
        import traceback
        print(f"进程 {process_id} 错误: {str(e)}")
        traceback.print_exc()
        return process_id, 0


# 测试NVIDIA GPU支持
def check_nvidia_gpu():
    """检测系统是否有NVIDIA GPU并支持NVENC"""
    try:
        # 首先检查ffmpeg是否支持nvenc编码器
        result = subprocess.run(
            ["ffmpeg", "-encoders"], 
            capture_output=True, 
            text=True
        )
        
        if "h264_nvenc" not in result.stdout:
            print("您的ffmpeg不支持h264_nvenc编码器")
            return False
            
        # 然后尝试运行一个简单的测试命令
        test_cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "color=black:s=32x32:r=1:d=1",
            "-c:v", "h264_nvenc", "-f", "null", "-"
        ]
        
        test_result = subprocess.run(
            test_cmd,
            capture_output=True,
            text=True
        )
        
        # 如果命令成功且没有错误输出，说明GPU可用
        if test_result.returncode == 0 and not test_result.stderr:
            print("NVENC硬件编码器测试成功")
            return True
        else:
            print(f"NVENC硬件编码器测试失败: {test_result.stderr}")
            return False
            
    except Exception as e:
        print(f"检测GPU失败: {e}")
        return False

def get_color(color_value):

    if isinstance(color_value, (list, tuple)) and len(color_value) >= 3:
        # 如果是RGB或RGBA格式
        color_value = color_value
    elif isinstance(color_value, dict) and 'value' in color_value:
        # 如果是{value: [r,g,b]}格式
        color_value = color_value['value']
    elif isinstance(color_value, str):
        # 如果是十六进制颜色字符串，转换为RGB
        try:
            if color_value.startswith('#'):
                color_value = color_value[1:]
            if len(color_value) == 6:
                r = int(color_value[0:2], 16)
                g = int(color_value[2:4], 16)
                b = int(color_value[4:6], 16)
                color_value = (r, g, b)
        except Exception as e:
            print(f"解析颜色错误: {e}")
            color_value = (255, 255, 255)  # 默认白色

    return color_value

def parse_effects_from_config(config):
    """
    从配置文件解析特效参数，转换为MoviePy可用的特效配置
    
    参数:
        config: 包含频道配置和特效参数的字典
        
    返回:
        effects_params: 包含Effects和ChannelContent的特效参数字典
    """
    try:
        # 初始化特效参数
        effects_params = {
            'Effects': {},
            'ChannelContent': {}
        }
        
        # 处理频道内容配置
        contents = config.get('contents', {})
        
        # 处理logo设置
        logo_url = None
        if 'channelLogo' in contents and 'logo' in contents['channelLogo']:
            logo_config = contents['channelLogo']['logo']
            for logo_type, type_config in logo_config.items():
                if isinstance(type_config, dict) and 'url' in type_config:
                    logo_url = type_config['url']
                    break
                elif logo_type == 'url' and isinstance(type_config, str):
                    logo_url = type_config
                    break

            effects_params['ChannelContent']['logo'] = {
                'location': logo_config.get('position', 'bottom-right'),
                'url': logo_url,
                'x': logo_config.get('x', 0),
                'y': logo_config.get('y', 0),
                'size': logo_config.get('size', 0)
            }
        
        # 处理频道名称设置
        channel_name = {
            'content': config['channel'].get('title', '频道名称')
        }
        if 'channelName' in contents and isinstance(contents['channelName'], dict):
            name_config = contents.get('channelName', {})
            # 处理颜色配置
            if 'nameStyle_color' in name_config:
                color_value = parse_rgba_string(name_config['nameStyle_color'])
                channel_name['nameStyle_color'] = color_value

            if 'nameStyle_bgColor' in name_config:
                color_value = parse_rgba_string(name_config['nameStyle_bgColor'])
                channel_name['nameStyle_bgColor'] = color_value

            if 'nameStyle_transparent' in name_config:
                channel_name['nameStyle_transparent'] = name_config['nameStyle_transparent']

            # 处理字体大小
            if 'nameSize' in name_config:
                size_value = name_config['nameSize']
                channel_name['nameSize'] = size_value

            if 'namePosition' in name_config:
                position = name_config['namePosition']
                channel_name['namePosition'] = position

            effects_params['ChannelContent']['name'] = channel_name

        # 设置基本效果配置
        uuid_str = str(uuid.uuid4())
        seed = uuid_to_seed(uuid_str)
        random.seed(seed)  # 保证每个 uuid 一致生成相同风格

        effects_params['Effects'] = {}
        
        # 应用来自配置的自定义效果
        custom_effects = config.get('effects', {})
        for effect_name, effect_data in custom_effects.items():
            category = effect_data.get('category', '')
            params = effect_data.get('parameters', {})
            
            # 基于效果类别和名称映射到MoviePy效果
            if category == 'Resize':
                effects_params['Effects'] = {
                    'Resize': {}
                }
                effects_params['Effects']['Resize']['new_size'] = float(params['scale'])
                effects_params['Effects']['Resize']['switch'] = True
            
            elif category == 'Rotate':
                effects_params['Effects'] = {
                    'Rotate': {}
                }
                effects_params['Effects']['Rotate']['switch'] = True
                effects_params['Effects']['Rotate']['angle'] = float(params['angle'])

            elif category == 'Margin':
                effects_params['Effects'] = {
                    'Margin': {}
                }
                effects_params['Effects']['Margin']['switch'] = True
                effects_params['Effects']['Margin']['marginSize'] = int(params['size'])
                effects_params['Effects']['Margin']['marginColor'] = params.get('color', get_warm_soft_margin_color(seed))

            elif category == 'Speed':
                effects_params['Effects'] = {
                    'Speed': {}
                }
                # 速度调整 (需添加处理逻辑)
                if effect_name.lower() in ['speed', '速度调整']:
                    # 检查是否有时间段设置
                    if 'speedRanges' in params and isinstance(params['speedRanges'], list):
                        effects_params['Effects']['Speed'] = {
                            'switch': True,
                            'speedRanges': params['speedRanges']
                        }

                    print(f"速度调整: {effects_params['Effects']['Speed']}")

            elif category == 'LumContrast':
                # 亮度调整
                effects_params['Effects'] = {
                    'LumContrast': {}
                }
                if effect_name.lower() in ['brightness', '亮度-对比度调整']:
                    effects_params['Effects']['LumContrast']['switch'] = True
                    effects_params['Effects']['LumContrast']['lum'] = float(params.get('lum', 0))
                    effects_params['Effects']['LumContrast']['contrast'] = float(params.get('contrast', 0))
                    print(f"亮度-对比度调整: {effects_params['Effects']['LumContrast']}")

            #
            elif category == 'GammaCorrection':
                effects_params['Effects'] = {
                    'GammaCorrection': {}
                }
                # 伽马校正
                if effect_name.lower() in ['gamma', 'gammacorrection', '伽马校正']:
                    effects_params['Effects']['GammaCorrection']['switch'] = True
                    effects_params['Effects']['GammaCorrection']['gamma'] = float(params.get('gamma', 1.0))
                    print(f"伽马校正: {effects_params['Effects']['GammaCorrection']}")

            #
            #     # 颜色乘数
            #     elif effect_name.lower() in ['color', 'multiplycolor', '颜色乘数'] and ('factor' in params or 'value' in params):
            #         effects_params['Effects']['MultiplyColor']['factor'] = float(params.get('factor', params.get('value', 1.0)))
            #         effects_params['Effects']['MultiplyColor']['switch'] = True
            #
            #     # 饱和度调整 (需要添加到特效中)
            #     elif effect_name.lower() in ['saturation', '饱和度调整']:
            #         # 饱和度目前没有直接对应的MoviePy特效，需要扩展特效库
            #         print(f"饱和度调整暂未实现: {params}")
            #
            #     # 色相调整 (需要添加到特效中)
            #     elif effect_name.lower() in ['hue', '色相调整']:
            #         # 色相目前没有直接对应的MoviePy特效，需要扩展特效库
            #         print(f"色相调整暂未实现: {params}")
            #
            # elif category == 'filter':
            #     # 模糊效果 (需要添加到特效中)
            #     if effect_name.lower() in ['blur', '模糊效果']:
            #         # 模糊效果暂未实现
            #         print(f"模糊效果暂未实现: {params}")
            #
            #     # 锐化效果 (需要添加到特效中)
            #     elif effect_name.lower() in ['sharpen', '锐化效果']:
            #         # 锐化效果暂未实现
            #         print(f"锐化效果暂未实现: {params}")
            #
            #     # 噪点效果 (需要添加到特效中)
            #     elif effect_name.lower() in ['noise', '噪点效果']:
            #         # 噪点效果暂未实现
            #         print(f"噪点效果暂未实现: {params}")

            else:
                print(f"未知特效类别: {category}, 特效名称: {effect_name}, 参数: {params}")
                
        return effects_params
        
    except Exception as e:
        print(f"解析特效配置出错: {str(e)}")
        traceback.print_exc()
        # 返回默认配置
        return {
            'Effects': {
                'FadeinFadeout': {'switch': True, 'fadein': 1, 'fadeout': 1},
                'Resize': {'switch': True, 'new_size': 1.0}
            },
            'ChannelContent': {}
        }

def apply_magic_from_config(input_video_path, output_path, config):
    """
    根据传入的配置参数对视频应用魔法效果并输出
    
    参数:
        input_video_path: 输入视频文件路径
        output_path: 输出视频文件路径
        config: 包含频道配置和魔法效果的字典
        
    返回:
        包含处理结果的字典 {'success': bool, 'error': str}
    """
    try:
        print(f"开始应用魔法效果，输入视频: {input_video_path}, 输出: {output_path}")
        print(f"配置参数: {json.dumps(config, indent=2, ensure_ascii=False)}")
        
        # 加载视频
        video_clip = VideoFileClip(filename=input_video_path)
        
        # 从配置中解析特效参数
        effects_params = parse_effects_from_config(config)
        
        # 应用魔法效果
        video_with_effects = apply_magic_effects_with_uuid(effects_params, video_clip)
        
        # 检测是否有NVIDIA GPU支持
        has_nvidia = check_nvidia_gpu()
        
        # 为合成视频准备导出
        try:
            # 直接使用MoviePy自带的write_videofile方法，避免多进程序列化问题
            print("开始导出视频...")
            
            # 设置合适的预设和编码器
            codec = "h264_nvenc" if has_nvidia else "libx264"
            preset = "fast"
            
            # 如果使用NVIDIA GPU
            if has_nvidia:
                print(f"使用NVIDIA硬件加速编码: {codec}")
            else:
                print(f"使用CPU编码: {codec}")
            
            # 导出视频
            video_with_effects.write_videofile(
                output_path,
                fps=30,
                codec=codec,
                preset=preset,
                audio=video_clip.audio is not None,
                threads=4,  # 使用4个线程进行编码
                bitrate="5M"
            )
            
            print(f"视频已成功导出到: {output_path}")
        except Exception as e:
            error_msg = f"标准方法导出视频失败: {str(e)}"
            print(error_msg)
            traceback.print_exc()
            
            # 使用更简单的备用方法
            print("尝试使用备用方法导出视频...")
            try:
                video_with_effects.write_videofile(
                    output_path,
                    fps=30,
                    codec="libx264",
                    preset="ultrafast",  # 使用最快的预设
                    audio=video_clip.audio is not None
                )
                print(f"备用方法成功: 视频已导出到 {output_path}")
            except Exception as e2:
                error_msg = f"备用方法也失败: {str(e2)}"
                print(error_msg)
                traceback.print_exc()
                return {'success': False, 'error': f"{error_msg}. 原始错误: {str(e)}"}
        
        # 清理资源
        try:
            video_clip.close()
            video_with_effects.close()
        except:
            pass
            
        return {'success': True, 'error': None}
        
    except Exception as e:
        error_msg = f"应用魔法效果失败: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        return {'success': False, 'error': error_msg}


if __name__ == '__main__':
    # 检测是否有NVIDIA GPU支持
    has_nvidia = check_nvidia_gpu()
    if has_nvidia:
        print("检测到NVIDIA GPU，将启用硬件加速编码 (NVENC)")
    else:
        print("未检测到支持NVENC的NVIDIA GPU，将使用CPU编码")
    
    # video = VideoFileClip(filename="template_video.mp4", audio=False)

    # 使用新的多进程方法
    # write_videofile_mp(
    #     video,
    #     "output.mp4",
    #     codec="libx264" if not has_nvidia else None,  # GPU模式下使用h264_nvenc
    #     preset="ultrafast",             # 更快的编码
    #     num_processes=8,                # 通常是CPU核心数的一半效果最佳
    #     fps=30,                         # 降低帧率加快处理
    #     resize_factor=0.5,              # 缩小一半尺寸，加快处理，减小文件体积
    #     use_gpu=has_nvidia,             # 自动检测是否使用GPU
    #     bitrate="8M"                    # 设置较高码率保证质量
    # )
    preview_filename = f"preview_{7}_template_video.mp4"  # _{int(time.time())}
    output_path = os.path.join("previews", preview_filename)

    # 应用魔法效果
    config = {
  "channel": {
    "id": 7,
    "language": "en",
    "title": "HapiiXu"
  },
  "contents": {
    "channelLogo": {
      "logo": {
        "position": "left-top",
        "size": {
          "height": 150,
          "width": 150
        },
        "url": "https://common-1256967975.cos.ap-seoul.myqcloud.com/thumbnail/logo.png",
        "x": 22,
        "y": 100
      }
    },
    "channelName": {
      "namePosition": {
        "horizontalAlign": "center",
        "margin_x": 0,
        "margin_y": 0,
        "position": "right-bottom",
        "textAlign": "center",
        "verticalAlign": "center"
      },
      "nameSize": 37,
      "nameStyle_bgColor": None,
      "nameStyle_color": "rgba(245, 0, 0, 0.5)",
      "nameStyle_useBgColor": False,
      "nameStyle_useColor": True
    },
    "preview": {
      "previewUrl": "http://127.0.0.1:5000/api/channel/preview/7\\preview_7_material.mp4"
    }
  },
  "effects": {
    "亮度-对比度调整": {
      "category": "LumContrast",
      "parameters": {
        "contrast": 0.18,
        "lum": 0.08
      }
    },
    "伽马校正": {
      "category": "GammaCorrection",
      "parameters": {
        "gamma": 1.1
      }
    },
    "放大缩小": {
      "category": "Resize",
      "parameters": {
        "scale": 1.2
      }
    },
    "旋转": {
      "category": "Rotate",
      "parameters": {
        "angle": 2
      }
    },
    "视频边框": {
      "category": "Margin",
      "parameters": {
        "color": "#FF0000",
        "size": 5
      }
    },
    "速度调整": {
      "category": "Speed",
      "parameters": {
        "speedRanges": [
          {
            "end": 2,
            "start": 2,
            "value": 2
          },
          {
            "value": 1
          }
        ],
        "speedranges": 0
      }
    }
  }
}
    video_path = "./materials/material.mp4"
    result = apply_magic_from_config(video_path, output_path, config)