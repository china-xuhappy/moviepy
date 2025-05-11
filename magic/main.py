import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from moviepy import VideoFileClip, ImageClip, VideoClip, CompositeVideoClip, TextClip, concatenate_videoclips, vfx, afx
import numpy as np
import multiprocessing
from functools import partial
from proglog import ProgressBarLogger
import subprocess
import tempfile
import time
import pickle
import json

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


def common_template(params: dict, video: VideoClip) -> VideoClip:
    """
    公共模板，logo，吉祥物gif，频道名

    吉祥物需要一个大绿幕，然后给识别出来去除绿幕
    :return:
    """
    channel_content = params['ChannelContent']

    logo_image = ImageClip(
        img="logo.png",
        duration=video.duration
    ).resized(new_size=(150, 150)).with_position(("right", "bottom"))
    logo_image = logo_image.with_effects([vfx.CrossFadeIn(2), vfx.CrossFadeOut(2)])

    # 这个可以搞个动画，从中间移动下去
    channel_name = TextClip(
        font="./ttf/Roboto_SemiCondensed-Black.ttf",
        text=channel_content['name']['content'],
        font_size=channel_content['name']['fontSize'],
        color=channel_content['name']['color'],
        text_align="center",
        horizontal_align="center",
        vertical_align="center",
        duration=video.duration,
        margin=(None, video.h * 0.01)
    ).with_position(("center", "bottom"))
    channel_name = channel_name.with_effects([vfx.CrossFadeIn(2), vfx.CrossFadeOut(2)])

    clip = CompositeVideoClip([video, channel_name, logo_image])
    return clip


def apply_magic_effects_with_uuid(params: dict, video_clip: VideoClip) -> VideoClip:
    """
    第一个模版，使用效果以动画效果为主。
    1. 在开头淡入

    2. 前面半段加速一点，后半段减速一点

    2. 在结束淡出
    :return:
    """
    duration = video_clip.duration
    effects_params = params['Effects']

    effects = []
    # speed = effects_params['Speed']
    # if speed['switch']:
    #     time_node = speed['timeNode']
    #     first_half_speed = speed['firstHalf']
    #     second_half_speed = speed['secondHalf']

    #     first_half_duration = int(duration / time_node) # 开头的时间， 用来区分前半段和后半段

    #     first_half = video_clip.subclipped(0, first_half_duration).with_effects([
    #         vfx.MultiplySpeed(factor=first_half_speed),
    #         # vfx.AccelDecel(abruptness=0.9)
    #     ])
    #     second_half = video_clip.subclipped(first_half_duration).with_effects([vfx.MultiplySpeed(second_half_speed)])
    #     video_clip = concatenate_videoclips([first_half, second_half])


    # fadein_fadeout = effects_params['FadeinFadeout']
    # if fadein_fadeout['switch']:
    #     fadein = fadein_fadeout['fadein']
    #     fadeout = fadein_fadeout['fadeout']
    #     effects.extend([
    #         vfx.FadeIn(fadein),
    #         vfx.FadeOut(fadeout),

    #         afx.AudioFadeIn(fadein),
    #         afx.AudioFadeOut(fadeout),
    #     ])

    resize = effects_params['Resize']
    if resize['switch']:
        new_size = resize['new_size']
        effects.extend([
            vfx.Resize(new_size=new_size)
        ])


    rotate = effects_params['Rotate']
    if rotate['switch']:
        angle = rotate['angle']
        effects.extend([
            vfx.Rotate(angle=angle)
        ])


    margin = effects_params['Margin']
    if margin['switch']:
        margin_size = margin['marginSize']
        margin_color = margin['marginColor']
        effects.extend([
            vfx.Margin(margin_size=margin_size, color=margin_color)
        ])


    lum_contrast = effects_params['LumContrast']
    if lum_contrast['switch']:
        lum = lum_contrast['lum']
        contrast = lum_contrast['contrast']
        effects.extend([
            vfx.LumContrast(lum=lum, contrast=contrast)
        ])

    gamma_correction = effects_params['GammaCorrection']
    if gamma_correction['switch']:
        gamma = gamma_correction['gamma']
        effects.extend([
            vfx.GammaCorrection(gamma=gamma)
        ])

    multiply_color = effects_params['MultiplyColor']
    if multiply_color['switch']:
        factor = multiply_color['factor']
        effects.extend([
            vfx.MultiplyColor(factor=factor)
        ])

    # super_sample = effects_params['SuperSample']
    # if super_sample['switch']:
    #     d = super_sample['d']
    #     effects.extend([
    #         vfx.SuperSample(d=d)
    #     ])

    video_clip = video_clip.with_effects(effects=effects)

    video_clip = common_template(params, video_clip)
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


def write_images_sequence_mp(clip, output_dir, fps=None, num_processes=None, start_number=0, jpeg_quality=95):
    """
    多进程高性能版本的write_images_sequence
    
    参数:
        clip: 视频剪辑
        output_dir: 输出目录
        fps: 帧率 
        num_processes: 进程数
        start_number: 起始帧编号
        jpeg_quality: JPEG压缩质量
    """
    if fps is None:
        fps = clip.fps
        
    if num_processes is None:
        num_processes = max(1, multiprocessing.cpu_count() - 1)
    
    # 限制最大进程数
    num_processes = min(num_processes, 8)
    
    # 计算所有时间点
    duration = clip.duration
    total_frames = int(duration * fps)
    time_points = [(i + start_number, i/fps) for i in range(total_frames)]
    
    print(f"准备提取 {len(time_points)} 帧, fps={fps:.2f}")
    
    # 分割时间点到各进程
    chunks = []
    frames_per_process = len(time_points) // num_processes
    
    if frames_per_process == 0:
        frames_per_process = 1
        num_processes = min(len(time_points), num_processes)
    
    for i in range(num_processes):
        start_idx = i * frames_per_process
        end_idx = min(start_idx + frames_per_process, len(time_points))
        
        if start_idx >= len(time_points):
            break
            
        process_points = time_points[start_idx:end_idx]
        chunks.append((clip, output_dir, process_points, i, jpeg_quality))
    
    print(f"启动 {len(chunks)} 个进程来并行处理所有帧")
    
    # 使用进程池并行处理
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = list(pool.starmap(save_frames_parallel, chunks))
    
    # 计算总计处理的帧数
    total_processed = sum(count for _, count in results)
    print(f"共处理 {total_processed}/{len(time_points)} 帧")
    
    return [os.path.join(output_dir, f"frame_{i+start_number:06d}.jpg") for i in range(total_processed)]


# 用于子进程的加载和处理视频帧的函数
def process_frames_chunk(args):
    """
    子进程的帧处理函数，负责加载视频、应用特效、计算并保存帧
    """
    source_path, chunk_times, output_dir, chunk_id, uuid_str, fps = args
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 计算时间范围
        if len(chunk_times) == 0:
            print(f"进程 {chunk_id}: 没有要处理的帧")
            return chunk_id, 0
            
        start_time = chunk_times[0]
        end_time = chunk_times[-1] + 1/fps  # 确保包含最后一帧
        
        print(f"\n进程 {chunk_id}: 开始处理时间段 {start_time:.2f}s 到 {end_time:.2f}s, 共 {len(chunk_times)} 帧")
        
        # 加载视频片段
        video = VideoFileClip(filename=source_path, audio=False)
        print(f"进程 {chunk_id}: 成功加载视频，原始时长: {video.duration:.2f}s")
        
        # 裁剪到需要的时间段
        if start_time > 0 or end_time < video.duration:
            video = video.subclipped(start_time, min(end_time, video.duration))
            print(f"进程 {chunk_id}: 创建子剪辑 {start_time:.2f}s-{min(end_time, video.duration):.2f}s")
        
        # 应用特效
        print(f"进程 {chunk_id}: 开始应用特效...")
        magic_params = get_video_params_from_uuid(uuid_str)
        video = apply_magic_effects_with_uuid(magic_params, video)
        print(f"进程 {chunk_id}: 特效应用完成，开始保存帧...")
        
        # 创建相对时间点列表 - 将chunk_times转换为索引和时间的元组
        # 因为我们已经裁剪了视频，所以需要调整时间点
        relative_time_points = []
        for i, t in enumerate(chunk_times):
            relative_t = t - start_time
            # 确保时间在有效范围内
            if 0 <= relative_t <= video.duration:
                relative_time_points.append((i, relative_t))
        
        # 使用我们自己的高性能多进程帧保存函数
        # 不需要多进程，因为外部已经有多进程了
        result = save_frames_parallel(
            video, 
            output_dir, 
            relative_time_points, 
            chunk_id,
            quality=90
        )
        
        # 清理资源
        video.close()
        
        return chunk_id, len(relative_time_points)
        
    except Exception as e:
        import traceback
        print(f"处理帧出错 (块 {chunk_id}): {str(e)}")
        traceback.print_exc()
        return chunk_id, 0


class FastProcessLogger(ProgressBarLogger):
    """自定义进度记录器，支持多进程进度跟踪"""
    def __init__(self, total_frames):
        super().__init__()
        self.total_frames = total_frames
        self.completed_frames = 0
        self.start_time = time.time()
        
    def callback(self, result):
        """每当一个进程块完成时调用"""
        chunk_id, frames_count = result
        self.completed_frames += frames_count
        elapsed = time.time() - self.start_time
        fps = self.completed_frames / elapsed if elapsed > 0 else 0
        
        percent = 100 * self.completed_frames / self.total_frames
        remaining_time = "计算中..." if fps <= 0 else f"{(self.total_frames-self.completed_frames)/fps/60:.1f}分钟"
        
        print(f"\r生成帧: {percent:.1f}% | {self.completed_frames}/{self.total_frames} 帧完成 | "
              f"{fps:.2f} fps | 预计剩余时间: {remaining_time}", end="")


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


def write_videofile_mp(
    clip, 
    filename, 
    fps=None, 
    codec=None, 
    bitrate=None, 
    audio=True, 
    audio_codec=None,
    preset="medium", 
    pixel_format=None, 
    num_processes=None,
    resize_factor=None,  # 新增: 缩放因子，如0.5表示缩小一半
    use_gpu=False,      # 新增: 是否使用GPU加速
    verbose=True
):
    """
    使用多进程并行生成帧，显著提高渲染速度
    
    参数:
        clip: 要渲染的视频剪辑
        filename: 输出文件名
        fps: 帧率
        codec: 视频编码器，如 'libx264'
        bitrate: 视频比特率，如 '5000k'
        audio: 是否包含音频
        audio_codec: 音频编码器
        preset: 编码预设，如 'ultrafast', 'fast', 'medium'
        pixel_format: 像素格式
        num_processes: 进程数量，默认为CPU核心数
        resize_factor: 视频尺寸缩放因子，如0.5表示缩小一半
        use_gpu: 是否使用GPU加速编码
        verbose: 是否显示进度
    """
    if fps is None:
        fps = clip.fps
    
    if num_processes is None:
        num_processes = max(1, multiprocessing.cpu_count() - 1)
    
    # 降低进程数，避免同时启动太多
    num_processes = min(num_processes, 8)  # 限制最大进程数为8
    
    # 临时文件夹
    temp_dir = tempfile.mkdtemp()
    source_video_path = os.path.abspath(clip.filename)  # 获取源视频路径
    print(f"使用临时目录: {temp_dir}")
    print(f"源视频路径: {source_video_path}")
    print(f"视频信息: 时长={clip.duration:.2f}s, 尺寸={clip.size}, FPS={fps}")
    
    # 获取唯一ID用于生成一致的特效
    uuid_str = str(uuid.uuid4())
    
    try:
        # 创建帧存储目录
        frames_dir = os.path.join(temp_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        
        # 使用我们的高性能帧提取方法，直接处理整个视频，避免分割和合并过程
        if num_processes <= 1:
            # 加载和应用特效
            magic_params = get_video_params_from_uuid(uuid_str)
            video_with_effects = apply_magic_effects_with_uuid(magic_params, clip)
            
            # 单进程方式处理
            print("使用单进程模式处理帧...")
            write_images_sequence_mp(
                video_with_effects, 
                frames_dir, 
                fps=fps, 
                num_processes=1
            )
        else:
            # 计算总帧数和时间点
            duration = clip.duration
            total_frames = int(duration * fps)
            times = [i/fps for i in range(total_frames)]
            
            # 将时间点分成适合多进程数量的组
            frames_per_process = total_frames // num_processes
            if frames_per_process == 0:
                frames_per_process = 1
                num_processes = min(num_processes, total_frames)
            
            # 准备任务参数
            tasks = []
            for i in range(num_processes):
                start_idx = i * frames_per_process
                end_idx = min(start_idx + frames_per_process, total_frames)
                
                if start_idx >= total_frames:
                    break
                    
                chunk_times = times[start_idx:end_idx]
                chunk_dir = os.path.join(temp_dir, f"chunk_{i}")
                
                # 每个进程接收: (源视频路径, 时间段, 输出目录, 块ID, UUID字符串, 帧率)
                tasks.append((source_video_path, chunk_times, chunk_dir, i, uuid_str, fps))
            
            # 创建进度记录器
            logger = FastProcessLogger(total_frames)
            
            # 使用进程池并行处理
            print(f"启动 {len(tasks)} 个进程处理 {total_frames} 帧...")
            print("处理中...首次生成可能需要较长时间，请耐心等待")
            print("=" * 50)
            
            with multiprocessing.Pool(processes=num_processes) as pool:
                for result in pool.imap_unordered(process_frames_chunk, tasks):
                    logger.callback(result)
                    
                # 确保所有进程完成
                pool.close()
                pool.join()
            
            print("\n" + "=" * 50)
            print("所有帧处理完成！准备合成视频...")
            
            # 合成所有帧到最终目录
            frame_count = 0
            for i in range(len(tasks)):
                chunk_dir = os.path.join(temp_dir, f"chunk_{i}")
                if not os.path.exists(chunk_dir):
                    continue
                    
                chunk_files = sorted([f for f in os.listdir(chunk_dir) if f.startswith("frame_")])
                
                for frame_file in chunk_files:
                    src_path = os.path.join(chunk_dir, frame_file)
                    dst_path = os.path.join(frames_dir, f"frame_{frame_count:06d}.jpg")
                    os.rename(src_path, dst_path)
                    frame_count += 1
        
        # 检查是否有帧生成
        frame_files = [f for f in os.listdir(frames_dir) if f.startswith("frame_")]
        if not frame_files:
            raise Exception("没有生成任何帧！请检查视频处理过程。")
            
        print(f"总共生成了 {len(frame_files)} 帧")
        
        # 获取第一帧尺寸
        import imageio
        first_frame_path = os.path.join(frames_dir, frame_files[0])
        first_frame = imageio.imread(first_frame_path)
        frame_height, frame_width = first_frame.shape[:2]
        print(f"帧尺寸: {frame_width}x{frame_height}")
        
        # 确保尺寸是偶数 (H.264编码器要求)
        width_scale = frame_width
        height_scale = frame_height
        
        # 如果指定缩放因子，应用缩放
        if resize_factor is not None and resize_factor > 0 and resize_factor != 1.0:
            width_scale = int(frame_width * resize_factor)
            height_scale = int(frame_height * resize_factor)
            print(f"应用缩放因子 {resize_factor}, 新尺寸: {width_scale}x{height_scale}")
        
        # 确保宽高都是偶数
        width_scale = width_scale - (width_scale % 2)
        height_scale = height_scale - (height_scale % 2)
        
        if width_scale <= 0:
            width_scale = 2
        if height_scale <= 0:
            height_scale = 2
            
        print(f"最终视频尺寸: {width_scale}x{height_scale}")
        
        # 使用ffmpeg直接合成视频
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", os.path.join(frames_dir, "frame_%06d.jpg"),
        ]
        
        # 添加缩放滤镜
        filter_complex = f"scale={width_scale}:{height_scale}:flags=lanczos"
        cmd.extend(["-vf", filter_complex])
        
        # 设置编码器 - 如果启用GPU加速，使用NVENC硬件编码器
        if use_gpu:
            try:
                # 直接强制使用NVENC编码器
                cmd.extend(["-c:v", "h264_nvenc"])
                
                # NVENC特定参数
                if preset == "ultrafast" or preset == "superfast":
                    nvenc_preset = "p1"  # 最低延迟
                elif preset == "veryfast" or preset == "faster":
                    nvenc_preset = "p2"  # 低延迟
                elif preset == "fast" or preset == "medium":
                    nvenc_preset = "p4"  # 中等质量
                else:
                    nvenc_preset = "p7"  # 高质量
                    
                cmd.extend([
                    "-preset", nvenc_preset,
                    "-tune", "hq",        # 高质量模式
                    "-rc", "vbr_hq",      # 高质量可变比特率
                    "-b:v", bitrate if bitrate else "8M",  # 设置比特率
                    "-maxrate", "10M",
                    "-qmin", "0",
                    "-qmax", "51"
                ])
                
                print("使用NVIDIA GPU硬件加速编码: h264_nvenc")
                print("NVIDIA编码参数:")
                print(f"  - 预设: {nvenc_preset}")
                print(f"  - 比特率: {bitrate if bitrate else '8M'}")
                
            except Exception as e:
                print(f"GPU编码器初始化失败，回退到CPU编码: {str(e)}")
                use_gpu = False
                # 下面会进入CPU编码逻辑
        
        if not use_gpu:
            # 使用指定的编码器，或默认使用libx264
            actual_codec = codec if codec else "libx264"
            cmd.extend(["-c:v", actual_codec])
            
            if preset:
                cmd.extend(["-preset", preset])
                
            # 添加比特率设置
            if bitrate:
                cmd.extend(["-b:v", bitrate])
            
        # 设置像素格式
        if pixel_format:
            cmd.extend(["-pix_fmt", pixel_format])
        else:
            cmd.extend(["-pix_fmt", "yuv420p"])  # 广泛兼容
            
        # 处理音频
        if audio and clip.audio is not None:
            audio_file = os.path.join(temp_dir, "audio.mp3")
            clip.audio.write_audiofile(audio_file, verbose=False, logger=None)
            cmd.extend(["-i", audio_file])
            
            if audio_codec:
                cmd.extend(["-c:a", audio_codec])
            
            cmd.extend(["-shortest"])
            
        cmd.append(filename)
        
        print("\n执行ffmpeg命令合成视频...")
        print(" ".join(cmd))
        
        # 执行ffmpeg命令
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # 实时显示FFmpeg输出，方便排查问题
        for line in process.stderr:
            if "encoder" in line.lower() or "gpu" in line.lower() or "nvenc" in line.lower():
                print(f"FFMPEG: {line.strip()}")
        
        process.wait()
        
        if process.returncode != 0:
            print("视频编码过程中出现错误，请检查FFmpeg输出")
        else:
            print(f"视频已成功保存到: {filename}")
        
    finally:
        # 清理临时文件
        import shutil
        print(f"清理临时文件...")
        shutil.rmtree(temp_dir)


if __name__ == '__main__':
    # 检测是否有NVIDIA GPU支持
    has_nvidia = check_nvidia_gpu()
    if has_nvidia:
        print("检测到NVIDIA GPU，将启用硬件加速编码 (NVENC)")
    else:
        print("未检测到支持NVENC的NVIDIA GPU，将使用CPU编码")
    
    video = VideoFileClip(filename="template_video.mp4", audio=False)

    # 使用新的多进程方法
    write_videofile_mp(
        video,
        "output.mp4",
        codec="libx264" if not has_nvidia else None,  # GPU模式下使用h264_nvenc
        preset="ultrafast",             # 更快的编码
        num_processes=8,                # 通常是CPU核心数的一半效果最佳
        fps=30,                         # 降低帧率加快处理
        resize_factor=0.5,              # 缩小一半尺寸，加快处理，减小文件体积
        use_gpu=has_nvidia,             # 自动检测是否使用GPU
        bitrate="8M"                    # 设置较高码率保证质量
    )
