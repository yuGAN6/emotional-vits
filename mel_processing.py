import math
import os
import random
import torch
import torchaudio
from torch import nn
import torch.nn.functional as F
import torch.utils.data
import numpy as np

import logging

numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.WARNING)
import warnings
warnings.filterwarnings('ignore')
import librosa
import librosa.util as librosa_util
from librosa.util import normalize, pad_center, tiny
from scipy.signal import get_window
from scipy.io.wavfile import read
from librosa.filters import mel as librosa_mel_fn

MAX_WAV_VALUE = 32768.0


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    '''对输入张量进行动态范围压缩（压缩幅度较大的值），返回压缩后的张量。

    参数：
     - `x`: 输入的张量，可以是任意形状
     - `C`: 压缩因子，用于控制压缩的程度，默认为 1
     - `clip_val`: 压缩后的张量中的值小于该阈值时进行裁剪，防止出现无限大的值，默认为 1e-5

    返回：
    compressed_x: 压缩后的张量，与输入张量 x 形状相同
    '''

    return torch.log(torch.clamp(x, min=clip_val) * C)
    # 使用 torch.clamp 函数将输入张量 x 中的所有值都限制在大于等于 clip_val 的范围内
    # 然后将结果乘以压缩因子 C，再使用 torch.log 函数对结果进行对数运算
    # 返回最终的压缩后的张量 compressed_x


def dynamic_range_decompression_torch(x, C=1):
    '''对经过动态范围压缩后的张量进行解压缩，返回解压缩后的张量。

    参数：
    - `x`: 经过动态范围压缩后的张量，可以是任意形状
    - `C`: 用于压缩的压缩因子，用于恢复压缩前的值，默认为 1

    返回：
    decompressed_x: 解压缩后的张量，与输入张量 x 形状相同
    '''

    return torch.exp(x) / C
    # 首先使用 torch.exp 函数对输入张量 x 进行指数运算
    # 然后除以压缩因子 C，以恢复压缩前的值
    # 返回最终的解压缩后的张量 decompressed_x


def spectral_normalize_torch(magnitudes):
    '''
    对谱图进行动态压缩和对数
    '''
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}


def spectrogram_torch(y, n_fft, sampling_rate, hop_size, win_size, center=False):
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global hann_window
    dtype_device = str(y.dtype) + '_' + str(y.device)
    wnsize_dtype_device = str(win_size) + '_' + dtype_device
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(dtype=y.dtype, device=y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[wnsize_dtype_device],
                      center=center, pad_mode='reflect', normalized=False, onesided=True)

    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
    return spec


def spec_to_mel_torch(spec, n_fft, num_mels, sampling_rate, fmin, fmax):
    '''
    ### 将线性频谱图转换为梅尔频谱图的函数
    - spec: 输入的频谱图，是一个二维张量，shape 为 (batch_size, num_freq_bins)
    - n_fft: FFT 窗口大小
    - num_mels: 转换后的梅尔频谱图的频率分辨率（频道数）
    - sampling_rate: 采样率
    - fmin, fmax: 转换后的梅尔频谱图的最低和最高频率（对应于梅尔刻度）
    '''

    global mel_basis
    # 定义一个全局变量 mel_basis，用于缓存梅尔滤波器组
    # 由于线性谱到梅尔谱计算是相对固定的，不会随着输入数据的变化而变化。因此，将计算得到的梅尔滤波器组进行缓存可以在后续的计算中节省时间，避免重复计算

    dtype_device = str(spec.dtype) + '_' + str(spec.device)
    # 获取输入频谱图的数据类型和设备信息，用于构建缓存的键

    fmax_dtype_device = str(fmax) + '_' + dtype_device
    # 根据 fmax、dtype 和 device 构建缓存的键

    if fmax_dtype_device not in mel_basis:
        # 如果缓存中没有对应的梅尔滤波器组

        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        # 使用 librosa 库的 mel_fn 函数计算梅尔滤波器组
        # 这里的 mel_fn 函数是用于计算梅尔滤波器组的，参数含义见函数定义

        mel_basis[fmax_dtype_device] = torch.from_numpy(mel).to(dtype=spec.dtype, device=spec.device)
        # 将计算得到的 mel 数组转换为 PyTorch 张量，并存储在缓存中
        # 确保与输入 spec 张量的数据类型和设备一致

    spec = torch.matmul(mel_basis[fmax_dtype_device], spec)
    # 将输入的频谱图与 mel_basis 相乘，将频谱图转换为梅尔频谱图
    # 这里的 matmul 表示矩阵相乘，即进行线性变换

    spec = spectral_normalize_torch(spec)
    # 对转换后的梅尔频谱图进行归一化处理

    return spec
    # 返回转换后的梅尔频谱图


def mel_spectrogram_torch(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    '''计算输入音频信号的梅尔频谱图。
    
    参数：
    - `y`: 输入的音频信号，是一个一维张量，shape 为 (num_samples,)
    - `n_fft`: FFT 窗口大小
    - `num_mels`: 转换后的梅尔频谱图的频率分辨率（频道数）
    - `sampling_rate`: 采样率
    - `hop_size`: 滑动窗口的步长
    - `win_size`: 窗口大小
    - `fmin, fmax`: 转换后的梅尔频谱图的最低和最高频率（对应于梅尔刻度）
    - `center`: 是否对信号进行中心化（默认为 False）

    返回：
    spec: 输入音频信号的梅尔频谱图，是一个二维张量，shape 为 (num_mels, num_frames)
    '''

    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global mel_basis, hann_window
    # 定义全局变量 mel_basis 和 hann_window，用于缓存梅尔滤波器组和汉宁窗函数

    dtype_device = str(y.dtype) + '_' + str(y.device)
    # 获取输入音频信号的数据类型和设备信息，用于构建缓存的键

    fmax_dtype_device = str(fmax) + '_' + dtype_device
    wnsize_dtype_device = str(win_size) + '_' + dtype_device
    # 根据 fmax、win_size、dtype 和 device 构建缓存的键

    if fmax_dtype_device not in mel_basis:
        # 如果缓存中没有对应的梅尔滤波器组

        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        # 使用 librosa 库的 mel_fn 函数计算梅尔滤波器组
        # 这里的 mel_fn 函数是用于计算梅尔滤波器组的，参数含义见函数定义

        mel_basis[fmax_dtype_device] = torch.from_numpy(mel).to(dtype=y.dtype, device=y.device)
        # 将计算得到的 mel 数组转换为 PyTorch 张量，并存储在缓存中
        # 确保与输入 y 张量的数据类型和设备一致

    if wnsize_dtype_device not in hann_window:
        # 如果缓存中没有对应的汉宁窗函数

        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(dtype=y.dtype, device=y.device)
        # 使用 PyTorch 的 hann_window 函数计算汉宁窗函数，并存储在缓存中
        # 确保与输入 y 张量的数据类型和设备一致

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    # 在信号的开头和结尾进行填充，使其与 STFT 的长度对齐
    # 这里的填充长度为 (n_fft - hop_size) / 2，使用 reflect 模式填充

    y = y.squeeze(1)
    # 去除新增的维度，将 y 转换为一维张量

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[wnsize_dtype_device],
                      center=center, pad_mode='reflect', normalized=False, onesided=True)
    # 计算 STFT（短时傅里叶变换），得到频谱图
    # 这里使用 hann_window 作为窗口函数，指定是否中心化由 center 参数决定

    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
    # 对 STFT 的实部和虚部进行平方和的开方运算，得到幅度谱

    spec = torch.matmul(mel_basis[fmax_dtype_device], spec)
    # 将幅度谱与 mel_basis 相乘，将频谱图转换为梅尔频谱图
    # 这里的 matmul 表示矩阵相乘，即进行线性变换

    spec = spectral_normalize_torch(spec)
    # 对转换后的梅尔频谱图进行归一化处理

    return spec
    # 返回转换后的梅尔频谱图



def mel_spectrogram_torchaudio(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    '''计算输入音频信号的梅尔频谱图，使用 torchaudio 库进行计算。
    
    参数：
    - `y`: 输入的音频信号，是一个一维张量，shape 为 (num_samples,)
    - `n_fft`: FFT 窗口大小
    - `num_mels`: 转换后的梅尔频谱图的频率分辨率（频道数）
    - `sampling_rate`: 采样率
    - `hop_size`: 滑动窗口的步长
    - `win_size`: 窗口大小
    - `fmin, fmax`: 转换后的梅尔频谱图的最低和最高频率（对应于梅尔刻度）
    - `center`: 是否对信号进行中心化（默认为 False）

    返回：
    spec: 输入音频信号的梅尔频谱图，是一个二维张量，shape 为 (num_mels, num_frames)
    '''

    global mel_basis
    # 定义全局变量 mel_basis，用于缓存梅尔滤波器组

    dtype_device = str(y.dtype) + '_' + str(y.device)
    # 获取输入音频信号的数据类型和设备信息，用于构建缓存的键

    fmax_dtype_device = str(fmax) + '_' + dtype_device
    wnsize_dtype_device = str(win_size) + '_' + dtype_device
    # 根据 fmax、win_size、dtype 和 device 构建缓存的键

    if fmax_dtype_device not in mel_basis:
        # 如果缓存中没有对应的梅尔滤波器组

        mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sampling_rate,
            n_fft=n_fft,
            win_length=win_size,
            hop_length=hop_size,
            n_mels=num_mels,
            f_min=fmin,
            f_max=fmax,
        )
        mel_basis[fmax_dtype_device] = mel.to(dtype=y.dtype, device=y.device)
        # 使用 torchaudio 的 MelSpectrogram 函数计算梅尔滤波器组
        # 并存储在缓存中，确保与输入 y 张量的数据类型和设备一致

    y = torch.nn.functional.pad(y.unsqueeze(0), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    # 在信号的开头和结尾进行填充，使其与 STFT 的长度对齐
    # 这里的填充长度为 (n_fft - hop_size) / 2，使用 reflect 模式填充

    spec = mel(y)
    # 调用存储在缓存中的梅尔滤波器组，计算输入音频信号的梅尔频谱图

    spec = spectral_normalize_torch(spec)
    # 对转换后的梅尔频谱图进行归一化处理

    return spec
    # 返回转换后的梅尔频谱图
