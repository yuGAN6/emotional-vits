import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


def convert_pad_shape(pad_shape):
    l = pad_shape[::-1]
    pad_shape = [item for sublist in l for item in sublist]
    return pad_shape


def intersperse(lst, item):
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result


def kl_divergence(m_p, logs_p, m_q, logs_q):
    """KL(P||Q)"""
    kl = (logs_q - logs_p) - 0.5
    kl += (
        0.5 * (torch.exp(2.0 * logs_p) + ((m_p - m_q) ** 2)) * torch.exp(-2.0 * logs_q)
    )
    return kl


def rand_gumbel(shape):
    """Sample from the Gumbel distribution, protect from overflows."""
    uniform_samples = torch.rand(shape) * 0.99998 + 0.00001
    return -torch.log(-torch.log(uniform_samples))


def rand_gumbel_like(x):
    g = rand_gumbel(x.size()).to(dtype=x.dtype, device=x.device)
    return g


def slice_segments(x, ids_str, segment_size=4):
    """
    ### 从输入的频谱图张量中按照给定的片段长度切片
    函数通过遍历每个样本的起始索引，从输入的频谱图张量中切取指定长度的片段，并将所有样本的切片结果拼接成一个新的张量返回。这个函数通常在语音或音频数据的处理过程中使用，用于将长时间的音频数据切割成小片段，方便进行后续处理和模型训练。

    - x: 输入的频谱图张量，是一个三维张量，shape 为 (batch_size, num_freq_bins, num_frames)
    - ids_str: 每个样本切片的起始索引，是一个一维张量，shape 为 (batch_size,)
    - segment_size: 切片的长度，默认为 4
    """

    ret = torch.zeros_like(x[:, :, :segment_size])
    # 创建一个全零张量 ret，其形状为 (batch_size, num_freq_bins, segment_size)
    # 这里 segment_size 取值范围为 [0, 4)，即切片的长度不包含结束索引 4

    for i in range(x.size(0)):
        # 遍历每个样本的索引 i

        idx_str = ids_str[i]
        # 获取第 i 个样本的切片起始索引

        idx_end = idx_str + segment_size
        # 计算切片的结束索引，即起始索引加上切片长度

        ret[i] = x[i, :, idx_str:idx_end]
        # 将第 i 个样本在频谱维度上，从 idx_str 到 idx_end 的片段赋值给 ret 的第 i 个样本

    return ret
    # 返回切片后的结果 ret，shape 为 (batch_size, num_freq_bins, segment_size)


def rand_slice_segments(x, x_lengths=None, segment_size=4):
    """
    - 定义一个函数`rand_slice_segments`，它接受三个参数：`x`，`x_lengths`和`segment_size`。
    - `x`是一个三维的张量，表示一批数据，第一维是批次大小，第二维是特征维，第三维是序列长度。
    - `x_lengths`是一个一维的张量，表示每个批次中数据的有效长度，如果为None，则默认为序列长度`t`。
    - `segment_size`是一个整数，表示要切片的长度，默认为4。
    - 计算每个批次中切片的最大起始位置`ids_str_max`，它等于有效长度减去切片长度加一。
    - 生成一个随机的一维张量`ids_str`，它的每个元素是一个介于0和`ids_str_max`之间的整数，表示每个批次中切片的起始位置。
    - 调用之前定义的函数`slice_segments`，传入参数`x`，`ids_str`和`segment_size`，得到一个新的张量`ret`，它包含了每个批次中随机切出的片段。
    - 返回新的张量`ret`和切片起始位置`ids_str`。
    """
    b, d, t = x.size()
    if x_lengths is None:
        x_lengths = t
    ids_str_max = x_lengths - segment_size + 1
    ids_str = (torch.rand([b]).to(device=x.device) * ids_str_max).to(dtype=torch.long)
    ret = slice_segments(x, ids_str, segment_size)
    return ret, ids_str


def get_timing_signal_1d(length, channels, min_timescale=1.0, max_timescale=1.0e4):
    position = torch.arange(length, dtype=torch.float)
    num_timescales = channels // 2
    log_timescale_increment = math.log(float(max_timescale) / float(min_timescale)) / (
        num_timescales - 1
    )
    inv_timescales = min_timescale * torch.exp(
        torch.arange(num_timescales, dtype=torch.float) * -log_timescale_increment
    )
    scaled_time = position.unsqueeze(0) * inv_timescales.unsqueeze(1)
    signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], 0)
    signal = F.pad(signal, [0, 0, 0, channels % 2])
    signal = signal.view(1, channels, length)
    return signal


def add_timing_signal_1d(x, min_timescale=1.0, max_timescale=1.0e4):
    b, channels, length = x.size()
    signal = get_timing_signal_1d(length, channels, min_timescale, max_timescale)
    return x + signal.to(dtype=x.dtype, device=x.device)


def cat_timing_signal_1d(x, min_timescale=1.0, max_timescale=1.0e4, axis=1):
    b, channels, length = x.size()
    signal = get_timing_signal_1d(length, channels, min_timescale, max_timescale)
    return torch.cat([x, signal.to(dtype=x.dtype, device=x.device)], axis)


def subsequent_mask(length):
    mask = torch.tril(torch.ones(length, length)).unsqueeze(0).unsqueeze(0)
    return mask


@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    n_channels_int = n_channels[0]
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    acts = t_act * s_act
    return acts


def convert_pad_shape(pad_shape):
    l = pad_shape[::-1]
    pad_shape = [item for sublist in l for item in sublist]
    return pad_shape


def shift_1d(x):
    x = F.pad(x, convert_pad_shape([[0, 0], [0, 0], [1, 0]]))[:, :, :-1]
    return x


def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


def generate_path(duration, mask):
    """
    duration: [b, 1, t_x]
    mask: [b, 1, t_y, t_x]
    """
    device = duration.device

    b, _, t_y, t_x = mask.shape
    cum_duration = torch.cumsum(duration, -1)

    cum_duration_flat = cum_duration.view(b * t_x)
    path = sequence_mask(cum_duration_flat, t_y).to(mask.dtype)
    path = path.view(b, t_x, t_y)
    path = path - F.pad(path, convert_pad_shape([[0, 0], [1, 0], [0, 0]]))[:, :-1]
    path = path.unsqueeze(1).transpose(2, 3) * mask
    return path


def clip_grad_value_(parameters, clip_value, norm_type=2):
    '''
    ### 对梯度进行裁剪，以防止梯度爆炸问题

    - `parameters`: 要进行梯度裁剪的参数列表或张量
    - `clip_value`: 梯度的绝对值上限，超过该值的梯度将被裁剪
    - `norm_type`: 计算梯度范数的类型，默认为 2

    `梯度裁剪`:
    常用于训练深度神经网络，特别是在使用梯度下降优化算法（如随机梯度下降）时，有时会出现梯度过大的情况。梯度过大可能导致训练不稳定甚至发散，梯度裁剪的目的就是限制梯度的大小，使其在一个可控的范围内。

    `梯度范数`是指梯度向量中各个分量的平方和再开根号，用来衡量梯度向量的大小。梯度的大小可以反映模型参数的更新幅度，即在当前参数值下，对损失函数的更新方向和强度。

    在深度学习中，常常使用梯度下降优化算法来更新模型参数，使得损失函数逐渐降低，从而得到更好的模型。然而，如果梯度过大，即梯度范数过大，会导致模型参数的更新过大，可能使得模型在参数空间中跳跃较大的距离，从而导致训练不稳定，甚至出现发散的情况。

    因此，为了避免梯度过大而引发的问题，一种常见的方法是对梯度进行裁剪，即限制梯度的大小，使其在一个可控的范围内。裁剪梯度的操作可以通过梯度范数来实现。如果梯度范数超过了预设的阈值（裁剪值），就将梯度向量等比例地缩放到阈值以内，从而限制了梯度的大小。这样做可以使模型的训练更加稳定，避免出现梯度爆炸问题，同时也有助于防止参数更新过大而跳过最优点。
    '''

    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    # 如果 parameters 是一个张量，则将其转换为列表，以便后续处理

    parameters = list(filter(lambda p: p.grad is not None, parameters))
    # 过滤出梯度不为 None 的参数

    norm_type = float(norm_type)
    # 将 norm_type 转换为浮点数类型

    if clip_value is not None:
        clip_value = float(clip_value)
    # 将 clip_value 转换为浮点数类型

    total_norm = 0
    # 初始化 total_norm，用于累加梯度的范数

    for p in parameters:
        # 遍历参数列表

        param_norm = p.grad.data.norm(norm_type)
        # 计算参数 p 的梯度的范数

        total_norm += param_norm.item() ** norm_type
        # 将参数 p 的梯度范数的 norm_type 次方加入 total_norm 中

        if clip_value is not None:
            # 如果 clip_value 不为 None，则对梯度进行裁剪
            p.grad.data.clamp_(min=-clip_value, max=clip_value)
            # 将参数 p 的梯度的绝对值限制在 [-clip_value, clip_value] 范围内

    total_norm = total_norm ** (1.0 / norm_type)
    # 将 total_norm 开根号，得到整体的梯度范数

    return total_norm
    # 返回裁剪后的梯度范数
