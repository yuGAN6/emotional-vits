import os
import json
import argparse
import itertools
import math
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler

import commons
import utils
from data_utils import (
    TextAudioSpeakerLoader,
    TextAudioSpeakerCollate,
    DistributedBucketSampler
)
from models import (
    SynthesizerTrn,
    MultiPeriodDiscriminator,
)
from losses import (
    generator_loss,
    discriminator_loss,
    feature_loss,
    kl_loss
)
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from text.symbols import symbols

torch.backends.cudnn.benchmark = False
global_step = 0


def main():
    """Assume Single Node Multi GPUs Training Only"""
    assert torch.cuda.is_available(), "CPU training is not allowed."

    n_gpus = torch.cuda.device_count()
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '8899'

    # 获取超参数
    hps = utils.get_hparams()
    # 多卡多进程训练
    # mp.spawn(run, nprocs=n_gpus, args=(n_gpus, hps,))
    # 单卡训练
    run(rank=0, n_gpus=1, hps=hps)


def run(rank, n_gpus, hps):
    '''
    分布式训练流程
    - rank 进程的编号
    - n_gpus 使用的GPU数量
    - hps 参数的字典
    '''

    global global_step
    # 声明全局变量global_step

    if rank == 0:
        # 只在rank=0的进程上进行以下操作

        logger = utils.get_logger(hps.model_dir)
        # 获取日志记录器，用于记录训练过程中的信息
        logger.info(hps)
        # 将配置参数打印到日志中
        utils.check_git_hash(hps.model_dir)
        # 检查代码的git哈希是否与训练时一致，用于确保代码版本一致性
        writer = SummaryWriter(log_dir=hps.model_dir)
        # 创建一个TensorBoard的SummaryWriter，用于将训练过程中的数据写入到TensorBoard中
        writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))
        # 创建一个TensorBoard的SummaryWriter，用于将评估过程中的数据写入到TensorBoard中

    dist.init_process_group(backend='gloo', init_method='env://', world_size=n_gpus, rank=rank)
    # 初始化分布式进程组，使用gloo后端，分布式训练的总进程数为n_gpus，当前进程的编号为rank
    # Linux - nccl更好  Windows - 只能gloo

    torch.manual_seed(hps.train.seed)
    # 设置随机数种子，用于复现训练结果
    torch.cuda.set_device(rank)
    # 将当前进程的设备（GPU）设置为rank指定的GPU

    train_dataset = TextAudioSpeakerLoader(hps.data.training_files, hps.data)
    # 加载训练数据集
    train_sampler = DistributedBucketSampler(
        train_dataset,
        hps.train.batch_size,
        [32, 300, 400, 500, 600, 700, 800, 900, 1000],
        num_replicas=n_gpus,
        rank=rank,
        shuffle=True)
    # 创建分布式的BucketSampler，用于在分布式环境下按照bucket大小对训练数据进行采样
    collate_fn = TextAudioSpeakerCollate()
    # 创建数据处理函数，用于将样本组合成batch
    train_loader = DataLoader(train_dataset, num_workers=8, shuffle=False, pin_memory=True,
                              collate_fn=collate_fn, batch_sampler=None)
    # 创建训练数据的DataLoader

    if rank == 0:
        # 只在rank=0的进程上进行以下操作

        eval_dataset = TextAudioSpeakerLoader(hps.data.validation_files, hps.data)
        # 加载评估数据集
        eval_loader = DataLoader(eval_dataset, num_workers=8, shuffle=False,
                                 batch_size=hps.train.batch_size, pin_memory=True,
                                 drop_last=False, collate_fn=collate_fn)
        # 创建评估数据的DataLoader

    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length, # 8192 // 256 = 32
        n_speakers=hps.data.n_speakers,
        **hps.model).cuda(rank)
    # 创建并初始化声音合成模型，并将其放置在指定的GPU上
    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).cuda(rank)
    # 创建并初始化多周期鉴别器模型，并将其放置在指定的GPU上
    optim_g = torch.optim.AdamW(
        net_g.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps)
    # 创建生成器模型的优化器
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps)
    # 创建鉴别器模型的优化器
    net_g = DDP(net_g, device_ids=[rank])
    # 使用DistributedDataParallel将生成器模型封装起来，实现分布式训练
    net_d = DDP(net_d, device_ids=[rank])
    # 使用DistributedDataParallel将鉴别器模型封装起来，实现分布式训练
    ckptG = hps.ckptG
    ckptD = hps.ckptD
    # 从配置参数中获取生成器和鉴别器的checkpoint路径

    try:
        # 尝试加载模型的checkpoint

        if ckptG is not None:
            assert os.path.exists(ckptG), f"Path of checkpoint D:{ckptG} does not exist!"
            # 断言生成器checkpoint路径存在
            _, _, _, epoch_str = utils.load_checkpoint(ckptG, net_g, optim_g, is_old=True)
            # 加载生成器模型和优化器的参数，并获取训练的epoch数
            print("加载原版VITS模型G记录点成功")
        else:
            _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g,
                                                       optim_g)
            print("从头训练初始化生成模型")
            # 加载最新的生成器checkpoint，如果没有最新checkpoint，则从头开始训练
        if ckptD is not None:
            assert os.path.exists(ckptD), f"Path of checkpoint D:{ckptD} does not exist!"
            # 断言鉴别器checkpoint路径存在
            _, _, _, epoch_str = utils.load_checkpoint(ckptG, net_g, optim_g, is_old=True)
            # 加载鉴别器模型和优化器的参数，并获取训练的epoch数
            print("加载原版VITS模型D记录点成功")
        else:
            _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "D_*.pth"), net_d,
                                                       optim_d)
            print("从头训练初始化判别模型")
            # 加载最新的鉴别器checkpoint，如果没有最新checkpoint，则从头开始训练
        global_step = (epoch_str - 1) * len(train_loader)
        # 计算全局的训练步数，用于统计训练的进度
    except:
        # 如果加载checkpoint出现异常，则从头开始训练
        epoch_str = 1
        global_step = 0
        # 将epoch和训练步数设置为初始值

    if ckptG is not None or ckptD is not None:
        # 如果生成器或鉴别器有checkpoint，则从头开始训练
        epoch_str = 1
        global_step = 0

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2)
    # 创建生成器模型的学习率衰减策略，指数衰减
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2)
    # 创建鉴别器模型的学习率衰减策略，指数衰减

    scaler = GradScaler(enabled=hps.train.fp16_run)
    # 创建一个GradScaler对象，用于在使用混合精度训练时动态调整梯度的大小

    for epoch in range(epoch_str, hps.train.epochs + 1):
        # 进行epoch轮的训练和评估

        if rank == 0:
            # 只在rank=0的进程上进行记录summary

            train_and_evaluate(rank, epoch, hps, [net_g, net_d], [optim_g, optim_d], [scheduler_g, scheduler_d], scaler,
                               [train_loader, eval_loader], logger, [writer, writer_eval])
            # 执行训练和评估的函数，完成一个epoch的训练和评估
        else:
            train_and_evaluate(rank, epoch, hps, [net_g, net_d], [optim_g, optim_d], [scheduler_g, scheduler_d], scaler,
                               [train_loader, None], None, None)
            # 执行训练的函数，完成一个epoch的训练
        scheduler_g.step()
        # 更新生成器模型的学习率
        scheduler_d.step()
        # 更新鉴别器模型的学习率


def train_and_evaluate(rank, epoch, hps, nets, optims, schedulers, scaler, loaders, logger, writers):
    '''
    ### 训练与验证代码
    - rank: 进程的编号，
    - epoch: 当前的训练epoch数，
    - hps: 配置参数字典，
    - nets: 模型列表，
    - optims: 优化器列表，
    - schedulers: 学习率调度器列表，
    - scaler: GradScaler对象，
    - loaders: 数据加载器列表，
    - logger: 日志记录器，
    - writers: TensorBoard的SummaryWriter列表
    '''

    net_g, net_d = nets
    # 获取生成器和鉴别器模型
    optim_g, optim_d = optims
    # 获取生成器和鉴别器模型的优化器
    scheduler_g, scheduler_d = schedulers
    # 获取生成器和鉴别器模型的学习率调度器
    train_loader, eval_loader = loaders
    # 获取训练数据和评估数据的加载器
    if writers is not None:
        writer, writer_eval = writers
    # 获取TensorBoard的SummaryWriter，用于记录训练和评估的数据

    # train_loader.batch_sampler.set_epoch(epoch)
    # （注释掉的代码）用于在每个epoch开始时，对分布式训练的数据重新进行shuffle，以保证每个epoch数据顺序的一致性

    global global_step
    # 声明全局变量global_step，用于统计训练的总步数

    net_g.train()
    net_d.train()
    # 将生成器和鉴别器设置为训练模式
    for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths, speakers, emo) in enumerate(train_loader):
        # 在训练数据加载器中迭代，获取一个batch的数据
        # print(batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths, speakers, emo))
        x, x_lengths = x.cuda(rank, non_blocking=True), x_lengths.cuda(rank, non_blocking=True)
        spec, spec_lengths = spec.cuda(rank, non_blocking=True), spec_lengths.cuda(rank, non_blocking=True)
        y, y_lengths = y.cuda(rank, non_blocking=True), y_lengths.cuda(rank, non_blocking=True)
        speakers = speakers.cuda(rank, non_blocking=True)
        emo = emo.cuda(rank, non_blocking=True)

        with autocast(enabled=hps.train.fp16_run):
            '''
            autocast 允许在其内部的代码块中，自动将计算的输入和输出张量转换为给定的浮点精度（通常为半精度浮点数），
            从而实现混合精度训练。在这个上下文中 PyTorch 会根据输入的张量类型自动选择合适的精度执行计算，
            并且会在需要时自动执行类型转换，使得计算在不同精度之间进行。
            '''
            y_hat, l_length, attn, ids_slice, x_mask, z_mask, \
            (z, z_p, m_p, logs_p, m_q, logs_q) = net_g(x, x_lengths, spec, spec_lengths, speakers, emo)
            # 生成 net_g 通过生成器模型，生成虚假的语音

            # print("spec:",spec.shape) 谱图

            mel = spec_to_mel_torch(
                spec,
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.mel_fmin,
                hps.data.mel_fmax)
            # 输入的线性谱 -> 梅尔谱转换


            y_mel = commons.slice_segments(mel, ids_slice, hps.train.segment_size // hps.data.hop_length)
            # 真实语音y的梅尔谱切片 从输入的频谱图张量中按照给定的片段长度切片 
            y_hat_mel = mel_spectrogram_torch(
                y_hat.squeeze(1),
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                hps.data.mel_fmin,
                hps.data.mel_fmax
            )
            # 生成语音y_hat 的梅尔谱


            y = commons.slice_segments(y, ids_slice * hps.data.hop_length, hps.train.segment_size)  
            # 对真实语音进行截取，使其与虚假语音长度一致
            

            # Discriminator 判别器
            y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
            # 第一次判别
            # 输入：真实语音 y 和生成器生成的虚假语音 y_hat（但不进行梯度传递，即detach()）
            # 目的：鉴别器用于学习区分真实语音和虚假语音，即区分真实数据和生成数据，此时对生成器的参数不进行更新，只进行判别器的反向传播和梯度计算。

            with autocast(enabled=False):
                loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
                # 计算鉴别器的损失
                loss_disc_all = loss_disc

        optim_d.zero_grad()
        # 将鉴别器的梯度清零，准备反向传播
        scaler.scale(loss_disc_all).backward()
        # 反向传播计算损失相对于鉴别器模型的梯度，使用GradScaler进行梯度缩放，以避免梯度溢出
        scaler.unscale_(optim_d)
        # 取消梯度缩放
        grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
        # 对鉴别器模型的梯度进行裁剪，以防止梯度爆炸
        scaler.step(optim_d)
        # 更新鉴别器模型的参数
        # 注意：这里的更新是使用未缩放的梯度，
        # 因为在step时GradScaler会自动将梯度缩放应用到优化器的参数更新上

        with autocast(enabled=hps.train.fp16_run):
            # 开启混合精度训练的上下文管理器

            # Generator
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
            # 第二次判别
            # 用参数更新后的判别器再次判别
            # 输入：真实语音 y 和生成器生成的虚假语音 y_hat（更新生成器参数）
            # 目的：鉴别器用于学习区分生成器生成的虚假语音和真实语音，即区分生成数据和真实数据。这一步骤是为了提供生成器优化的反馈信号，对生成器的参数进行更新，使其生成的虚假语音更逼真。
            with autocast(enabled=False):
                loss_dur = torch.sum(l_length.float())
                # 计算时长损失 l_length
                loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
                # 计算梅尔频谱损失
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
                # 计算KL散度损失
                loss_fm = feature_loss(fmap_r, fmap_g)
                # 计算特征匹配损失
                loss_gen, losses_gen = generator_loss(y_d_hat_g)
                # 计算生成器损失
                loss_gen_all = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl
                # 计算生成器的总损失

        optim_g.zero_grad()
        # 将生成器的梯度清零，准备反向传播
        scaler.scale(loss_gen_all.float()).backward()
        # 反向传播计算损失相对于生成器模型的梯度，使用GradScaler进行梯度缩放，以避免梯度溢出
        scaler.unscale_(optim_g)
        # 取消梯度缩放
        grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
        # 对生成器模型的梯度进行裁剪，以防止梯度爆炸
        scaler.step(optim_g)
        # 更新生成器模型的参数
        # 注意：这里的更新是使用未缩放的梯度，因为在step时GradScaler会自动将梯度缩放应用到优化器的参数更新上
        scaler.update()
        # 更新GradScaler内部的状态，以便下一次迭代时进行正确的梯度缩放

        if rank == 0:
            # 只在rank=0的进程上进行训练过程记录
            if global_step % hps.train.log_interval == 0:
                # 每隔一定的步数进行训练信息的记录和打印
                lr = optim_g.param_groups[0]['lr']
                # 获取生成器优化器的学习率
                losses = [loss_disc, loss_gen, loss_fm, loss_mel, loss_dur, loss_kl]
                # 将损失值放入一个列表中，便于后续记录和打印
                logger.info('Train Epoch: {} [{:.0f}%]'.format(
                    epoch,
                    100. * batch_idx / len(train_loader)))
                # 打印当前的训练epoch和完成的进度
                logger.info(f"Losses: {[x.item() for x in losses]} Steps: {global_step}, Lr: {lr}")
                # 打印损失值和当前训练步数
                scalar_dict = {"loss/g/total": loss_gen_all, "loss/d/total": loss_disc_all, "learning_rate": lr,
                               "grad_norm_d": grad_norm_d, "grad_norm_g": grad_norm_g}
                scalar_dict.update(
                    {"loss/g/fm": loss_fm, "loss/g/mel": loss_mel, "loss/g/dur": loss_dur, "loss/g/kl": loss_kl})

                scalar_dict.update({"loss/g/{}".format(i): v for i, v in enumerate(losses_gen)})
                scalar_dict.update({"loss/d_r/{}".format(i): v for i, v in enumerate(losses_disc_r)})
                scalar_dict.update({"loss/d_g/{}".format(i): v for i, v in enumerate(losses_disc_g)})
                # 将损失和相关信息记录到一个字典中，用于后续写入TensorBoard
    

                image_dict = { 
                    "slice/mel_org": utils.plot_spectrogram_to_numpy(y_mel[0].data.cpu().numpy()),
                    "slice/mel_gen": utils.plot_spectrogram_to_numpy(y_hat_mel[0].data.cpu().numpy()),
                    "all/mel": utils.plot_spectrogram_to_numpy(mel[0].data.cpu().numpy()),
                    "all/attn": utils.plot_alignment_to_numpy(attn[0, 0].data.cpu().numpy())
                }
                # 使用matplotlib画 Mel 图，并记录到tensorboard summary。
                utils.summarize(
                    writer=writer,
                    global_step=global_step,
                    images=image_dict,
                    scalars=scalar_dict)
                # 将损失值和图片写入TensorBoard
            if global_step % hps.train.eval_interval == 0:
                # 每隔一定的步数进行模型评估
                print("== Evaluating ==")
                evaluate(hps, net_g, eval_loader, writer_eval)
                # 执行评估函数，对模型进行评估
                utils.save_checkpoint(net_g, optim_g, hps.train.learning_rate, epoch,
                                      os.path.join(hps.model_dir, "G_{}.pth".format(global_step)))
                # 将生成器模型的参数保存为checkpoint
                utils.save_checkpoint(net_d, optim_d, hps.train.learning_rate, epoch,
                                      os.path.join(hps.model_dir, "D_{}.pth".format(global_step)))
                # 将鉴别器模型的参数保存为checkpoint
        global_step += 1
        # 更新全局的训练步数，用于统计训练的进度

    if rank == 0:
        logger.info('====> Epoch: {}'.format(epoch))


def evaluate(hps, 
             generator:SynthesizerTrn, 
             eval_loader:DataLoader, 
             writer_eval):
    '''对模型进行评估，并将评估结果记录到 TensorBoard 中。

    Args:
    - hps (argparse.Namespace): 包含各种超参数的命名空间。
    - generator (torch.nn.Module): 要评估的生成器模型。
    - eval_loader (torch.utils.data.DataLoader): 用于评估的数- 据加载器。
    - writer_eval (tensorboard.SummaryWriter): 用于记录评估结- 果的 TensorBoard SummaryWriter。

    Returns:
        None
    '''

    generator.eval()
    # 将生成器模型设为评估模式，这样可以关闭 Batch Normalization 和 Dropout 等模型中的随机性。

    with torch.no_grad():
        # 禁用梯度计算，不需要进行反向传播。

        for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths, speakers, emo) in enumerate(eval_loader):
            # 遍历评估数据加载器中的每个批次。

            x, x_lengths = x.cuda(0), x_lengths.cuda(0)
            spec, spec_lengths = spec.cuda(0), spec_lengths.cuda(0)
            y, y_lengths = y.cuda(0), y_lengths.cuda(0)
            speakers = speakers.cuda(0)
            emo = emo.cuda(0)
            # 将输入数据移动到 GPU 上，通常用于 GPU 加速计算。

            # 移除了原有的 else 分支，该分支没有被执行的必要。
            # remove else
            x = x[:1]
            x_lengths = x_lengths[:1]
            spec = spec[:1]
            spec_lengths = spec_lengths[:1]
            y = y[:1]
            y_lengths = y_lengths[:1]
            speakers = speakers[:1]
            emo = emo[:1]
            # 仅使用第一个样本进行评估，这里可能是为了节省计算资源，只用一个样本来展示评估结果。

            break
            # 在此处进行了 break，即只评估了第一个样本。

        y_hat, attn, mask, *_ = generator.module.infer(x, x_lengths, speakers, emo, max_len=1000)
        # 使用生成器模型进行推断，生成输出序列 y_hat。
        # module 表示生成器模型已经封装成了一个 DataParallel 模型，因此需要使用 .module 来调用其方法。

        y_hat_lengths = mask.sum([1, 2]).long() * hps.data.hop_length
        # 计算输出序列 y_hat 的长度。

        mel = spec_to_mel_torch(
            spec,
            hps.data.filter_length,
            hps.data.n_mel_channels,
            hps.data.sampling_rate,
            hps.data.mel_fmin,
            hps.data.mel_fmax)
        # 将输入音频信号的频谱转换为梅尔频谱图。

        y_hat_mel = mel_spectrogram_torch(
            y_hat.squeeze(1).float(),
            hps.data.filter_length,
            hps.data.n_mel_channels,
            hps.data.sampling_rate,
            hps.data.hop_length,
            hps.data.win_length,
            hps.data.mel_fmin,
            hps.data.mel_fmax
        )
        # 将生成的音频序列 y_hat 转换为对应的梅尔频谱图。

    image_dict = {
        "gen/mel": utils.plot_spectrogram_to_numpy(y_hat_mel[0].cpu().numpy())
    }
    # 将生成的梅尔频谱图转换为可视化的图像，用于记录到 TensorBoard。

    audio_dict = {
        "gen/audio": y_hat[0, :, :y_hat_lengths[0]]
    }
    # 将生成的音频序列转换为可视化的音频，用于记录到 TensorBoard。

    if global_step == 0:
        image_dict.update({"gt/mel": utils.plot_spectrogram_to_numpy(mel[0].cpu().numpy())})
        audio_dict.update({"gt/audio": y[0, :, :y_lengths[0]]})
    # 如果全局步数 global_step 为 0，则也记录原始梅尔频谱图和音频序列（标签）到 TensorBoard。

    utils.summarize(
        writer=writer_eval,
        global_step=global_step,
        images=image_dict,
        audios=audio_dict,
        audio_sampling_rate=hps.data.sampling_rate
    )
    # 调用 utils.summarize 函数，将评估结果写入 TensorBoard。

    generator.train()
    # 将生成器模型设回训练模式，即启用 Batch Normalization 和 Dropout 等模型中的随机性，以便进行训练。


if __name__ == "__main__":
    main()
