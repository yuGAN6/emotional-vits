import time
import os
import random
import numpy as np
import torch, torchaudio
import torch.utils.data

import commons 
from mel_processing import spectrogram_torch
from utils import load_wav_to_torch, load_filepaths_and_text
from text import text_to_sequence, cleaned_text_to_sequence


class TextAudioLoader(torch.utils.data.Dataset):
    """
        1) loads audio, text pairs
        2) normalizes text and converts them to sequences of integers
        3) computes spectrograms from audio files.
    """
    def __init__(self, audiopaths_and_text, hparams):
        self.audiopaths_and_text = load_filepaths_and_text(audiopaths_and_text)
        self.text_cleaners  = hparams.text_cleaners
        self.max_wav_value  = hparams.max_wav_value
        self.sampling_rate  = hparams.sampling_rate
        self.filter_length  = hparams.filter_length 
        self.hop_length     = hparams.hop_length 
        self.win_length     = hparams.win_length
        self.sampling_rate  = hparams.sampling_rate 

        self.cleaned_text = getattr(hparams, "cleaned_text", False)

        self.add_blank = hparams.add_blank
        self.min_text_len = getattr(hparams, "min_text_len", 1)
        self.max_text_len = getattr(hparams, "max_text_len", 190)

        random.seed(1234)
        random.shuffle(self.audiopaths_and_text)
        self._filter()


    def _filter(self):
        """
        Filter text & store spec lengths
        """
        # Store spectrogram lengths for Bucketing
        # wav_length ~= file_size / (wav_channels * Bytes per dim) = file_size / (1 * 2)
        # spec_length = wav_length // hop_length

        audiopaths_and_text_new = []
        lengths = []
        for audiopath, text in self.audiopaths_and_text:
            if self.min_text_len <= len(text) and len(text) <= self.max_text_len:
                audiopaths_and_text_new.append([audiopath, text])
                lengths.append(os.path.getsize(audiopath) // (2 * self.hop_length))
        self.audiopaths_and_text = audiopaths_and_text_new
        self.lengths = lengths

    def get_audio_text_pair(self, audiopath_and_text):
        # separate filename and text
        audiopath, text = audiopath_and_text[0], audiopath_and_text[1]
        text = self.get_text(text)
        spec, wav = self.get_audio(audiopath)
        emo = torch.FloatTensor(np.load(audiopath+".emo.npy"))
        return (text, spec, wav, emo)

    def get_audio(self, filename):
        audio, sampling_rate = load_wav_to_torch(filename)
        if sampling_rate != self.sampling_rate:
            raise ValueError("{} {} SR doesn't match target {} SR".format(
                sampling_rate, self.sampling_rate))
        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        spec_filename = filename.replace(".wav", ".spec.pt")
        if os.path.exists(spec_filename):
            spec = torch.load(spec_filename)
        else:
            spec = spectrogram_torch(audio_norm, self.filter_length,
                self.sampling_rate, self.hop_length, self.win_length,
                center=False)
            spec = torch.squeeze(spec, 0)
            torch.save(spec, spec_filename)
        return spec, audio_norm

    def get_text(self, text):
        if self.cleaned_text:
            text_norm = cleaned_text_to_sequence(text)
        else:
            text_norm = text_to_sequence(text, self.text_cleaners)
        if self.add_blank:
            text_norm = commons.intersperse(text_norm, 0)
        text_norm = torch.LongTensor(text_norm)
        return text_norm

    def __getitem__(self, index):
        return self.get_audio_text_pair(self.audiopaths_and_text[index])

    def __len__(self):
        return len(self.audiopaths_and_text)


class TextAudioCollate():
    """ Zero-pads model inputs and targets
    """
    def __init__(self, return_ids=False):
        self.return_ids = return_ids

    def __call__(self, batch):
        """Collate's training batch from normalized text and aduio
        PARAMS
        ------
        batch: [text_normalized, spec_normalized, wav_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[1].size(1) for x in batch]),
            dim=0, descending=True)

        max_text_len = max([len(x[0]) for x in batch])
        max_spec_len = max([x[1].size(1) for x in batch])
        max_wav_len = max([x[2].size(1) for x in batch])

        text_lengths = torch.LongTensor(len(batch))
        spec_lengths = torch.LongTensor(len(batch))
        wav_lengths = torch.LongTensor(len(batch))

        text_padded = torch.LongTensor(len(batch), max_text_len)
        spec_padded = torch.FloatTensor(len(batch), batch[0][1].size(0), max_spec_len)
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)
        emo = torch.FloatTensor(len(batch), 1024)

        text_padded.zero_()
        spec_padded.zero_()
        wav_padded.zero_()
        emo.zero_()

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            text = row[0]
            text_padded[i, :text.size(0)] = text
            text_lengths[i] = text.size(0)

            spec = row[1]
            spec_padded[i, :, :spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)

            wav = row[2]
            wav_padded[i, :, :wav.size(1)] = wav
            wav_lengths[i] = wav.size(1)

            emo[i, :] = row[3]

        if self.return_ids:
            return text_padded, text_lengths, spec_padded, spec_lengths, wav_padded, wav_lengths, ids_sorted_decreasing
        return text_padded, text_lengths, spec_padded, spec_lengths, wav_padded, wav_lengths, emo


"""Multi speaker version"""
class TextAudioSpeakerLoader(torch.utils.data.Dataset):
    """
    ### 用于加载音频、说话者ID和文本对，对文本进行归一化，并将其转换为整数序列，从音频文件计算频谱图。

    1) loads audio, speaker_id, text pairs 加载 音频, 说话人ID, 文本对
    2) normalizes text and converts them to sequences of integers 文本归一化并转换为整数序列
    3) computes spectrograms from audio files. 计算音频频谱图

    Args:
    - audiopaths_sid_text (list): 包含音频文件路径、说话者ID和- 文本的列表。
    - hparams (argparse.Namespace): 包含各种超参数的命名空间。

    Attributes:
    - audiopaths_sid_text (list): 加载的音频文件路径、说话者ID- 和文本的列表。
    - text_cleaners (str): 文本清理器的名称，用于对文本进行清理和- 处理。
    - max_wav_value (int): 音频文件的最大值（用于归一化）。
    - sampling_rate (int): 采样率，用于音频处理。
    - filter_length (int): 滤波器长度，用于计算频谱图。
    - hop_length (int): 帧移长度，用于计算频谱图。
    - win_length (int): 窗口长度，用于计算频谱图。
    - cleaned_text (bool): 是否使用清理后的文本。
    - add_blank (bool): 是否在文本中添加空白标记。
    - min_text_len (int): 文本最小长度，用于过滤文本。
    - max_text_len (int): 文本最大长度，用于过滤文本。
    - min_audio_len (int): 音频最小长度，用于过滤音频。

    Methods:
    - _filter(): 过滤文本并存储频谱图长度。
    - get_audio_text_speaker_pair(audiopath_sid_text): 获取- 音频、文本和说话者ID对。
    - get_audio(filename): 从文件中加载音频并计算频谱图。
    - get_text(text): 将文本转换为整数序列。
    - get_sid(sid): 将说话者ID转换为整数张量。
    - __getitem__(index): 获取给定索引处的音频、文本和说话者ID- 对。
    - __len__(): 返回数据集中的样本数量。
    """
    def __init__(self, audiopaths_sid_text, hparams):
        '''
        - audiopaths_sid_text (list): 包含音频文件路径、说话者ID和文本的列表。
        - hparams (argparse.Namespace): 包含各种超参数的命名空间。
        '''
        self.audiopaths_sid_text = load_filepaths_and_text(audiopaths_sid_text)
        self.text_cleaners = hparams.text_cleaners
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.filter_length  = hparams.filter_length
        self.hop_length     = hparams.hop_length
        self.win_length     = hparams.win_length
        self.sampling_rate  = hparams.sampling_rate

        self.cleaned_text = getattr(hparams, "cleaned_text", False)

        self.add_blank = hparams.add_blank
        self.min_text_len = getattr(hparams, "min_text_len", 1)
        self.max_text_len = getattr(hparams, "max_text_len", 190)
        self.min_audio_len = getattr(hparams, "min_audio_len", 200)

        random.seed(1234)
        random.shuffle(self.audiopaths_sid_text)
        self._filter()

    def _filter(self):
        """
        Filter text & store spec lengths
        """
        # Store spectrogram lengths for Bucketing
        # wav_length ~= file_size / (wav_channels * Bytes per dim) = file_size / (1 * 2)
        # 通过文件大小估算wav文件的长度【不太理想】
        # spec_length = wav_length // hop_length

        # 存储经过过滤的音频文件路径、说话者ID和文本的新列表
        audiopaths_sid_text_new = []
        # 存储频谱图的长度
        lengths = []
        for audiopath, sid, text in self.audiopaths_sid_text:
            # 过滤文本，仅保留长度在min_text_len和max_text_len之间的样本
            if self.min_text_len <= len(text) and len(text) <= self.max_text_len:
                # 将通过过滤的样本添加到新列表中
                audiopaths_sid_text_new.append([audiopath, sid, text])
                # 计算并存储频谱图的长度, 1音频长度 = 2字节 所以x2
                lengths.append(os.path.getsize(audiopath) // (2 * self.hop_length))
        # 将过滤后的列表和频谱图长度存储到对象的属性中
        self.audiopaths_sid_text = audiopaths_sid_text_new
        self.lengths = lengths

    def get_audio_text_speaker_pair(self, audiopath_sid_text):
        '''
        ### 从文本形式数据读取拆分为:
        - text: 文本
        - sepc, wav: 谱图, 音频
        - sid: 说话人id
        - emo: 情绪嵌入
        
        Args:
        audiopath_sid_text (list): 包含音频文件路径、说话者ID和文本的列表。

        Returns:
        tuple: 包含以下五个元素的元组：
            - 文本（整数序列）
            - 谱图（频谱图），shape为（频道数，频谱长度）
            - 音频，shape为（1，音频长度）
            - 说话人ID（整数张量）
            - 情绪嵌入（浮点张量）
        '''
        
        audiopath, sid, text = audiopath_sid_text[0], audiopath_sid_text[1], audiopath_sid_text[2]
        # print(audiopath, text)
        text = self.get_text(text)
        spec, wav = self.get_audio(audiopath)
        sid = self.get_sid(sid)
        emo = torch.FloatTensor(np.load(audiopath+".emo.npy"))
        # print((text, spec, wav, sid, emo))
        return (text, spec, wav, sid, emo)

    def get_audio(self, filename):
        """
        从音频文件中加载音频并计算频谱图。

        Args:
            filename (str): 音频文件的路径。

        Returns:
            tuple: 包含以下两个元素的元组：
                - 频谱图（频谱矩阵），shape为（频道数，频谱长度）
                - 归一化后的音频，shape为（1，音频长度）
        """
        # 从音频文件中加载音频和采样率
        audio, sampling_rate = load_wav_to_torch(filename)
        # 若采样率与目标采样率不同，则进行重采样
        if sampling_rate != self.sampling_rate:
            audio = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=self.sampling_rate)(audio)
            # 如果需要，可以打印采样率相关信息，或者抛出异常
            # print(f"sr:{sampling_rate}, target sr:{self.sampling_rate}")
            # raise ValueError("{} {} SR doesn't match target {} SR".format(
            #     sampling_rate, self.sampling_rate))
        # 对音频进行归一化处理，使其值范围在[-1, 1]之间
        audio_norm = audio / self.max_wav_value # 32768
        # 将音频张量的维度由(音频长度,)调整为(1, 音频长度)，以符合频谱图计算的输入格式
        audio_norm = audio_norm.unsqueeze(0)
        # 使用频谱图文件来存储已经计算过的频谱图，如果文件已存在，则直接加载
        spec_filename = filename.replace(".wav", ".spec.pt")
        if os.path.exists(spec_filename):
            spec = torch.load(spec_filename)
        else:
            # 如果频谱图文件不存在，则计算频谱图并保存到文件
            spec = spectrogram_torch(audio_norm, self.filter_length, # n_fft - self.filter_length
                self.sampling_rate, self.hop_length, self.win_length,
                center=False)
            spec = torch.squeeze(spec, 0)
            torch.save(spec, spec_filename)
        # 返回计算得到的频谱图和归一化后的音频
        return spec, audio_norm

    def get_text(self, text):
        # 文本 --> token
        if self.cleaned_text:
            text_norm = cleaned_text_to_sequence(text)
        else:
            text_norm = text_to_sequence(text, cleaner_names=self.text_cleaners)
        if self.add_blank:
            text_norm = commons.intersperse(text_norm, 0)
        text_norm = torch.LongTensor(text_norm)
        return text_norm

    def get_sid(self, sid):
        sid = torch.LongTensor([int(sid)])
        return sid

    def __getitem__(self, index):
        return self.get_audio_text_speaker_pair(self.audiopaths_sid_text[index])

    def __len__(self):
        return len(self.audiopaths_sid_text)


class TextAudioSpeakerCollate():
    """ Zero-pads model inputs and targets
    """
    def __init__(self, return_ids=False):
        self.return_ids = return_ids

    def __call__(self, batch):
        """Collate's training batch from normalized text, audio and speaker identities
        PARAMS
        ------
        batch: [text_normalized, spec_normalized, wav_normalized, sid]
        """
        # Right zero-pad all one-hot text sequences to max input length
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[1].size(1) for x in batch]),
            dim=0, descending=True)

        max_text_len = max([len(x[0]) for x in batch])
        max_spec_len = max([x[1].size(1) for x in batch])
        max_wav_len = max([x[2].size(1) for x in batch])

        text_lengths = torch.LongTensor(len(batch))
        spec_lengths = torch.LongTensor(len(batch))
        wav_lengths = torch.LongTensor(len(batch))
        sid = torch.LongTensor(len(batch))

        text_padded = torch.LongTensor(len(batch), max_text_len)
        spec_padded = torch.FloatTensor(len(batch), batch[0][1].size(0), max_spec_len)
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)
        emo = torch.FloatTensor(len(batch), 1024)

        text_padded.zero_()
        spec_padded.zero_()
        wav_padded.zero_()
        emo.zero_()

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            text = row[0]
            text_padded[i, :text.size(0)] = text
            text_lengths[i] = text.size(0)

            spec = row[1]
            spec_padded[i, :, :spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)

            wav = row[2]
            wav_padded[i, :, :wav.size(1)] = wav
            wav_lengths[i] = wav.size(1)

            sid[i] = row[3]

            emo[i, :] = row[4]

        if self.return_ids:
            # print(f"wav padded shape: {wav_padded.shape}, wav lengths: {wav_lengths}")
            return text_padded, text_lengths, spec_padded, spec_lengths, wav_padded, wav_lengths, sid, ids_sorted_decreasing
        # print(f"wav padded shape: {wav_padded.shape}, wav lengths: {wav_lengths}")
        return text_padded, text_lengths, spec_padded, spec_lengths, wav_padded, wav_lengths, sid, emo


class DistributedBucketSampler(torch.utils.data.distributed.DistributedSampler):
    """
    Maintain similar input lengths in a batch.
    Length groups are specified by boundaries.
    Ex) boundaries = [b1, b2, b3] -> any batch is included either {x | b1 < length(x) <=b2} or {x | b2 < length(x) <= b3}.
  
    It removes samples which are not included in the boundaries.
    Ex) boundaries = [b1, b2, b3] -> any x s.t. length(x) <= b1 or length(x) > b3 are discarded.
    """
    def __init__(self, dataset, batch_size, boundaries, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.lengths = dataset.lengths
        self.batch_size = batch_size
        self.boundaries = boundaries
  
        self.buckets, self.num_samples_per_bucket = self._create_buckets()
        self.total_size = sum(self.num_samples_per_bucket)
        self.num_samples = self.total_size // self.num_replicas
  
    def _create_buckets(self):
        buckets = [[] for _ in range(len(self.boundaries) - 1)]
        for i in range(len(self.lengths)):
            length = self.lengths[i]
            idx_bucket = self._bisect(length)
            if idx_bucket != -1:
                buckets[idx_bucket].append(i)
  
        for i in range(len(buckets) - 1, 0, -1):
            if len(buckets[i]) == 0:
                buckets.pop(i)
                self.boundaries.pop(i+1)
  
        num_samples_per_bucket = []
        for i in range(len(buckets)):
            len_bucket = len(buckets[i])
            total_batch_size = self.num_replicas * self.batch_size
            rem = (total_batch_size - (len_bucket % total_batch_size)) % total_batch_size
            num_samples_per_bucket.append(len_bucket + rem)
        return buckets, num_samples_per_bucket
  
    def __iter__(self):
      # deterministically shuffle based on epoch
      g = torch.Generator()
      g.manual_seed(self.epoch)
  
      indices = []
      if self.shuffle:
          for bucket in self.buckets:
              indices.append(torch.randperm(len(bucket), generator=g).tolist())
      else:
          for bucket in self.buckets:
              indices.append(list(range(len(bucket))))
  
      batches = []
      for i in range(len(self.buckets)):
          bucket = self.buckets[i]
          len_bucket = len(bucket)
          ids_bucket = indices[i]
          num_samples_bucket = self.num_samples_per_bucket[i]
  
          # add extra samples to make it evenly divisible
          rem = num_samples_bucket - len_bucket
          ids_bucket = ids_bucket + ids_bucket * (rem // len_bucket) + ids_bucket[:(rem % len_bucket)]
  
          # subsample
          ids_bucket = ids_bucket[self.rank::self.num_replicas]
  
          # batching
          for j in range(len(ids_bucket) // self.batch_size):
              batch = [bucket[idx] for idx in ids_bucket[j*self.batch_size:(j+1)*self.batch_size]]
              batches.append(batch)
  
      if self.shuffle:
          batch_ids = torch.randperm(len(batches), generator=g).tolist()
          batches = [batches[i] for i in batch_ids]
      self.batches = batches
  
      assert len(self.batches) * self.batch_size == self.num_samples
      return iter(self.batches)
  
    def _bisect(self, x, lo=0, hi=None):
      if hi is None:
          hi = len(self.boundaries) - 1
  
      if hi > lo:
          mid = (hi + lo) // 2
          if self.boundaries[mid] < x and x <= self.boundaries[mid+1]:
              return mid
          elif x <= self.boundaries[mid]:
              return self._bisect(x, lo, mid)
          else:
              return self._bisect(x, mid + 1, hi)
      else:
          return -1

    def __len__(self):
        return self.num_samples // self.batch_size
