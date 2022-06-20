import numpy as np
from soundfile import read
import librosa
import torch
import soundfile as sf


def get_mask_from_lengths(lengths):
    max_len = int(torch.max(lengths))
    ids = torch.arange(0, max_len, out=torch.LongTensor(max_len)).cuda()
    mask = (ids < lengths.unsqueeze(1)).bool()
    return mask


def load_wav_to_torch(full_path, use_librosa=True, audio_dtype='np.int16', final_sr=22050):
    if audio_dtype != 'np.int16':
        audio, sampling_rate = sf.read(full_path, dtype='int16')
        audio = librosa.resample(audio.astype(np.float32), sampling_rate, final_sr)
        data = audio.astype(np.int16)
    else:
        if use_librosa:
            data, _final_sr = librosa.load(full_path)
            data = librosa.resample(data, orig_sr=_final_sr, target_sr=final_sr)
        else:
            final_sr, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), final_sr


def load_filepaths_and_text(filename, split="|"):
    with open(filename, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text


def to_gpu(x):
    x = x.contiguous()

    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return torch.autograd.Variable(x)
