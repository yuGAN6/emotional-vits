import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description='Emotion Extraction Preprocess')
parser.add_argument('--train-txt', required=True, type=str, help='path of the filelists')
parser.add_argument('--val-txt', required=True, type=str, help='path of the filelists')
ARGS = parser.parse_args()

import pandas as pd
train = pd.read_table(Path(ARGS.train_txt),sep='|', header=None)
val = pd.read_table(Path(ARGS.val_txt),sep='|', header=None)
df = pd.concat([train, val])
unique_chars = ''.join(sorted(set(''.join(df[2]))))

# 完成后
'''
" .Nabdefghijklmnoprstv~æɑɒɔəɪɯɹʃʊʦʧˇˉˊˋ˙↑↓ㄅㄆㄇㄈㄉㄊㄋㄌㄍㄎㄏㄐㄑㄒㄓㄔㄕㄖㄗㄘㄙㄚㄛㄜㄝㄞㄟㄠㄡㄢㄣㄤㄥㄦㄧㄨㄩ，。？"
'''