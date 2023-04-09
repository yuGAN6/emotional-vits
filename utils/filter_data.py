import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description='Emotion Extraction Preprocess')
parser.add_argument('--data-txt', required=True, type=str, help='path of the filelists')
ARGS = parser.parse_args()

import pandas as pd
df = pd.read_table(ARGS.data_txt,sep='|')
df['length'] = df[0].apply(lambda x:float(str(Path(x).stem).split('-')[-1]) - float(str(Path(x).stem).split('-')[-2]))
df = df[df['length']>=500]
df = df[[0, 1, 2]]
df.to_csv(Path(r"C:\Users\63537\Desktop\projects\emotional-vits-main\filelists\yua\20221110\train-filtered.txt"),index=None,header=None,sep='|')