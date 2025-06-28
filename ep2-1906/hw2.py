import os
import pandas as pd
import sys

df = pd.read_json('news_category_dataset_v3.json', lines=True)
print(df)

df_headline = df['headline']

# ep1-1706 klasörünü path'e ekle
current_dir = os.path.dirname(os.path.abspath(__file__))
hw1_path = os.path.abspath(os.path.join(current_dir, "..", "ep1-1706"))
sys.path.append(hw1_path)

from hw1 import text_pipeline