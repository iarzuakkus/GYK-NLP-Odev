import pandas as pd
import os

data_dir = "ep3-0307/data/full_dataset/"
filenames = ["goemotions_1.csv", "goemotions_2.csv", "goemotions_3.csv"]

dfs = [pd.read_csv(os.path.join(data_dir, f)) for f in filenames]

merged_df = pd.concat(dfs, ignore_index=True)
merged_df.drop_duplicates(inplace=True)

output_path = "ep3-0307/data/goemotions_merged.csv"
merged_df.to_csv(output_path, index=False)

print(f"Birleştirme tamamlandı: {output_path}")
