import pandas as pd
from pathlib import Path

df_sample = pd.read_csv('data/raw/BindingDB_All.tsv', sep='\t', nrows=5)
print(df_sample.columns)

#RAW_PATH= Path("data/raw/bindingdb_raw.tsv")