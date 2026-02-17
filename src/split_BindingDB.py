import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

#For reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

#Load cleaned dataset
df = pd.read_csv('data/processed/bindingdb_clean.csv')
# print(f"Total entries in cleaned dataset: {len(df)}")
# print(f"Columns: {list(df.columns)}")

##PROTEIN LEVEL SPLITTING

unique_proteins = df['BindingDB Target Chain Sequence 1'].unique()
# print(f"Total unique proteins: {len(unique_proteins)}")

#Split unique proteins into 70/30 train/temp, then 15/15 val/test
train_proteins, temp_proteins = train_test_split(
    unique_proteins,
    test_size=0.3,
    random_state=RANDOM_STATE
)

val_proteins, test_proteins = train_test_split(
    temp_proteins,
    test_size=0.5,
    random_state=RANDOM_STATE
)

#Make train test and val dataframes
train_df = df[df['BindingDB Target Chain Sequence 1'].isin(train_proteins)]
# print(f"Train samples: {len(train_df)}")

val_df = df[df['BindingDB Target Chain Sequence 1'].isin(val_proteins)]
# print(f"Val samples: {len(val_df)}")

test_df = df[df['BindingDB Target Chain Sequence 1'].isin(test_proteins)]
# print(f"Test samples: {len(test_df)}")


#Ensure no overlap and save files
train_set = set(train_proteins)
val_set = set(val_proteins)
test_set = set(test_proteins)

train_df.to_csv('data/splits/train.csv', index=False)
val_df.to_csv('data/splits/val.csv', index=False)
test_df.to_csv('data/splits/test.csv', index=False)