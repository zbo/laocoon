import pandas as pd
import seaborn as sns

def normalize_feature(df):
    return df.apply(lambda column:(column-column.mean()) / column.std())

sns.set(context="notebook",style="whitegrid",palette="dark")
df1 = pd.read_csv("data1.csv",names=["square","price"])
df = normalize_feature(df1)
print(df.head())
# sns.lmplot("square","price",df0,height=6,fit_reg=True)
