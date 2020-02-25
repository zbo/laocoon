#https://www.cnblogs.com/mfryf/p/7904044.html
import pandas as pd
import seaborn as sns
sns.set(context="notebook",style="whitegrid",palette="dark")
df0 = pd.read_csv("data.csv",names=["square","price"])
print(df0.head())