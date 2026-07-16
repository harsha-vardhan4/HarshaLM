import pandas as pd

df = pd.read_csv("datasets/daily_dialog/dailydialog.csv")

print(repr(df.loc[0, "dialog"]))
print(repr(df.loc[0, "act"]))
print(repr(df.loc[0, "emotion"]))