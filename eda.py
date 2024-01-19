import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

plt.style.use("seaborn-v0_8")
sns.set(font_scale=2.5)

pd.options.display.max_columns = 12
pd.options.display.width = 70

np.set_printoptions(linewidth=70)

train_df = pd.read_csv("/Users/jongbeomkim/Desktop/workspace/Kaggle-Titanic/dataset/train.csv")

train_df.head()
train_df.info()
train_df.describe()

# Missing value
train_df.isna().sum(axis=0) / len(train_df)
msno.matrix(train_df, figsize=(12, 6), fontsize=8)
plt.tight_layout()
plt.show()

# "Sex" column

# "Age" column
age_cnts = train_df["Age"].value_counts().to_dict()
for age, cnt in age_cnts.items():
    if age - int(age) != 0:
        print(age, cnt)
# 정수로 변환할 수 없는 "Age" 값은 전부 1명 또는 2명만 가지고 있습니다.

# "Embarked" column

# "Survived" column
train_df["Survived"].value_counts() / len(train_df)
train_df["Survived"].value_counts().plot.pie(autopct="%.1f%%")
plt.show()

# "Pclass" column
train_df["Pclass", "Survived"].value_counts()
train_df[["Pclass", "Survived"]].groupby(["Pclass", "Survived"]).count()
train_df[["Pclass", "Survived"]]
ct = pd.crosstab(index=train_df["Pclass"], columns=train_df["Survived"])
ct["total"] = ct[0] + ct[1]
ct["ratio"] = ct[1] / ct["total"]
ct
