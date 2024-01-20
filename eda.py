# References:
    # https://kaggle-kr.tistory.com/17?category=868316

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import re

plt.style.use("seaborn-v0_8")
sns.set(font_scale=1)

pd.options.display.max_columns = 12
pd.options.display.width = 70

np.set_printoptions(linewidth=70)


train_df = pd.read_csv("/Users/jongbeomkim/Desktop/workspace/Kaggle-Titanic/dataset/train.csv")
test_df = pd.read_csv("/Users/jongbeomkim/Desktop/workspace/Kaggle-Titanic/dataset/test.csv")

train_df.head()
train_df.info()
train_df.describe()

### "Survived" column
train_df["Survived"].value_counts() / len(train_df)
train_df["Survived"].value_counts().plot.pie(autopct="%.1f%%")
plt.show()

### Missing value
train_df.isna().sum(axis=0) / len(train_df)
msno.matrix(train_df, figsize=(12, 6), fontsize=8)
plt.tight_layout()
plt.show()

msno.matrix(test_df, figsize=(12, 6), fontsize=8)
plt.tight_layout()
plt.show()

### "Cabin" column
train_df["Cabin"].unique()

# 뭔가 상관관계를 찾으려고 했지만
train_df["Cabin"].fillna("", inplace=True)
train_df["new_Cabin"] = train_df["Cabin"].apply(lambda x: re.sub(pattern=r"[0-9 ]", repl="", string=x))
train_df["new_Cabin"] = train_df["new_Cabin"].apply(lambda x: re.sub(pattern=r"(.)\1*", repl=r"\1", string=x))

sns.countplot(data=train_df, x="new_Cabin", hue="Survived")
plt.tight_layout()
plt.show()
# 딱히 찾을 수 없었습니다.

train_df["Cabin"].fillna("", inplace=True)
train_df["new_Cabin"] = train_df["Cabin"].apply(lambda x: 0 if not x else 1)

sns.countplot(data=train_df, x="new_Cabin", hue="Survived")
plt.tight_layout()
plt.show()
# "Cabin" column이 결측치가 아닌 경우의 생존률이 더 높았으므로 이를 활용할 수 있을 것으로 보입니다.

### "Sex" column
fig, axes = plt.subplots(1, 1, figsize=(7, 7))
sns.countplot(ax=axes, data=train_df, x="Sex", hue="Survived")
plt.tight_layout()
plt.show()

### "Age" column
age_cnts = train_df["Age"].value_counts().to_dict()
for age, cnt in age_cnts.items():
    if age - int(age) != 0:
        print(age, cnt)
# 정수로 변환할 수 없는 "Age" 값은 전부 1명 또는 2명만 가지고 있습니다.

### "Embarked" column

### "Pclass" column
train_df[["Pclass", "Survived"]].groupby(["Pclass", "Survived"]).count()
ct = pd.crosstab(index=train_df["Pclass"], columns=train_df["Survived"])
ct["total"] = ct[0] + ct[1]
ct["ratio"] = ct[1] / ct["total"]
print(ct)

sns.countplot(data=train_df, x="Pclass", hue="Survived")
plt.tight_layout()
plt.show()

### "Pclass" and "Sex" column
# fig, axes = plt.subplots(1, 1, figsize=(7, 7))
sns.catplot(data=train_df, x="Pclass", y="Survived", hue="Sex", kind="point", hue_order=[0, 1])
plt.tight_layout()
plt.show()

### "Fare" column
train_df["log_Fare"] = train_df["Fare"].apply(lambda x: np.log(x) if x > 0 else 0)
train_df["Fare"].skew(), train_df["log_Fare"].skew()

# sns.displot(train_df["Fare"], kind="hist")
sns.displot(train_df["Fare"], kind="kde")
sns.displot(train_df["log_Fare"], kind="kde")
plt.tight_layout()
plt.show()

### "Name" column
train_df["Name"].unique()
#  # 여는 괄호와 닫는 괄호가 짝이 맞지 않는 경우는 없습니다.
# (train_df["Name"].str.count("\(") - train_df["Name"].str.count("\)")).unique()
train_df["title"] = train_df["Name"].apply(lambda x: re.search(pattern="[a-zA-Z]+\.", string=x).group(0))
train_df["title"].unique()
train_df["title"].value_counts()
ct = pd.crosstab(index=train_df["title"], columns=train_df["Sex"])
print(ct)
