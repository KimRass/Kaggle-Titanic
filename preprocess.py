import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from sklearn.preprocessing import StandardScaler

train_df = pd.read_csv("/Users/jongbeomkim/Desktop/workspace/Kaggle-Titanic/dataset/train.csv")
test_df = pd.read_csv("/Users/jongbeomkim/Desktop/workspace/Kaggle-Titanic/dataset/test.csv")

train_df.head()
train_df.info()
train_df.describe()

test_df.head()
test_df.info()
test_df.describe()

# train_df.drop(["Name"], axis=1, inplace=True)

train_df["Sex"] = train_df["Sex"].map({"male": 0, "female": 1})

train_df["Embarked"] = train_df["Embarked"].astype("category").cat.codes


train_df["Cabin"].fillna("", inplace=True)
train_df["Cabin"] = train_df["Cabin"].apply(lambda x: 0 if not x else 1)


sns.heatmap(train_df[["Survived", "Pclass"]].corr())
plt.show()

scaler = StandardScaler()
# train_df[["Pclass"]] = scaler.fit_transform(train_df[["Pclass"]])
# train_df[["Survived"]] = scaler.fit_transform(train_df[["Survived"]])
# train_df[["Survived", "Pclass"]].corr()

sns.displot(train_df["Survived"], kind="kde")
plt.tight_layout()
plt.show()


def preprocess(train_df):
    new_train_df = train_df.copy()

    # Decision tree 계열의 모델에서는 효과가 없습니다.
    # train_df["Fare"] = train_df["Fare"].apply(lambda x: np.log(x) if x > 0 else 0)
    # train_df["Fare"] = train_df["Fare"].apply(lambda x: x ** 0.213)

    scaler = StandardScaler()

    return new_train_df
