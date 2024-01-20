import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from preprocess import preprocess

SEED = 883

train_df = pd.read_csv("/Users/jongbeomkim/Desktop/workspace/Kaggle-Titanic/dataset/train.csv")

train_df = preprocess(train_df)

train_val_input = train_df[["Fare"]].values
train_val_gt = train_df["Survived"].values
train_input, val_input, train_gt, val_gt = train_test_split(
    train_val_input, train_val_gt, test_size=0.1, random_state=SEED,
)

model = RandomForestClassifier(random_state=SEED)
model.fit(train_input, train_gt)
train_acc = (train_gt == model.predict(train_input)).mean()
val_acc = (val_gt == model.predict(val_input)).mean()
print(f"{train_acc: .3f}, {val_acc: .3f}")
