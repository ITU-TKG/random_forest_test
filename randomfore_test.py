import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# from transformers.trainer_utils import set_seed
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import os


df = pd.read_csv(
    "svm_test/depression_dataset_reddit_cleaned.csv"
    )

text = df["clean_text"].to_list()
label = df["is_depression"].to_list()

# set_seed(42)
x_train, x_val, y_train, y_val = train_test_split(
    text,
    label,
    test_size=0.2
    )

# テキストデータを数値ベクトルに変換
vectorizer = TfidfVectorizer()
x_train_tfidf = vectorizer.fit_transform(x_train)
x_val_tfidf = vectorizer.transform(x_val)


model = RFC()
model.fit(x_train_tfidf, y_train)


if not os.path.exists("画像"):
    os.makedirs("画像")


# トレーニングデータに対する精度
pred = model.predict(x_val_tfidf)
accuracy = accuracy_score(y_val, pred)
print(f'{accuracy:.4f}')

# 混同行列を計算して表示
cm = confusion_matrix(y_val, pred)
# 混同行列をヒートマップとして表示
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['0', '1'],
    yticklabels=['0', '1'],
    annot_kws={"size": 10}
    )

plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig("画像/confusion_matrix.png")
# plt.show()
