# Kaggle: Titanic - Machine Learning from Disaster
# Date: Oct 9, 2021

from datetime import datetime
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import colors
from sklearn import preprocessing
from sklearn import cluster
from scipy import stats
import tensorflow as tf


pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)


# ////////////////////////////// [Preprocessing] Read dataset and parse attributes.
train = pd.read_csv("datasets/train.csv")
print("\n\n" + "=" * 16, "Attributes Summary", "=" * 16)
train.info()
test = pd.merge(left=pd.read_csv("datasets/gender_submission.csv"), right=pd.read_csv("datasets/test.csv"), on="PassengerId")
print("-" * 64)
test.info()

# ////////////////////////////// [Preprocessing] Working with Missing Data.
print("\n\n" + "=" * 16, "Working with Missing Data", "=" * 16)
train["Cabin"] = train["Cabin"].fillna("None")
mean_age = train["Age"].mean()
train["Age"] = train["Age"].fillna(mean_age)
train = train.dropna()
print(train.describe())
test["Cabin"] = test["Cabin"].fillna("None")
test["Age"] = test["Age"].fillna(mean_age)
test = test.dropna()
print("-" * 64)
print(test.describe())

# ////////////////////////////// [Preprocessing] Noise Reduction.
# Because     [1] train["Parch"].mean() == test["Parch"].mean()
#             [2] train["Parch"].std() == test["Parch"].std()
#             [3] train["Parch"].max() != train["Parch"].max()
# Therefore   There exists noise in the attribute `Parch`.
print("\n\n" + "=" * 16, "Noise Reduction", "=" * 16)
test = test[test["Parch"] <= 8]
# Because     train["Fare"].std() >> 1
# Therefore   There exists noise in the attribute `Fare`.
noise_labels = cluster.KMeans(
    n_clusters=2, random_state=0
).fit_predict(
    np.stack([train["Fare"].values, np.zeros_like(train["Fare"].values)], axis=1)
)
if 2 * np.sum(noise_labels) < noise_labels.__len__():
    non_noise_indices = np.arange(noise_labels.__len__())[np.where(noise_labels == 0)]
else:
    non_noise_indices = np.arange(noise_labels.__len__())[np.where(noise_labels == 1)]
train = train.iloc[non_noise_indices, :]
print(train.describe())
print("-" * 64)
noise_labels = cluster.KMeans(
    n_clusters=2, random_state=0
).fit_predict(
    np.stack([test["Fare"].values, np.zeros_like(test["Fare"].values)], axis=1)
)
if 2 * np.sum(noise_labels) < noise_labels.__len__():
    non_noise_indices = np.arange(noise_labels.__len__())[np.where(noise_labels == 0)]
else:
    non_noise_indices = np.arange(noise_labels.__len__())[np.where(noise_labels == 1)]
test = test.iloc[non_noise_indices, :]
print(test.describe())

# ////////////////////////////// [Data Visualization] Generate descriptive statistics.
# -------------------- `Group By` Operator + `Basic Pie Chart` --------------------
for column in ["Survived", "Pclass", "Sex", "SibSp", "Parch", "Cabin", "Embarked"]:
    series = train[column].value_counts()
    fig, ax = plt.subplots()
    ax.set_title(column)
    ax.pie(
        x=series.values,
        explode=0.1 * (series.values == np.max(series.values)),
        labels=series.index.tolist(),
        autopct='%1.1f%%', shadow=True, startangle=90)
    ax.axis('equal')
# -------------------- `Box Plot` --------------------
fig, (ax1, ax2) = plt.subplots(1, 2, tight_layout=True)
ax1.set_title('box plot')
ax1.yaxis.grid(True)
ax1.boxplot(
    list(train[["Age"]].values.T),
    vert=True,  # vertical box alignment
    patch_artist=True,  # fill with color
    labels=["Age"]
)
ax2.set_title('Notched box plot')
ax2.yaxis.grid(True)
box_plot = ax2.boxplot(
    list(train[["SibSp", "Parch"]].values.T),
    notch=True,  # notch shape
    vert=True,  # vertical box alignment
    patch_artist=True,  # fill with color
    labels=["SibSp", "Parch"]
)
for patch, color in zip(box_plot['boxes'], ['lightblue', 'lightgreen']):
    patch.set_facecolor(color)
# -------------------- `Histogram` --------------------
fig, (ax1, ax2) = plt.subplots(1, 2, tight_layout=True)
ax1.yaxis.grid(True)
ax1.set_title("Age")
N, bins, patches = ax1.hist(train["Age"].values)
fracs = N / N.max()
norm = colors.Normalize(fracs.min(), fracs.max())
for thisfrac, thispatch in zip(fracs, patches):
    color = plt.cm.viridis(norm(thisfrac))
    thispatch.set_facecolor(color)
ax2.yaxis.grid(True)
ax2.set_title("Fare")
N, bins, patches = ax2.hist(train["Fare"].values)
fracs = N / N.max()
norm = colors.Normalize(fracs.min(), fracs.max())
for thisfrac, thispatch in zip(fracs, patches):
    color = plt.cm.viridis(norm(thisfrac))
    thispatch.set_facecolor(color)

plt.show()

# ////////////////////////////// [Preprocessing] Data Reduction
print("\n\n" + "=" * 16, "Data Reduction", "=" * 16)
train = train.drop(columns=["PassengerId", "Name", "Ticket"])
test = test.drop(columns=["PassengerId", "Name", "Ticket"])
# The attribute `Cabin` contains too many classes, make the correlation analysis.
cross_table = pd.crosstab(train["Cabin"].values, train["Survived"].values)
if stats.chi2_contingency(observed=cross_table.values)[1] < 0.05:
    train = train.drop(columns=["Cabin"])
    test = test.drop(columns=["Cabin"])
train.info()
print("-" * 64)
test.info()
print("-" * 64)

# ////////////////////////////// [Preprocessing] Encoding Categorical Attributes.
print("\n\n" + "=" * 16, "Encoding Categorical Features", "=" * 16)
encoder = preprocessing.OneHotEncoder()
encoder.fit(train[["Pclass", "Sex", "Embarked"]].values)
print("[[3, 'male', 'Q'], [3, 'female', 'S']] ->\n",
      encoder.transform([[3, "male", "Q"], [3, "female", "S"]]).toarray())

# ////////////////////////////// [Classification] Decision Tree
print("\n\n" + "=" * 16, "Decision Tree", "=" * 16)

x_train = np.concatenate([
    encoder.transform(train[["Pclass", "Sex", "Embarked"]].values).toarray(),  # Discrete Attributes
    train[["Age", "SibSp", "Parch", "Fare"]].values  # Continuous Attributes
], axis=1)
y_train = train["Survived"].values

x_test = np.concatenate([
    encoder.transform(test[["Pclass", "Sex", "Embarked"]].values).toarray(),  # Discrete Attributes
    test[["Age", "SibSp", "Parch", "Fare"]].values  # Continuous Attributes
], axis=1)
y_test = test["Survived"].values

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(2)
])
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.005),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])
model.fit(
    x_train, y_train, batch_size=4, epochs=20,
    validation_data=(x_test,  y_test),
    callbacks=[
        tf.keras.callbacks.LearningRateScheduler(lambda epoch, lr: 0.75 ** (epoch // 10) * lr, verbose=1),
        tf.keras.callbacks.TensorBoard(log_dir="logs/" + datetime.now().strftime("%Y%m%d-%H%M%S"), histogram_freq=1)
    ],
)
# tensorboard --logdir logs
