import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("WineDataset.csv")

print(df.isna().sum())

df.hist(bins=60, figsize=(20, 10))

plt.show()

X = df.drop(columns=['Wine'])
y = df['Wine']


# Масштабирование
def MinMaxScaler(A):
    for i in A.columns:
        maxi = A[i].max()
        mini = A[i].min()
        A[i] = (A[i] - mini) / (maxi - mini)
    return A


# Разделение на обучающий и тестовый наборы
def train_test_split_custom(X, y, test_size=0.2):
    num_samples = X.shape[0]
    num_test_samples = int(test_size * num_samples)

    # Генерация случайных индексов для тестового набора
    test_indices = np.random.choice(num_samples, num_test_samples, replace=False)

    # Индексы для обучающего набора
    train_indices = np.setdiff1d(np.arange(num_samples), test_indices)

    X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
    y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]

    return X_train, X_test, y_train, y_test


X = MinMaxScaler(X)
X_train, X_test, y_train, y_test = train_test_split_custom(X, y, test_size=0.2)
X_train_np = X_train.to_numpy()
X_test_np = X_test.to_numpy()
y_train_np = y_train.to_numpy()
y_test_np = y_test.to_numpy()


class KNN:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    # Евклидово расстояние
    def distance(self, x0, x1):
        return np.sqrt(np.sum((x0 - x1) ** 2))

    # Наиболее частый класс
    def most_common(self, y):
        labels = np.unique(y)
        count = [list(y).count(i) for i in labels]
        return labels[np.argmax(count)]

    def predict(self, X_test):
        # Предсказываем метки классов
        labels = [self.find_labels(x) for x in X_test]
        return np.array(labels)

    def find_labels(self, x):
        # Считаем расстояние
        distances = [self.distance(x, x_train) for x_train in self.X_train]
        # Берем индексы наблюдений
        k_nearest = np.argsort(distances)[:self.k]
        # По индексам берем метки классов
        labels = [self.y_train[i] for i in k_nearest]
        return self.most_common(labels)


def f1_score(y_test, pred):
    # Classes = [TP, FP, TN, FN]
    y_test = y_test.flatten()
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    classes = list(set(y_test.tolist()))
    classes_stats = [[0, 0, 0, 0]] * len(classes)
    for i, cur_class in enumerate(classes):
        for idx, (el1, el2) in enumerate(zip(y_test, pred)):
            if el1 == el2 == cur_class:
                TP += 1
                classes_stats[i][0] += 1
            if el2 == cur_class and el1 != el2:
                FP += 1
                classes_stats[i][1] += 1
            if el1 == el2 and el1 != cur_class:
                classes_stats[i][2] += 1
                TN += 1
            if el1 != el2 and el1 != cur_class:
                classes_stats[i][3] += 1
                FN += 1

    return TP / (TP + (1 / (len(classes))) * (FP + FN))


def confusion_matrix(y_test, pred):
    y_test = y_test.flatten()

    classes = list(set(y_test.tolist()))
    num_classes = len(classes)
    confusion_matrix = np.zeros((num_classes, num_classes))
    for true_label, predicted_label in zip(y_test, pred):
        true_label_index = classes.index(true_label)
        predicted_label_index = classes.index(predicted_label)
        confusion_matrix[true_label_index][predicted_label_index] += 1

    return confusion_matrix


def show_cf_matrix(cf_matrix):
    fig, AX = plt.subplots(figsize=(3, 2))
    ax = sns.heatmap(cf_matrix, annot=True, cmap='Purples')
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ');
    ax.xaxis.set_ticklabels(['1', '2', '3'])
    ax.yaxis.set_ticklabels(['1', '2', '3'])

    plt.show()


# Модель со случайными признаками
tags = ["Alcohol", "Malic Acid", "Ash", "Alcalinity of ash", "Magnesium", "Total phenols", "Flavanoids",
        "Nonflavanoid phenols", "Proanthocyanins", "Color intensity", "Hue", "OD280/OD315 of diluted wines", "Proline"]
n = random.randint(1, len(tags))
tags_1 = random.sample(tags, n)
print(tags_1)

X_test_rand = X_test[tags_1]
X_train_rand = X_train[tags_1]
X_test_rand = X_test_rand.to_numpy()
X_train_rand = X_train_rand.to_numpy()
k = []
test_score = []
for i in range(3, 21):
    clf = KNN(k=i)
    clf.fit(X_train_rand, y_train_np)
    y_pred = clf.predict(X_test_rand)
    show_cf_matrix(confusion_matrix(y_test_np, y_pred))
    test_score.append(f1_score(y_test_np, y_pred))
    k.append(i)

plt.plot(k, test_score)
plt.show()
