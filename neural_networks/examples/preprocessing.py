import numpy as np
from sklearn.preprocessing import StandardScaler as SklSS, MinMaxScaler as SklMM

from neural_networks.utils import StandardScaler as MySS, MinMaxScaler as MyMM, train_test_split

data = np.array([
    [1, 2, 3],
    [0, -4, 10],
    [12, 3, 6],
    [12, 3, 3],
    [0, 0, 0],
    [12, 3, 10]
])

my = MySS(ddof=0)
sk = SklSS()

print('My StandardScaler:\n', my.fit_transform(data))
print('Sklearn StandardScaler:\n', sk.fit_transform(data))

my = MyMM()
sk = SklMM()

print('My MinMaxScaler:\n', my.fit_transform(data))
print('Sklearn MinMaxScaler:\n', sk.fit_transform(data))

y = np.array([
    [1],
    [0],
    [15],
    [12],
    [11],
    [-2]
])
x_train, x_test, y_train, y_test = train_test_split(data, y, test_size=0.5, random_state=42)

print(x_train)
print(x_test)

print(y_train)
print(y_test)
