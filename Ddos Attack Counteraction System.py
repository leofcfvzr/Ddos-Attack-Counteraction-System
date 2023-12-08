import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Загрузка данных
data = pd.read_csv('ddos_data.csv')

# Подготовка данных
X = data.drop(['Label'], axis=1)
y = data['Label']

# Разделение данных на тренировочную и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создание и обучение модели
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Прогнозирование
predictions = model.predict(X_test)

# Оценка точности модели
accuracy = accuracy_score(y_test, predictions)
print("Точность модели: ", accuracy)
