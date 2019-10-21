import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

# Read the sonar dataset
input_data = pd.read_csv('data/sonar.csv', header=None)
input_data.rename(columns={60: 'label'}, inplace=True)
input_data.rename(columns={60: 'label'}, inplace=True)
print(input_data.head(2))

input_data.plot.box(figsize=(12, 7), xticks=[])
plt.title('Boxplots of all frequency bins')
plt.xlabel('Frequency bin')
plt.ylabel('Power spectral density (normalized)')
# plt.show()

plt.figure(figsize=(8, 5))
plt.plot(input_data[input_data['label'] == 'R'].values[0][:-1], label='Rock', color='black')
plt.plot(input_data[input_data['label'] == 'M'].values[0][:-1], label='Metal', color='lightgray', linestyle='--')
plt.legend()
plt.title('Example of both classes')
plt.xlabel('Frequency bin')
plt.ylabel('Power spectral density (normalized)')
plt.tight_layout()
# plt.show()

# dataset already normalized
# we just have to encode the classes of type string to integer then split the dataset (80%/20%)
y = input_data['label'].copy()
y = LabelEncoder().fit_transform(y)
y = to_categorical(y)

X_df = input_data.copy()
X_df.drop(['label'], inplace=True, axis=1)

X_train, X_test, y_train, y_test = train_test_split(X_df.values, y, test_size=0.2, shuffle=True, random_state=42)


def build_baseline_model_60_1_layer_3_hidden_units(input_dim):
    model = Sequential()
    model.add(Dense(input_dim, input_dim=input_dim, activation='sigmoid'))
    model.add(Dense(3, activation='sigmoid'))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


model = build_baseline_model_60_1_layer_3_hidden_units(60)
model.fit(X_train, y_train, epochs=250, batch_size=32)

# evaluate the keras model
_, accuracy = model.evaluate(X_test, y_test)
print('Accuracy: %.2f' % (accuracy*100))
