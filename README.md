# mineDetect
using NN (with keras) to classify Mines and Rocks from the Sonar Dataset

## Data observation
The file "sonar.mines" contains 111 patterns obtained by bouncing sonar signals off a metal cylinder at various angles and under various conditions. The file "sonar.rocks" contains 97 patterns obtained from rocks under similar conditions. The transmitted sonar signal is a frequency-modulated chirp, rising in frequency. The data set contains signals obtained from a variety of different aspect angles, spanning 90 degrees for the cylinder and 180 degrees for the rock.

Each pattern is a set of 60 numbers in the range 0.0 to 1.0. 
Each number represents the energy within a particular frequency band, integrated over a certain period of time. The integration aperture for higher frequencies occur later in time, since these frequencies are transmitted later during the chirp.

According to the following boxplots of the frequency bins we can say that the data are more or less evenly distributed.

![alt text](/res/boxplot_frequency.png "Boxplot of frequency")

![alt text](/res/psd.png "Power Spectral Density")

The features are already normalized, therefore, the only preprocessing needed is to encode the classes from String to Integer and split the dataset in training dataset and testing dataset.

```python
y = input_data['label'].copy()
y = LabelEncoder().fit_transform(y)
y = to_categorical(y)

X_df = input_data.copy()
X_df.drop(['label'], inplace=True, axis=1)

X_train, X_test, y_train, y_test = train_test_split(X_df.values, y, test_size=0.2, shuffle=True, random_state=42)
```

## Model used
```python
model = Sequential()
model.add(Dense(input_dim, input_dim=input_dim, activation='sigmoid'))
model.add(Dense(3, activation='sigmoid'))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'categorical_crossentropy'])
model.fit(X_train, y_train, epochs=250, batch_size=32, validation_data=(X_test, y_test), verbose=1)
```

|  Layer (type)  |  Output Shape  |  Param #  |
|----------------|----------------|-----------|
|dense_1 (Dense) |(None, 60)      |3660       |
|dense_2 (Dense) |(None, 3)       |183        |
|dense_3 (Dense) |(None, 2)       |8          |

**Total params:** 3,851

**Trainable params:** 3,851

**Non-trainable params:** 0

Train on 166 samples and validate on 42 samples.

**Training:**
+ 250 epochs
+ batch of 32

![alt text](/res/training.png "Training")

**Test loss:** around 0.31

**Test accuracy:** around 0.92