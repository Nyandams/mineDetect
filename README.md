# mineDetect
using NN (with keras) to classify Mines and Rocks from the Sonar Dataset

## Model used
```python
model = Sequential()
    model.add(Dense(input_dim, input_dim=input_dim, activation='sigmoid'))
    model.add(Dense(3, activation='sigmoid'))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'categorical_crossentropy'])
```

|  Layer (type)  |  Output Shape  |  Param #  |
|----------------|----------------|-----------|
|dense_1 (Dense) |(None, 60)      |3660       |
|dense_2 (Dense) |(None, 3)       |183        |
|dense_3 (Dense) |(None, 2)       |8          |

**Total params:** 3,851

**Trainable params:** 3,851

**Non-trainable params:** 0