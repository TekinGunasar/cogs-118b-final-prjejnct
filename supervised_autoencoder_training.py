from tensorflow.keras.layers import Conv1D,Conv2D,Dense,Flatten,MaxPooling2D,BatchNormalization,Dropout
from tensorflow.keras import Sequential
from tensorflow_addons.losses import ContrastiveLoss

import tensorflow as tf
from numpy import array

from helper_functions import parse_dataset,flattenMatrixDataset
from sklearn.preprocessing import StandardScaler

from pickle import dump

#Doing inter-subject classification with subject 'B'
training_data_path_trial_1 = 'Raw Mat Files (CLA)/CLASubjectB1510193StLRHand.mat'
training_data_path_trial_2 = 'Raw Mat Files (CLA)/CLASubjectB1510193StLRHand.mat'

window_size = 50
limit = 10*3*190

training_set_1 = parse_dataset(training_data_path_trial_1,window_size,limit)
training_set_2 = parse_dataset(training_data_path_trial_2,window_size,limit)

X_1 = list(StandardScaler().fit_transform(flattenMatrixDataset(array(training_set_1[0]))))
X_2 = list(StandardScaler().fit_transform(flattenMatrixDataset(array(training_set_2[0]))))

y_1 = training_set_1[1]
y_2 = training_set_1[1]

X = X_1 + X_2
y = y_1 + y_2

latent_dims = 2
NUM_CHANNELS = 22
batch_size = 32

train_dataset = tf.data.Dataset.from_tensor_slices((X,y))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

auto_encoder = Sequential([
    Dense(NUM_CHANNELS*window_size*2,activation='tanh', bias_initializer='random_normal'),
    Dropout(rate=0.6),

    Dense(300, activation='tanh', bias_initializer='random_normal'),
    Dropout(rate=0.3),

    Dense(latent_dims, activation='sigmoid', bias_initializer='random_normal'),

    Dense(300, activation='tanh', bias_initializer='random_normal'),
    Dropout(rate=0.3),

    Dropout(rate=0.6),
    Dense(NUM_CHANNELS*window_size*2, activation='tanh', bias_initializer='random_normal')
])

auto_encoder.build(input_shape=[batch_size,NUM_CHANNELS*window_size*2])
auto_encoder.summary()

epochs = 1
optimizer = tf.keras.optimizers.Adadelta(learning_rate=0.003)
loss_fn = ContrastiveLoss()

training_losses = []

for epoch in range(epochs):

    for step,(x_batch_train,y_batch_train) in enumerate(train_dataset):

        with tf.GradientTape() as tape:
            x_batch_train = tf.cast(x_batch_train, tf.float32)
            embedding_matrices = auto_encoder(x_batch_train)
            y_pred = tf.linalg.norm(embedding_matrices - x_batch_train, axis=1)
            c_loss = loss_fn(y_batch_train, y_pred)

            grads = tape.gradient(c_loss, auto_encoder.trainable_weights)
            optimizer.apply_gradients(zip(grads, auto_encoder.trainable_weights))

        if step % 10 == 0:
            print(
                f'Step {step} - Training Loss: {c_loss}'
            )
        training_losses.append(c_loss)

        if step == 1000:
            break


encoder_idx = int(((len(auto_encoder.layers) + 1) / 2))

encoder = Sequential(
    auto_encoder.layers[0:encoder_idx]
)

encoder.build(input_shape=[1,NUM_CHANNELS*window_size*2])
encoder.save('Models/autoencoder_supervised.h5')

with open('supervised_losses.pkl','wb') as pickle_file:
    dump(training_losses,pickle_file)
    pickle_file.close()

