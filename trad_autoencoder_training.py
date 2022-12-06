from tensorflow.keras.layers import Conv1D,Conv2D,Dense,Flatten,MaxPooling2D,BatchNormalization,Dropout
from tensorflow.keras import Sequential
from tensorflow.keras.losses import MeanSquaredError

import tensorflow as tf
from numpy import array

from helper_functions import parse_dataset,flattenMatrixDataset
from sklearn.preprocessing import StandardScaler

from pickle import dump

#Doing inter-subject classification with subject 'B'
training_data_path_trial_1 = 'Raw Mat Files (CLA)/CLASubjectB1510193StLRHand.mat'
training_data_path_trial_2 = 'Raw Mat Files (CLA)/CLASubjectB1510193StLRHand.mat'

window_size = 50
limit = 10**3

training_set_1 = parse_dataset(training_data_path_trial_1,window_size,limit)
training_set_2 = parse_dataset(training_data_path_trial_2,window_size,limit)

X_1 = list(StandardScaler().fit_transform(flattenMatrixDataset(array(training_set_1[0]))))
X_2 = list(StandardScaler().fit_transform(flattenMatrixDataset(array(training_set_2[0]))))

X = X_1 + X_2


latent_dims = 2
NUM_CHANNELS = 22
batch_size = 32

train_dataset = tf.data.Dataset.from_tensor_slices(X)
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
loss_fn = tf.losses.MeanSquaredError()

training_losses = []

for epoch in range(epochs):

    for step,x_batch_train in enumerate(train_dataset):

        with tf.GradientTape() as tape:
            reconstructed_x_batch = auto_encoder(x_batch_train)
            loss = loss_fn(reconstructed_x_batch,x_batch_train)

            grads = tape.gradient(loss, auto_encoder.trainable_weights)
            optimizer.apply_gradients(zip(grads, auto_encoder.trainable_weights))

        if step % 10 == 0:
            print(
                f'Step {step} - Training Loss: {loss}'
            )
        training_losses.append(loss)

        if step == 100:
            break


encoder_idx = int(((len(auto_encoder.layers) + 1) / 2))

encoder = Sequential(
    auto_encoder.layers[0:encoder_idx]
)

encoder.build(input_shape=[1,NUM_CHANNELS*window_size*2])
encoder.save('Models/autoencoder_trad.h5')

with open('losses.pkl','wb') as pickle_file:
    dump(training_losses,pickle_file)
    pickle_file.close()

