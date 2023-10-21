import keras
# import tensorflow as tf
import numpy as np
from keras.layers import Input, Conv2D, Dense, Flatten, Multiply
from keras.optimizers import Adam
from keras.losses import MeanSquaredError
from ER import ER

# tf.config.experimental.enable_tensor_float_32_execution(
#     False
# )
def create( action_space_size, input_shape ):
    layers = {
        "input": Input( shape=input_shape ),
        "flatten": Flatten(),
        "dense1": Dense(512, activation="relu", kernel_initializer="he_uniform"),
        "dense2": Dense(256, activation="relu", kernel_initializer="he_uniform"),
        "dense3": Dense(64, activation="relu", kernel_initializer="he_uniform"),
        "dense4": Dense( action_space_size,
                    activation="linear",
                    kernel_initializer="he_uniform")
    }

    dqn = keras.Sequential()
    dqn.add( layers["input"] )
    dqn.add( layers["flatten"] )
    dqn.add( layers["dense1"] )
    dqn.add( layers["dense2"] )
    dqn.add( layers["dense3"] )
    dqn.add( layers["dense4"] )
    dqn.compile(loss=MeanSquaredError(), optimizer=Adam(lr=0.00005))
    dqn.summary()
    return {
        "model": dqn,
        "layers": layers
    }
def train_single( model, target_model, state, next_state, action, reward, done, gamma ):
    target = model.predict( state, verbose=0 )
    target_next = model.predict( next_state, verbose=0 )
    target_val = target_model.predict( next_state, verbose=0 )

    if done:
        target[0][action] = reward
    else:
        a = np.argmax(target_next[0])
        target[0][action] = reward + gamma * target_val[0][a]

    model.fit( state, target, batch_size=1, verbose=0 )

def train( model, target_model, er, batch_size, gamma ):
    states, actions, rewards, next_states, dones = ER.sample(
        er, batch_size
    )
    if len( states ) == 0:
        return

    # do batch prediction to save speed
    # predict Q-values for starting state using the main network
    target = model.predict( states, verbose=0 )
    # predict best action in ending state using the main network
    target_next = model.predict( next_states, verbose=0 )
    # predict Q-values for ending state using the target network
    target_val = target_model.predict( next_states, verbose=0 )


    for i in range(len(states)):
        # correction on the Q value for the action used
        if dones[i]:
            target[i][actions[i]] = rewards[i]
        else:
            # the key point of Double DQN
            # selection of action is from model
            # update is from target model
            # current Q Network selects the action
            # a'_max = argmax_a' Q(s', a')
            a = np.argmax( target_next[i] )
            # target Q Network evaluates the action
            # Q_max = Q_target(s', a'_max)
            target[i][actions[i]] = rewards[i] + gamma * target_val[i][a]

    # if self.USE_PER:
    #     indices = np.arange(self.batch_size, dtype=np.int32)
    #     absolute_errors = np.abs(target_old[indices, action]-target[indices, action])

    #     # Update priority
    #     self.MEMORY.batch_update(tree_idx, absolute_errors)

    # Train the Neural Network with batches
    model.fit( states, target, batch_size=batch_size, verbose=0 )

def copy_weights( from_model, target_model ):
    target_model.set_weights( from_model.get_weights() )

def save( model, save_path ):
    model.save( save_path )

def pongFrame2input( frame ):
    # croping frame to 80x80 size
    frame_cropped = frame[35:195:2, ::2,:]
    if frame_cropped.shape[0] != 80 or frame_cropped.shape[1] != 80:
        # OpenCV resize function
        frame_cropped = cv2.resize(frame, (80, 80), interpolation=cv2.INTER_CUBIC)

    # converting to RGB (numpy way)
   #  frame_rgb = 0.299*frame_cropped[:,:,0] + 0.587*frame_cropped[:,:,1] + 0.114*frame_cropped[:,:,2]
    # converting to RGB (OpenCV way)
    # frame_rgb = cv2.cvtColor(frame_cropped, cv2.COLOR_RGB2GRAY)

    # dividing by 255 we expresses value to 0-1 representation
    return np.array(frame_cropped).astype(np.float32) / 255.0
