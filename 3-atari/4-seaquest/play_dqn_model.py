import gym
import random
import numpy as np
import tensorflow as tf
from skimage.color import rgb2gray
from skimage.transform import resize
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv2D
from keras import backend as K

EPISODES = 50000


class TestAgent:
    def __init__(self, action_size):
        self.state_size = (84, 84, 4)
        self.action_size = action_size

        self.model = self.build_model()

        self.sess = tf.InteractiveSession()
        K.set_session(self.sess)

        self.avg_q_max, self.avg_loss = 0, 0
        self.sess.run(tf.global_variables_initializer())

    def build_model(self):
        model = Sequential()
        model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu',
                         input_shape=self.state_size))
        model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
        model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.action_size))
        model.summary()

        return model

    def get_action(self, history):
        history = np.float32(history / 255.0)
        if np.random.rand() <= 0.1:
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(history)
            return np.argmax(q_value[0])

    def load_model(self, filename):
        self.model.load_weights(filename)


def pre_processing(observe):
    processed_observe = np.uint8(
        resize(rgb2gray(observe), (84, 84), mode='constant') * 255)
    return processed_observe


if __name__ == "__main__":
    env = gym.make('SeaquestDeterministic-v4')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = TestAgent(action_size=action_size)
    agent.load_model("./save_model/seaquest_ddqn_1500.h5")

    for e in range(EPISODES):
        done = False
        dead = False

        step, score, start_life = 0, 0, 3
        observe = env.reset()

        state = pre_processing(observe)
        history = np.stack((state, state, state, state), axis=2)
        history = np.reshape([history], (1, 84, 84, 4))

        while not done:
            env.render()
            step += 1

            action = agent.get_action(history)
            observe, reward, done, info = env.step(action)

            next_state = pre_processing(observe)
            next_state = np.reshape([next_state], (1, 84, 84, 1))
            next_history = np.append(next_state, history[:, :, :, :3], axis=3)

            if start_life > info['ale.lives']:
                dead = True
                start_life = info['ale.lives']

            score += reward

            history = next_history

            if dead:
                dead = False
            else:
                history = next_history

            if done:
                print("episode:", e, "  score:", score)

