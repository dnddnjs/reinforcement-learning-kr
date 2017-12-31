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
from keras.layers.convolutional import Conv2D
from keras.layers import Dense, Flatten, Input, Lambda, merge
from keras.optimizers import RMSprop
from keras.models import Model


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

    # 상태가 입력, 큐함수가 출력인 인공신경망 생성
    def build_model(self):
        input = Input(shape=self.state_size)
        shared = Conv2D(32, (8, 8), strides=(4, 4), activation='relu')(input)
        shared = Conv2D(64, (4, 4), strides=(2, 2), activation='relu')(shared)
        shared = Conv2D(64, (3, 3), strides=(1, 1), activation='relu')(shared)
        flatten = Flatten()(shared)

        # network separate state value and advantages
        advantage_fc = Dense(512, activation='relu')(flatten)
        advantage = Dense(self.action_size)(advantage_fc)
        advantage = Lambda(lambda a: a[:, :] - K.mean(a[:, :], keepdims=True),
                           output_shape=(self.action_size,))(advantage)

        value_fc = Dense(512, activation='relu')(flatten)
        value = Dense(1)(value_fc)
        value = Lambda(lambda s: K.expand_dims(s[:, 0], -1),
                       output_shape=(self.action_size,))(value)

        # network merged and make Q Value
        q_value = merge([value, advantage], mode='sum')
        model = Model(inputs=input, outputs=q_value)
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

