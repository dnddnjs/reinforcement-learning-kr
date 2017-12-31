import os
import sys
import gym
import pylab
import random
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from collections import deque

EPISODES = 10000000

model_path = os.path.join(os.getcwd(), 'save_model')
graph_path = os.path.join(os.getcwd(), 'save_graph')

if not os.path.isdir(model_path):
    os.mkdir(model_path)
if not os.path.isdir(graph_path):
    os.mkdir(graph_path)


# 카트폴 예제에서의 액터-크리틱(A2C) 에이전트
class ACERAgent:
    def __init__(self, state_size, action_size):
        self.render = False
        self.load_model = False
        # 상태와 행동의 크기 정의
        self.state_size = state_size
        self.action_size = action_size
        self.value_size = 1
        self.train_start = 1000
        self.batch_size = 64

        # 액터-크리틱 하이퍼파라미터
        self.discount_factor = 0.99
        self.actor_lr = 0.001
        self.critic_lr = 0.005

        self.memory = deque(maxlen=2000)

        # 정책신경망과 가치신경망 생성
        self.actor = self.build_actor()
        self.critic = self.build_critic()

        if self.load_model:
            self.actor.load_weights("./save_model/driving_actor.h5")
            self.critic.load_weights("./save_model/driving_critic.h5")

    # actor 와 critic의 네트워크 앞의 레이어를 공유하면 조금 더 안정적으로 학습함
    def build_actor(self):
        actor = Sequential()
        actor.add(Dense(30, input_dim=self.state_size, activation='relu',
                        kernel_initializer='he_uniform'))
        actor.add(Dense(self.action_size, activation='softmax',
                        kernel_initializer='he_uniform'))
        actor.summary()
        # See note regarding crossentropy in cartpole_reinforce.py
        actor.compile(loss='categorical_crossentropy',
                      optimizer=Adam(lr=self.actor_lr))
        return actor

    # critic은 상태를 input으로 받아 상태에 대한 value 를 출력으로 하는 모델입니다.
    def build_critic(self):
        critic = Sequential()
        critic.add(Dense(30, input_dim=self.state_size, activation='relu',
                         kernel_initializer='he_uniform'))
        critic.add(Dense(30, activation='relu',
                         kernel_initializer='he_uniform'))
        critic.add(Dense(self.value_size, activation='linear',
                         kernel_initializer='he_uniform'))
        critic.summary()
        critic.compile(loss="mse", optimizer=Adam(lr=self.critic_lr))
        return critic

    # 정책신경망의 출력을 받아 확률적으로 행동을 선택
    def get_action(self, state):
        policy = self.actor.predict(state, batch_size=1).flatten()
        return np.random.choice(self.action_size, 1, p=policy)[0], policy

    # 샘플 <s, a, r, s'>을 리플레이 메모리에 저장
    def append_sample(self, state, action, reward, next_state, done, policy):
        self.memory.append((state, action, reward, next_state, done, policy))

    # 매 에피소드마다 모델을 학습시킵니다.
    def train_model(self):
        mini_batch = random.sample(self.memory, self.batch_size)

        # target과 advantange를 numpy zero로 선언해줍니다.
        targets = np.zeros((self.batch_size, self.value_size))
        advantages = np.zeros((self.batch_size, self.action_size))

        states = np.zeros((self.batch_size, self.state_size))
        next_states = np.zeros((self.batch_size, self.state_size))
        actions, rewards, dones, act_policies = [], [], [], []

        for i in range(self.batch_size):
            states[i] = mini_batch[i][0]
            actions.append(mini_batch[i][1])
            rewards.append(mini_batch[i][2])
            next_states[i] = mini_batch[i][3]
            dones.append(mini_batch[i][4])
            act_policies.append(mini_batch[i][5])

        # critic 모델로 부터 현재 state에 대한 value와 next_state에 대한 value를 산출합니다.
        values = self.critic.predict(states)
        next_values = self.critic.predict(next_states)
        target_policies = self.actor.predict(states)

        for i in range(self.batch_size):
            importance = target_policies[i][actions[i]] / act_policies[i][actions[i]]
            importance_1 = np.minimum(importance, 1)
            importance_2 = np.maximum((1 - 1/importance), 0)

            if dones[i]:
                # [[a0, a1, a2]]
                advantages[i][actions[i]] = (importance_1 + importance_2) * (rewards[i] - values[i])
                # [[value]]
                targets[i][0] = rewards[i]
            else:
                advantages[i][actions[i]] = (importance_1 + importance_2) * (rewards[i] +
                                     self.discount_factor * (next_values[i]) - values[i])
                targets[i][0] = rewards[i] + self.discount_factor * next_values[i]

        self.actor.fit(states, advantages, epochs=1, verbose=0)
        self.critic.fit(states, targets, epochs=1, verbose=0)

if __name__ == "__main__":
    # CartPole-v1 환경, 최대 타임스텝 수가 500
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # DQN 에이전트 생성
    agent = ACERAgent(state_size, action_size)

    # 액터-크리틱(A2C) 에이전트 생성
    scores, episodes = [], []

    for e in range(EPISODES):
        done = False
        score = 0
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        while not done:
            if agent.render:
                env.render()

            action, policy = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            reward = reward if not done or score == 499 else -1

            agent.append_sample(state, action, reward, next_state, done, policy)

            if len(agent.memory) >= agent.train_start:
                agent.train_model()

            score += reward
            state = next_state

            if done:
                score = score if score == 500 else score + 1
                # 에피소드마다 학습 결과 출력
                scores.append(score)
                episodes.append(e)
                pylab.plot(episodes, scores, 'b')
                pylab.savefig("./save_graph/cartpole_acer.png")
                print("episode:", e, "  score:", score, "  memory length:",
                      len(agent.memory))

                # 이전 10개 에피소드의 점수 평균이 490보다 크면 학습 중단
                if np.mean(scores[-min(10, len(scores)):]) > 490:
                    agent.actor.save_weights("./save_model/acer_actor.h5")
                    agent.critic.save_weights("./save_model/acer_critic.h5")
                    sys.exit()
