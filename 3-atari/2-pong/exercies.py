import gym
import random

if __name__ == "__main__":
    # 환경과 에이전트 생성
    env = gym.make('Pong-v0')
    # env.action_space = 6

    for episode in range(10):
        print('episode ' + str(episode) + ' ----------------')
        done = False
        state = env.reset()

        while not done:
            env.render()
            # 랜덤하게 행동 선택
            # 0, 1 = 가만히, 2, 4 = 위로, 3, 5 = 아래로
            action = random.choice([1, 2, 3])
            # 선택한 행동으로 환경에서 한 타임스텝 진행
            state, reward, done, info = env.step(action)

            print('action: ', action, ' reward: ', reward, ' life: ', done)

            if done:
                break