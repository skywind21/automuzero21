import gym
import numpy as np
import tensorflow as tf
import upbit

# 업비트 API 접근 정보
access_key = 'YOUR ACCESS KEY'
secret_key = 'YOUR SECRET KEY'
upbit_api = upbit.Upbit(access_key, secret_key)

# Lunar Lander 게임 환경
env = gym.make('LunarLander-v2')

# 강화 학습 모델
class MuZero:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
      # 네트워크 모델 정의
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(state_size,)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(action_size + 1)
        ])

        # 옵티마이저 설정
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    def predict(self, state):
        # 예측하기
        policy_logits, value = self.model(state)
        policy = tf.nn.softmax(policy_logits)
        return policy, value

    def train(self, states, actions, rewards, next_states):
        # 손실함수 정의
        def loss_fn(model, states, actions, rewards, next_states):
            policy_logits, value = model(states)
            next_policy, next_value = model(next_states)
            td_targets = rewards + next_value
            td_errors = td_targets - value

            # 정책 손실
            policy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=actions, logits=policy_logits)
            policy_loss = tf.reduce_mean(policy_loss)

            # 가치 손실
            value_loss = tf.reduce_mean(tf.square(td_errors))

            # 정책 엔트로피
            entropy_loss = tf.reduce_mean(
                tf.reduce_sum(next_policy * tf.math.log(next_policy), axis=1))

            # 총 손실
            total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_loss
            return total_loss

        # 손실 최소화
        with tf.GradientTape() as tape
         gradients = tape.gradient(loss_fn(self.model, states, actions, rewards, next_states),
                              self.model.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
     # MuZero 모델
    self.model = MuZero(state_size, action_size)

def act(self, state):
    # 정책과 가치 예측
    policy, value = self.model.predict(np.array(state).reshape(-1, self.state_size))

    # 확률적으로 행동 선택
    action = np.random.choice(self.action_size, p=np.squeeze(policy))
    return action, value

def train(self, states, actions, rewards, next_states):
    # MuZero 모델 학습
    self.model.train(states, actions, rewards, next_states)
    states = []
actions = []
rewards = []
next_states = []

# 게임 진행
while True:
    # 상태 저장
    states.append(state)

    # 행동 선택
    action, value = agent.act(state)
    actions.append(action)

    # 행동 실행
    next_state, reward, done, _ = env.step(action)
    rewards.append(reward)
       # 다음 상태 저장
    next_states.append(next_state)

    # 게임 종료 시
    if done:
        # 마지막 상태의 가치
        final_value = reward

        # 반환 계산
        returns = [final_value]
        for reward in reversed(rewards):
            returns.append(reward + 0.99 * returns[-1])
        returns.reverse()

        # 게임 데이터 반환
        return states, actions, returns[:-1], next_states

    # 상태 업데이트
    state = next_state
    # 게임 반복 횟수
num_episodes = 1000

# 각 게임의 최대 단계 수
max_steps_per_episode = 1000

# 게임 반복
for i in range(num_episodes):
    # 게임 실행
    states, actions, rewards, next_states = play_game(agent)

    # 에이전트 학습
    agent.train(states, actions, rewards, next_states)

    # 현재 잔액
    balance = upbit_api.get_balance('KRW')

    # 수익률
    returns = sum(rewards) / len(rewards)
        # 수익이 발생하면 매도
    if returns >= 0.005:
        coin_balance = upbit_api.get_balance('BTC')
        if coin_balance > 0:
            upbit_api.sell_market_order('BTC-KRW', coin_balance)
            print(f'{i+1}/{num_episodes}: Profit! Sold BTC {coin_balance} at {upbit_api.get_current_price("BTC-KRW")} KRW')
    # 손실이 발생하면 매도
    elif returns <= -0.005:
        coin_balance = upbit_api.get_balance('BTC')
        if coin_balance >
                  0:
            upbit_api.sell_market_order('BTC-KRW', coin_balance)
            print(f'{i+1}/{num_episodes}: Loss! Sold BTC {coin_balance} at {upbit_api.get_current_price("BTC-KRW")} KRW')

    # 잔액 업데이트
    balance = upbit_api.get_balance('KRW')

    # 현재 상태 출력
    print(f'{i+1}/{num_episodes}: Balance: {balance} KRW, Returns: {returns}')

    # 최대 단계 수 초과 시 종료
    if len(states) >= max_steps_per_episode:
        print(f'{i+1}/{num_episodes}: Max steps reached, terminating game')
        break

# 최종 잔액 출력
balance = upbit_api.get_balance('KRW')
print(f'Final balance: {balance} KRW')
# 게임 하이퍼파라미터 설정
num_episodes = 100  # 에피소드 수
max_steps_per_episode = 50  # 에피소드 당 최대 단계 수
budget = 10000  # 초기 예산
win_reward = 0.005  # 이기는 게임 보상 (0.5% 수익)
lose_penalty = 0.005  # 지는 게임 패널티 (0.5% 손실)

# 게임 실행
play_game(upbit_api, num_episodes, max_steps_per_episode, budget, win_reward, lose_penalty)

# 프로그램 종료
print('Program terminated')
import jwt
import uuid
import hashlib
import requests
import json
from urllib.parse import urlencode

class UpbitAPI:
    BASE_URL = 'https://api.upbit.com'

    def __init__(self, access_key, secret_key):
        self.access_key = access_key
        self.secret_key = secret_key

    def get_balance(self, ticker):
        query = {
            'currency': ticker
        }
        return self._request_get('/v1/accounts', query)[0]['balance']

    def get_current_price(self, ticker):
        query = {
            'markets': ticker
        }
        return self._request_get('/v1/ticker', query)[0]['trade_price']
       def buy_market_order(self, ticker, budget):
        query = {
            'market': ticker,
            'side': 'bid',
            'price': None,
            'ord_type': 'price',
            'volume': None,
            'funds': budget
        }
        return self._request_post('/v1/orders', query)['uuid']

    def sell_market_order(self, ticker, volume):
        query = {
            'market': ticker,
            'side': 'ask',
            'price': None,
            'ord_type': 'market',
            'volume': volume,
            'funds': None
        }
        return self._request_post('/v1/orders', query)['uuid']

    def _request_get(self, path, query=None):
        url = self.BASE_URL + path
        if query is not None:
            url += '?' + urlencode(query)
               payload = {
            'access_key': self.access_key,
            'nonce': str(uuid.uuid4())
        }

        jwt_token = jwt.encode(payload, self.secret_key, algorithm='HS256').decode('utf-8')
        authorization_token = f'Bearer {jwt_token}'

        headers = {'Authorization': authorization_token}

        response = requests.get(url, headers=headers)

        return json.loads(response.content)

    def _request_post(self, path, query=None):
        url = self.BASE_URL + path
        if query is not None:
            query_string = urlencode(query)
        else:
            query_string = ''

        payload = {
            'access_key': self.access_key,
            'nonce': str(uuid.uuid4()),
            'query': query_string
        }
              jwt_token = jwt.encode(payload, self.secret_key, algorithm='HS256').decode('utf-8')
        authorization_token = f'Bearer {jwt_token}'

        headers = {
            'Authorization': authorization_token,
            'Content-Type': 'application/json'
        }

        response = requests.post(url, headers=headers, data=json.dumps(query))

        return json.loads(response.content)
    from typing import Tuple
import math
import numpy as np
from muzero_core.mcts import MCTS, Node
from muzero_core.replay_buffer import ReplayBuffer
from upbit_api import UpbitAPI

class LunarLanderEnv:
    def __init__(self):
        self.observation_space = [0.0, 10000.0]  # [balance, price]
        self.action_space = [0, 1]  # 0: hold, 1: trade
        self.current_step = 0
        self.max_steps = 100

    def reset(self):
        self.current_step = 0
        self.balance = 10000.0
        self.price = 0.0
        return [self.balance, self.price]
     def step(self, action):
        self.current_step += 1
        reward = 0.0
        done = False
        if action == 1:
            # Trade
            price = api.get_current_price('KRW-BTC')
            if price is None:
                price = self.price
            else:
                self.price = price
            trade_budget = self.balance * 0.99
            volume = trade_budget / price
            order_id = api.buy_market_order('KRW-BTC', trade_budget)
            if order_id is None:
                # Failed to place an order
                reward = -10.0
                          else:
                # Wait for the order to be filled
                filled = False
                while not filled:
                    status = api.get_order_status(order_id)
                    if status == 'filled':
                        filled = True
                        break
                    elif status == 'canceled' or status == 'expired':
                        # The order is canceled or expired
                        reward = -10.0
                        break
                if filled:
                    self.balance -= trade_budget
                    self.balance += volume * price
                    if price >= self.observation_space[1] * 1.005:
                        reward = 1.0
                    elif price <= self.observation_space[1] * 0.995:
                        reward = -1.0
                    else:
                        reward = 0.1
                               else:
            # Hold
            reward = 0.0

        self.observation_space = [self.balance, self.price]
        if self.current_step >= self.max_steps:
            done = True
        return self.observation_space, reward, done

class MuZeroConfig:
    def __init__(self):
        self.seed = 0
        self.max_num_episodes = 1000
        self.max_num_moves = 100
        self.discount_rate = 0.99
        self.num_simulations = 50
        self.temperature = 1.0
        self.dirichlet_alpha = 0.25
        self.exploration_fraction = 0.25
        self.training_steps_per_iteration = 100
        self.checkpoint_interval = 10
        self.num_actors = 1
        self.training_batch_size = 128
        self.num_training_epochs
        class MuZeroConfig:
    def __init__(self):
        self.seed = 0
        self.max_num_episodes = 1000
        self.max_num_moves = 100
        self.discount_rate = 0.99
        self.num_simulations = 50
        self.temperature = 1.0
        self.dirichlet_alpha = 0.25
        self.exploration_fraction = 0.25
        self.training_steps_per_iteration = 100
        self.checkpoint_interval = 10
        self.num_actors = 1
        self.training_batch_size = 128
        self.num_training_epochs = 5

        # Define the dynamic range of the reward
        self.min_reward = -10.0
        self.max_reward = 1.0
            # Define the input/output sizes of the neural network
        self.observation_shape = (2,)
        self.action_space_size = len(LunarLanderEnv().action_space)
        self.encoding_size = 32
        self.hidden_size = 32
        self.value_support_size = 10
        self.reward_support_size = 10
        self.train_replay_buffer_size = 10000

        # Define the network architecture
        self.network_args = {
            'encoding_size': self.encoding_size,
            'hidden_size': self.hidden_size,
            'value_support_size': self.value_support_size,
            'reward_support_size': self.reward_support_size,
            'observation_shape': self.observation_shape,
            'action_space_size': self.action_space_size,
        }
            # Define the optimizer parameters
        self.weight_decay = 1e-4
        self.momentum = 0.9
        self.lr_init = 0.05
        self.lr_decay_rate = 0.1
        self.lr_decay_steps = 10000

        # Define the initial and final exploration rates
        self.init_exploration_rate = 0.9
        self.final_exploration_rate = 0.0
        self.exploration_decay_rate = 0.9999
        def train(api: UpbitAPI):
    env = LunarLanderEnv()
    config = MuZeroConfig()
    replay_buffer = ReplayBuffer(config.train_replay_buffer_size, config.seed)

    for episode in range(config.max_num_episodes):
        # Reset the environment
        observation = env.reset()
        done = False
        total_reward = 0.0

        # Initialize the root node of the search tree
        root = Node(0)
        root.expand(env, config.network_args)

        for step in range(config.max_num_moves):
            # Run Monte Carlo Tree Search to select an action
            mcts = MCTS(config)
            action = mcts.search(root, env, observation)

            # Execute the action
            next_observation, reward, done = env.step(action)
            total_reward += reward
            replay_buffer.add(observation, action, reward, next_observation, done)
               # Update the root node
            if done:
                root = Node(0)
                root.expand(env, config.network_args)
            else:
                root = root.children[action]

            # Train the neural network
            if len(replay_buffer)