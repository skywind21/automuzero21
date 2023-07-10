import time
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import requests

class UpbitAPI:
    def __init__(self, access_key, secret_key):
        self.access_key = access_key
        self.secret_key = secret_key

    def get_current_price(self, market):
        url = f"https://api.upbit.com/v1/ticker?markets={market}"
        headers = {"Authorization": f"Bearer {self.access_key}"}
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            if len(data) > 0:
                return float(data[0]['trade_price'])
        return None

# 업비트 API 접근 정보
access_key = 'YOUR ACCESS KEY'
secret_key = 'YOUR SECRET KEY'
upbit_api = UpbitAPI(access_key, secret_key)

# 비트코인 자동매매 환경 설정
initial_balance = 10000.0  # 초기 자산
max_steps_per_episode = 1000  # 에피소드 당 최대 단계 수

# MuZero 신경망 구조
class MuZeroNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(MuZeroNetwork, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = 64

        # 상태 인코더
        self.state_encoder = nn.Sequential(
            nn.Linear(state_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU()
        )

        # 동적 모델
        self.dynamic_model = nn.GRUCell(self.hidden_size, self.hidden_size)

        # 상태 모델
        self.state_model = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, state_size)
        )

        # 정책 모델
        self.policy_model = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, action_size),
            nn.Softmax(dim=1)
        )

        # 가치 모델
        self.value_model = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1)
        )

    def forward(self, state):
        encoded_state = self.state_encoder(state)
        hidden_state = torch.zeros(1, self.hidden_size)
        predicted_state = None
        predicted_policy = None
        predicted_value = None

        for _ in range(self.action_size):
            hidden_state = self.dynamic_model(encoded_state, hidden_state)
            predicted_state = self.state_model(hidden_state)
            predicted_policy = self.policy_model(hidden_state)
            predicted_value = self.value_model(hidden_state)

        return predicted_state, predicted_policy, predicted_value

# MuZero 알고리즘
class MuZero:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.model = MuZeroNetwork(state_size, action_size)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def predict(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        predicted_state, predicted_policy, predicted_value = self.model(state_tensor)
        return predicted_policy, predicted_value

    def train(self, states, actions, rewards, next_states):
        self.optimizer.zero_grad()

        loss = self.calculate_loss(states, actions, rewards, next_states)

        loss.backward()
        self.optimizer.step()

    def calculate_loss(self, states, actions, rewards, next_states):
        loss = 0
        discounted_reward = 0
        hidden_state = None

        for t in reversed(range(len(states))):
            state = states[t]
            action = actions[t]
            reward = rewards[t]
            next_state = next_states[t]

            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)

            predicted_state, predicted_policy, predicted_value = self.model(state_tensor)
            predicted_next_state, _, _, _ = self.model(next_state_tensor)

            loss += (predicted_value - reward) ** 2

            # 상태 모델 손실
            loss += nn.MSELoss()(predicted_next_state, Variable(torch.FloatTensor(next_state)))

            # 정책 모델 손실
            target_policy = self.get_target_policy(predicted_policy, action)
            loss += nn.MSELoss()(predicted_policy, Variable(torch.FloatTensor(target_policy)))

            # 가치 모델 손실
            target_value = self.get_target_value(rewards[t:])
            loss += nn.MSELoss()(predicted_value, Variable(torch.FloatTensor(target_value)))

            discounted_reward = reward + discounted_reward * 0.99

        return loss

    def get_target_policy(self, predicted_policy, action):
        target_policy = np.zeros(self.action_size)
        target_policy[action] = 1.0
        return target_policy

    def get_target_value(self, rewards):
        target_value = np.zeros(len(rewards) + 1)
        cumulative_reward = 0
        for i in reversed(range(len(rewards))):
            cumulative_reward = rewards[i] + cumulative_reward * 0.99
            target_value[i] = cumulative_reward
        return target_value[:-1]

    def act(self, state):
        policy, value = self.predict(state)

        action = np.random.choice(self.action_size, p=np.squeeze(policy.detach().numpy()))
        return action, value.item()

# MuZero 모델 초기화
state_size = 2  # 상태 공간 크기: [balance, price]
action_size = 2  # 행동 공간 크기: [hold, trade]
agent = MuZero(state_size, action_size)

# 게임 데이터 생성
def play_game(agent):
    states = []
    actions = []
    rewards = []
    next_states = []

    # 게임 진행
    state = [initial_balance, upbit_api.get_current_price('BTC-KRW')]
    done = False
    while not done:
        # 상태 저장
        states.append(state)

        # 행동 선택
        action, value = agent.act(state)
        actions.append(action)

        # 행동 실행
        if action == 1:
            # Trade
            price = upbit_api.get_current_price('BTC-KRW')
            trade_budget = state[0] * 0.99
            volume = trade_budget / price
            order_id = upbit_api.buy_market_order('BTC-KRW', trade_budget)
            if order_id is None:
                # Failed to place an order
                reward = -10.0
            else:
                # Wait forthe order to be filled
                filled = False
                while not filled:
                    status = upbit_api.get_order_status(order_id)
                    if status == 'filled':
                        filled = True
                        break
                    elif status == 'canceled' or status == 'expired':
                        # The order is canceled or expired
                        reward = -10.0
                        break
                if filled:
                    state[0] -= trade_budget
                    state[0] += volume * price
                    if price >= states[-1][1] * 1.005:
                        reward = 1.0
                    elif price <= states[-1][1] * 0.995:
                        reward = -1.0
                    else:
                        reward = 0.1
        else:
            # Hold
            reward = 0.0

        # 보상 저장
        rewards.append(reward)

        # 다음 상태 저장
        next_state = [state[0], upbit_api.get_current_price('BTC-KRW')]
        next_states.append(next_state)

        # 상태 업데이트
        state = next_state

        # 최대 단계 수 초과 시 종료
        if len(states) >= max_steps_per_episode:
            done = True

        # 일정 시간 간격으로 반복
        time.sleep(1)

    # 게임 데이터 반환
    return states, actions, rewards, next_states

# 게임 반복 횟수
num_episodes = 1000

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
        if coin_balance > 0:
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

    # 일정 시간 간격으로 반복
    time.sleep(1)

# 최종 잔액 출력
balance = upbit_api.get_balance('KRW')
print(f'Final balance: {balance} KRW')
