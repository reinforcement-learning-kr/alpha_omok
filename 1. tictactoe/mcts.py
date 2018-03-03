# -*- coding: utf-8 -*-
import time
import xxhash
import numpy as np
import dill as pickle
import tictactoe_env
from collections import deque, defaultdict

np.set_printoptions(suppress=True)

PLAYER = 0
OPPONENT = 1

N, W, Q, P = 0, 1, 2, 3
PLANE = np.zeros((3, 3), 'int').flatten()
EPISODE = 1600
SAVE_CYCLE = 1600


class MCTS(object):
    """몬테카를로 트리 탐색 클래스. (PUCT-MCTS)

    시뮬레이션을 통해 train 데이터 생성. (state, pi, z)

    state_new
    ----------
    환경이 주는 state를 각 주체당 4수까지 저장해서 state_new 로 만듦.

    type: 9x3x3 numpy array -> 1x81 numpy array (flatten)
        0번 평면: 플레이어의 현재 착수 상황.
        1 ~ 3번: 플레이어의 3수 전까지의 착수 정보.
        4번 평면: 상대의 현재 착수 상황.
        5 ~ 7번: 상대의 3수 전까지의 착수 정보.
        8번 평면: 첫 턴(O표시)인 주체와 동기화.

    edge
    -----
    현재 state에서 착수 가능한 모든 action자리에 4개의 정보(N, W, Q, P) 저장.

    type: 3x3x4 numpy array.

        9개 좌표에 4개의 정보 N, W, Q, P 매칭.
        N: edge 방문횟수, W: 보상누적값, Q: 보상평균값(W/N), P: 선택 확률 추정 백터.
        edge[보드의 x좌표][보드의 y좌표][알파벳] 으로 접근.

    """

    def __init__(self):
        # memories
        # self.state_memory = deque(maxlen=9 * EPISODE)
        self.tree_memory = defaultdict(lambda: np.zeros((3, 3, 4), 'float'))

        # hyperparameter
        self.c_puct = 5
        self.epsilon = 0.25
        self.alpha = 0.7

        # reset_step member
        self.node = None
        self.edge = None
        self.puct = None
        self.total_visit = None
        self.empty_loc = None
        self.legal_move_n = None
        self.pr = None
        self.state = None
        self.state_new = None

        # reset_episode member
        self.player_history = None
        self.opponent_history = None
        self.node_memory = None
        self.edge_memory = None
        self.action_memory = None
        self.action_count = None
        self.board = None
        self.first_turn = None
        self.user_type = None

        # member 초기화
        self._reset_step()
        self._reset_episode()

    def _reset_step(self):
        self.node = None
        self.edge = np.zeros((3, 3, 4), 'float')
        self.puct = np.zeros((3, 3), 'float')
        self.total_visit = 0
        self.empty_loc = None
        self.legal_move_n = 0
        self.pr = 0.
        self.state = None
        self.state_new = None

    def _reset_episode(self):
        self.player_history = deque([PLANE] * 4, maxlen=4)
        self.opponent_history = deque([PLANE] * 4, maxlen=4)
        self.node_memory = deque(maxlen=9)
        self.edge_memory = deque(maxlen=9)
        self.action_memory = deque(maxlen=9)
        self.action_count = 0
        self.board = None
        self.first_turn = None
        self.user_type = None

    def select_action(self, state):
        """raw state를 받아 변환 및 저장 후 action을 리턴하는 외부 메소드.

        node
        ------
        state_new -> node

            type: string. (hash)
                state_new를 string으로 바꾼 후 hash 생성.
                tree dict의 key로 사용.

        action
        -------
        puct 값이 가장 높은 곳을 선택함. 동점인 곳은 랜덤 선택.

            action: 1x3 tuple.
                action = (주체 인덱스, 보드의 x좌표, 보드의 y좌표)
                주체 인덱스: 플레이어는 0, 상대는 1.
                좌표:
                        [0,0][0,1][0,2]
                        [1,0][1,1][1,2]
                        [2,0][2,1][2,2]

        """
        #  action 수 세기
        self.action_count += 1

        # 호출될 때마다 첫턴 기준 교대로 행동주체 바꿈, 최종 action에 붙여줌
        self.user_type = (self.first_turn + self.action_count - 1) % 2

        # state 변환
        self.state = state
        self.state_new = self._convert_state(state)
        # 새로운 state 저장
        # self.state_memory.appendleft(self.state_new)

        # state를 문자열 -> hash로 변환 (dict의 key로 사용)
        self.node = xxhash.xxh64(self.state_new.tostring()).hexdigest()
        self.node_memory.appendleft(self.node)

        # edge 세팅: tree 탐색 -> edge 생성 or 세팅 -> PUCT 점수 계산
        self._set_edge()

        # PUCT 점수 출력
        # print('***  PUCT Score  ***')
        # print(self.puct.round(decimals=2))
        # print('')

        # 빈자리가 아닌 곳은 PUCT값으로 -9999를 넣어 빈자리가 최댓값 되는 것 방지
        puct = self.puct.tolist()
        for i, v in enumerate(puct):
            for j, _ in enumerate(v):
                if [i, j] not in self.empty_loc.tolist():
                    puct[i][j] = -9999

        # PUCT가 최댓값인 곳 찾기
        self.puct = np.asarray(self.puct)
        puct_max = np.argwhere(self.puct == self.puct.max()).tolist()

        # 동점 처리
        move_target = puct_max[np.random.choice(len(puct_max))]

        # 최종 action 구성 (배열 접붙히기)
        action = np.r_[self.user_type, move_target]

        # action 저장 및 step 초기화
        self.action_memory.appendleft(action)
        self._reset_step()

        # action 튜플 리턴
        return tuple(action)

    def _convert_state(self, state):
        """state변환 메소드: action 주체별 최대 4수까지 history를 저장하여 새로운 state로 구성.

        최대 길이 4의 deque 이용함. 빈평면 4개를 채워놓고 밀어내기.

        """
        if abs(self.user_type) == PLAYER:
            self.opponent_history.appendleft(state[OPPONENT].flatten())
        else:
            self.player_history.appendleft(state[PLAYER].flatten())
        state_new = np.r_[np.array(self.player_history).flatten(),
                          np.array(self.opponent_history).flatten(),
                          self.state[2].flatten()]
        return state_new

    def _set_edge(self):
        """확장할 edge를 초기화 하는 메소드.

        dict{node: edge}인 MCTS Tree 구성
        edge의 Q, P를 계산하여 9개의 좌표에 PUCT값을 계산하여 매칭.

        """
        # tree에서 현재 node를 검색하여 해당 edge의 누적정보 가져오기
        self.edge = self.tree_memory[self.node]

        # edge의 총 방문횟수 계산
        for i in range(3):
            for j in range(3):
                self.total_visit += self.edge[i][j][N]

        # 방문횟수 출력
        # print('(visit count: {:0.0f})\n'.format(self.total_visit + 1))

        # 현재 보드에서 착수가능한 수를 알아내어 랜덤 확률 P 계산
        self.board = self.state[PLAYER] + self.state[OPPONENT]
        self.empty_loc = np.argwhere(self.board == 0)
        self.legal_move_n = self.empty_loc.shape[0]
        prob = 1 / self.legal_move_n

        # root node면 P에 노이즈 (탐험)
        if self.action_count == 1:
            self.pr = (1 - self.epsilon) * prob + self.epsilon * \
                                                  np.random.dirichlet(
                                                      self.alpha * np.ones(
                                                          self.legal_move_n))
        else:  # 아니면 n분의 1
            self.pr = prob * np.ones(self.legal_move_n)

        # P값 배치
        for i in range(self.legal_move_n):
            self.edge[tuple(self.empty_loc[i])][P] = self.pr[i]

        # Q값 계산 후 배치
        for i in range(3):
            for j in range(3):
                if self.edge[i][j][N] != 0:
                    self.edge[i][j][Q] = self.edge[i][j][W] / \
                                         self.edge[i][j][N]

                # 각자리의 PUCT 계산
                self.puct[i][j] = self.edge[i][j][Q] + \
                                  self.c_puct * \
                                  self.edge[i][j][P] * \
                                  np.sqrt(self.total_visit) / (
                                  1 + self.edge[i][j][N])

        # Q, P값을 배치한 edge 저장
        self.edge_memory.appendleft(self.edge)

    def backup(self, reward):
        steps = self.action_count
        for i in range(steps):
            # W 배치
            # 내가 지나온 edge는 +reward 로
            if self.action_memory[i][0] == PLAYER:
                self.edge_memory[i][tuple(
                    self.action_memory[i][1:])][W] += reward
            # 상대가 지나온 edge는 -reward 로
            else:
                self.edge_memory[i][tuple(
                    self.action_memory[i][1:])][W] -= reward

            # N 배치
            self.edge_memory[i][tuple(self.action_memory[i][1:])][N] += 1

            # N, W, Q, P 가 계산된 edge들을 tree에 최종 업데이트
            self.tree_memory[self.node_memory[i]] = self.edge_memory[i]

        self._reset_episode()


if __name__ == '__main__':
    start = time.time()
    env = tictactoe_env.TicTacToeEnv()

    agent = MCTS()

    # 통계용
    result = {1: 0, 0: 0, -1: 0}
    win_mark_O = 0
    step = 0

    # 에피소드 시작
    for e in range(EPISODE):
        state = env.reset()
        agent.first_turn = (PLAYER + e) % 2
        print(agent.first_turn)
        env.player_color = agent.first_turn

        done = False

        while not done:
            print('Episode: {}'.format(e + 1))
            step += 1
            print(state)
            # action = agent.select_action(state)
            action = input()
            # step 진행
            state, reward, done, info = env.step(action)

        # 에피소드가 끝나면
        if done:
            # 승부난 보드 보기
            # print('- FINAL BOARD -')
            # print(state[PLAYER] + state[OPPONENT] * 2.0)
            # print('')
            print(reward)
            # 보상과 방문카운트를 edge에 백업
            agent.backup(reward)

            # 결과 체크 (통계용)
            result[reward] += 1

            # 선공으로 이긴 경우 체크 (밸런스 판단용)
            if reward == 1:
                if env.player_color == 0:
                    win_mark_O += 1

        # 데이터 저장
        if (e + 1) % SAVE_CYCLE == 0:
            # with open('data/state_memory_e{}.pkl'.format(e + 1), 'wb') as f:
            #   pickle.dump(zero_play.state_memory, f)
            with open('data/tree_memory_e{}.pkl'.format(e + 1), 'wb') as f:
                pickle.dump(agent.tree_memory, f, pickle.HIGHEST_PROTOCOL)

            # 시간 측정
            finish = round(float(time.time() - start))

            # 저장 완료 메시지 출력
            print('[{} Episode Data Saved]'.format(e + 1))

            # 에피소드 통계 문자열 생성
            statics = ('\nWin: {}  Lose: {}  Draw: {}  Winrate: {:0.1f}% '
                       'WinMarkO: {}'.format(result[1], result[-1], result[0],
                                             1 / (1 + np.exp(result[-1] / EPISODE) /
                          np.exp(result[1] / EPISODE)) * 100,
                     win_mark_O))

            # 통계 화면 출력
            print('=' * 65, statics)
