# -*- coding: utf-8 -*-
import tictactoe_env
import neural_network

import time
from collections import deque, defaultdict

import torch
from torch.autograd import Variable
# from torch.optim import lr_scheduler

# import slackweb
import xxhash
import dill as pickle
import numpy as np
np.set_printoptions(suppress=True)

PLAYER, OPPONENT = 0, 1
MARK_O, MARK_X = 0, 1
N, W, Q, P = 0, 1, 2, 3
PLANE = np.zeros((3, 3), 'int').flatten()

NUM_CHANNEL = 128
GAME = 1
SIMULATION = 800


class MCTS(object):
    """몬테카를로 트리 탐색 클래스.

    셀프플레이를 통해 train 데이터 생성

    state
    ------
    각 주체당 4수까지 저장해서 state_new 로 만듦

        9x3x3 numpy array -> 1x81 tuple

    edge
    -----
    현재 state에서 착수 가능한 모든 action자리에 4개의 정보 저장

    type: 3x3x4 numpy array

        9개 좌표에 4개의 정보 N, W, Q, P 매칭
        N: edge 방문횟수, W: 보상누적값, Q: 보상평균값(W/N), P: 선택 확률 추정치
        edge[좌표행][좌표열][번호]로 접근

    Warning: action의 현재 주체인 current_user를 step마다 제공해야 함.

    """

    def __init__(self, tree=None, model=None):
        # tree
        if tree is None:
            self.tree = defaultdict(lambda: np.zeros((3, 3, 4), 'float'))
        else:
            with open(tree, 'rb') as f:
                self.tree = pickle.load(f)

        # model
        if model is None:
            self.pv_net = neural_network.PolicyValueNet(NUM_CHANNEL)
        else:
            self.pv_net = torch.load(self.pv_net.state_dict(), model)

        # hyperparameter
        self.c_puct = 5
        self.epsilon = 0.25
        self.alpha = 0.7

        # loop controller
        self.done = None

        # reset_step member
        self.total_visit = None
        self.edge = None
        self.legal_move = None
        self.no_legal_move = None
        self.state = None
        self.state_new = None
        self.state_tensor = None
        self.state_variable = None
        self.pr = None
        self.current_user = None

        # reset_episode member
        self.player_history = None
        self.opponent_history = None
        self.node_memory = None
        self.edge_memory = None
        self.action_memory = None
        self.action_count = None
        self.p_theta = None
        self.value_theta = None
        self.root_state = None

        # member init
        self.reset_step()
        self.reset_episode()

    def reset_step(self):
        self.edge = np.zeros((3, 3, 4), 'float')
        self.total_visit = 0
        self.legal_move = None
        self.no_legal_move = None
        self.state = None
        self.state_new = None
        self.state_tensor = None
        self.state_variable = None
        self.pr = np.zeros((3, 3), 'float')
        self.current_user = None

    def reset_episode(self):
        self.player_history = deque([PLANE] * 4, maxlen=4)
        self.opponent_history = deque([PLANE] * 4, maxlen=4)
        self.node_memory = deque(maxlen=9)
        self.edge_memory = deque(maxlen=9)
        self.action_memory = deque(maxlen=9)
        self.p_theta = None
        self.value_theta = None
        self.action_count = 0
        self.board_fill = None

    def select_action(self, state):
        """raw state를 받아 변환 및 저장 후 action을 리턴하는 외부 메소드.

        state 변환
        ---------
        state_new -> node & state_variable

            state_new: 9x3x3 numpy array.
                유저별 최근 4-histroy 저장하여 재구성. (저장용)

            state_variable: 1x9x3x3 torch.autograd.Variable.
                신경망의 인수로 넣을 수 있게 조정. (학습용)

            node: string. (xxhash)
                state_new를 string으로 바꾼 후 hash 생성. (탐색용)

        action 선택
        -----------
        puct 값이 가장 높은 곳을 선택함, 동점이면 랜덤 선택.

            action: 1x3 tuple.
            action = (피아식별, 보드의 x좌표, 보드의 y좌표)

        """
        self.action_count += 1

        # state 변환
        self.state = state
        self.state_new = self._convert_state(state)

        # root state면 저장
        if self.action_count == 1:
            self.root_state = self.state_new

        # state -> 문자열 -> hash로 변환 (new state 대신 dict의 key로 사용)
        node = xxhash.xxh64(self.state_new.tostring()).hexdigest()

        self.node_memory.appendleft(node)

        # 현재 보드에서 착수가능한 곳 검색
        board_fill = self.state[PLAYER] + self.state[OPPONENT]
        self.legal_move = np.argwhere(board_fill == 0)
        self.no_legal_move = np.argwhere(board_fill != 0)

        # tree 탐색 -> edge 호출 or 생성
        self._tree_search(node)

        # edge의 puct 계산
        puct = self._puct(self.edge)

        # PUCT가 최댓값인 곳 찾기
        puct_max = np.argwhere(puct == puct.max())
        # 동점 처리
        move_target = puct_max[np.random.choice(len(puct_max))]

        # 최종 action 구성 (현재 행동주체 + 좌표) 접붙히기
        action = np.r_[self.current_user, move_target]

        # action 저장
        self.action_memory.appendleft(action)

        # tuple로 action 리턴
        return tuple(action)

    def _convert_state(self, state):
        """state변환 메소드: action 주체별 최대 4수까지 history를 저장하여 새로운 state로 변환.

            state -> state_new

        """
        if self.current_user == OPPONENT:
            self.player_history.appendleft(state[PLAYER].flatten())
        else:
            self.opponent_history.appendleft(state[OPPONENT].flatten())
        state_new = np.r_[np.array(self.player_history).flatten(),
                          np.array(self.opponent_history).flatten(),
                          self.state[2].flatten()]
        return state_new

    def _tree_search(self, node):
        """tree search를 통해 선택, 확장을 진행하는 메소드.

        {node: edge}인 Tree 구성
        edge에 있는 Q, P를 이용하여 PUCT값을 계산한 뒤 모든 좌표에 매칭.

            puct: 3x3 numpy array. (float)

        """
        # tree에서 현재 node를 검색하여 존재하면 해당 edge 불러오기
        if node in self.tree:
            self.edge = self.tree[node]
            print('"Select"')
            edge_n = np.zeros((3, 3), 'float')
            for i in range(3):
                for j in range(3):
                    self.pr[i][j] = self.edge[i][j][P]
                    edge_n[i][j] = self.edge[i][j][N]
            self.total_visit = np.sum(edge_n)
            self.done = False

        else:  # 없으면 child node 이므로 edge 초기화하여 달아 주기
            self._expand(node)
        # edge의 총 방문횟수 출력
        print('(visit count: {:0.0f})'.format(self.total_visit))

        # root node면 edge의 P에 노이즈
        if self.action_count == 1:
            print('(root node noise)')
            for i, move in enumerate(self.legal_move):
                self.edge[tuple(move)][P] = (1 - self.epsilon) * self.pr[tuple(move)] + \
                    self.epsilon * np.random.dirichlet(
                        self.alpha * np.ones(len(self.legal_move)))[i]
        else:
            for move in self.legal_move:
                self.edge[tuple(move)][P] = self.pr[tuple(move)]

        # Q, P값을 배치한 edge를 백업 전 까지 저장
        self.edge_memory.appendleft(self.edge)

    def _puct(self, edge):
        # 모든 edge의 PUCT 계산
        puct = np.zeros((3, 3), 'float')
        for move in self.legal_move:
                puct[tuple(move)] = edge[tuple(move)][Q] + \
                    self.c_puct * edge[tuple(move)][P] * \
                    np.sqrt(self.total_visit) / (1 + edge[tuple(move)][N])

        # 착수 불가능한 곳엔 PUCT에 -inf를 넣어 최댓값 되는 것 방지
        for move in self.no_legal_move:
            puct[tuple(move)] = -np.inf

        # 보정한 PUCT 점수 출력
        print('***  PUCT Score  ***')
        print(puct.round(decimals=2))

        return puct

    def _expand(self, node):
        """ 기존 tree에 없는 노드가 선택됐을때 사용되는 메소드.

        현재 node의 모든 좌표의 edge를 생성.
        state_new를 신경망에 넣고 p, v 얻음.
        edge에 p를 넣어 초기화.
        이후 과정에서 edge 중 하나를 선택한 후 v로 백업하도록 알림.

        """
        # edge를 생성
        self.edge = self.tree[node]
        print('"Expand"')

        # state에 Variable 씌워서 신경망에 넣기
        self.state_tensor = torch.from_numpy(self.state_new)
        self.state_variable = Variable(self.state_tensor.view(9, 3, 3).float().unsqueeze(0))
        self.p_theta, self.value_theta = self.pv_net(self.state_variable)
        self.pr = self.p_theta.data.numpy().reshape(3, 3)
        self.value = self.value_theta.data.numpy()[0]
        print('"Evaluate"')

        # 이번 액션 후 백업할 것 알림
        self.done = True

    def backup(self, reward):
        """search가 끝나면 지나온 edge의 N, W, Q를 업데이트."""
        steps = self.action_count
        for i in range(steps):
            # W 배치
            # 내가 지나온 edge에는 v 로
            if self.action_memory[i][0] == PLAYER:
                self.edge_memory[i][tuple(
                    self.action_memory[i][1:])][
                    W] += reward
            # 상대가 지나온 edge는 -v 로
            else:
                self.edge_memory[i][tuple(
                    self.action_memory[i][1:])][
                    W] -= reward
            # N 배치 후 Q 배치
            self.edge_memory[i][tuple(self.action_memory[i][1:])][N] += 1
            self.edge_memory[i][tuple(
                self.action_memory[i][1:])][Q] = self.edge_memory[i][tuple(
                    self.action_memory[i][1:])][W] / self.edge_memory[i][tuple(
                        self.action_memory[i][1:])][N]
            # N, W, Q 트리에 최종 추가
            self.tree[self.node_memory[i]] = self.edge_memory[i]
        print('"Backup"')

    def play(self):
        """root node의 pi를 계산하고 최댓값을 찾아 action을 return함."""
        root_node = xxhash.xxh64(self.root_state.tostring()).hexdigest()
        edge = self.tree[root_node]

        pi = np.zeros((3, 3), 'float')
        total_visit = 0
        for i in range(3):
            for j in range(3):
                total_visit += edge[i][j][N]
        for i in range(3):
            for j in range(3):
                pi[i][j] = edge[i][j][N] / total_visit
        pi_max = np.argwhere(pi == pi.max())
        final_move = pi_max[np.random.choice(len(pi_max))]
        action = np.r_[self.current_user, final_move]
        return tuple(action)


if __name__ == "__main__":
    # 시작 시간 측정
    start = time.time()

    # 환경 생성
    env_game = tictactoe_env.TicTacToeEnv()
    env_simul = tictactoe_env.TicTacToeEnv()

    # mcts 생성
    MCTS = MCTS()

    # game 통계용
    result_game = {-1: 0, 0: 0, 1: 0}
    win_mark_o = 0

    for g in range(GAME):
        # state 생성
        state_game = env_game.reset()
        # 환경에 플레이어의 컬러 알림 (O, X 교대)
        env_game.player_color = (MARK_O + g) % 2

        # 루프 컨트롤러
        done_game = False
        done_mcts = False
        step_game = 0
        while not done_game:
            # Game 순번 출력
            print('=' * 70, '\nGame: {}'.format(g + 1))
            # 보드 상황 출력
            print('----- ROOT -----')
            print(state_game[PLAYER] + state_game[OPPONENT] * 2.0)

            if done_mcts:
                MCTS.current_user = (PLAYER + step_game) % 2
                step_game += 1
                action_game = MCTS.play()
                state_game, reward_game, done_game, _ = env_game.step(
                    action_game)
                # 보드 상황 출력
                print('--- SELF PLAY ---')
                print(state_game[PLAYER] + state_game[OPPONENT] * 2.0)
                MCTS.reset_step()
            if done_game:
                MCTS.reset_episode()
                result_game[reward_game] += 1
                if reward_game == 1:
                    if env_game.player_color == MARK_O:
                        win_mark_o += 1
                if (g + 1) % GAME == 0:
                    static_game = ('\n[GAME] Win: {}  Lose: {}  Draw: {}  Winrate: {:0.1f}%  \
WinMarkO: {}'.format(result_game[1], result_game[-1], result_game[0],
                     1 / (1 + np.exp(result_game[-1] / GAME) /
                          np.exp(result_game[1] / GAME)) * 100,
                     win_mark_o))
                    print('=' * 70, static_game)

            # simul 통계용
            result_simul = {-1: 0, 0: 0, 1: 0}
            terminal_n = 0
            backup_n = 0
            for s in range(SIMULATION):
                # root state 받아오기
                state_simul = env_simul.reset(state_game.copy())
                # 환경 설정 똑같이 맞추기
                env_simul.player_color = env_game.player_color

                # simul 순번 출력
                print('-' * 70, '\nSimulation: {}'.format(s + 1))
                done_simul = False
                step_simul = 0
                while not done_simul:
                    print('-- SIMULATION --')
                    print(state_simul[PLAYER] + state_simul[OPPONENT] * 2.0)
                    MCTS.current_user = (PLAYER + step_game + step_simul) % 2
                    step_simul += 1
                    action_simul = MCTS.select_action(state_simul)
                    state_simul, reward_simul, terminal, _ = env_simul.step(
                        action_simul)
                    MCTS.reset_step()
                    done_simul = MCTS.done or terminal
                    done_mcts = done_simul
                if done_simul:
                    if terminal:
                        print('--- TERMINAL ---')
                        print(state_simul[PLAYER] +
                              state_simul[OPPONENT] * 2.0)
                        result_simul[reward_simul] += 1
                        MCTS.backup(reward_simul)
                        MCTS.reset_episode()
                        terminal_n += 1
                    else:
                        print('---- BACKUP ----')
                        print(state_simul[PLAYER] +
                              state_simul[OPPONENT] * 2.0)
                        print('v: {}'.format(MCTS.value))
                        MCTS.backup(MCTS.value)
                        MCTS.reset_episode()
                        backup_n += 1
                if (s + 1) % SIMULATION == 0:

                    static_simul = ('\n[SIMULATION] Win: {}  Lose: {}  Draw: {}  Backup: {}  \
Terminal: {}'.format(result_simul[1], result_simul[-1], result_simul[0],
                     backup_n, terminal_n))
                    print('=' * 70, static_simul)
