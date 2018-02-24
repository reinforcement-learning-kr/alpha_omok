# -*- coding: utf-8 -*-
import tictactoe_env
import neural_network

import time
from collections import deque, defaultdict

import torch
from torch.autograd import Variable

import slackweb
import xxhash
import dill as pickle
import numpy as np
np.set_printoptions(suppress=True)

PLAYER, OPPONENT = 0, 1
MARK_O, MARK_X = 0, 1
N, W, Q, P = 0, 1, 2, 3
PLANE = np.zeros((3, 3), 'int').flatten()

NUM_CHANNEL = 128
GAME = 10
SIMULATION = 60


class MCTS(object):
    """몬테카를로 트리 탐색 클래스.

    셀프플레이를 통해 train 데이터 생성 (s, pi, z)

    state
    ------
    observation에서 각 주체당 4수까지 저장해서 state로 만듦

        9x3x3 numpy array -> 1x81 tuple

    edge
    -----
    현재 observation에서 착수 가능한 모든 action자리에 4개의 정보 저장

    type: 3x3x4 numpy array

        9개 좌표에 4개의 정보 N, W, Q, P 매칭
        N: edge 방문횟수, W: 보상누적값, Q: 보상평균값(W/N), P: 선택 확률 추정치
        edge[좌표행][좌표열][번호]로 접근

    Warning: action의 현재 주체인 current_user를 reset_step()에서 제공해야 함.

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
        self.observation = None
        self.state = None
        self.state_tensor = None
        self.state_variable = None
        self.prob = None
        self.value = None
        self.current_user = None

        # reset_episode member
        self.player_history = None
        self.opponent_history = None
        self.node_memory = None
        self.edge_memory = None
        self.action_memory = None
        self.action_count = None
        self.p_theta = None
        self.v_theta = None

        # member init
        self.reset_step()
        self._reset_episode()

    def reset_step(self, current_user=None):
        self.edge = np.zeros((3, 3, 4), 'float')
        self.total_visit = 0
        self.legal_move = None
        self.no_legal_move = None
        self.observation = None
        self.state = None
        self.state_tensor = None
        self.state_variable = None
        self.prob = np.zeros((3, 3), 'float')
        self.value = None
        self.current_user = current_user

    def _reset_episode(self):
        self.player_history = deque([PLANE] * 4, maxlen=4)
        self.opponent_history = deque([PLANE] * 4, maxlen=4)
        self.node_memory = deque(maxlen=9)
        self.edge_memory = deque(maxlen=9)
        self.action_memory = deque(maxlen=9)
        self.p_theta = None
        self.v_theta = None
        self.action_count = 0

    def select_action(self, observation):
        """observation을 받아 변환 및 저장 후 action을 리턴하는 외부 메소드.

        observation 변환
        ----------
        observation -> state -> node & state_variable

            state: 9x3x3 numpy array.
                유저별 최근 4-histroy 저장하여 재구성.

            state_variable: 1x9x3x3 torch.autograd.Variable.
                신경망의 인수로 넣을 수 있게 조정. (학습용)

            node: string. (xxhash)
                state를 string으로 바꾼 후 hash 생성. (탐색용)

        action 선택
        -----------
        puct 값이 가장 높은 곳을 선택함, 동점이면 랜덤 선택.

            action: 1x3 tuple.
            action = (유저타입, 보드의 좌표행, 보드의 좌표열)

        """
        # 현재 주체 설정 여부 필터링
        if self.current_user is None:
            raise NotImplementedError("Set Current User!")

        self.action_count += 1

        # observation 변환
        self.observation = observation
        self.state = self._generate_state(observation)

        # state -> 문자열 -> hash로 변환 (state 대신 tree dict의 key로 사용)
        node = xxhash.xxh64(self.state.tostring()).hexdigest()

        self.node_memory.appendleft(node)

        # 현재 보드에서 착수가능한 곳 검색
        board_fill = self.observation[PLAYER] + self.observation[OPPONENT]
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

    def _generate_state(self, observation):
        """observation 변환 메소드: action 주체별 최대 4수까지 history를 저장하여 state로 생성.

            observation -> state

        """

        if self.current_user == OPPONENT:
            self.player_history.appendleft(observation[PLAYER].flatten())
        else:
            self.opponent_history.appendleft(observation[OPPONENT].flatten())
        state = np.r_[np.array(self.player_history).flatten(),
                      np.array(self.opponent_history).flatten(),
                      self.observation[2].flatten()]
        return state

    def _tree_search(self, node):
        """tree search를 통해 선택, 확장을 진행하는 메소드.

        {node: edge}인 Tree 구성
        edge에 있는 Q, P를 이용하여 PUCT값을 계산한 뒤 모든 좌표에 매칭.

            puct: 3x3 numpy array. (float)

        """

        # tree에서 현재 node를 검색하여 존재하면 해당 edge 불러오기
        if node in self.tree:
            self.edge = self.tree[node]
            print('"Select"\n')
            edge_n = np.zeros((3, 3), 'float')
            for i in range(3):
                for j in range(3):
                    self.prob[i][j] = self.edge[i][j][P]
                    edge_n[i][j] = self.edge[i][j][N]
            self.total_visit = np.sum(edge_n)
            # 계속 진행
            self.done = False

        else:  # 없으면 child node 이므로 edge 초기화하여 달아 주기
            self._expand(node)
        # edge의 총 방문횟수 출력
        print('(visit count: {:0.0f})\n'.format(self.total_visit))

        # root node면 edge의 P에 노이즈
        if self.action_count == 1:
            print('(root node noise)\n')
            for i, move in enumerate(self.legal_move):
                self.edge[tuple(move)][P] = (1 - self.epsilon) * self.prob[tuple(move)] + \
                    self.epsilon * np.random.dirichlet(
                        self.alpha * np.ones(len(self.legal_move)))[i]
        else:
            for move in self.legal_move:
                self.edge[tuple(move)][P] = self.prob[tuple(move)]

        # Q, P값을 배치한 edge를 담아둠. 백업할 때 사용
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
        print('***  PUCT SCORE  ***')
        print(puct.round(decimals=2))
        print('')

        return puct

    def _expand(self, node):
        """ 기존 tree에 없는 노드가 선택됐을때 사용되는 메소드.

        현재 node의 모든 좌표의 edge를 생성.
        state 텐서화 하여 신경망에 넣고 p_theta, v_theta 얻음.
        edge의 P에 p_theta를 넣어 초기화.
        select에서 edge 중 하나를 선택한 후 v로 백업하도록 알림.

        """

        # edge를 생성
        self.edge = self.tree[node]
        print('"Expand"')

        # state에 Variable 씌워서 신경망에 넣기
        self.state_tensor = torch.from_numpy(self.state)
        self.state_variable = Variable(self.state_tensor.view(9, 3, 3).float().unsqueeze(0))
        self.p_theta, self.v_theta = self.pv_net(self.state_variable)
        self.prob = self.p_theta.data.numpy().reshape(3, 3)
        self.value = self.v_theta.data.numpy()[0]
        print('"Evaluate"\n')

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

            # N, W, Q 배치한 edge 트리에 최종 업데이트
            self.tree[self.node_memory[i]] = self.edge_memory[i]

        print('"Backup"\n\n')

        self._reset_episode()

    def play(self, root_state):
        """root node의 pi를 계산하고 최댓값을 찾아 action을 return함."""

        root_node = xxhash.xxh64(root_state.tostring()).hexdigest()
        edge = self.tree[root_node]

        pi = np.zeros((3, 3), 'float')
        total_visit = 0

        for i in range(3):
            for j in range(3):
                total_visit += edge[i][j][N]

        for i in range(3):
            for j in range(3):
                pi[i][j] = edge[i][j][N] / total_visit

        print('=*=*=*=   Pi   =*=*=*=')
        print(pi.round(decimals=2))
        print('')

        pi_max = np.argwhere(pi == pi.max())
        final_move = pi_max[np.random.choice(len(pi_max))]
        action = np.r_[self.current_user, final_move]

        return tuple(action), pi.flatten()


if __name__ == "__main__":

    start = time.time()

    env_game = tictactoe_env.TicTacToeEnv()
    env_simul = tictactoe_env.TicTacToeEnv()

    result_game = {-1: 0, 0: 0, 1: 0}
    win_mark_o = 0
    step_game = 0
    step_total_simul = 0

    print("=" * 30, " Game Start ", "=" * 30)
    print('')
    for game in range(GAME):
        player_color = (MARK_O + game) % 2
        observation_game = env_game.reset(player_color=player_color)
        mcts = MCTS()
        root_state = None
        done_game = False
        step_play = 0

        while not done_game:
            print("=" * 27, " Simulation Start ", "=" * 27)
            print('')

            current_user_play = ((PLAYER if player_color == MARK_O else OPPONENT) + step_play) % 2
            result_simul = {-1: 0, 0: 0, 1: 0}
            terminal_n = 0
            backup_n = 0
            step_simul = 0

            for simul in range(SIMULATION):
                print('#######   Simulation: {}   #######'.format(simul + 1))
                print('')

                observation_simul = env_simul.reset(
                    observation_game.copy(), player_color=player_color)
                done_simul = False
                step_mcts = 0

                while not done_simul:
                    print('---- BOARD ----')
                    print(observation_simul[PLAYER] + observation_simul[OPPONENT] * 2.0)
                    print('')

                    current_user_mcts = (current_user_play + step_mcts) % 2
                    mcts.reset_step(current_user_mcts)
                    action_simul = mcts.select_action(observation_simul)
                    observation_simul, z_env, done_env, _ = env_simul.step(action_simul)
                    step_mcts += 1
                    step_simul += 1
                    step_total_simul += 1

                    if step_mcts == 1:
                        root_state = mcts.state

                    done_mcts = mcts.done
                    v = mcts.value
                    done_simul = done_mcts or done_env

                if done_simul:
                    if done_mcts:
                        print('==== BACKUP ====')
                        print(observation_simul[PLAYER] + observation_simul[OPPONENT] * 2.0)
                        print('')
                        print('(v: {})'.format(v))
                        print('')

                        mcts.backup(v)
                        backup_n += 1

                    else:
                        print('=== TERMINAL ===')
                        print(observation_simul[PLAYER] + observation_simul[OPPONENT] * 2.0)
                        print('')
                        print('(v: {})'.format(v))
                        print('')

                        mcts.backup(z_env)
                        result_simul[z_env] += 1
                        terminal_n += 1

            finish_simul = round(float(time.time() - start))

            print("=" * 25, " {} Simulations End ".format(simul + 1), "=" * 25)
            print('Win: {}  Lose: {}  Draw: {}  Backup: {}  Terminal: {}  Step: {}\n'.format(
                result_simul[1], result_simul[-1], result_simul[0], backup_n, terminal_n,
                step_simul))
            print('##########    Game: {}    ##########'.format(game + 1))
            print('')
            print('`*`*` ROOT `*`*`')
            print(observation_game[PLAYER] + observation_game[OPPONENT] * 2.0)
            print('')

            mcts.reset_step(current_user_play)
            action_game, pi = mcts.play(root_state)

            observation_game, z, done_game, _ = env_game.step(action_game)
            step_play += 1
            step_game += 1

            print('`*`*` PLAY `*`*`')
            print(observation_game[PLAYER] + observation_game[OPPONENT] * 2.0)
            print('')

        if done_game:
            result_game[z] += 1
            if z == 1:
                if env_game.player_color == MARK_O:
                    win_mark_o += 1

    finish_game = round(float(time.time() - start))

    print("=" * 27, " {}  Game End  ".format(game + 1), "=" * 27)
    stat_game = ('[GAME] Win: {}  Lose: {}  Draw: {}  Winrate: {:0.1f}%  WinMarkO: {}'.format(
        result_game[1], result_game[-1], result_game[0],
        1 / (1 + np.exp(result_game[-1] / (game + 1)) / np.exp(result_game[1] / (game + 1))) * 100,
        win_mark_o))
    print(stat_game)

    slack = slackweb.Slack(
        url="https://hooks.slack.com/services/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")
    slack.notify(
        text="Finished: [{} Game/{} Step] in {}s [Mac]".format(
            game + 1, step_game + step_total_simul, finish_game))
    slack.notify(text=stat_game)
