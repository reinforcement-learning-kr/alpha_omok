# -*- coding: utf-8 -*-
import numpy as np
import gym

PLAYER = 0
OPPONENT = 1
USER_TYPE = 0  # action index
MARK_O = 0
MARK_X = 1


class TicTacToeEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}
    reward = (-1, 0, 1)

    def __init__(self):
        self.state = None
        self.viewer = None
        self.player_color = None  # 나의 "OX"
        self.step_count = 0
        self.reset()

    def reset(self):
        self.state = np.zeros((3, 3, 3), 'int')
        self.viewer = None
        self.step_count = 0
        self.player_color = None
        return self.state

    def step(self, action):
        # 규칙 위반 필터링: action 자리에 이미 자리가 차있으면 착수 금지
        if self.state[action] == 1:
            if action[USER_TYPE] == PLAYER:  # 근데 그게 플레이어가 한 짓이면 반칙패
                reward = -1
                done = True
                info = {}
                # print('@@ Illegal Lose! @@')  # 출력
                return self.state, reward, done, info
            elif action[USER_TYPE] == OPPONENT:  # 상대가 한 짓이면 반대
                reward = 1
                done = True
                info = {}
                # print('@@ Illegal Win! @@')
                return self.state, reward, done, info

        # action 적용
        self.state[action] = 1

        # 연속 두번 하기, player_color 비설정 시 오류 발생시키기
        redupl = np.sum(self.state[PLAYER]) - np.sum(self.state[OPPONENT])
        if abs(redupl) > 1:
            raise NotImplementedError("Place Once!")
        if self.player_color is None:
            raise NotImplementedError("Set Player Color!")
        # "O"가 아닌데 처음에 하면 오류 발생시키기
        if self.player_color != MARK_O:
            if np.sum(self.state) == 1 and action[USER_TYPE] == PLAYER:
                raise NotImplementedError("Not Your Turn!")
        else:
            if np.sum(self.state) == 1 and action[USER_TYPE] == OPPONENT:
                raise NotImplementedError("Not Your Turn!")

        # 2번 보드("O") 동기화
        if self.player_color == MARK_O:
            self.state[2] = self.state[PLAYER]
        else:
            self.state[2] = self.state[OPPONENT]

        return self._check_win()  # 승패 체크해서 리턴

    def _check_win(self):
        """state 승패체크용 내부 함수."""
        # 승리패턴 8가지 구성 (1:돌이 있는 곳, 0: 돌이 없는 곳)
        win_pattern = np.array([[[1, 1, 1], [0, 0, 0], [0, 0, 0]],
                                [[0, 0, 0], [1, 1, 1], [0, 0, 0]],
                                [[0, 0, 0], [0, 0, 0], [1, 1, 1]],
                                [[1, 0, 0], [1, 0, 0], [1, 0, 0]],
                                [[0, 1, 0], [0, 1, 0], [0, 1, 0]],
                                [[0, 0, 1], [0, 0, 1], [0, 0, 1]],
                                [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                                [[0, 0, 1], [0, 1, 0], [1, 0, 0]]])
        # 0,1번 보드가 승리패턴과 일치하면
        for i in range(2):
            for k in range(8):
                # 2진 배열은 패턴을 포함할때 서로 곱(행렬곱 아님)하면 패턴 자신이 나옴
                if np.array_equal(
                        self.state[i] * win_pattern[k], win_pattern[k]):
                    if i == PLAYER:  # i가 플레이어면 승리
                        reward = 1  # 보상 1
                        done = True  # 게임 끝
                        info = {}
                        print('## You Win! ##')  # 승리 메세지 출력
                        return self.state, reward, done, info  # 필수 요소 리턴!
                    else:  # i가 상대면 패배
                        reward = -1  # 보상 -1
                        done = True  # 게임 끝
                        info = {}
                        print('## You Lose! ##')  # 너 짐
                        return self.state, reward, done, info  # 필수 요소 리턴!
        # 다 돌려봤는데 승부난게 없더라 근데 "O"식별용 2번보드에 들어있는게 5개면? 비김
        if np.count_nonzero(self.state[2]) == 5:
            reward = 0  # 보상 0
            done = True  # 게임 끝
            info = {}
            print('##  Draw! ##')  # 비김
            return self.state, reward, done, info
        # 이거 다~~~ 아니면 다음 수 둬야지
        else:
            reward = 0
            done = False  # 안 끝남!
            info = {'steps': self.step_count}
            return self.state, reward, done, info

    def render(self, mode='human', close=False):
        """현재 state를 그려주는 함수."""
        if close:  # 클로즈값이 참인데
            if self.viewer is not None:  # 뷰어가 비어있지 않으면
                self.viewer.close()   # 뷰어를 닫고
                self.viewer = None   # 뷰어 지우기
            return

        if self.viewer is None:
            from gym.envs.classic_control import rendering  # 렌더링 모듈 임포트
            # 뷰어의 좌표 딕트로 구성
            render_loc = {0: (50, 250), 1: (150, 250), 2: (250, 250),
                          3: (50, 150), 4: (150, 150), 5: (250, 150),
                          6: (50, 50), 7: (150, 50), 8: (250, 50)}

            # -------------------- 뷰어 생성 --------------------- #
            # 캔버스 역할의 뷰어 초기화 가로 세로 300
            self.viewer = rendering.Viewer(300, 300)
            # 가로 세로 선 생성 (시작점좌표, 끝점좌표), 색정하기 (r, g, b)
            line_1 = rendering.Line((0, 100), (300, 100))
            line_1.set_color(0, 0, 0)
            line_2 = rendering.Line((0, 200), (300, 200))
            line_2.set_color(0, 0, 0)
            line_a = rendering.Line((100, 0), (100, 300))
            line_a.set_color(0, 0, 0)
            line_b = rendering.Line((200, 0), (200, 300))
            line_b.set_color(0, 0, 0)
            # 뷰어에 선 붙이기
            self.viewer.add_geom(line_1)
            self.viewer.add_geom(line_2)
            self.viewer.add_geom(line_a)
            self.viewer.add_geom(line_b)

            # ----------- OX 마크 이미지 생성 및 위치 지정 -------------- #
            # 9개의 위치에 O,X 모두 위치지정해 놓음 (18장)
            # 그림파일 위치는 이 파일이 있는 폴더 내부의 img 폴더

            # 그림 객체 생성
            self.image_O1 = rendering.Image("img/O.png", 96, 96)
            # 위치 컨트롤 하는 객체 생성
            trans_O1 = rendering.Transform(render_loc[0])
            # 이놈을 이미지에 붙혀서 위치 지정
            # (이미지를 뷰어에 붙이기 전까진 렌더링 안됨)
            self.image_O1.add_attr(trans_O1)

            self.image_O2 = rendering.Image("img/O.png", 96, 96)
            trans_O2 = rendering.Transform(render_loc[1])
            self.image_O2.add_attr(trans_O2)

            self.image_O3 = rendering.Image("img/O.png", 96, 96)
            trans_O3 = rendering.Transform(render_loc[2])
            self.image_O3.add_attr(trans_O3)

            self.image_O4 = rendering.Image("img/O.png", 96, 96)
            trans_O4 = rendering.Transform(render_loc[3])
            self.image_O4.add_attr(trans_O4)

            self.image_O5 = rendering.Image("img/O.png", 96, 96)
            trans_O5 = rendering.Transform(render_loc[4])
            self.image_O5.add_attr(trans_O5)

            self.image_O6 = rendering.Image("img/O.png", 96, 96)
            trans_O6 = rendering.Transform(render_loc[5])
            self.image_O6.add_attr(trans_O6)

            self.image_O7 = rendering.Image("img/O.png", 96, 96)
            trans_O7 = rendering.Transform(render_loc[6])
            self.image_O7.add_attr(trans_O7)

            self.image_O8 = rendering.Image("img/O.png", 96, 96)
            trans_O8 = rendering.Transform(render_loc[7])
            self.image_O8.add_attr(trans_O8)

            self.image_O9 = rendering.Image("img/O.png", 96, 96)
            trans_O9 = rendering.Transform(render_loc[8])
            self.image_O9.add_attr(trans_O9)

            self.image_X1 = rendering.Image("img/X.png", 96, 96)
            trans_X1 = rendering.Transform(render_loc[0])
            self.image_X1.add_attr(trans_X1)

            self.image_X2 = rendering.Image("img/X.png", 96, 96)
            trans_X2 = rendering.Transform(render_loc[1])
            self.image_X2.add_attr(trans_X2)

            self.image_X3 = rendering.Image("img/X.png", 96, 96)
            trans_X3 = rendering.Transform(render_loc[2])
            self.image_X3.add_attr(trans_X3)

            self.image_X4 = rendering.Image("img/X.png", 96, 96)
            trans_X4 = rendering.Transform(render_loc[3])
            self.image_X4.add_attr(trans_X4)

            self.image_X5 = rendering.Image("img/X.png", 96, 96)
            trans_X5 = rendering.Transform(render_loc[4])
            self.image_X5.add_attr(trans_X5)

            self.image_X6 = rendering.Image("img/X.png", 96, 96)
            trans_X6 = rendering.Transform(render_loc[5])
            self.image_X6.add_attr(trans_X6)

            self.image_X7 = rendering.Image("img/X.png", 96, 96)
            trans_X7 = rendering.Transform(render_loc[6])
            self.image_X7.add_attr(trans_X7)

            self.image_X8 = rendering.Image("img/X.png", 96, 96)
            trans_X8 = rendering.Transform(render_loc[7])
            self.image_X8.add_attr(trans_X8)

            self.image_X9 = rendering.Image("img/X.png", 96, 96)
            trans_X9 = rendering.Transform(render_loc[8])
            self.image_X9.add_attr(trans_X9)

        # ------------ state 정보에 맞는 이미지를 뷰어에 붙이는 과정 ------------- #
        # 좌표번호마다 "OX"가 있는지 확인하여 해당하는 이미지를 뷰어에 붙임 (렌더링 때 보임)
        # "OX"의 정체성 설정!
        render_O = None
        render_X = None
        if self.player_color == MARK_O:
            render_O = PLAYER
            render_X = OPPONENT
        else:
            render_O = OPPONENT
            render_X = PLAYER

        if self.state[render_O][0][0] == 1:
            self.viewer.add_geom(self.image_O1)
        elif self.state[render_X][0][0] == 1:
            self.viewer.add_geom(self.image_X1)

        if self.state[render_O][0][1] == 1:
            self.viewer.add_geom(self.image_O2)
        elif self.state[render_X][0][1] == 1:
            self.viewer.add_geom(self.image_X2)

        if self.state[render_O][0][2] == 1:
            self.viewer.add_geom(self.image_O3)
        elif self.state[render_X][0][2] == 1:
            self.viewer.add_geom(self.image_X3)

        if self.state[render_O][1][0] == 1:
            self.viewer.add_geom(self.image_O4)
        elif self.state[render_X][1][0] == 1:
            self.viewer.add_geom(self.image_X4)

        if self.state[render_O][1][1] == 1:
            self.viewer.add_geom(self.image_O5)
        elif self.state[render_X][1][1] == 1:
            self.viewer.add_geom(self.image_X5)

        if self.state[render_O][1][2] == 1:
            self.viewer.add_geom(self.image_O6)
        elif self.state[render_X][1][2] == 1:
            self.viewer.add_geom(self.image_X6)

        if self.state[render_O][2][0] == 1:
            self.viewer.add_geom(self.image_O7)
        elif self.state[render_X][2][0] == 1:
            self.viewer.add_geom(self.image_X7)

        if self.state[render_O][2][1] == 1:
            self.viewer.add_geom(self.image_O8)
        elif self.state[render_X][2][1] == 1:
            self.viewer.add_geom(self.image_X8)

        if self.state[render_O][2][2] == 1:
            self.viewer.add_geom(self.image_O9)
        elif self.state[render_X][2][2] == 1:
            self.viewer.add_geom(self.image_X9)

        # rgb 모드면 뷰어를 렌더링해서 리턴
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')
