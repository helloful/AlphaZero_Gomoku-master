# -*- coding: utf-8 -*-
"""
@author: Junxiao Song
"""

from __future__ import print_function
import numpy as np
import pygame

EMPTY=0
BLACK=1
WHITE=2

black_color = [0, 0, 0]
# 定义黑色（黑棋用，画棋盘）
white_color = [255, 255, 255]
# 定义白色（白棋用）

class Board(object):
    """board for the game"""

    def __init__(self, **kwargs):
        self.width = int(kwargs.get('width', 15)) # get获取字典中的键值，如果不存在，那么使用默认值8
        self.height = int(kwargs.get('height', 15))
        # board states stored as a dict,
        # key: move as location on the board,
        # value: player as pieces type
        self.states = {}
        # need how many pieces in a row to win
        self.n_in_row = int(kwargs.get('n_in_row', 5))
        self.players = [1, 2]  # player1 and player2
        self.grahpic=[[]]*15#制作图形化的棋盘
        self.reset() #重置图像化界面
    def reset(self):
        for i in range(len(self.grahpic)):
            self.grahpic[i]=[EMPTY]*15

    def init_board(self, start_player=0):
        if self.width < self.n_in_row or self.height < self.n_in_row:
            raise Exception('board width and height can not be '
                            'less than {}'.format(self.n_in_row))
        self.current_player = self.players[start_player]  # start player
        # keep available moves in a list
        self.availables = list(range(self.width * self.height)) # 有效步骤
        self.states = {} # 表示字典
        self.last_move = -1

    def move_to_location(self, move):
        """
        3*3 board's moves like:
        6 7 8
        3 4 5
        0 1 2
        and move 5's location is (1,2)
        """
        h = move // self.width # 双斜杠是整除
        w = move % self.width
        return [h, w] # 返回走子的位置

    def location_to_move(self, location):
        if len(location) != 2:
            return -1
        h = location[0]
        w = location[1]
        move = h * self.width + w
        if move not in range(self.width * self.height): # 再这样的情况下，类似于每一个位置都给出了标号
            return -1
        return move # 返回走子的键值

    def current_state(self):
        """return the board state from the perspective of the current player.
        从当前玩家的角度返回棋盘的状态
        返回的大小是：
        state shape: 4*width*height
        """
        '''
        # 这个语句创建的数组是4个width*height的矩阵，然后，四个矩阵又组合成一个1维的矩阵
        # 这个1*4的矩阵中，第一个矩阵表示一个棋手的棋，第二个矩阵表示第二个棋手的棋
        # 第三个矩阵标记上一次走的位置为1
        # 第四个矩阵，如果是当前当前棋手，那么标记全部为1
        
        '''
        square_state = np.zeros((4, self.width, self.height)) # 默认浮点数
        if self.states: # 状态非空
            moves, players = np.array(list(zip(*self.states.items()))) # zip打包字典中的，并且字典只有两个items
            move_curr = moves[players == self.current_player] # 当前棋手
            move_oppo = moves[players != self.current_player] # 对手
            square_state[0][move_curr // self.width,
                            move_curr % self.height] = 1.0 # 当前棋手的下棋位置标记为1
            square_state[1][move_oppo // self.width, # 对方棋手下棋的位置标记为1
                            move_oppo % self.height] = 1.0
            # indicate the last move location
            square_state[2][self.last_move // self.width,
                            self.last_move % self.height] = 1.0
        if len(self.states) % 2 == 0:
            square_state[3][:, :] = 1.0  # indicate the colour to play 把最后一个width*height的数组全部标记为1
        return square_state[:, ::-1, :] # 返回square_state

    '''
    states字典中，存储的格式是move:plays
    即是把棋盘全部标号，然后，看玩家走的是哪个标号
    '''
    def do_move(self, move):
        self.states[move] = self.current_player # 字典的访问形式
        self.availables.remove(move) # 走了以后，把可以走的位置从有效列表中除去
        self.current_player = (
            self.players[0] if self.current_player == self.players[1]
            else self.players[1]
        ) # 轮流走棋，轮到下一个玩家
        self.last_move = move # 每次都记下当前玩家走的位置
        # 画图
        h = move // self.width  # 双斜杠是整除
        w = move % self.width
        if self.grahpic[h][w]==EMPTY:
            self.grahpic[h][w]=self.current_player

    def draw(self,screen):
        for h in range(1,16):
            pygame.draw.line(screen,black_color,[40,h*40],[600,h*40],1)
            pygame.draw.line(screen,black_color,[40*h,40],[h*40,600],1)

        # 给棋盘加一个外框，使得美观
        pygame.draw.rect(screen,black_color,[36,36,568,568],3)
        # 两次for循环取得棋盘上所有交叉点的坐标
        for row in range(len(self.grahpic)):
            for col in range(len(self.grahpic[row])):
                # 画出棋子
                if self.grahpic[row][col]!=EMPTY:
                    color=black_color if self.grahpic[row][col]==BLACK else white_color
                    pos=[40*(col+1),40*(row+1)]
                    pygame.draw.circle(screen,color,pos,18,0)



    '''
    检查是否有赢家
    '''
    def has_a_winner(self):
        width = self.width
        height = self.height
        states = self.states # 字典
        n = self.n_in_row

        moved = list(set(range(width * height)) - set(self.availables)) # 获得已经走过的位置
        if len(moved) < self.n_in_row *2-1:
            return False, -1

        for m in moved:
            h = m // width
            w = m % width
            player = states[m]

            if (w in range(width - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n))) == 1):
                return True, player

            if (h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * width, width))) == 1):
                return True, player

            if (w in range(width - n + 1) and h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (width + 1), width + 1))) == 1):
                return True, player

            if (w in range(n - 1, width) and h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (width - 1), width - 1))) == 1):
                return True, player

        return False, -1

    def game_end(self):
        """Check whether the game is ended or not"""
        win, winner = self.has_a_winner()
        if win:
            return True, winner
        elif not len(self.availables):
            return True, -1
        return False, -1

    def get_current_player(self):
        return self.current_player


class Game(object):
    """game server"""

    def __init__(self, board, **kwargs):
        self.board = board

    def graphic(self, board, player1, player2):
        """Draw the board and show game info"""
        width = board.width
        height = board.height

        print("Player", player1, "with X".rjust(3))
        print("Player", player2, "with O".rjust(3))
        print()
        for x in range(width):
            print("{0:8}".format(x), end='')
        print('\r\n')
        for i in range(height - 1, -1, -1):
            print("{0:4d}".format(i), end='')
            for j in range(width):
                loc = i * width + j
                p = board.states.get(loc, -1)
                if p == player1:
                    print('X'.center(8), end='')
                elif p == player2:
                    print('O'.center(8), end='')
                else:
                    print('_'.center(8), end='')
            print('\r\n\r\n')

    def start_play(self, player1, player2, start_player=0, is_shown=1):
        """start a game between two players"""
        if start_player not in (0, 1):
            raise Exception('start_player should be either 0 (player1 first) '
                            'or 1 (player2 first)')
        self.board.init_board(start_player)
        p1, p2 = self.board.players
        player1.set_player_ind(p1) # Human1
        player2.set_player_ind(p2) # MCTS 2
        players = {p1: player1, p2: player2}

        # if is_shown:
        #     self.graphic(self.board, player1.player, player2.player)

        pygame.init()  # pygame初始化函数，固定写法
        pygame.display.set_caption("模型评估")
        screen = pygame.display.set_mode((650, 650))
        screen.fill([182, 131, 73])
        self.board.draw(screen)  # 画出棋盘
        pygame.display.flip()  # 刷新

        while True:
            current_player = self.board.get_current_player()
            player_in_turn = players[current_player]
            move = player_in_turn.get_action(self.board)
            self.board.do_move(move)

            self.board.draw(screen)  # 画出棋盘
            pygame.display.flip()  # 刷新

            end, winner = self.board.game_end()
            if end:
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is", players[winner])
                    else:
                        print("Game end. Tie")
                # pygame.quit()
                return winner

    def start_self_play(self, player, is_shown=0, temp=1e-3):
        """ start a self-play game using a MCTS player, reuse the search tree,
        and store the self-play data: (state, mcts_probs, z) for training
        """

        self.board.init_board()
        p1, p2 = self.board.players
        states, mcts_probs, current_players = [], [], []

        pygame.init()  # pygame初始化函数，固定写法
        pygame.display.set_caption("五子棋(400次模拟后的结果)")
        screen = pygame.display.set_mode((650, 650))
        screen.fill([182, 131, 73])
        self.board.draw(screen)  # 画出棋盘
        pygame.display.flip()  # 刷新

        while True:

            move, move_probs = player.get_action(self.board,
                                                 temp=temp,
                                                 return_prob=1)
            # store the data
            states.append(self.board.current_state())
            mcts_probs.append(move_probs)
            current_players.append(self.board.current_player)
            # perform a move
            self.board.do_move(move)

            self.board.draw(screen)  # 画出棋盘
            pygame.display.flip()  # 刷新

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()

            if is_shown:
                self.graphic(self.board, p1, p2)
            end, winner = self.board.game_end()
            if end:
                self.board.reset()
                # winner from the perspective of the current player of each state
                winners_z = np.zeros(len(current_players))
                if winner != -1:
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0
                # reset MCTS root node
                player.reset_player()
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is player:", winner)
                    else:
                        print("Game end. Tie")
                pygame.quit()
                return winner, zip(states, mcts_probs, winners_z)
