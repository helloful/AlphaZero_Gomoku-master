import pygame

EMPTY=0
BLACK=1
WHITE=2

black_color = [0, 0, 0]
# 定义黑色（黑棋用，画棋盘）
white_color = [255, 255, 255]
# 定义白色（白棋用）

class Show(object):
    def __init__(self,board):
        pygame.init()  # pygame初始化函数，固定写法
        pygame.display.set_caption("五子棋")
        self.screen = pygame.display.set_mode((650, 650))
        self.screen.fill([182, 131, 73])
        # self.board.draw(screen)  # 画出棋盘
        # pygame.display.flip()  # 刷新
        self.graphic=board.grahpic
        self.draw()
    def draw(self):
        for h in range(1, 16):
            pygame.draw.line(self.screen, black_color, [40, h * 40], [600, h * 40], 1)
            pygame.draw.line(self.screen, black_color, [40 * h, 40], [h * 40, 600], 1)

            # 给棋盘加一个外框，使得美观
        pygame.draw.rect(self.screen, black_color, [36, 36, 568, 568], 3)
        # 两次for循环取得棋盘上所有交叉点的坐标
        for row in range(len(self.grahpic)):
            for col in range(len(self.grahpic[row])):
                # 画出棋子
                if self.grahpic[row][col] != EMPTY:
                    color = black_color if self.grahpic[row][col] == BLACK else white_color
                    pos = [40 * (col + 1), 40 * (row + 1)]
                    pygame.draw.circle(self.screen, color, pos, 18, 0)



