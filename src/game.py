import pygame
import random
import numpy as np
from enum import IntEnum
from collections import namedtuple

pygame.init()
font = pygame.font.Font('src/arial.ttf', 25)


class Direction(IntEnum):
    RIGHT = 0
    DOWN = 1
    LEFT = 2
    UP = 3


Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

BLOCK_SIZE = 20
SPEED = 150


def distance(point_a, point_b):
    return np.sqrt((np.power((point_a.x - point_b.x), 2) + np.power((point_a.y - point_b.y), 2)))


class SnakeGameIA:

    def __init__(self, w=640, h=480, display=True):
        self.w = w
        self.h = h

        self.food = None
        self.score = None
        self.snake = None
        self.head = None
        self.direction = None
        self.game_over = None
        self.frameCounter = 0

        # init display
        if display:
            self.display = pygame.display.set_mode((self.w, self.h))
            pygame.display.set_caption('Snake')
            self.clock = pygame.time.Clock()

        self.reset()

    def reset(self):
        # init game state
        self.direction = Direction.RIGHT

        self.head = Point(self.w / 2, self.h / 2)
        self.snake = [self.head,
                      Point(self.head.x - BLOCK_SIZE, self.head.y),
                      Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)]

        self.score = 0
        self.food = None
        self.game_over = False
        self._place_food()

    def copy(self):
        new_game = SnakeGameIA(display=False)
        new_game.direction = self.direction
        new_game.head = self.head
        new_game.snake = self.snake
        new_game.score = self.score
        new_game.food = self.food
        new_game.game_over = self.game_over
        if new_game.food is None:
            new_game._place_food()

        if new_game.game_over:
            new_game.reset()

        return new_game

    def _place_food(self):
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def play_step(self, action):
        self.frameCounter += 1
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # 2. move
        self.play(action)  # update the head
        self.snake.insert(0, self.head)

        # 3. check if game over
        game_over = False
        reward = 0
        if self.is_collision(self.head) or self.frameCounter > 25 * len(self.snake):
            game_over = True
            self.frameCounter = 0
            reward = -10
            return reward, game_over, self.score

        # 4. place new food or just move
        if self.head == self.food:
            reward = 10
            self.frameCounter = 0
            self.score += 1
            self._place_food()
        else:
            self.snake.pop()

        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)

        # 6. return game over and score
        return reward, game_over, self.score

    def is_collision(self, point):
        # hits boundary
        if point.x > self.w - BLOCK_SIZE or point.x < 0 or point.y > self.h - BLOCK_SIZE or point.y < 0:
            return True
        # hits itself
        if point in self.snake[1:]:
            return True

        return False

    def _update_ui(self):
        self.display.fill(BLACK)

        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def play(self, action):
        pos = action.index(1)
        plus = 0
        match pos:
            case 1:
                plus = 1
            case 2:
                plus = -1

        final_dir = (self.direction + plus) % 4
        self._move(Direction(final_dir))

    def yourself(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self.play_step([0, 0, 1])
                elif event.key == pygame.K_RIGHT:
                    self.play_step([0, 1, 0])
                else:
                    self.play_step([1, 0, 0])

    def _move(self, direction):
        x = self.head.x
        y = self.head.y
        self.direction = direction
        if direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)

    def get_state(self):
        point_l = Point(self.head.x - BLOCK_SIZE, self.head.y)
        point_r = Point(self.head.x + BLOCK_SIZE, self.head.y)
        point_u = Point(self.head.x, self.head.y - BLOCK_SIZE)
        point_d = Point(self.head.x, self.head.y + BLOCK_SIZE)

        dir_l = self.direction == Direction.LEFT
        dir_r = self.direction == Direction.RIGHT
        dir_u = self.direction == Direction.UP
        dir_d = self.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and self.is_collision(point_r)) or
            (dir_l and self.is_collision(point_l)) or
            (dir_u and self.is_collision(point_u)) or
            (dir_d and self.is_collision(point_d)),

            # Danger right
            (dir_u and self.is_collision(point_r)) or
            (dir_d and self.is_collision(point_l)) or
            (dir_l and self.is_collision(point_u)) or
            (dir_r and self.is_collision(point_d)),

            # Danger left
            (dir_d and self.is_collision(point_r)) or
            (dir_u and self.is_collision(point_l)) or
            (dir_r and self.is_collision(point_u)) or
            (dir_l and self.is_collision(point_d)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            self.food.x < self.head.x,  # food left
            self.food.x > self.head.x,  # food right
            self.food.y < self.head.y,  # food up
            self.food.y > self.head.y  # food down
        ]

        return np.array(state, dtype=int)
