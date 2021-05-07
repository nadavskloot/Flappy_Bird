import random
import numpy as np
# from prettyprinter import pprint

import pygame
from pygame.locals import *
from datetime import datetime

from turtle import Turtle, Screen
import time

bird_img = pygame.image.load('images/birdup.png')
bg_img = pygame.image.load('images/birdupbg.png')
pipe_img = pygame.image.load('images/birduppipe.png')
ground_img = pygame.image.load('images/birdupground.png')


class Bird:
    def __init__(self, jump_speed=100, x_pos=40, init_y_pos=110, width=20, height=20, v_y=0, gravity=-150):
        self.x_pos = x_pos
        self.init_y_pos = init_y_pos
        self.y_pos = init_y_pos
        self.width = width
        self.height = height
        self.v_y = v_y
        self.jump_speed = jump_speed
        self.gravity = gravity

    def jump(self):
        self.v_y = self.jump_speed

    def proceed(self, t_step):
        self.v_y += self.gravity * t_step
        self.y_pos += self.v_y * t_step

    def get_bounds(self):
        b_l = (self.x_pos, self.y_pos)
        t_r = (self.x_pos + self.width, self.y_pos + self.height)
        return (b_l, t_r)

    def get_pos(self):
        return (self.x_pos, self.y_pos)

    def draw_self(self, env, win):
        bird = bird_img
        bird = pygame.transform.scale(bird, (self.width, self.height))
        win.blit(bird, (self.x_pos, env.screen_height - self.y_pos - self.height))


class PipePair:
    PIPE_THICKNESS = 50

    def __init__(self, x_pos, top_end, bottom_end, v_x=-50):
        self.x_pos = x_pos
        self.top_end = top_end
        self.bottom_end = bottom_end
        self.v_x = v_x
        self.width = self.PIPE_THICKNESS

    def proceed(self, t_step):
        self.x_pos += self.v_x * t_step

    def check_pos_collision(self, pos):
        if pos[0] > self.x_pos and pos[0] < self.x_pos + self.width:
            if pos[1] > top_end or pos[1] < bottom_end:
                return True
        return False

    def check_bound_collision(self, bounds):
        bl = bounds[0]
        tr = bounds[1]
        if tr[0] > self.x_pos and bl[0] < self.x_pos + self.width:
            if tr[1] > self.top_end or bl[1] < self.bottom_end:
                return True
        return False

    def check_passed(self, x_pos):
        if x_pos > self.x_pos + self.width:
            return True
        else:
            False

    def draw_self(self, env, win):
        top = pipe_img
        top = pygame.transform.scale(
            top, (self.width, env.roof - self.top_end))
        top = pygame.transform.flip(top, False, True)
        win.blit(top, (self.x_pos, env.roof-env.screen_height))
        bottom = pipe_img
        bottom = pygame.transform.scale(
            bottom, (self.width, self.bottom_end - env.ground))
        win.blit(bottom, (self.x_pos, env.screen_height - self.bottom_end))


class Environment:
    MIN_X_GAP = 100

    def __init__(self, decision_rate, t_step, gravity, v_pipe, x_grid, y_grid, fov, screen_width, screen_height, gap_range=0, min_pipe_gap=50):
        # how often the agent gets to perform an action in seconds
        self.decision_rate = decision_rate
        self.t_step = t_step  # length of a single frame of animation in seconds
        self.gravity = gravity
        self.x_grid = x_grid
        self.y_grid = y_grid
        self.flappy = Bird(gravity=gravity)
        self.fov = fov
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.ground = 0
        self.roof = screen_height

        self.gap_range = gap_range
        self.min_pipe_gap = min_pipe_gap
        self.v_pipe = v_pipe
        self.back_pipes = []
        self.front_pipes = []
        self.pipeCount = 0
        last_pos = screen_width
        for i in range(20):
            self.front_pipes.append(self.createRandomPipe(
                x_pos=last_pos, v_pipe=v_pipe, min_gap=min_pipe_gap, gap_range=gap_range))
            last_pos = last_pos + PipePair.PIPE_THICKNESS + \
                random.randrange(1, 4) * self.MIN_X_GAP

        self.action_space = ['J', 'F']

    def createRandomPipe(self, x_pos, v_pipe, min_gap, gap_range):
        gap = min_gap
        if gap_range > 0:
            gap = gap + random.randrange(gap_range + 1)
        top_end = random.randrange(self.ground + gap, self.roof)
        return PipePair(x_pos=x_pos, top_end=top_end, bottom_end=top_end-gap, v_x=v_pipe)

    def run(self, num_time_steps, win=None):
        pause_time = 1
        for step in range(num_time_steps):
            self.flappy.proceed(self.t_step)
            for pipe in self.back_pipes + self.front_pipes:
                pipe.proceed(self.t_step)
            if len(self.back_pipes):
                if self.back_pipes[0].x_pos + self.back_pipes[0].width < 0:
                    self.front_pipes.append(self.back_pipes.pop(0))

            if self.front_pipes[0].check_bound_collision(self.flappy.get_bounds()):
                self.flappy.v_y = 0
                self.flappy.jump_speed = 0
                self.flappy.gravity = 0
                for p in self.back_pipes + self.front_pipes:
                    p.v_x = 0
                if win != None:
                    print("Collision detected\n", "Flappy Bounds: ", self.flappy.get_bounds(
                    ), "\nPipe X-Pos: ", self.front_pipes[0].x_pos, "\nPipe Top: ", self.front_pipes[0].top_end, "\nPipe Bottom: ", self.front_pipes[0].bottom_end, sep="")
                    time.sleep(pause_time)
                return 'Dead'
            elif self.flappy.y_pos <= self.ground or self.flappy.y_pos + self.flappy.height >= self.roof:

                self.flappy.v_y = 0
                self.flappy.jump_speed = 0
                self.flappy.gravity = 0
                for p in self.back_pipes + self.front_pipes:
                    p.v_x = 0
                if win != None:
                    print("Flappy splat ground or bump roof\n",
                          "Flappy Bounds: ", self.flappy.get_bounds(), sep="")
                    time.sleep(pause_time)

                return 'Dead'
            elif self.front_pipes[0].check_passed(self.flappy.x_pos):
                self.back_pipes.append(self.front_pipes.pop(0))
                return 'Success'
            if win != None:
                self.draw_bg(win)
                self.flappy.draw_self(self, win)
                for pipe in self.back_pipes + self.front_pipes:
                    pipe.draw_self(self, win)
                pygame.display.update()
                time.sleep(self.t_step)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()

        return 'Alive'

    def draw_bg(self, win):
        bg = bg_img
        bg = pygame.transform.scale(
            bg, (self.screen_width, self.screen_height))
        win.blit(bg, (0, 0))

    def gridifyX(self, x):
        if x > self.flappy.x_pos + self.fov:
            return float('Inf')
        return int(x//(self.fov/self.x_grid))

    def gridifyY(self, y):
        return int(y//(self.screen_height/self.y_grid))

    def get_state(self):
        pipeBottom_hieght = self.gridifyY(self.front_pipes[0].bottom_end)
        pipeTop_hieght = self.gridifyY(self.front_pipes[0].top_end)
        ground_clearance = self.gridifyY(self.flappy.y_pos)
        x_dist = self.gridifyX(
            self.front_pipes[0].x_pos) - self.gridifyX(self.flappy.x_pos)
        return (ground_clearance, x_dist, pipeBottom_hieght, pipeTop_hieght)

    def reset(self):
        self.flappy = Bird(gravity=self.gravity)
        self.back_pipes = []
        self.front_pipes = []
        self.pipeCount = 0
        last_pos = self.screen_width
        for i in range(20):
            self.front_pipes.append(self.createRandomPipe(
                x_pos=last_pos, v_pipe=self.v_pipe, min_gap=self.min_pipe_gap, gap_range=self.gap_range))
            last_pos = last_pos + PipePair.PIPE_THICKNESS + \
                random.randrange(1, 4) * self.MIN_X_GAP
        return self.get_state()

    def summarize_state(self):
        state = self.get_state()
        output = '*'*20 + "\nGround clearance: " + str(state[0])
        output += "\nx distance: " + str(state[1])
        output += "\nbottom clearance: " + str(state[2])
        output += "\ntop clearance: " + str(state[3])
        return output

    def get_qtable(self):
        Q = {}
        for d_x in range(self.x_grid):
            for flapp_y in range(self.y_grid):
                for pipe_top in range(self.gridifyY(self.screen_height-self.min_pipe_gap)+1):
                    for pipe_bottom in range(pipe_top + self.gridifyY(self.min_pipe_gap), min(pipe_top + self.gridifyY(self.min_pipe_gap+self.gap_range), self.gridifyY(self.screen_height)) + 1):
                        Q[(flapp_y, d_x, pipe_bottom, pipe_top)] = {
                            'J': 0, 'F': 0}
        return Q

    def step(self, action, win=None):
        
        if action == 'J':
            self.flappy.jump()
        condition = self.run(int(self.decision_rate//self.t_step), win=win)
        if condition == 'Dead':
            reward = -1
        elif condition == 'Alive':
            reward = 0
        elif condition == 'Success':
            reward = 1
            self.pipeCount += 1

        s_next = self.get_state()
        info = {
            'Pipe Ahead': {
                'Top end acc': self.front_pipes[0].top_end,
                'Bottom end acc': self.front_pipes[0].bottom_end,
                'X dist acc': self.front_pipes[0].x_pos - self.flappy.x_pos,
                'Y height acc': self.flappy.y_pos
            }
        }
        return s_next, reward, condition, info


def drawPipes(env, win, pipe):
    top = pipe_img
    top = pygame.transform.scale(top, (pipe.width, env.roof - pipe.top_end))
    top = pygame.transform.flip(top, ybool=True)
    win.blit(top, (pipe.x_pos, env.roof-env.screen_height))
    bottom = pipe_img
    bottom = pygame.transform.scale(
        bottom, (pipe.width, pipe.bottom_end - env.ground))
    win.blit(bottom, (pipe.x_pos, env.screen_height - pipe.bottom_end))


def get_best_action(env, s, Q):
    if s in Q:
        best_a = None
        best_q = float('-inf')
        for a in env.action_space:
            if Q[s].get(a, 0) > best_q:
                best_a = a
                best_q = Q[s].get(a, 0)
        return best_a
    else:
        return random.choice(env.action_space)


def get_eps_greedy_prob(env, eps, a_best):
    prob = {}
    for a in env.action_space:
        prob[a] = eps/len(env.action_space)
    prob[a_best] += 1 - eps
    return prob


def choose_action(env, s, policy):
    if s in policy:
        return np.random.choice(env.action_space, p=[policy[s][a] for a in env.action_space])
    else:
        return np.random.choice(env.action_space)


def update_action_value(env, s, a, s_next, reward, Q, alpha, gamma):
    if s not in Q:
        Q[s] = {a: 0 for a in env.action_space}
    best_Q_next = Q.get(s_next, {a: 0 for a in env.action_space})[
        get_best_action(env, s_next, Q)]
    Q[s][a] = Q[s].get(a, 0) + alpha * (reward + gamma *
                                        best_Q_next - Q[s].get(a, 0))


def q_learning(env, gamma, eps, alpha, n_iter, win=None):
    Q = {}
    policy = {}
    # s = env.reset()
    s = env.get_state()
    for itr in range(n_iter):
        if itr % (n_iter//100) == 0:
            print((itr*100)/n_iter, '% complete', sep="")
        condition = 'Alive'
        while condition == 'Alive':
            # print('Iter:', itr, '| S:', s)

            a_best = get_best_action(env, s, Q)
            policy[s] = get_eps_greedy_prob(env, eps, a_best)
            a = choose_action(env, s, policy)
            s_next, reward, condition, info = env.step(a, win=win)

            update_action_value(env, s, a, s_next, reward, Q, alpha, gamma)
            s = s_next
            if condition == 'Dead':
                env.reset()
            # print('Loop final', Q)
            # print('*'*70)
    return policy, Q


def play_game(env, Q, win=None):
    env.reset()
    s = env.get_state()
    condition = 'Alive'
    while condition != 'Dead':
        # print(s, Q.get(s, 'New State'))
        a_best = get_best_action(env, s, Q)
        s_next, reward, condition, info = env.step(a_best, win=win)
        s = s_next
        if condition == 'Dead':
            # time.sleep(10)
            # print(s, Q.get(s, 'New State'))
            # print("pipeCount: ", env.pipeCount)
            pipeCount1 = env.pipeCount
            env.reset()
            condition == 'Alive'
            return pipeCount1


def main():
    seed = datetime.now()
    np.random.seed(seed.second)
    random.seed(seed)

    pygame.init()

    env = Environment(decision_rate=0.5, t_step=0.05, gravity=-150, v_pipe=-50,
                      x_grid=20, y_grid=20, fov=300, screen_width=400, screen_height=250, gap_range=0, min_pipe_gap=140)

    win = pygame.display.set_mode((env.screen_width, env.screen_height))

    # policy, q = q_learning(env=env, gamma=1, eps=0.2,
    #                        alpha=0.1, n_iter=10000, win=win)

    n_iterList = [1000,10000,50000,100000]
    pipeCountAveragesList = []
    for i in range(4):
        policy, q = q_learning(env=env, gamma=1, eps=0.4,
                            alpha=0.1, n_iter=n_iterList[i])

        pipeCount = play_game(env, q, win)
        print(pipeCount)

        pipeCountList = []
        for i in range(100):
            pipeCount = play_game(env, q)
            pipeCountList.append(pipeCount)
        print(pipeCountList)
        print(np.mean(pipeCountList))
        # pprint(q, width=150)
        # print(len(q))
        pipeCountAveragesList.append(np.mean(pipeCountList))

    print(pipeCountAveragesList)
    import matplotlib.pyplot as plt
    plt.plot(n_iterList, pipeCountAveragesList)
    plt.show()


if __name__ == "__main__":
    main()
