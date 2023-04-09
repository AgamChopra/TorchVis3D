# -*- coding: utf-8 -*-
"""
Created on Thu Apr 8 2023

@author: Agamdeep Chopra, achopra4@uw.edu
"""
import pygame
import numpy as np
from torch import tensor, flip, cat, nan_to_num, sum, ones, einsum, sqrt, randint, float64, int64, stack, sort, zeros, max, min
from math import sin, cos, pi
from tqdm import tqdm
from scipy.ndimage import zoom

pygame.init()

T = 1E-2  # step constant.
FPS = 30  # frame per second
SPF = (1/3)*T/FPS  # step per frame
RADIUS = 100
EPSILON = 1E0

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
WIDTH = 1900
HEIGHT = 1080
SCREEN = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
pygame.display.set_caption('PyTorchVis 3D')
FONT = pygame.font.SysFont("Arial", 18, bold=True)
clock = pygame.time.Clock()

CAM, F, TRANS, ROT, BH = [
    int(WIDTH/2), int(HEIGHT/2), 0], 1000, [0, 0, 200], [0., 0., 0.], False


def event_handler(event):
    global TRANS, ROT, CAM, BH
    if event.type == pygame.KEYDOWN:
        if event.key == pygame.K_w:
            TRANS[2] -= 10
        if event.key == pygame.K_s:
            TRANS[2] += 10
        if event.key == pygame.K_a:
            TRANS[0] += 10
        if event.key == pygame.K_d:
            TRANS[0] -= 10
        if event.key == pygame.K_q:
            TRANS[1] -= 10
        if event.key == pygame.K_e:
            TRANS[1] += 10

        if event.key == pygame.K_k:
            ROT[0] -= pi/29.1
        if event.key == pygame.K_i:
            ROT[0] += pi/29.1
        if event.key == pygame.K_l:
            ROT[1] += pi/29.2
        if event.key == pygame.K_j:
            ROT[1] -= pi/29.2
        if event.key == pygame.K_u:
            ROT[2] -= pi/29.3
        if event.key == pygame.K_o:
            ROT[2] += pi/29.3

        if event.key == pygame.K_c:
            if BH:
                BH = False
            else:
                BH = True

        if event.key == pygame.K_b:
            CAM, TRANS, ROT = [int(WIDTH/2), int(HEIGHT/2),
                               0], [0, 0, 100], [0, 0, 0]


def get_scale(R, Z):
    scale = 10 * R / ((Z - CAM[2]))
    if scale < 1:
        scale = 1
    if scale > 100:
        scale = 100
    return scale


def fps_counter():
    flag = int(clock.get_fps())
    c = "RED" if flag < 5 else "YELLOW" if flag < 10 else "GREEN"
    fps = str(flag)
    fps_t = FONT.render(fps, 1, pygame.Color(c))
    SCREEN.blit(fps_t, (0, 0))


def draw_window(arty, lighting):
    SCREEN.fill(lighting)
    _, idx = sort(arty[:, 2], descending=True)
    [pygame.draw.rect(SCREEN, [cell[3], cell[4], cell[5]], (cell[0], cell[1], get_scale(RADIUS, cell[2]), get_scale(
        RADIUS, cell[2]))) if cell[2] > CAM[2] else '' for cell in arty[idx].detach().cpu().numpy()]
    fps_counter()
    pygame.display.update()


def norm(X):
    return (X - min(X))/(max(X) - min(X))


def get_rot_matx(psi, theta, phi):
    r11 = cos(theta) * cos(phi)
    r12 = sin(psi) * sin(theta) * cos(phi) - cos(psi) * sin(phi)
    r13 = cos(psi) * sin(theta) * cos(phi) + sin(psi) * sin(phi)

    r21 = cos(theta) * sin(phi)
    r22 = sin(psi) * sin(theta) * sin(phi) + cos(psi) * cos(phi)
    r23 = cos(psi) * sin(theta) * sin(phi) - sin(psi) * cos(phi)

    r31 = -sin(theta)
    r32 = sin(psi) * cos(theta)
    r33 = cos(psi) * cos(theta)

    return [r11, r12, r13, r21, r22, r23, r31, r32, r33]


def proj_3_to_2(cam_coord, obj_coord, f, trans, rot_ang):
    r = get_rot_matx(rot_ang[0], rot_ang[1], rot_ang[2])

    obj_1 = ones(obj_coord.shape[0], 1).to(dtype=float64).cuda()

    X = cat((obj_coord, obj_1), dim=1).T
    K = tensor(((f, 0, cam_coord[0]), (0, f, cam_coord[1]), (0, 0, 1))).to(
        dtype=float64).cuda()
    E = tensor(((r[0], r[1], r[2], trans[0]), (r[3], r[4], r[5], trans[1]),
               (r[6], r[7], r[8], trans[2]))).to(dtype=float64).cuda()

    X_ = K @ E @ X

    x = X_[0]/X_[2]
    y = X_[1]/X_[2]
    lam = X_[2]

    return stack((x, y, lam), dim=1).cpu()


def acceleration_step_update(x, y, M, counter,):
    '''
    Custom uppdate rules. 
        Ex: Grav dynamics-
            G = 1E-1
            R = sqrt(einsum('ijk, ijk->ij', x-y, x-y))  
            M_ = M @ M.T
            R_ = stack([(x - x[i]) / (R[i].reshape(R.shape[0], 1) + EPSILON) for i in counter],dim=0)

            a = sum(nan_to_num((1/(M * ones((R.shape[0], R.shape[0])).cuda())) * G * (M_/((R ** 2) + EPSILON))).reshape(R.shape[0], R.shape[0], 1) * R_, axis=1)
    '''
    a = 0.
    return a


def dynamics(ringo, color, counter, M, SPF=1/144, WIDTH=600, HEIGHT=600, N=2000):
    x = ringo[:, :3]
    y = x.reshape(x.shape[0], 1, x.shape[1])

    a = acceleration_step_update(x, y, M, counter)

    v = ringo[:, 3:] + (a * SPF)
    x = x + (v * SPF)

    # project 3d to 2d camera plane
    x_ = proj_3_to_2(CAM, x, F, TRANS, ROT)

    # Outputs
    ringo = cat((x, v), axis=1)
    arty = cat((x_.to(dtype=int64), color), axis=1)
    return arty, ringo, M


def color_jet(x):
    r, g, b = int(255 * (((sin((pi * x) - (pi/2)))/2) + .5)), int(255 *
                                                                  (((sin((pi * x)))/2) + .5)), int(255 * (((sin((pi * x) + (pi/2)))/2) + .5))
    return [r, g, b]


def color_gray(x):
    r, g, b = int(255 * (((sin((pi * x) - (pi/2)))/2) + .5)), int(255 *
                                                                  (((sin((pi * x) - (pi/2)))/2) + .5)), int(255 * (((sin((pi * x) - (pi/2)))/2) + .5))
    return [r, g, b]


def norm(x):
    return (x - min(x)) / (max(x) - min(x))


def main(data=norm(flip(tensor(zoom(np.load('R:/img (2).pkl', allow_pickle=True)[0].squeeze(), [.5, .5, .5], order=0)), [2])).cuda(), color_map=color_jet):
    print(data.shape)
    global CAM, HEIGHT, WIDTH
    run = True
    width, depth, height, M, vx, vy, vz, color = [], [], [], [], [], [], [], []

    for i, a in enumerate(tqdm(data)):
        for j, b in enumerate(a):
            for k, c in enumerate(b):
                if c > 0:
                    width.append(i - int(data.shape[0]/2))
                    depth.append(j - int(data.shape[1]/2))
                    height.append(k - int(data.shape[2]/2))
                    color.append(color_map(c))
                    M.append(0)
                    vx.append(0)
                    vy.append(0)
                    vz.append(0)

    N = len(width)
    M = tensor(M).view((N, 1)).to(dtype=float64).cuda()
    vx = tensor(vx).view((N, 1)).to(dtype=float64).cuda()
    vz = tensor(vy).view((N, 1)).to(dtype=float64).cuda()
    vy = tensor(vz).view((N, 1)).to(dtype=float64).cuda()
    width = tensor(width).view((N, 1)).to(dtype=float64).cuda()
    depth = tensor(depth).view((N, 1)).to(dtype=float64).cuda()
    height = tensor(height).view((N, 1)).to(dtype=float64).cuda()
    color = tensor(color).view((N, 3)).to(dtype=int64).cpu()

    ringo = cat((width, height, depth, vx, vy, vz), axis=1).cuda()

    time_step = 0
    lighting = BLACK
    counter = range(N)

    while run:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

            event_handler(event)

        arty, ringo, M = dynamics(
            ringo, color, counter, M=M, WIDTH=WIDTH, HEIGHT=HEIGHT, SPF=SPF, N=N)
        draw_window(arty, lighting)
        time_step += 1

    pygame.quit()


if __name__ == '__main__':
    main(color_map=color_jet)