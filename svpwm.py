import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


def abc_to_dq():
    theta = np.linspace(0, 4 * np.pi, 3000)
    sin_cos = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    sin_cos = np.transpose(sin_cos, (2, 0, 1))
    # 频率为1，变换后为直线，频率大于1，变换后为频率减少1的正弦函数
    a = np.cos(theta)
    b = np.cos(theta - 2 / 3 * np.pi)
    c = np.cos(theta + 2 / 3 * np.pi)

    # 3D数组的维度(2, 1, 3000)表示2个1*3000的矩阵，在计算中转换成(3000, 2, 1)
    alpha_beta = 2 / 3 * np.array([[1, -0.5, -0.5], [0, np.sqrt(3) / 2, -np.sqrt(3) / 2]]) @ np.transpose(
        np.array([[a], [b], [c]]), (2, 0, 1))

    dq = sin_cos @ alpha_beta
    xd = dq[:, 0, :]
    xq = dq[:, 1, :]

    fig, ax = plt.subplots()
    ax.plot(theta, xd, label='xq')
    ax.plot(theta, xq, label='xd')

    ax.legend(handlelength=4)  # 不加这行没图例
    plt.show()


def dq_to_abc():
    N = 3000
    theta = np.linspace(0, 4 * np.pi, N)
    xd = np.ones(N)
    xq = np.zeros(N)

    # abc为倍频正弦函数
    # xd = np.cos(theta)
    # xq = np.sin(theta)
    dq = np.array([[xd], [xq]])

    sin_cos = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    alpha_beta = np.transpose(sin_cos, (2, 0, 1)) @ np.transpose(dq, (2, 0, 1))

    abc = np.array([[1, 0], [-0.5, np.sqrt(3) / 2], [-0.5, -np.sqrt(3) / 2]]) @ alpha_beta
    a = abc[:, 0, :]
    b = abc[:, 1, :]
    c = abc[:, 2, :]

    fig, ax = plt.subplots()
    ax.plot(theta, a, label='a')
    ax.plot(theta, b, label='b')
    ax.plot(theta, c, label='c')
    ax.legend(handlelength=4)
    plt.show()


def svpwm():
    n = 3000
    theta = np.linspace(-np.pi / 2, 4 * np.pi, n)
    sin_cos = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    sin_cos = np.transpose(sin_cos, (2, 0, 1))

    a = 350 * np.cos(theta)
    b = 350 * np.cos(theta - 2 / 3 * np.pi)
    c = 350 * np.cos(theta + 2 / 3 * np.pi)
    vdc = 700

    alpha_beta = 2 / 3 * np.array([[1, -0.5, -0.5], [0, np.sqrt(3) / 2, -np.sqrt(3) / 2]]) @ np.transpose(
        np.array([[a], [b], [c]]), (2, 0, 1))

    dq = sin_cos @ alpha_beta
    xd = dq[:, 0, :]
    xq = dq[:, 1, :]

    alpha = alpha_beta[:, 0, :]
    beta = alpha_beta[:, 1, :]

    mag = np.sqrt(np.square(alpha) + np.square(beta))
    sector = np.ceil(np.remainder(np.arctan2(beta, alpha), 2 * np.pi) / np.pi * 3)
    sector[sector == 0] = 6
    sector_theta = np.remainder(np.arctan2(beta, alpha), 2 * np.pi) - (sector - 1) * np.pi / 3
    tx = np.sqrt(3) * np.multiply(np.sin(np.pi / 3 - sector_theta).reshape(n, 1), mag) / vdc
    ty = np.sqrt(3) * np.sin(sector_theta).reshape(n, 1) * mag / vdc
    t0 = 1 - tx - ty

    va = []
    vb = []
    vc = []

    for i in range(n):
        if sector[i] == 1:
            va.append((tx[i] + ty[i] + t0[i] / 2)[0])
            vb.append((ty[i] + t0[i] / 2)[0])
            vc.append((t0[i] / 2)[0])
        elif sector[i] == 2:
            va.append((tx[i] + t0[i] / 2)[0])
            vb.append((tx[i] + ty[i] + t0[i] / 2)[0])
            vc.append((t0[i] / 2)[0])
        elif sector[i] == 3:
            va.append((t0[i] / 2)[0])
            vb.append((tx[i] + ty[i] + t0[i] / 2)[0])
            vc.append((ty[i] + t0[i] / 2)[0])
        elif sector[i] == 4:
            va.append((t0[i] / 2)[0])
            vb.append((tx[i] + t0[i] / 2)[0])
            vc.append((tx[i] + ty[i] + t0[i] / 2)[0])
        elif sector[i] == 5:
            va.append((ty[i] + t0[i] / 2)[0])
            vb.append((t0[i] / 2)[0])
            vc.append((tx[i] + ty[i] + t0[i] / 2)[0])
        elif sector[i] == 6:
            va.append((tx[i] + ty[i] + t0[i] / 2)[0])
            vb.append((t0[i] / 2)[0])
            vc.append((tx[i] + t0[i] / 2)[0])

    signal_a = []
    signal_b = []
    signal_c = []

    triangle = signal.sawtooth(200 * np.pi * theta, 0.5)
    for i in range(n):
        signal_a.append(int(va[i] > triangle[i]))
        signal_b.append(int(vb[i] > triangle[i]))
        signal_c.append(int(vc[i] > triangle[i]))


if __name__ == '__main__':
    dq_to_abc()
