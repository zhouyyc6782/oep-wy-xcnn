import matplotlib.pyplot as plt
import numpy as np


def load_p(fn):
    coords = np.load(fn[0])
    values = np.load(fn[1])

    k = 0
    for coord in coords:
        if abs(coord[0]) < 1.e-8 and abs(coord[1]) < 1.e-8:
            k += 1
    print k
    d = np.zeros((k, 2))

    k = 0
    for i, coord in enumerate(coords):
        if abs(coord[0]) < 1.e-8 and abs(coord[1]) < 1.e-8:
            d[k][0] = coord[2]
            if values.ndim == 2:
                d[k][1] = values[i][-1]
            else:
                d[k][1] = values[i]
            k += 1

    d = np.array(sorted(d.tolist(), key=lambda a:a[0]))
    return d


def draw(ax, d, label, ls="-"):
    ax.plot(d[:, 0], d[:, 1], ls=ls, label=label)


def main():
    vref = load_p(["d0704_single_ll_L3_0.9_9_coords.npy", "d0704_single_ll_L3_0.9_9.npy"])
    w1 = load_p(["coords_nn.npy", "xc_w_iter_01.npy"])
    w2 = load_p(["coords_nn.npy", "xc_w_iter_02.npy"])
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    draw(ax, vref, "vref", ls="-")
    draw(ax, w1, "w1", ls="--")
    draw(ax, w2, "w2", ls="-.")

    ax.grid(ls="--")
    ax.legend()

    plt.show()


if __name__ == "__main__":
    main()



