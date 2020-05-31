import matplotlib.pyplot as plt
import numpy as np

def draw_heatmaps(res_path, save_path):
    #a = np.loadtxt(res_path)
    a = np.random.random(32*32*3)

    plt.imshow(a[::3].reshape(32, 32), cmap='Reds', interpolation='nearest')
    plt.show()
    plt.savefig(save_path + "1.png")

    plt.imshow(a[1::3].reshape(32, 32), cmap='Greens', interpolation='nearest')
    plt.show()
    plt.savefig(save_path + "2.png")

    plt.imshow(a[2::3].reshape(32, 32), cmap='Blues', interpolation='nearest')
    plt.show()
    plt.savefig(save_path + "3.png")

if __name__ == "__main__":
    draw_heatmaps("test1.txt", "res")