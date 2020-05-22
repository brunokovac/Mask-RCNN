import matplotlib.pyplot as plt
import numpy as np
import config

def draw_losses_graphs(title, file):
    losses = np.loadtxt(file)

    num_epochs = len(losses)
    epochs = range(1, num_epochs + 1)

    rpn_obj_losses = losses[:, 0]
    rpn_bbox_losses = losses[:, 1]
    mask_rcnn_cls_losses = losses[:, 2]
    mask_rcnn_bbox_losses = losses[:, 3]

    fig, ax = plt.subplots(2, 2)
    #fig.suptitle(title)

    ax[0, 0].set_title("RPN object loss")
    ax[0, 0].plot(epochs, rpn_obj_losses, color="orange")

    ax[0, 1].set_title("RPN bbox loss")
    ax[0, 1].plot(epochs, rpn_bbox_losses, color="orange")

    ax[1, 0].set_title("Mask-RCNN class loss")
    ax[1, 0].plot(epochs, mask_rcnn_cls_losses, color="orange")

    ax[1, 1].set_title("Mask-RCNN bbox loss")
    ax[1, 1].plot(epochs, mask_rcnn_bbox_losses, color="orange")

    plt.tight_layout()
    plt.savefig("{}.png".format(title))
    plt.show()

if __name__ == "__main__":
    draw_losses_graphs("TRAIN LOSSES", config.TRAIN_LOSSES_FILE)
    draw_losses_graphs("VALID LOSSES", config.VALID_LOSSES_FILE)