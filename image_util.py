import numpy as np
from PIL import Image, ImageDraw
from skimage import io
import config
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def load_image(path):
    img = io.imread(path)
    res = np.zeros((config.IMAGE_SIZE[0], config.IMAGE_SIZE[1], 3))
    res[:img.shape[0], :img.shape[1]] = img
    return res.astype("float32"), img.shape[0], img.shape[1]

def load_mask(path):
    return np.array(Image.open(path))

def load_images(paths):
    images = []

    for path in paths:
        images.append(load_image(path))

    return np.array(images)

def draw_bounding_boxes(save_path, path, bboxes, texts=None):
    img = Image.open(path).convert("RGBA")

    draw = ImageDraw.Draw(img)
    for i in range(len(bboxes)):
        x1, y1, x2, y2 = bboxes[i]
        draw.rectangle(((x1, y1), (x2, y2)), outline="red")
        if texts:
            draw.text((x1, y1), texts[i], fill="white")

    img.save(save_path, "PNG")
    return

def draw_bounding_boxes_from_array(save_path, img, bboxes, texts=None):
    img = img.astype(np.uint8)
    img = Image.fromarray(img).convert("RGBA")

    draw = ImageDraw.Draw(img)
    for i in range(len(bboxes)):
        x1, y1, x2, y2 = bboxes[i]
        draw.rectangle(((x1, y1), (x2, y2)), outline="red")
        if texts:
            draw.text((x1, y1), texts[i], fill="white")

    img.save(save_path, "PNG")
    return

def draw_bounding_boxes_and_masks_from_array(save_path, img, bboxes, masks, texts):
    img = img.astype(np.uint8)
    plt.figure()
    plt.imshow(img)

    mask_img = img.copy()
    for i in range(len(bboxes)):
        color = np.random.uniform(0, 100, 3)
        x1, y1, x2, y2 = bboxes[i]
        x1, y1, x2, y2 = x1.numpy(), y1.numpy(), x2.numpy(), y2.numpy()

        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor="green", linewidth=2)
        plt.gca().add_patch(rect)

        mask = masks[i]
        extracted_mask = mask_img[y1:y2, x1:x2]
        mask_img[y1:y2, x1:x2] = np.where(mask == 1, mask * color, extracted_mask)
        plt.text(x1, y1, texts[i], fontsize=8, color="white", bbox=dict(facecolor='green', alpha=0.7))
        #plt.text(x1, y1, texts[i])

    plt.imshow(mask_img, alpha=0.7)
    save_path = save_path if save_path.endswith(".png") else save_path + ".png"

    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    return

def draw_masks_from_array(save_path, img, bboxes, masks, texts):
    img = img.astype(np.uint8)
    plt.figure()
    plt.imshow(img)

    mask_img = img.copy()
    for i in range(len(bboxes)):
        color = np.array([0, 0, 0])
        x1, y1, x2, y2 = bboxes[i]
        x1, y1, x2, y2 = x1.numpy(), y1.numpy(), x2.numpy(), y2.numpy()

        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor="green", linewidth=2)
        plt.gca().add_patch(rect)

        mask = masks[i]
        extracted_mask = mask_img[y1:y2, x1:x2]
        mask_img[y1:y2, x1:x2] = np.where(mask == 1, mask * color, extracted_mask)

    plt.imshow(mask_img, alpha=0.7)
    save_path = save_path if save_path.endswith(".png") else save_path + ".png"

    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    return

if __name__ == "__main__":
    img, h, w = load_image("dataset/VOC2012/JPEGImages/2007_000032.jpg")

    draw_bounding_boxes_and_masks_from_array("", img[:h, :w], [[50, 10, 200, 200]], [np.random.randint(0, 2, [190, 150])], ["t1"])