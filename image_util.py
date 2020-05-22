import numpy as np
from PIL import Image, ImageDraw
from skimage import io

def load_image(path):
    img = io.imread(path)
    res = np.zeros((512, 512, 3))
    res[:img.shape[0], :img.shape[1]] = img
    return res.astype("float32"), img.shape[0], img.shape[1]

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
            draw.text((x1, y1), texts[i], fill="black")

    img.save(save_path, "PNG")
    return

def draw_bounding_boxes_from_array(save_path, img, bboxes, texts=None):
    img = Image.fromarray(img).convert("RGBA")

    draw = ImageDraw.Draw(img)
    for i in range(len(bboxes)):
        x1, y1, x2, y2 = bboxes[i]
        draw.rectangle(((x1, y1), (x2, y2)), outline="red")
        if texts:
            draw.text((x1, y1), texts[i], fill="black")

    img.save(save_path, "PNG")
    return

if __name__ == "__main__":
    img, h, w = load_image("dataset/VOC2012/JPEGImages/2007_000032.jpg")

    draw_bounding_boxes("dataset/VOC2012/JPEGImages/2007_000032.jpg", [[0, 0, 100, 20], [200, 200, 220, 230]], ["text1", "text2"])

    i2 = Image.fromarray(img.astype(np.uint8)).convert("RGBA")
    i2.show()