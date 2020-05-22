CLASSES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor"
]

MAX_OBJECTS_PER_IMAGE = 70

POSITIVE_ANCHOR_THRESHOLD = 0.7
NEGATIVE_ANCHOR_THRESHOLD = 0.3
MAX_ANCHORS = 128

TRAIN_PRE_NMS_TOP_N_PER_IMAGE = 5000
TEST_PRE_NMS_TOP_N_PER_IMAGE = 1000

TRAIN_POST_NMS_TOP_N_PER_IMAGE = 1000
TEST_POST_NMS_TOP_N_PER_IMAGE = 500

POSITIVE_ROI_THRESHOLD = 0.5
MAX_ROIS = 200
POSITIVE_ROIS_RATIO = 1/4

IMAGE_SIZE = (512, 512)
ANCHOR_SCALES = [32, 64, 128, 256, 512]
ANCHOR_RATIOS = [(1, 1), (1, 2), (2, 1)]

ROIS_SHAPE = (7, 7)
FPN_NUM_CHANNELS = 256

TEST_NMS_IOU_THRESHOLD = 0.3
TEST_CLASSIFICATION_SCORE_THRESHOLD = 0.8
TEST_MAX_BOXES_PER_CLASS = 10

WEIGHTS_DIR = "ckpts"

TRAIN_LOSSES_FILE = "train-losses.txt"
VALID_LOSSES_FILE = "valid-losses.txt"
