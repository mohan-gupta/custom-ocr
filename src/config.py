DATA_PATH = "../data"
DEVICE = "cuda"
PAD_VAL = 0

IMAGE_HEIGHT = 75
IMAGE_WIDTH = 300

MAX_LEN = 96

TRAIN_BATCH_SIZE = 64
VALID_BATCH_SIZE = 16
TEST_BATCH_SIZE = 16

LR = 1e-3
DR = 0.03

NUM_CLASSES = 95

EPOCHS = 10

LABEL_MAPPER = "../model/label_mapper.bin"
RECOGNIZER_PATH = "../model/model.bin"