from colossalai.amp import AMP_TYPE

# hyperparameters
# BATCH_SIZE is as per GPU
# global batch size = BATCH_SIZE x data parallel size
BATCH_SIZE = 100
LEARNING_RATE = 3e-3
WEIGHT_DECAY = 0.3
NUM_EPOCHS = 10
WARMUP_EPOCHS = 2
NW = 8

# model config
IMG_SIZE = 224
PATCH_SIZE = 16
HIDDEN_SIZE = 384
DEPTH = 12
NUM_HEADS = 6
MLP_RATIO = 4
NUM_CLASSES = 10
CHECKPOINT = False

USE_DDP = True
TP_WORLD_SIZE = 2
TP_TYPE = 'row'
parallel = dict(tensor=dict(mode="1d", size=TP_WORLD_SIZE),)

fp16 = dict(mode=AMP_TYPE.NAIVE)
clip_grad_norm = 1.0
gradient_accumulation = 8

LOG_PATH = "./log"