"""
配置文件 - 可以在这里修改超参数
"""

# 数据配置
DATA_ROOT_TRAIN = 'MMAR/train_500'
DATA_ROOT_TEST = 'MMAR/test_200'
VIDEO_LIST_TRAIN = 'MMAR/train_500/train_videofolder_500.txt'
VIDEO_LIST_TEST = 'MMAR/test_200/test_videofolder_200.txt'

# 模型配置
NUM_CLASS = 20
NUM_SEGMENTS = 8
BASE_MODEL = 'resnet18'  # 'resnet18' 或 'resnet50'
FUSION_METHOD = 'late'  # 'early', 'late', 'attention'

# 训练配置
BATCH_SIZE = 4
EPOCHS = 50
LEARNING_RATE = 0.001
VAL_RATIO = 0.2
DROPOUT = 0.5

# 优化器配置
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
LR_STEP_SIZE = 20
LR_GAMMA = 0.1

# 其他配置
SAVE_DIR = 'checkpoints'
GPU_ID = 0
NUM_WORKERS = 4

# 20个行为类别
CLASS_NAMES = [
    'switch light',      # 0
    'up the stairs',     # 1
    'pack backpack',     # 2
    'ride a bike',       # 3
    'turn around',       # 4
    'fold clothes',      # 5
    'hug somebody',      # 6
    'long jump',         # 7
    'move the chair',    # 8
    'open the umbrella', # 9
    'orchestra conducting', # 10
    'rope skipping',     # 11
    'shake hands',       # 12
    'squat',             # 13
    'swivel',            # 14
    'tie shoes',         # 15
    'tie hair',          # 16
    'twist waist',       # 17
    'wear hat',          # 18
    'down the stairs',   # 19
]




