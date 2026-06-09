# Dataset config
TRAIN_PATH = ""
DEV_PATH = ""
TEST_PATH = ""

# Tokenizer config
VOCAB_SIZE = 32000
SPM_MODEL_PATH = ""
PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3

# Training config
BATCH_SIZE = 32
NUM_EPOCHS = 10
LAST_CHECKPOINT_PATH = "last_checkpoint.pt"
BEST_MODEL_PATH = "best_model.pt"
LEARNING_RATE = 3e-4
RESUME_TRAINING = False

# Warmup config
WARMUP_STEPS = 4000  # number of warmup steps for LR scheduler

# Model Config
TYPE="seq2seq"
MODEL_DIM = 256
HEAD_DIM = 8
EXPANSION_FACTOR = 2
NUM_LAYERS = 4

# Infer config
MAX_NEW_TOKENS = 150
PREDS_PATH = "preds.csv"

# Beam search config
BEAM_SIZE = 4           # number of beams (1 = greedy)
BEAM_LENGTH_PENALTY = 0.6  # alpha in length penalty: score /= len(seq)^alpha
