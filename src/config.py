# Epsilon fuzz factor as used by keras backend, see
# https://github.com/keras-team/keras/blob/ceabd61f89572a4b83c0348a96c5a8fd8af25b87/keras/src/backend/config.py#L9
EPSILON = 1e-8

# Hyperparameters for network training
EPOCHS = 150
LEARNING_RATE = 0.001
LEARNING_RATE_HALVE_AFTER = 5
BATCH_SIZE = 16
CHANCE_OF_ALTERING_DATA = 0.8
PATIENCE = 15
MIN_DELTA_REL = 0.005

# Logging
LOG_FILE = "learning.log"

# For debug
DEBUGGING = False
