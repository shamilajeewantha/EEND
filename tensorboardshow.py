import tensorboard
import tensorflow as tf

# Set the log directory path
log_dir = "./tensorboard"

# Start TensorBoard
tensorboard.notebook.start("--logdir " + log_dir)
