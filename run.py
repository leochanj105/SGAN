import os
import tensorflow as tf
from model import SpatialGAN
flags = tf.app.flags
flags.DEFINE_float("learning_rate", 0.0005, "Adam alpha")
flags.DEFINE_float("beta1", 0.5, "Adam beta1")
flags.DEFINE_float("beta2", 0.999, "Adam beta2")
flags.DEFINE_float("bn_epsilon", 0.00005, "bn_epsilon")
flags.DEFINE_float("bn_momentum", 0.9, "bn_momentum")
flags.DEFINE_float("init_std", 0.02, "initialization std")
flags.DEFINE_integer("batch_size", 32, "batch_size")
flags.DEFINE_integer("spatial_size", 9, "spatial size")
flags.DEFINE_integer("z_dim", 100, "noise dimension")
flags.DEFINE_integer("iters", 2001, "iterations")
flags.DEFINE_integer("kernel_size", 5, "CNN kernel size")
flags.DEFINE_integer("img_interval", 10, "image saving interval")
flags.DEFINE_integer("model_save_interval", 500, "model_save interval")
flags.DEFINE_integer("depth", 5, "depth of network")
flags.DEFINE_integer("num_samples", 1, "num of samples")
flags.DEFINE_integer("sample_width", 36, "w of samples")
flags.DEFINE_integer("sample_height", 36, "h of samples")
flags.DEFINE_integer("max_to_keep", 5, "max ckpts to store")
flags.DEFINE_string("dataset_dir", "./dataset", "dateset dir")
flags.DEFINE_string("model_dir", "./models", "model dir")
flags.DEFINE_string("sample_dir", "./samples", "sample dir")
flags.DEFINE_boolean("training", True, "training flag")
flags.DEFINE_boolean("reshape", True, "reshape")
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config = config) as sess:
    gan = SpatialGAN(sess)
    if flags.FLAGS.training:
        if not os.path.exists(flags.FLAGS.model_dir):
            os.makedirs(flags.FLAGS.model_dir)
        if not os.path.exists(flags.FLAGS.sample_dir):
            os.makedirs(flags.FLAGS.sample_dir)
        gan.setup()
        gan.train()
    else:
        gan.generate()
