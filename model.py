import os
import cv2
import numpy as np
import tensorflow as tf
class SpatialGAN(object):
    def __init__(self, sess):
        self.opt = tf.app.flags.FLAGS
        self.kernel_size = self.opt.kernel_size
        self.sess = sess
        self.batch_size = self.opt.batch_size
        self.image_size = (2 ** self.opt.depth) * (self.opt.spatial_size - 1) + 1
        self.image_shape = [self.image_size, self.image_size, 3]
        self.z_dim = [self.opt.spatial_size, self.opt.spatial_size, self.opt.z_dim]
            
    def get_batch(self):
        batch = np.zeros((self.batch_size, self.image_size, self.image_size, 3))
        for i in range(self.batch_size):
            image = self.images[np.random.randint(len(self.images))]
            h = np.random.randint(image.shape[0] - self.image_size)
            w = np.random.randint(image.shape[1] - self.image_size)
            batch[i] = image[h:h + self.image_size, w:w + self.image_size, :]
        return batch
    
    def lrelu(self, x, k = 0.2):
        return tf.maximum(x, k * x)
    def save_image(self, image, filename):
        img = np.array(image)
        img = np.uint8((img + 1.) * 127.5)
        cv2.imwrite(filename, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    def generator(self, z, reuse = False, training = True):
        self.G_layers = []
        self.G_weights =[]
        with tf.variable_scope("generator", reuse = reuse) as scope:
            if training:
                _, h, w, in_channels = [i.value for i in z.get_shape()]
            else:
                shape = tf.shape(z)
                h = tf.cast(shape[1], tf.int32)
                w = tf.cast(shape[2], tf.int32)
            self.G_layers.append(z)
            for i in range(0, self.opt.depth):
                if(i == self.opt.depth - 1):
                    num_filters = 3
                else:
                    num_filters = 2 ** (self.opt.depth - i + 4)
                output_shape = [self.batch_size, (2 ** (i + 1)) * (h - 1) + 1, (2 ** (i + 1)) * (w - 1) + 1, num_filters]
                with tf.variable_scope('G_h' + str(i)):
                    weight = tf.get_variable('W_' + str(i),
                                             [self.kernel_size, self.kernel_size, num_filters, self.G_layers[-1].get_shape()[-1]],
                                             initializer = tf.random_normal_initializer(stddev = self.opt.init_std))
                    deconv = tf.nn.conv2d_transpose(self.G_layers[-1], weight, output_shape = output_shape, 
                                                    strides=[1, 2, 2, 1], padding = 'SAME')
                self.G_weights.append(weight)
                if(i == self.opt.depth - 1):
                    self.G_layers.append(tf.nn.tanh(deconv, name = 'output'))
                else:
                    self.G_layers.append(tf.nn.relu(tf.contrib.layers.batch_norm(deconv, decay = self.opt.bn_momentum,
                                                                                 updates_collections = None, epsilon = self.opt.bn_epsilon,
                                                                                 scale = True, is_training = training,
                                                                                 scope = 'G_bn' + str(i + 1))))
            return self.G_layers[-1]

    def discriminator(self, image, reuse = False):
        self.D_layers = []
        self.D_weights = []
        with tf.variable_scope("discriminator", reuse = reuse) as scope:
            logit = None
            for i in range(0, self.opt.depth):
                if(i == self.opt.depth - 1):
                    num_filters = 1
                else:
                    num_filters = 2 ** (i + 6)
                if(i == 0):
                    in_layer = image
                else:
                    in_layer = self.D_layers[-1]
                with tf.variable_scope('D_h' + str(i) + '_conv'):
                    weight = tf.get_variable('W',
                                            [self.kernel_size, self.kernel_size, in_layer.get_shape()[-1], num_filters],
                                            initializer = tf.truncated_normal_initializer(stddev = self.opt.init_std))

                    conv = tf.nn.conv2d(in_layer, weight, strides = [1, 2, 2, 1], padding = 'SAME')
                if not reuse:
                    self.D_weights.append(weight)
                if(i < self.opt.depth - 2):
                    if(i == 0):
                        self.D_layers.append(self.lrelu(conv))
                    else:
                        self.D_layers.append(self.lrelu(tf.contrib.layers.batch_norm(conv, decay = self.opt.bn_momentum,
                                                                                     updates_collections = None,
                                                                                     epsilon = self.opt.bn_epsilon, scale = True,
                                                                                     is_training = True, scope = 'D_bn' + str(i))))
                else:
                    logit = conv
                    self.D_layers.append(tf.nn.sigmoid(logit))
        return self.D_layers[-1], logit



    def setup(self):
        self.real_images = tf.placeholder(tf.float32, [self.batch_size] + self.image_shape, name='real_images')
        self.noise = tf.placeholder(tf.float32, [self.batch_size] + self.z_dim, name='z')
        self.G_fake = self.generator(self.noise)
        self.D_real, self.D_logits_real = self.discriminator(self.real_images)
        self.D_fake, self.D_logits_fake = self.discriminator(self.G_fake, reuse = True)
        if self.opt.training:
            self.images = []
            files = os.listdir(self.opt.dataset_dir)
            for file in files:
                image = cv2.cvtColor(cv2.imread(os.path.join(self.opt.dataset_dir, file)), cv2.COLOR_BGR2RGB)
                self.images.append(np.array(image)/127.5 - 1.)
            self.D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = self.D_logits_real,
                                              labels = tf.ones_like(self.D_logits_real)))
            self.D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = self.D_logits_fake,
                                              labels = tf.zeros_like(self.D_logits_fake)))
            self.G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = self.D_logits_fake,
                                         labels = tf.ones_like(self.D_logits_fake)))
            self.D_loss = self.D_loss_real + self.D_loss_fake
            variables = tf.trainable_variables()
            self.D_vars = [v for v in variables if 'D_' in v.name]
            self.G_vars = [v for v in variables if 'G_' in v.name]
        self.saver = tf.train.Saver(reshape = self.opt.reshape, max_to_keep = self.opt.max_to_keep)
                                           
    def train(self):
        self.D_optim = tf.train.AdamOptimizer(self.opt.learning_rate, beta1 = self.opt.beta1, beta2 = self.opt.beta2).minimize(self.D_loss, var_list = self.D_vars)
        self.G_optim = tf.train.AdamOptimizer(self.opt.learning_rate, beta1 = self.opt.beta1, beta2 = self.opt.beta2).minimize(self.G_loss, var_list = self.G_vars)
        tf.global_variables_initializer().run()
        sample_noise = np.random.uniform(-1, 1, [self.opt.batch_size] + self.z_dim).astype(np.float32)
        epoch = 1
        print("started...")
        for epoch in range(1, self.opt.iters):
            batch_real = self.get_batch()
            batch_z = np.random.uniform(-1, 1, [self.opt.batch_size] + self.z_dim).astype(np.float32)
            if epoch % 2 != 0:
                _ = self.sess.run([self.D_optim], feed_dict={self.real_images: batch_real, self.noise: batch_z})
            else:
                _ = self.sess.run([self.G_optim], feed_dict={self.noise: batch_z})
            if epoch % self.opt.img_interval == 0:
                save_image(self.G_fake.eval({self.noise: sample_noise})[0], os.path.join(self.opt.sample_dir, "epoch"+ str(epoch) + '.jpg'))
            if epoch % self.opt.model_save_interval == 0:
                self.saver.save(self.sess, os.path.join(self.opt.model_dir, 'model.ckpt'), global_step = epoch)
    def generate(self):
        self.z_dim = [self.opt.sample_height, self.opt.sample_width, self.opt.z_dim]
        self.batch_size = self.opt.num_samples
        self.setup()
        ckpt_name = os.path.basename(tf.train.get_checkpoint_state(self.opt.model_dir).model_checkpoint_path)
        self.saver.restore(self.sess, os.path.join(self.opt.model_dir, ckpt_name))
        z_sample = np.random.uniform(-1, 1, [self.batch_size] + self.z_dim).astype(np.float32)
        samples = self.sess.run(self.G_fake, feed_dict = {self.noise: z_sample})
        for i in range(0, self.batch_size):
            self.save_image(samples[i], 'sample_' + str(i + 1) + '.jpg')