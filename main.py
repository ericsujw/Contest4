import pandas as pd
import pickle
import numpy as np
import random
import tensorflow as tf
import os
import copy
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from tqdm import trange
import scipy

def get_hparas():
    hparas = {
        'TRAIN_COEFF_KL' : 2.0,
        'GAN_EMBEDDING_DIM' : 128,
        'GAN_Z_DIM' : 100,
        'GF_DIM' : 64, 
        'DF_DIM' : 64,
        'BATCH_SIZE' : 32,
        'UNCOND_LOSS_COEFF' : 1.0,
        'LR': 0.002,
        'BETA': 0.5,
        'N_EPOCH': 5,
        'DECAY_EVERY': 100,
        'LR_DECAY': 0.5,
        'N_SAMPLE': 7370,
        'DISPLAY_NUM' : 100
    }
    return hparas

def GLU(input_tensor):
    # tensor shape should be [batch, ... , channels]
    channels_num = input_tensor.get_shape().as_list()[-1]
    #print(channels_num)
    assert channels_num %2 == 0
    channels_num = int(channels_num/2)
    return input_tensor[...,:channels_num] * tf.sigmoid(input_tensor[...,channels_num:])
    

def conv3x3(input_tensor, out_planes):
    # tensor shape should be [batch, height, width, channels]
    "3x3 convolution with padding"
    return tf.layers.conv2d(input_tensor, out_planes, kernel_size=3, strides=1,
                     padding="same", use_bias=False, reuse=False)

def upBlock(input_tensor, out_planes, traing_phase):
    # tensor shape should be [batch, height, width, channels]
    assert isinstance(traing_phase, bool)
    new_height = input_tensor.get_shape().as_list()[1] * 2
    new_width = input_tensor.get_shape().as_list()[2] * 2
    upimg = tf.image.resize_nearest_neighbor(images = input_tensor, 
                                             size = tf.constant([new_height, new_width]))
    convimg = conv3x3(upimg, out_planes *2)
    batch_norm_img = tf.layers.batch_normalization(convimg, 
                                                   momentum = 0.1, 
                                                   epsilon=0.00001, 
                                                   training = traing_phase)
    output_tensor = GLU(batch_norm_img)
    return output_tensor

def resBlock(intput_tensor, out_planes, traing_phase):
    assert isinstance(traing_phase, bool)
    convimg = conv3x3(intput_tensor, out_planes * 2)
    batch_norm_img = tf.layers.batch_normalization(convimg, 
                                                   momentum = 0.1, 
                                                   epsilon=0.00001, 
                                                   training = traing_phase)
    output_tensor = GLU(batch_norm_img)
    convimg2 = conv3x3(output_tensor, out_planes)
    output_tensor = tf.layers.batch_normalization(convimg2, 
                                                   momentum = 0.1, 
                                                   epsilon=0.00001, 
                                                   training = traing_phase)
    output_tensor += intput_tensor
    return output_tensor

def Block3x3_leakRelu(input_tensor, out_planes, traing_phase):
    assert isinstance(traing_phase, bool)
    convimg = conv3x3(input_tensor, out_planes)
    batch_norm_img = tf.layers.batch_normalization(convimg, 
                                                   momentum = 0.1, 
                                                   epsilon=0.00001, 
                                                   training = traing_phase)
    
    return tf.nn.leaky_relu(batch_norm_img)

def downBlock(input_tensor, out_planes, traing_phase):
    assert isinstance(traing_phase, bool)
    convimg = tf.layers.conv2d(input_tensor, out_planes, kernel_size=4, strides=2,
                               padding="same", use_bias=False)
    batch_norm_img = tf.layers.batch_normalization(convimg, 
                                                   momentum = 0.1, 
                                                   epsilon=0.00001, 
                                                   training = traing_phase)
    
    return tf.nn.leaky_relu(batch_norm_img)

def encode_image_by_16times(input_tensor, ndf, traing_phase):
    assert isinstance(traing_phase, bool)
    with tf.variable_scope('encode_image', reuse=False):
        convimg1 = tf.layers.conv2d(input_tensor, ndf, kernel_size=4, strides=2,
                                    padding="same", use_bias=False, name='conv1')
        convimg1 = tf.nn.leaky_relu(convimg1)
        #======#
        convimg2 = tf.layers.conv2d(convimg1, ndf * 2, kernel_size=4, strides=2,
                                    padding="same", use_bias=False, name='conv2')
        batch_norm_img2 = tf.layers.batch_normalization(convimg2, 
                                                       momentum = 0.1, 
                                                       epsilon=0.00001, 
                                                       training = traing_phase)
        batch_norm_img2 = tf.nn.leaky_relu(batch_norm_img2)
        #======#
        convimg3 = tf.layers.conv2d(batch_norm_img2, ndf * 4, kernel_size=4, strides=2,
                                    padding="same", use_bias=False, name='conv3')
        batch_norm_img3 = tf.layers.batch_normalization(convimg3, 
                                                       momentum = 0.1, 
                                                       epsilon=0.00001, 
                                                       training = traing_phase)
        batch_norm_img3 = tf.nn.leaky_relu(batch_norm_img3)
        #======#
        convimg4 = tf.layers.conv2d(batch_norm_img3, ndf * 8, kernel_size=4, strides=2,
                                    padding="same", use_bias=False, name='conv4')
        batch_norm_img4 = tf.layers.batch_normalization(convimg4, 
                                                       momentum = 0.1, 
                                                       epsilon=0.00001, 
                                                       training = traing_phase)
        output_tensor = tf.nn.leaky_relu(batch_norm_img4)
        #======#
        return output_tensor

def KL_loss(mu, log_sigma):
    with tf.name_scope("KL_divergence"):
        loss = -log_sigma + .5 * (-1 + tf.exp(2. * log_sigma) + tf.square(mu))
        loss = tf.reduce_mean(loss)
        return loss
    
    
    

IMAGE_DEPTH = 3

def training_data_generator(embedding, image_path):
    # load in the image according to image path
    imagefile = tf.read_file('dataset' + image_path)
    image = tf.image.decode_image(imagefile, channels=3)
    float_img = tf.image.convert_image_dtype(image, tf.float32)
    float_img.set_shape([None, None, 3])
    image1 = tf.image.resize_images(float_img, size=[64, 64])
    image1.set_shape([64, 64, IMAGE_DEPTH])
    image2 = tf.image.resize_images(float_img, size=[128, 128])
    image2.set_shape([128, 128, IMAGE_DEPTH])
    image3 = tf.image.resize_images(float_img, size=[256, 256])
    image3.set_shape([256, 256, IMAGE_DEPTH])

    return image1, image2, image3 , embedding


def data_iterator(datapath, batch_size, data_generator, train_flag = True):
    # Load the training data into two NumPy arrays
    embedding_filename = ""
    ImagePath_filename = ""
    if train_flag == True:
        ImagePath_filename = os.path.join( datapath , "text2ImgData.pkl")
        embedding_filename = os.path.join( datapath , "train_embedding.npy")
        df = pd.read_pickle(ImagePath_filename)
        image_path = df['ImagePath'].values
    else:
        embedding_filename = os.path.join( datapath , "test_embedding.npy")
        
    embedding = np.load(embedding_filename).astype('float32')
    dataset = None
    if train_flag == True:
        dataset = tf.data.Dataset.from_tensor_slices((embedding, image_path))
        dataset = dataset.map(data_generator, num_parallel_calls=8)
        dataset = dataset.repeat()
        dataset = dataset.batch(batch_size)
    else:
        df_test = pd.read_pickle('dataset/testData.pkl')
        image_path_list = copy.deepcopy(df_test['ID']).tolist()
        dataset = tf.data.Dataset.from_tensor_slices((embedding, image_path_list))
        dataset = dataset.batch(91)
    
    iterator = dataset.make_initializable_iterator()
    output_types = dataset.output_types
    output_shapes = dataset.output_shapes

    return iterator, output_types, output_shapes

class CONDITION:
    """
    Encode Word Enbedding to Conditions
    """

    def __init__(self,
               embedding,
               hparas,
               training_phase=True,
               reuse=False):
        super(CONDITION, self).__init__()
        self.embedding = embedding
        self.hparas = hparas
        self.train = training_phase
        self.reuse = reuse
        self._build_model()
        
        
    def generate_condition(self, embeddings):
        em_flat = tf.layers.flatten(embeddings)
        conditions = tf.layers.dense(em_flat, 
                                self.hparas['GAN_EMBEDDING_DIM'] * 4,
                                kernel_initializer = tf.random_normal_initializer(stddev=0.02))
        conditions = GLU(conditions)
        mean = conditions[:, :self.hparas['GAN_EMBEDDING_DIM']]
        log_sigma = conditions[:, self.hparas['GAN_EMBEDDING_DIM']:]
        return [mean, log_sigma]
        
    def _build_model(self):
        with tf.variable_scope('em2condition', reuse=self.reuse):
            c_mean_logsigma = self.generate_condition(self.embedding)
            mean = c_mean_logsigma[0]
            if self.train:
                epsilon = tf.truncated_normal(tf.shape(mean))
                stddev = tf.exp(tf.multiply(c_mean_logsigma[1], 0.5))
                c = tf.add(mean , tf.multiply(stddev , epsilon))
            else:
                c = mean
                kl_loss = 0
            self.c_code = c
            self.mu = mean
            self.log_var = c_mean_logsigma[1]
            
class INIT_GEN:
    
    def __init__(self, 
                 noise_z,
                 condition, 
                 training_phase, 
                 hparas,
                 ngf,
                 reuse):
        
        self.z = noise_z
        self.c = condition
        self.train = training_phase
        self.hparas = hparas
        self.gf_dim = ngf
        self.reuse = reuse
        self._build_model()

    def _build_model(self):
        with tf.variable_scope('initial_gen', reuse=self.reuse):
            in_code = tf.concat([self.c, self.z],1)
            fc_in_code = tf.layers.dense(in_code, self.gf_dim *4 *4 *2, use_bias=False)
            batch_norm_in_code = tf.layers.batch_normalization(fc_in_code, 
                                                               momentum = 0.1, 
                                                               epsilon=0.00001, 
                                                               training = self.train)
            
            fc_out = GLU(batch_norm_in_code)
            out_code = tf.reshape(fc_out, [tf.shape(fc_out)[0], 4, 4, self.gf_dim])
            out_code = upBlock(out_code,  self.gf_dim // 2, self.train)
            out_code = upBlock(out_code,  self.gf_dim // 4, self.train)
            out_code = upBlock(out_code,  self.gf_dim // 8, self.train)
            out_code = upBlock(out_code,  self.gf_dim // 16, self.train)
            self.out_code = out_code # [-1, 64, 64, self.gf_dim // 16]

class GET_IMAGE_G:
    
    def __init__(self, 
                 h_code,
                 hparas,
                 ngf,
                 scope_name = 'get_image'):
        with tf.variable_scope(scope_name):
            self.h_code = h_code
            self.hparas = hparas
            self.gf_dim = ngf
            conv_img = conv3x3(self.h_code, 3)
            self.outimg = tf.tanh(conv_img)
            
            
class NEXT_STAGE_G:
    def __init__(self, 
                 h_code, 
                 c_code, 
                 training_phase, 
                 hparas,
                 ngf,
                 reuse,
                 scope_name = 'next_stage_gen'):
        
        self.h_code = h_code
        self.c_code = c_code
        self.train = training_phase
        self.hparas = hparas
        self.gf_dim = ngf
        self.reuse = reuse
        self.scope_name = scope_name
        self._build_model()
        
        
    def _build_model(self):
        with tf.variable_scope(self.scope_name):
            # reshape condition from [batch, self.hparas['GAN_EMBEDDING_DIM']] to
            # [batch, 1, 1, self.hparas['GAN_EMBEDDING_DIM']]
            # channel_last for resize
            self.c_code = tf.reshape(self.c_code, [tf.shape(self.c_code)[0], 1, 1, self.hparas['GAN_EMBEDDING_DIM']])
            s_size = self.h_code.get_shape().as_list()[1]
            self.c_code = tf.tile(self.c_code, [1, s_size, s_size, 1])
            h_c_code = tf.concat([self.c_code, self.h_code], -1)
            convimg = conv3x3(h_c_code, self.gf_dim * 2)
            batch_norm_img = tf.layers.batch_normalization(convimg, 
                                                           momentum = 0.1, 
                                                           epsilon=0.00001, 
                                                           training = self.train)
            output_tensor = GLU(batch_norm_img)
            output_tensor = resBlock(output_tensor, self.gf_dim, self.train)
            output_tensor = resBlock(output_tensor, self.gf_dim, self.train)
            output_tensor = upBlock(output_tensor, self.gf_dim // 2, self.train)
            self.output = output_tensor
            
class G_NET:
    def __init__(self, 
                 embedding, 
                 noise_z,
                 training_phase, 
                 hparas,
                 ngf,
                 reuse):
        self.embedding = embedding
        self.noise_z = noise_z
        self.train = training_phase
        self.hparas = hparas
        self.gf_dim = ngf
        self.reuse = reuse
        self._build_model()
        
    def _build_model(self):
        with tf.variable_scope('treegen'):
            
            condition_return = CONDITION(embedding = self.embedding, 
                                         hparas = self.hparas, 
                                         training_phase = self.train, 
                                         reuse = self.reuse)
            self.c_code = condition_return.c_code
            mu = condition_return.mu
            logvar = condition_return.log_var
            
            fake_imgs = []
            init_gan = INIT_GEN(noise_z = self.noise_z,
                               condition = self.c_code, 
                               training_phase = self.train, 
                               hparas = self.hparas,
                               ngf = self.gf_dim  * 16,
                               reuse = self.reuse)
            h_code1 = init_gan.out_code
            get_image_1 = GET_IMAGE_G(h_code = h_code1,
                                    hparas = self.hparas,
                                    ngf = self.gf_dim)
            fake_img1 = get_image_1.outimg
            fake_imgs.append(fake_img1)
            next_stage_2 = NEXT_STAGE_G(h_code = h_code1, 
                                   c_code = self.c_code, 
                                   training_phase = self.train, 
                                   hparas = self.hparas,
                                   ngf = self.gf_dim,
                                   reuse = self.reuse)
            h_code2 = next_stage_2.output
            get_image_2 = GET_IMAGE_G(h_code = h_code2,
                                    hparas = self.hparas,
                                    ngf = self.gf_dim // 2,
                                     scope_name = 'get_image_2')
            fake_img2 = get_image_2.outimg
            fake_imgs.append(fake_img2)
            next_stage_3 = NEXT_STAGE_G(h_code = h_code2, 
                                   c_code = self.c_code, 
                                   training_phase = self.train, 
                                   hparas = self.hparas,
                                   ngf = self.gf_dim // 2,
                                   reuse = self.reuse,
                                       scope_name = 'next_stage_3')
            h_code3 = next_stage_3.output
            get_image_3 = GET_IMAGE_G(h_code = h_code3,
                                    hparas = self.hparas,
                                    ngf = self.gf_dim // 4,
                                     scope_name = 'get_image_3')
            fake_img3 = get_image_3.outimg
            fake_imgs.append(fake_img3)
            self.fake_imgs = fake_imgs
            self.mu = mu
            self.logvar = logvar
            
            
class D_NET64:
    def __init__(self, 
                 x_var, 
                 c_code, 
                 training_phase, 
                 hparas,
                 ngf,
                 reuse,
                 name = '64dis'):
        self.x_var = x_var
        self.c_code = c_code
        self.train = training_phase
        self.hparas = hparas
        self.gf_dim = ngf
        self.reuse = reuse
        self.ndf = self.hparas['DF_DIM']
        self.name = name
        self._build_model()
        
        
    def _build_model(self):
        with tf.variable_scope(self.name):
            x_code = encode_image_by_16times(self.x_var, self.ndf, self.train)
            self.c_code = tf.reshape(self.c_code, [tf.shape(x_code)[0], 1, 1, self.hparas['GAN_EMBEDDING_DIM']])
            self.c_code = tf.tile(self.c_code, [1, 4, 4, 1])
            h_c_code = tf.concat([self.c_code, x_code], -1)
            h_c_code = Block3x3_leakRelu(h_c_code, self.ndf * 8, self.train)
            output = tf.layers.conv2d(h_c_code, 1, kernel_size=4, strides=4)
            #output = tf.sigmoid(output)
            output = tf.reshape(output, [tf.shape(x_code)[0]])
            out_uncond = tf.layers.conv2d(x_code, 1, kernel_size=4, strides=4)
            #out_uncond = tf.sigmoid(out_uncond)
            out_uncond = tf.reshape(out_uncond, [tf.shape(x_code)[0]])
            self.output = [output, out_uncond]
            
            
class D_NET128:
    def __init__(self, 
                 x_var, 
                 c_code, 
                 training_phase, 
                 hparas,
                 ngf,
                 reuse,
                 name = '128dis'):
        self.x_var = x_var
        self.c_code = c_code
        self.train = training_phase
        self.hparas = hparas
        self.gf_dim = ngf
        self.reuse = reuse
        self.ndf = self.hparas['DF_DIM']
        self.name = name
        self._build_model()
        
    def _build_model(self):
        with tf.variable_scope(self.name):
            x_code = encode_image_by_16times(self.x_var, self.ndf, self.train)
            x_code = downBlock(x_code, self.ndf * 16, self.train)
            x_code = Block3x3_leakRelu(x_code, self.ndf * 8, self.train)
            self.c_code = tf.reshape(self.c_code, [tf.shape(x_code)[0], 1, 1, self.hparas['GAN_EMBEDDING_DIM']])
            self.c_code = tf.tile(self.c_code, [1, 4, 4, 1])
            h_c_code = tf.concat([self.c_code, x_code], -1)
            h_c_code = Block3x3_leakRelu(h_c_code, self.ndf * 8, self.train)
            output = tf.layers.conv2d(h_c_code, 1, kernel_size=4, strides=4)
            #output = tf.sigmoid(output)
            output = tf.reshape(output, [tf.shape(x_code)[0]])
            out_uncond = tf.layers.conv2d(x_code, 1, kernel_size=4, strides=4)
            #out_uncond = tf.sigmoid(out_uncond)
            out_uncond = tf.reshape(out_uncond, [tf.shape(x_code)[0]])
            self.output = [output, out_uncond]
            
            
class D_NET256:
    def __init__(self, 
                 x_var, 
                 c_code, 
                 training_phase, 
                 hparas,
                 ngf,
                 reuse,
                 name):
        self.x_var = x_var
        self.c_code = c_code
        self.train = training_phase
        self.hparas = hparas
        self.gf_dim = ngf
        self.reuse = reuse
        self.ndf = self.hparas['DF_DIM']
        self.name = name
        self._build_model()
        
    def _build_model(self):
        with tf.variable_scope(self.name):
            x_code = encode_image_by_16times(self.x_var, self.ndf, self.train)
            x_code = downBlock(x_code, self.ndf * 16, self.train)
            x_code = downBlock(x_code, self.ndf * 32, self.train)
            x_code = Block3x3_leakRelu(x_code, self.ndf * 16, self.train)
            x_code = Block3x3_leakRelu(x_code, self.ndf * 8, self.train)
            self.c_code = tf.reshape(self.c_code, [tf.shape(x_code)[0], 1, 1, self.hparas['GAN_EMBEDDING_DIM']])
            self.c_code = tf.tile(self.c_code, [1, 4, 4, 1])
            h_c_code = tf.concat([self.c_code, x_code], -1)
            h_c_code = Block3x3_leakRelu(h_c_code, self.ndf * 8, self.train)
            output = tf.layers.conv2d(h_c_code, 1, kernel_size=4, strides=4)
            #output = tf.sigmoid(output)
            output = tf.reshape(output, [tf.shape(x_code)[0]])
            out_uncond = tf.layers.conv2d(x_code, 1, kernel_size=4, strides=4)
            #out_uncond = tf.sigmoid(out_uncond)
            out_uncond = tf.reshape(out_uncond, [tf.shape(x_code)[0]])
            self.output =  [output, out_uncond]
            
            
class GAN:
    def __init__(self,
                 hparas,
                 training_phase,
                 dataset_path,
                 ckpt_path,
                 inference_path,
                 sess,
                 global_step,
                 recover=None):

        self.hparas = hparas
        self.train = training_phase
        self.dataset_path = dataset_path  # dataPath+'/text2ImgData.pkl'
        self.ckpt_path = ckpt_path
        self.sample_path = './samples'
        self.inference_path = './inference'
        self.sess = sess
        self.global_step = global_step
        #self._get_session()  # get session
        self._get_train_data_iter()  # initialize and get data iterator
        self._input_layer()  # define input placeholder
        self._get_inference()  # build generator and discriminator
        self._get_loss()  # define gan loss
        self._get_var_with_name()  # get variables for each part of model
        self._optimize()  # define optimizer
        self._init_vars()
        self._get_saver()

        if recover is not None:
            self._load_checkpoint(recover)

    def _get_session(self):
        self.sess = tf.Session()

    def _get_train_data_iter(self):
        if self.train:  # training data iteratot
            iterator_train, types, shapes = data_iterator(self.dataset_path , 
                                                          self.hparas['BATCH_SIZE'],
                                                          training_data_generator, 
                                                          train_flag = True)
            iter_initializer = iterator_train.initializer
            next_element = iterator_train.get_next()
            self.sess.run(iterator_train.initializer)
            self.iterator_train = iterator_train
        else:  # testing data iterator
            iterator_train, types, shapes = data_iterator(self.dataset_path,
                                                          self.hparas['BATCH_SIZE'],
                                                          training_data_generator,
                                                          train_flag = False)
            iter_initializer = iterator_train.initializer
            next_element = iterator_train.get_next()
            self.sess.run(iterator_train.initializer)
            self.iterator_test = iterator_train

    def _input_layer(self):
        if self.train:
            self.real_image1 = tf.placeholder('float32', 
                                              [self.hparas['BATCH_SIZE'], 64, 64, 3], 
                                              name='real_image_64x64')
            self.real_image2 = tf.placeholder('float32', 
                                              [self.hparas['BATCH_SIZE'], 128, 128, 3], 
                                              name='real_image_128x128')
            self.real_image3 = tf.placeholder('float32', 
                                              [self.hparas['BATCH_SIZE'], 256, 256, 3], 
                                              name='real_image_256x256')
            self.embedding = tf.placeholder('float32',
                                            shape=[self.hparas['BATCH_SIZE'], 10, 1024],
                                            name='origin_embedding')
            self.z_noise = tf.placeholder(tf.float32, 
                                          [self.hparas['BATCH_SIZE'], self.hparas['GAN_Z_DIM']],
                                          name='z_noise')
        else:
            self.embedding = tf.placeholder('float32',
                                            shape=[91, 10, 1024],
                                            name='origin_embedding')
            self.z_noise = tf.placeholder(tf.float32, 
                                          [91, self.hparas['GAN_Z_DIM']],
                                          name='z_noise')
    def _get_inference(self):
        if self.train:
            g_net = G_NET(embedding = self.embedding, 
                          noise_z = self.z_noise, 
                          training_phase = self.train, 
                          hparas = self.hparas, 
                          ngf = self.hparas['GF_DIM'], 
                          reuse = False)
            self.fake_imgs = g_net.fake_imgs
            self.mu = g_net.mu
            self.logvar = g_net.logvar
            d_net_64 = D_NET64(x_var = tf.concat([self.fake_imgs[0], self.real_image1], 0),
                               c_code = tf.concat([self.mu, self.mu], 0),
                               training_phase = self.train, 
                               hparas = self.hparas, 
                               ngf = self.hparas['GF_DIM'], 
                               reuse = False,
                               name = '64dis')
            self.dis1 = d_net_64.output
            
            d_net_128 = D_NET128(x_var = tf.concat([self.fake_imgs[1], self.real_image2], 0),
                               c_code = tf.concat([self.mu, self.mu], 0),
                               training_phase = self.train, 
                               hparas = self.hparas, 
                               ngf = self.hparas['GF_DIM'], 
                               reuse = False,
                               name = '128dis')
            self.dis2 = d_net_128.output
            
            d_net_256 = D_NET256(x_var = tf.concat([self.fake_imgs[2], self.real_image3], 0),
                               c_code = tf.concat([self.mu, self.mu], 0),
                               training_phase = self.train, 
                               hparas = self.hparas, 
                               ngf = self.hparas['GF_DIM'], 
                               reuse = False,
                               name = '256dis')
            self.dis3 = d_net_256.output
        else:
            g_net = G_NET(embedding = self.embedding, 
                          noise_z = self.z_noise, 
                          training_phase = self.train, 
                          hparas = self.hparas, 
                          ngf = self.hparas['GF_DIM'], 
                          reuse = False)
            self.fake_imgs = g_net.fake_imgs
            self.mu = g_net.mu
            self.logvar = g_net.logvar
    def _get_loss(self):
        if self.train:
            size_1 = self.dis1[0].get_shape().as_list()
            size_1[0] /= 2
            label_1 = tf.concat([tf.zeros(size_1, tf.float32), tf.ones(size_1, tf.float32)], 0)
            d_loss1_1 = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=self.dis1[0],
                                                        labels=label_1,
                                                        name='d_loss1_1'))
            d_loss1_2 = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=self.dis1[1],
                                                        labels=label_1,
                                                        name='d_loss1_2'))
            self.d_loss1 = d_loss1_1 + d_loss1_2# 64x64 loss
            
            size_2 = self.dis2[0].get_shape().as_list()
            size_2[0] /= 2
            label_2 = tf.concat([tf.zeros(size_2, tf.float32), tf.ones(size_2, tf.float32)], 0)

            d_loss2_1 = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=self.dis2[0],
                                                        labels=label_2,
                                                        name='d_loss2_1'))
            d_loss2_2 = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=self.dis2[1],
                                                        labels=label_2,
                                                        name='d_loss2_2'))
            self.d_loss2 = d_loss2_1 + d_loss2_2 # 128x128 loss
            
            size_3 = self.dis3[0].get_shape().as_list()
            size_3[0] /= 2
            label_3 = tf.concat([tf.zeros(size_3, tf.float32), tf.ones(size_3, tf.float32)], 0)

            d_loss3_1 = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=self.dis3[0],
                                                        labels=label_3,
                                                        name='d_loss3_1'))
            d_loss3_2 = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=self.dis3[1],
                                                        labels=label_3,
                                                        name='d_loss3_2'))
            self.d_loss3 = d_loss3_1 + d_loss3_2 # 256x256 loss
            tf.summary.scalar('d_loss1', self.d_loss1)
            tf.summary.scalar('d_loss2', self.d_loss2)
            tf.summary.scalar('d_loss3', self.d_loss3)
            batch_size1 = int(self.dis1[0].get_shape().as_list()[0] / 2)
            g_loss1_1 = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.slice(self.dis1[0], [0], [batch_size1]),
                                                        labels=tf.ones_like(tf.slice(self.dis1[0], 
                                                                                     [0], 
                                                                                     [batch_size1]), 
                                                                            dtype = tf.float32),
                                                        name='g_loss1_1'))
            g_loss1_2 = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.slice(self.dis1[1], [0], [batch_size1]),
                                                        labels=tf.ones_like(tf.slice(self.dis1[1], 
                                                                                     [0], 
                                                                                     [batch_size1]), 
                                                                            dtype = tf.float32),
                                                        name='g_loss1_2'))
            self.g_loss1 = g_loss1_1 + g_loss1_2 # 64x64 g_loss
            batch_size2 = int(self.dis2[0].get_shape().as_list()[0] / 2)
            g_loss2_1 = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.slice(self.dis2[0], [0], [batch_size2]),
                                                        labels=tf.ones_like(tf.slice(self.dis2[0], 
                                                                                     [0], 
                                                                                     [batch_size2]),
                                                                            dtype = tf.float32),
                                                        name='g_loss2_1'))
            g_loss2_2 = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.slice(self.dis2[1], [0], [batch_size2]),
                                                        labels=tf.ones_like(tf.slice(self.dis2[1], 
                                                                                     [0], 
                                                                                     [batch_size2]), 
                                                                            dtype = tf.float32),
                                                        name='g_loss2_2'))
            self.g_loss2 = g_loss2_1 + g_loss2_2 # 128x128 g_loss
            batch_size3 = int(self.dis3[0].get_shape().as_list()[0] / 2)
            g_loss3_1 = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.slice(self.dis3[0], [0], [batch_size3]),
                                                        labels=tf.ones_like(tf.slice(self.dis3[0], 
                                                                                     [0], 
                                                                                     [batch_size3]), 
                                                                            dtype = tf.float32),
                                                        name='g_loss3_1'))
            g_loss3_2 = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.slice(self.dis3[1], [0], [batch_size3]),
                                                        labels=tf.ones_like(tf.slice(self.dis3[1], 
                                                                                     [0],
                                                                                     [batch_size3]),
                                                                            dtype = tf.float32),
                                                        name='g_loss3_2'))
            self.g_loss3 = g_loss3_1 + g_loss3_2 # 64x64 g_loss
            self.KL_loss = KL_loss(self.mu, self.logvar)
            self.g_total_loss = self.g_loss1 + self.g_loss2 + self.g_loss3 + self.KL_loss
            tf.summary.scalar('KL_loss', self.KL_loss)
            tf.summary.scalar('g_total_loss', self.g_total_loss)
    def _get_var_with_name(self):
        self.generator_vars = tf.trainable_variables(scope='treegen')
        if self.train:
            self.discrim_vars1 = tf.trainable_variables(scope='64dis')
            self.discrim_vars2 = tf.trainable_variables(scope='128dis')
            self.discrim_vars3 = tf.trainable_variables(scope='256dis')

    def _optimize(self):
        if self.train:
            with tf.variable_scope('learning_rate'):
                self.lr_var = tf.Variable(self.hparas['LR'], trainable=False)
            discriminator_optimizer = tf.train.AdamOptimizer(self.lr_var, beta1=0.5, beta2=0.999)
            generator_optimizer = tf.train.AdamOptimizer(self.lr_var, beta1=0.5, beta2=0.999)
            self.d1_optim = discriminator_optimizer.minimize(self.d_loss1, var_list=self.discrim_vars1)
            self.d2_optim = discriminator_optimizer.minimize(self.d_loss2, var_list=self.discrim_vars2)
            self.d3_optim = discriminator_optimizer.minimize(self.d_loss3, var_list=self.discrim_vars3)
            self.g_optim = generator_optimizer.minimize(self.g_total_loss, var_list=self.generator_vars)
    def _init_vars(self):
        # Merge all summaries together
        self.merged_summary = tf.summary.merge_all()
        # Initialize the FileWriter
        self.writer_train = tf.summary.FileWriter('./tensorboard/train')
        self.writer_test = tf.summary.FileWriter('./tensorboard/test')
        self.sess.run(tf.global_variables_initializer())
        self.writer_train.add_graph(self.sess.graph)
        self.writer_test.add_graph(self.sess.graph)

    def _get_saver(self):
        if self.train:
            self.g_saver = tf.train.Saver(var_list=self.generator_vars)
            self.d_saver1 = tf.train.Saver(var_list=self.discrim_vars1)
            self.d_saver2 = tf.train.Saver(var_list=self.discrim_vars2)
            self.d_saver3 = tf.train.Saver(var_list=self.discrim_vars3)
        else:
            self.g_saver = tf.train.Saver(var_list=self.generator_vars)

    def _load_checkpoint(self, recover):
        if self.train:
            self.g_saver.restore(self.sess,
                               self.ckpt_path + 'g_model_' + str(recover) + '.ckpt')
            self.d_saver1.restore(self.sess,
                               self.ckpt_path + 'd_model_1_' + str(recover) + '.ckpt')
            self.d_saver2.restore(self.sess,
                               self.ckpt_path + 'd_model_2_' + str(recover) + '.ckpt')
            self.d_saver3.restore(self.sess,
                               self.ckpt_path + 'd_model_3_' + str(recover) + '.ckpt')
        else:
            self.g_saver.restore(self.sess,
                               self.ckpt_path + 'g_model_' + str(recover) + '.ckpt')
        print('-----success restored checkpoint--------')

    def _save_checkpoint(self, epoch):
        self.g_saver.save(self.sess,
                          self.ckpt_path + 'g_model_' + str(epoch) + '.ckpt')
        self.d_saver1.save(self.sess,
                          self.ckpt_path + 'd_model_1_' + str(epoch) + '.ckpt')
        self.d_saver2.save(self.sess,
                          self.ckpt_path + 'd_model_2_' + str(epoch) + '.ckpt')
        self.d_saver3.save(self.sess,
                          self.ckpt_path + 'd_model_3_' + str(epoch) + '.ckpt')
        print('-----success saved checkpoint--------')
        
    def evaluate(self):
        eval_size = 91
        for i in trange(9):
            embedding_batch, id_x = self.sess.run(self.iterator_test.get_next())
            b_z = np.random.normal(loc=0.0,
                                   scale=1.0, 
                                   size=(eval_size,
                                         self.hparas['GAN_Z_DIM'])).astype(np.float32)
            self.fake_img = self.sess.run(self.fake_imgs,
                                          feed_dict={self.embedding: embedding_batch,
                                                     self.z_noise: b_z})
            for i in range(eval_size):
                name = self.inference_path + '/inference_{:04d}.png'.format(id_x[i])
                out_img = scipy.misc.imresize(self.fake_img[2][i],(64,64))
                scipy.misc.imsave(name, out_img)
        #os.system("cd testing && python inception_score.py ../inference ../score.csv && kg submit ../score.csv")
        print('submit new score!')
        

    def training(self):
        global_step = self.global_step
        for _epoch in trange(self.hparas['N_EPOCH']):
            if _epoch != 0 and (_epoch % self.hparas['DECAY_EVERY'] == 0):
                new_lr_decay = self.hparas['LR_DECAY']**(_epoch // self.hparas['DECAY_EVERY'])
                self.sess.run(tf.assign(self.lr_var, self.hparas['LR'] * new_lr_decay))
                print("new lr %f" % (self.hparas['LR'] * new_lr_decay))
            n_batch_epoch = int(self.hparas['N_SAMPLE'] / self.hparas['BATCH_SIZE'])
            for _step in trange(n_batch_epoch):
                global_step += 1
                image_batch1, image_batch2, image_batch3, embedding_batch = self.sess.run(
                    self.iterator_train.get_next())
                b_z = np.random.normal(loc=0.0,
                                       scale=1.0, 
                                       size=(self.hparas['BATCH_SIZE'],
                                             self.hparas['GAN_Z_DIM'])).astype(np.float32)
                self.discriminator_error1, _ = self.sess.run([self.d_loss1, self.d1_optim],
                                                             feed_dict={self.real_image1: image_batch1,
                                                                        self.real_image2: image_batch2,
                                                                        self.real_image3: image_batch3,
                                                                        self.embedding: embedding_batch,
                                                                        self.z_noise: b_z})
                self.discriminator_error2, _ = self.sess.run([self.d_loss2, self.d2_optim],
                                                             feed_dict={self.real_image1: image_batch1,
                                                                        self.real_image2: image_batch2,
                                                                        self.real_image3: image_batch3,
                                                                        self.embedding: embedding_batch,
                                                                        self.z_noise: b_z})
                self.discriminator_error3, _ = self.sess.run([self.d_loss3, self.d3_optim],
                                                             feed_dict={self.real_image1: image_batch1,
                                                                        self.real_image2: image_batch2,
                                                                        self.real_image3: image_batch3,
                                                                        self.embedding: embedding_batch,
                                                                        self.z_noise: b_z})
                self.generator_error, _ = self.sess.run([self.g_total_loss, self.g_optim],
                                                        feed_dict={self.real_image1: image_batch1,
                                                                   self.real_image2: image_batch2,
                                                                   self.real_image3: image_batch3,
                                                                   self.embedding: embedding_batch,
                                                                   self.z_noise: b_z})
                if global_step % 50 == 0:
                    print("Epoch: [%2d/%2d] [%2d] d_loss1: %.3f, d_loss2: %.3f, d_loss3: %.3f, g_loss: %.3f" \
                          % (_epoch, self.hparas['N_EPOCH'], 
                             global_step,
                             self.discriminator_error1, 
                             self.discriminator_error2, 
                             self.discriminator_error3,
                             self.generator_error))
                if global_step % self.hparas['DISPLAY_NUM'] == 0:
                    summary = self.sess.run(self.merged_summary, feed_dict={self.real_image1: image_batch1,
                                                                       self.real_image2: image_batch2,
                                                                       self.real_image3: image_batch3,
                                                                       self.embedding: embedding_batch,
                                                                       self.z_noise: b_z})
                    self.writer_train.add_summary(summary, global_step)
                    print('write summary complete!')
        self._save_checkpoint(global_step)
        return global_step
    

tf.reset_default_graph()
checkpoint_path = './checkpoint/'
inference_path = './inference'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

global_step = 0

for i in range(120):
    tf.reset_default_graph()
    with tf.Session(config=config) as sess:
        try:
            if global_step==0:
                tmp = None
            else:
                tmp = global_step
            gan = GAN(
                get_hparas(),
                training_phase=True,
                dataset_path='dataset',
                ckpt_path=checkpoint_path,
                inference_path=inference_path,
                sess = sess,
                global_step = global_step,
                recover = tmp)
            global_step = gan.training()
        finally:
            sess.close()
    tf.reset_default_graph()
    with tf.Session(config=config) as sess:
        try:
            if global_step==0:
                tmp = None
            else:
                tmp = global_step
            gan = GAN(
                get_hparas(),
                training_phase=False,
                dataset_path='dataset',
                ckpt_path=checkpoint_path,
                inference_path=inference_path,
                sess = sess,
                global_step = global_step,
                recover = global_step)
            gan.evaluate()
        finally:
            sess.close()