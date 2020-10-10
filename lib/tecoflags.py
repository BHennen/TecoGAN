import tensorflow as tf
Flags = tf.app.flags

Flags.DEFINE_integer('rand_seed', 1 , 'random seed' )

# Directories
Flags.DEFINE_string('input_dir_LR', None, 'The directory of the input resolution input data, for inference mode')
Flags.DEFINE_integer('input_dir_len', -1, 'length of the input for inference mode, -1 means all')
Flags.DEFINE_string('input_dir_HR', None, 'The directory of the input resolution input data, for inference mode')
Flags.DEFINE_string('mode', 'inference', 'train, or inference')
Flags.DEFINE_string('output_dir', None, 'The output directory of the checkpoint')
Flags.DEFINE_string('output_pre', '', 'The name of the subfolder for the images')
Flags.DEFINE_string('output_name', 'output', 'The pre name of the outputs')
Flags.DEFINE_string('output_ext', 'jpg', 'The format of the output when evaluating')
Flags.DEFINE_string('summary_dir', None, 'The dirctory to output the summary')

# Models
Flags.DEFINE_string('checkpoint', None, 'If provided, the weight will be restored from the provided checkpoint')
Flags.DEFINE_integer('num_resblock', 16, 'How many residual blocks are there in the generator')
# Models for training
Flags.DEFINE_boolean('pre_trained_model', False, 'If True, the weight of generator will be loaded as an initial point'
                                                     'If False, continue the training')
Flags.DEFINE_string('vgg_ckpt', None, 'path to checkpoint file for the vgg19')

# Machine resources
Flags.DEFINE_string('cudaID', '0', 'CUDA devices')
Flags.DEFINE_integer('queue_thread', 6, 'The threads of the queue (More threads can speedup the training process.')
Flags.DEFINE_integer('name_video_queue_capacity', 512, 'The capacity of the filename queue (suggest large to ensure'
                                                  'enough random shuffle.')
Flags.DEFINE_integer('video_queue_capacity', 256, 'The capacity of the video queue (suggest large to ensure'
                                                   'enough random shuffle')
Flags.DEFINE_integer('video_queue_batch', 2, 'shuffle_batch queue capacity')
                                                   
# Training details
# The data preparing operation
Flags.DEFINE_integer('RNN_N', 10, 'The number of the rnn recurrent length')
Flags.DEFINE_integer('batch_size', 4, 'Batch size of the input batch')
Flags.DEFINE_boolean('flip', True, 'Whether random flip data augmentation is applied')
Flags.DEFINE_boolean('random_crop', True, 'Whether perform the random crop')
Flags.DEFINE_boolean('movingFirstFrame', True, 'Whether use constant moving first frame randomly.')
Flags.DEFINE_integer('crop_size', 32, 'The crop size of the training image')
# Training data settings
Flags.DEFINE_string('input_video_dir', '', 'The directory of the video input data, for training')
Flags.DEFINE_string('input_video_pre', 'scene', 'The pre of the directory of the video input data')
Flags.DEFINE_integer('str_dir', 1000, 'The starting index of the video directory')
Flags.DEFINE_integer('end_dir', 2000, 'The ending index of the video directory')
Flags.DEFINE_integer('end_dir_val', 2050, 'The ending index for validation of the video directory')
Flags.DEFINE_integer('max_frm', 119, 'The ending index of the video directory')
# The loss parameters
Flags.DEFINE_float('vgg_scaling', -0.002, 'The scaling factor for the VGG perceptual loss, disable with negative value')
Flags.DEFINE_float('warp_scaling', 1.0, 'The scaling factor for the warp')
Flags.DEFINE_boolean('pingpang', False, 'use bi-directional recurrent or not')
Flags.DEFINE_float('pp_scaling', 1.0, 'factor of pingpang term, only works when pingpang is True')
# Training parameters
Flags.DEFINE_float('EPS', 1e-12, 'The eps added to prevent nan')
Flags.DEFINE_float('learning_rate', 0.0001, 'The learning rate for the network')
Flags.DEFINE_integer('decay_step', 500000, 'The steps needed to decay the learning rate')
Flags.DEFINE_float('decay_rate', 0.5, 'The decay rate of each decay step')
Flags.DEFINE_boolean('stair', False, 'Whether perform staircase decay. True => decay in discrete interval.')
Flags.DEFINE_float('beta', 0.9, 'The beta1 parameter for the Adam optimizer')
Flags.DEFINE_float('adameps', 1e-8, 'The eps parameter for the Adam optimizer')
Flags.DEFINE_integer('max_epoch', None, 'The max epoch for the training')
Flags.DEFINE_integer('max_iter', 1000000, 'The max iteration of the training')
Flags.DEFINE_integer('display_freq', 20, 'The diplay frequency of the training process')
Flags.DEFINE_integer('summary_freq', 100, 'The frequency of writing summary')
Flags.DEFINE_integer('save_freq', 10000, 'The frequency of saving images')
# Dst parameters
Flags.DEFINE_float('ratio', 0.01, 'The ratio between content loss and adversarial loss')
Flags.DEFINE_boolean('Dt_mergeDs', True, 'Whether only use a merged Discriminator.')
Flags.DEFINE_float('Dt_ratio_0', 1.0, 'The starting ratio for the temporal adversarial loss')
Flags.DEFINE_float('Dt_ratio_add', 0.0, 'The increasing ratio for the temporal adversarial loss')
Flags.DEFINE_float('Dt_ratio_max', 1.0, 'The max ratio for the temporal adversarial loss')
Flags.DEFINE_float('Dbalance', 0.4, 'An adaptive balancing for Discriminators')
Flags.DEFINE_float('crop_dt', 0.75, 'factor of dt crop') # dt input size = crop_size*crop_dt
Flags.DEFINE_boolean('D_LAYERLOSS', True, 'Whether use layer loss from D')

FLAGS = Flags.FLAGS