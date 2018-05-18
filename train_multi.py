"""
Train multi-GPU
"""
import os
import sys
import importlib
import argparse
import json
from datetime import datetime
import numpy as np
import tensorflow as tf
import utils.metric as metric

# Uncomment to shut down TF warnings
# os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"

PARSER = argparse.ArgumentParser()
PARSER.add_argument('--config', type=str, default="config.json", metavar='N',
                    help='config file')
ARGS = PARSER.parse_args()
JSON_DATA_CUSTOM = open(ARGS.config).read()
CUSTOM = json.loads(JSON_DATA_CUSTOM)
JSON_DATA = open('default.json').read()
PARAMS = json.loads(JSON_DATA)

PARAMS.update(CUSTOM)

BATCH_SIZE = PARAMS['batch_size']
NUM_POINT = PARAMS['num_point']
MAX_EPOCH = PARAMS['max_epoch']
BASE_LEARNING_RATE = PARAMS['learning_rate']
GPU_INDEX = PARAMS['gpu']
MOMENTUM = PARAMS['momentum']
OPTIMIZER = PARAMS['optimizer']
DECAY_STEP = PARAMS['decay_step']
DECAY_RATE = PARAMS['learning_rate_decay_rate']
DATASET_NAME = PARAMS['dataset']
INPUT_DROPOUT = PARAMS['input_dropout']
BOX_SIZE = PARAMS['box_size']

# Fix GPU use
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_INDEX)
NUM_GPUS = 1
assert(BATCH_SIZE % NUM_GPUS == 0)
DEVICE_BATCH_SIZE = BATCH_SIZE / NUM_GPUS

# Import model
MODEL = importlib.import_module('models.'+PARAMS['model'])
LOG_DIR = PARAMS['logdir']
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)

# Batch normalisation
BN_INIT_DECAY = PARAMS['bn_init_decay']
BN_DECAY_DECAY_RATE = PARAMS['bn_decay_decay_rate']
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = PARAMS['bn_decay_clip']

# Import dataset
data = importlib.import_module('dataset.' + DATASET_NAME)
TRAIN_DATASET = data.Dataset(npoints=NUM_POINT, split='train', box_size=PARAMS['box_size'], use_color=PARAMS['use_color'],
                             dropout_max=INPUT_DROPOUT, path=PARAMS['data_path']
                             , accept_rate=PARAMS['accept_rate'])
TEST_DATASET = data.Dataset(npoints=NUM_POINT, split='test', box_size=PARAMS['box_size'], use_color=PARAMS['use_color'],
                             dropout_max=INPUT_DROPOUT, path=PARAMS['data_path']
                             , accept_rate=PARAMS['accept_rate'])
NUM_CLASSES = TRAIN_DATASET.num_classes

# Start logging
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')

EPOCH_CNT = 0

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

# update_progress() : Displays or updates a console progress bar
## Accepts a float between 0 and 1. Any int will be converted to a float.
## A value under 0 represents a 'halt'.
## A value at 1 or bigger represents 100%
def update_progress(progress):
    barLength = 10 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = round(float(progress),2)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\rProgress: [{0}] {1}% {2}".format( "#"*block + "-"*(barLength-block), progress*100, status)
    sys.stdout.write(text)
    sys.stdout.flush()

def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.
  Note that this function provides a synchronization point across all towers.
  From tensorflow tutorial: cifar10/cifar10_multi_gpu_train.py
  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    #for g, _ in grad_and_vars:
    for g, v in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(axis=0, values=grads)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads


def get_learning_rate(batch):
    """Compute the learning rate for a given batch size and global parameters
    
    Args:
        batch (tf.Variable): the batch size
    
    Returns:
        scalar tf.Tensor: the decayed learning rate
    """

    learning_rate = tf.train.exponential_decay(
        BASE_LEARNING_RATE,  # Base learning rate.
        batch * BATCH_SIZE,  # Current index into the dataset.
        DECAY_STEP,          # Decay step.
        DECAY_RATE,          # Decay rate.
        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate

def get_bn_decay(batch):
    """Compute the batch normalisation exponential decay
    
    Args:
        batch (tf.Variable): the batch size
    
    Returns:
        scalar tf.Tensor: the batch norm decay
    """
    
    bn_momentum = tf.train.exponential_decay(
    BN_INIT_DECAY,
    batch*BATCH_SIZE,
    BN_DECAY_DECAY_STEP,
    BN_DECAY_DECAY_RATE,
    staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def train():
    """Train the model on the training dataset GPU, and evaluate it on the test dataset
    """
    with tf.Graph().as_default():
        with tf.device('/cpu:0'):
            pointclouds_pl, labels_pl, smpws_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT, hyperparams=PARAMS)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            print (is_training_pl)

            # Note the global_step=batch parameter to minimize. 
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            #batch = tf.Variable(0)
            batch = tf.get_variable('batch', [],
                initializer=tf.constant_initializer(0), trainable=False)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            print ("--- Get training operator")
            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)

            # -------------------------------------------
            # Get model and loss on multiple GPU devices
            # -------------------------------------------
            # Allocating variables on CPU first will greatly accelerate multi-gpu training.
            # Ref: https://github.com/kuza55/keras-extras/issues/21
            print ("--- Get model")
            # Get model  
            pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl, NUM_CLASSES, hyperparams=PARAMS, bn_decay=bn_decay)
            
            tower_grads = []
            pred_gpu = []
            total_loss_gpu = []
            for i in range(NUM_GPUS):
                with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                    with tf.device('/gpu:%d'%(i)), tf.name_scope('gpu_%d'%(i)) as scope:
                        # Evenly split input data to each GPU
                        pc_batch = tf.slice(pointclouds_pl,
                            [i*DEVICE_BATCH_SIZE,0,0], [DEVICE_BATCH_SIZE,-1,-1])
                        label_batch = tf.slice(labels_pl,
                            [i*DEVICE_BATCH_SIZE,0], [DEVICE_BATCH_SIZE,-1])
                        smpws_batch = tf.slice(smpws_pl,
                            [i*DEVICE_BATCH_SIZE,0], [DEVICE_BATCH_SIZE,-1])
                        pred, end_points = MODEL.get_model(pc_batch, is_training_pl, NUM_CLASSES, hyperparams=PARAMS, bn_decay=bn_decay)

                        MODEL.get_loss(pred, label_batch, smpws_batch, end_points)
                        losses = tf.get_collection('losses', scope)
                        total_loss = tf.add_n(losses, name='total_loss')
                        for l in losses + [total_loss]:
                            tf.summary.scalar(l.op.name, l)

                        grads = optimizer.compute_gradients(total_loss)
                        tower_grads.append(grads)

                        pred_gpu.append(pred)
                        total_loss_gpu.append(total_loss)
            
            #print(tower_grads)
            # Merge pred and losses from multiple GPUs
            pred = tf.concat(pred_gpu, 0)
            total_loss = tf.reduce_mean(total_loss_gpu)

            # Get training operator 
            grads = average_gradients(tower_grads)
            train_op = optimizer.apply_gradients(grads, global_step=batch)

            # Compute accuracy
            correct = tf.equal(tf.argmax(pred, 2), tf.to_int64(labels_pl))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE)
            tf.summary.scalar('accuracy', accuracy)

            # Computer mean intersection over union
            mean_intersection_over_union, update_iou_op = tf.metrics.mean_iou(tf.to_int32(labels_pl), tf.to_int32(tf.argmax(pred, 2)), NUM_CLASSES)
            tf.summary.scalar('mIoU', tf.to_float(mean_intersection_over_union))

            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'), sess.graph)

        # Init variables
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer()) # important for mIoU

        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'smpws_pl': smpws_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': total_loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch,
               'end_points': end_points,
               'update_iou': update_iou_op}

        best_acc = -1

        # Train for MAX_EPOCH epochs
        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()

            # Train one epoch
            train_one_epoch(sess, ops, train_writer)

            # Evaluate, save, and compute the accuracy
            if epoch % 5 == 0:
                acc = eval_one_epoch(sess, ops, test_writer) 
            if acc > best_acc:
                best_acc = acc
                save_path = saver.save(sess, os.path.join(LOG_DIR, "best_model_epoch_%03d.ckpt"%(epoch)))
                log_string("Model saved in file: %s" % save_path)

            # Save the variables to disk.
            if epoch % 10 == 0:
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
                log_string("Model saved in file: %s" % save_path)

def train_one_epoch(sess, ops, train_writer):
    """Train one epoch
    
    Args:
        sess (tf.Session): the session to evaluate Tensors and ops
        ops (dict of tf.Operation): contain multiple operation mapped with with strings
        train_writer (tf.FileSaver): enable to log the training with TensorBoard
        compute_class_iou (bool): it takes time to compute the iou per class, so you can disable it here
    """

    is_training = True
    

    num_batches = TRAIN_DATASET.get_num_batches(BATCH_SIZE)

    log_string(str(datetime.now()))
    update_progress(0)
    # Reset metrics
    loss_sum = 0
    confusion_matrix = metric.ConfusionMatrix(NUM_CLASSES)

    # Train over num_batches batches
    for batch_idx in range(num_batches):
        progress = float(batch_idx)/float(num_batches)
        update_progress(round(progress,2))
        batch_data, batch_label, batch_weights = TRAIN_DATASET.next_batch(BATCH_SIZE,True,True)

        # Get predicted labels
        feed_dict = {ops['pointclouds_pl']: batch_data,
                     ops['labels_pl']: batch_label,
                     ops['smpws_pl']: batch_weights,
                     ops['is_training_pl']: is_training,}
        summary, step, _, loss_val, pred_val, _ = sess.run([ops['merged'], ops['step'],
                                                         ops['train_op'], ops['loss'], ops['pred'], ops['update_iou']], feed_dict=feed_dict)
        train_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 2)
        
        # Update metrics
        for i in range(len(pred_val)):
            for j in range(len(pred_val[i])):
                confusion_matrix.count_predicted(batch_label[i][j], pred_val[i][j])
        loss_sum += loss_val
    update_progress(1)    
    log_string('mean loss: %f' % (loss_sum / float(num_batches)))
    log_string("Overall accuracy : %f" %(confusion_matrix.get_overall_accuracy()))
    log_string("Average IoU : %f" %(confusion_matrix.get_average_intersection_union()))
    iou_per_class = confusion_matrix.get_intersection_union_per_class()
    for i in range(1,NUM_CLASSES):
        log_string("IoU of %s : %f" % (TRAIN_DATASET.labels_names[i],iou_per_class[i]))

def eval_one_epoch(sess, ops, test_writer):
    """Evaluate one epoch
    
    Args:
        sess (tf.Session): the session to evaluate tensors and operations
        ops (tf.Operation): the dict of operations
        test_writer (tf.summary.FileWriter): enable to log the evaluation on TensorBoard
    
    Returns:
        float: the overall accuracy computed on the test set
    """

    global EPOCH_CNT

    is_training = False

    num_batches = TEST_DATASET.get_num_batches(BATCH_SIZE)

    # Reset metrics
    loss_sum = 0
    confusion_matrix = metric.ConfusionMatrix(NUM_CLASSES)

    log_string(str(datetime.now()))

    log_string('---- EPOCH %03d EVALUATION ----'%(EPOCH_CNT))

    update_progress(0)

    for batch_idx in range(num_batches):
        progress = float(batch_idx)/float(num_batches)
        update_progress(round(progress,2))
        batch_data, batch_label, batch_weights = TEST_DATASET.next_batch(BATCH_SIZE,False,False)
        
        feed_dict = {ops['pointclouds_pl']: batch_data,
                     ops['labels_pl']: batch_label,
                     ops['smpws_pl']: batch_weights,
                     ops['is_training_pl']: is_training}
        summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
                                                      ops['loss'], ops['pred']], feed_dict=feed_dict)

        test_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 2) # BxN
        
        # Update metrics
        for i in range(len(pred_val)):
            for j in range(len(pred_val[i])):
                confusion_matrix.count_predicted(batch_label[i][j], pred_val[i][j])
        loss_sum += loss_val
    
    update_progress(1)

    iou_per_class = confusion_matrix.get_intersection_union_per_class()

    # Display metrics
    log_string('eval mean loss: %f' % (loss_sum / float(num_batches)))
    log_string("Overall accuracy : %f" %(confusion_matrix.get_overall_accuracy()))
    log_string("Average IoU : %f" %(confusion_matrix.get_average_intersection_union()))
    for i in range(1,NUM_CLASSES):
        log_string("IoU of %s : %f" % (TEST_DATASET.labels_names[i],iou_per_class[i]))
    
    EPOCH_CNT += 5
    return confusion_matrix.get_overall_accuracy()

if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    train()
    LOG_FOUT.close()
