"""
Train rework
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

# Shut down useless TF warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"

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
                             proba_terrain=PARAMS['proba_terrain'], dropout_max=PARAMS['dropout_max'], path=PARAMS['data_path'])
TEST_DATASET = data.Dataset(npoints=NUM_POINT, split='test', box_size=PARAMS['box_size'], use_color=PARAMS['use_color'],
                             proba_terrain=PARAMS['proba_terrain'], dropout_max=PARAMS['dropout_max'], path=PARAMS['data_path'])
NUM_CLASSES = TRAIN_DATASET.num_classes

# Start logging
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')

EPOCH_CNT = 0

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

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
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl, labels_pl, smpws_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            print is_training_pl

            # Note the global_step=batch parameter to minimize. 
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            print "--- Get model and loss"
            # Get model and loss 
            pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl, NUM_CLASSES, hyperparams=PARAMS, bn_decay=bn_decay)
            loss = MODEL.get_loss(pred, labels_pl, smpws_pl, end_points)
            tf.summary.scalar('loss', loss)

            # Compute accuracy
            correct = tf.equal(tf.argmax(pred, 2), tf.to_int64(labels_pl))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE*NUM_POINT)
            tf.summary.scalar('accuracy', accuracy)

            # Computer mean intersection over union
            mean_intersection_over_union, update_iou_op = tf.metrics.mean_iou(tf.to_int32(labels_pl), tf.to_int32(tf.argmax(pred, 2)), NUM_CLASSES)
            tf.summary.scalar('mIoU', tf.to_float(mean_intersection_over_union))

            print "--- Get training operator"
            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)

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
               'loss': loss,
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
    """

    is_training = True

    num_batches = TRAIN_DATASET.get_num_batches(BATCH_SIZE)

    log_string(str(datetime.now()))

    # Reset metrics
    loss_sum = 0
    confusion_matrix = metric.ConfusionMatrix(NUM_CLASSES)

    # Train over num_batches batches
    for batch_idx in range(num_batches):

        batch_data, batch_label, batch_weights = TRAIN_DATASET.next_batch(BATCH_SIZE,True,False)

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

        # Every 10 batches, print metrics and reset them
        if (batch_idx+1)%1 == 0:
            log_string(' -- %03d / %03d --' % (batch_idx+1, num_batches))
            log_string('mean loss: %f' % (loss_sum / 10))
            log_string("Overall accuracy : %f" %(confusion_matrix.get_overall_accuracy()))
            log_string("Average IoU : %f" %(confusion_matrix.get_average_intersection_union()))
            iou_per_class = confusion_matrix.get_intersection_union_per_class()
            for i in range(1,NUM_CLASSES):
                log_string("IoU of %s : %f" % (TRAIN_DATASET.labels_names[i],iou_per_class[i]))
            loss_sum = 0   
            confusion_matrix = metric.ConfusionMatrix(NUM_CLASSES)

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

    for batch_idx in range(num_batches):
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
    
    iou_per_class = confusion_matrix.get_intersection_union_per_class()

    # Display metrics
    log_string('eval mean loss: %f' % (loss_sum / float(num_batches)))
    log_string("Overall accuracy : %f" %(confusion_matrix.get_overall_accuracy()))
    log_string("Average IoU : %f" %(confusion_matrix.get_average_intersection_union()))
    for i in range(1,NUM_CLASSES):
        log_string("IoU of %s : %f" % (data.LABELS_NAMES[i],iou_per_class[i]))
    
    EPOCH_CNT += 5
    return confusion_matrix.get_overall_accuracy()

if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    train()
    LOG_FOUT.close()
