"""
Train
"""
import argparse
import math
from datetime import datetime
import numpy as np
import tensorflow as tf
import importlib
import os
import sys

import utils.provider as provider
import utils.tf_util as tf_util
import utils.pc_util as pc_util
import dataset.semantic as data

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet2_sem_seg', help='Model name [default: pointnet2_sem_seg]')
parser.add_argument('--logdir', default='log', help='Log name [default: log]')
parser.add_argument('--num_point', type=int, default=8192, help='Point Number [default: 8192]')
parser.add_argument('--max_epoch', type=int, default=201, help='Epoch to run [default: 201]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
FLAGS = parser.parse_args()

EPOCH_CNT = 0

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate

MODEL = importlib.import_module('models.'+FLAGS.model) # import network module
LOG_DIR = "logs/"+FLAGS.logdir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)

LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

NUM_CLASSES = 9 # SHOULD NOT BE IN HERE

TRAIN_DATASET = data.SemanticDataset(root="", npoints=NUM_POINT, split='train_short')
TEST_DATASET = data.SemanticDataset(root="", npoints=NUM_POINT, split='test_short')
TEST_DATASET_WHOLE_SCENE = data.SemanticDatasetWholeScene(root="", npoints=NUM_POINT, split='test_short')


def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
        BASE_LEARNING_RATE,  # Base learning rate.
        batch * BATCH_SIZE,  # Current index into the dataset.
        DECAY_STEP,          # Decay step.
        DECAY_RATE,          # Decay rate.
        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
    BN_INIT_DECAY,
    batch*BATCH_SIZE,
    BN_DECAY_DECAY_STEP,
    BN_DECAY_DECAY_RATE,
    staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def train():
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
            pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl, NUM_CLASSES, bn_decay=bn_decay)
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
        sess.run(tf.local_variables_initializer()) # fix for mIoU

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
        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()

            train_one_epoch(sess, ops, train_writer)
            if epoch%5==0:
                #acc = eval_one_epoch(sess, ops, test_writer) 
                acc = eval_whole_scene_one_epoch(sess, ops, test_writer) #it's too long
            if acc > best_acc:
                best_acc = acc
                save_path = saver.save(sess, os.path.join(LOG_DIR, "best_model_epoch_%03d.ckpt"%(epoch)))
                log_string("Model saved in file: %s" % save_path)

            # Save the variables to disk.
            if epoch % 10 == 0:
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
                log_string("Model saved in file: %s" % save_path)

def get_batch_wdp(dataset, idxs, start_idx, end_idx):
    bsize = end_idx-start_idx             #=BATCH_SIZE (default : 32)
    batch_data = np.zeros((bsize, NUM_POINT, 3))
    batch_label = np.zeros((bsize, NUM_POINT), dtype=np.int32)
    batch_smpw = np.zeros((bsize, NUM_POINT), dtype=np.float32)
    for i in range(bsize):
        ps,seg,smpw = dataset[idxs[i+start_idx]]#on parcourt un batch du dataset
        batch_data[i,...] = ps
        batch_label[i,:] = seg
        batch_smpw[i,:] = smpw

        dropout_ratio = np.random.random()*0.875 # 0-0.875
        drop_idx = np.where(np.random.random((ps.shape[0]))<=dropout_ratio)[0]
        batch_data[i,drop_idx,:] = batch_data[i,0,:]
        batch_label[i,drop_idx] = batch_label[i,0]
        batch_smpw[i,drop_idx] *= 0
    return batch_data, batch_label, batch_smpw

def get_batch(dataset, idxs, start_idx, end_idx):
    bsize = end_idx-start_idx
    batch_data = np.zeros((bsize, NUM_POINT, 3))
    batch_label = np.zeros((bsize, NUM_POINT), dtype=np.int32)
    batch_smpw = np.zeros((bsize, NUM_POINT), dtype=np.float32)#sample weights
    for i in range(bsize):
        ps,seg,smpw = dataset[idxs[i+start_idx]]
        batch_data[i,...] = ps
        batch_label[i,:] = seg
        batch_smpw[i,:] = smpw
    return batch_data, batch_label, batch_smpw

def train_one_epoch(sess, ops, train_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = True

    # Shuffle train samples
    train_idxs = np.arange(0, len(TRAIN_DATASET))
    np.random.shuffle(train_idxs)
    num_batches = len(TRAIN_DATASET)/BATCH_SIZE

    log_string(str(datetime.now()))

    total_correct = 0
    total_seen = 0
    loss_sum = 0
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE
        batch_data, batch_label, batch_smpw = get_batch_wdp(TRAIN_DATASET, train_idxs, start_idx, end_idx)
        # Augment batched point clouds by rotation
        #aug_data = provider.rotate_point_cloud_z(batch_data)
        #That function doesn't exist and I think it was supposed to do what this one does
        aug_data = provider.rotate_point_cloud(batch_data)
        feed_dict = {ops['pointclouds_pl']: aug_data,
                     ops['labels_pl']: batch_label,
                     ops['smpws_pl']:batch_smpw,
                     ops['is_training_pl']: is_training,}
        summary, step, _, loss_val, pred_val, _ = sess.run([ops['merged'], ops['step'],
                                                         ops['train_op'], ops['loss'], ops['pred'], ops['update_iou']], feed_dict=feed_dict)
        train_writer.add_summary(summary, step)
#        if batch_idx==0:
#            print(pred_val)
        pred_val = np.argmax(pred_val, 2)
        correct = np.sum(pred_val == batch_label)
        total_correct += correct
        total_seen += (BATCH_SIZE*NUM_POINT)
        loss_sum += loss_val
        if (batch_idx+1)%10 == 0:
            log_string(' -- %03d / %03d --' % (batch_idx+1, num_batches))
            log_string('mean loss: %f' % (loss_sum / 10))
            log_string('accuracy: %f' % (total_correct / float(total_seen)))
            total_correct = 0
            total_seen = 0
            loss_sum = 0

# evaluate on randomly chopped scenes
def eval_one_epoch(sess, ops, test_writer):
    """ ops: dict mapping from string to tf ops """
    global EPOCH_CNT
#    hasPrinted = False
    is_training = False
    test_idxs = np.arange(0, len(TEST_DATASET))
    num_batches = len(TEST_DATASET)/BATCH_SIZE

    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]

    total_correct_vox = 0
    total_seen_vox = 0
    total_seen_class_vox = [0 for _ in range(NUM_CLASSES)]
    total_correct_class_vox = [0 for _ in range(NUM_CLASSES)]

    log_string(str(datetime.now()))
    log_string('---- EPOCH %03d EVALUATION ----'%(EPOCH_CNT))

    labelweights = np.zeros(NUM_CLASSES)
    labelweights_vox = np.zeros(NUM_CLASSES)
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE
        batch_data, batch_label, batch_smpw = get_batch(TEST_DATASET, test_idxs, start_idx, end_idx)

        #aug_data = provider.rotate_point_cloud_z(batch_data)
        #That function doesn't exist and I think it was supposed to do what this one does
        aug_data = provider.rotate_point_cloud(batch_data)
        
        feed_dict = {ops['pointclouds_pl']: aug_data,
                     ops['labels_pl']: batch_label,
                     ops['smpws_pl']: batch_smpw,
                     ops['is_training_pl']: is_training}
        summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
                                                      ops['loss'], ops['pred']], feed_dict=feed_dict)

        test_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 2) # BxN
        correct = np.sum((pred_val == batch_label) & (batch_label>0) & (batch_smpw>0)) # evaluate only on 20 categories but not unknown
        total_correct += correct
        total_seen += np.sum((batch_label>0) & (batch_smpw>0))
        loss_sum += loss_val
        tmp,_ = np.histogram(batch_label,range(NUM_CLASSES+1))
        labelweights += tmp
        for l in range(NUM_CLASSES):
            total_seen_class[l] += np.sum((batch_label==l) & (batch_smpw>0))
            total_correct_class[l] += np.sum((pred_val==l) & (batch_label==l) & (batch_smpw>0))

        for b in xrange(batch_label.shape[0]):
            _, uvlabel, _ = pc_util.point_cloud_label_to_surface_voxel_label_fast(aug_data[b,batch_smpw[b,:]>0,:], np.concatenate((np.expand_dims(batch_label[b,batch_smpw[b,:]>0],1),np.expand_dims(pred_val[b,batch_smpw[b,:]>0],1)),axis=1), res=0.02)
            total_correct_vox += np.sum((uvlabel[:,0]==uvlabel[:,1])&(uvlabel[:,0]>0))
            total_seen_vox += np.sum(uvlabel[:,0]>0)
            tmp,_ = np.histogram(uvlabel[:,0],range(NUM_CLASSES+1))
            labelweights_vox += tmp
            for l in range(NUM_CLASSES):
                total_seen_class_vox[l] += np.sum(uvlabel[:,0]==l)
                total_correct_class_vox[l] += np.sum((uvlabel[:,0]==l) & (uvlabel[:,1]==l))
    print("num_batches ", num_batches, "total_seen_vox ", total_seen_vox, "total_correct_vox ", total_correct_vox)
    log_string('eval mean loss: %f' % (loss_sum / float(num_batches)))
    log_string('eval point accuracy vox: %f'% (total_correct_vox / float(total_seen_vox)))
    log_string('eval point avg class acc vox: %f' % (np.mean(np.array(total_correct_class_vox[1:])/(np.array(total_seen_class_vox[1:],dtype=np.float)+1e-6))))
    log_string('eval point accuracy: %f'% (total_correct / float(total_seen)))
    log_string('eval point avg class acc: %f' % (np.mean(np.array(total_correct_class[1:])/(np.array(total_seen_class[1:],dtype=np.float)+1e-6))))
    labelweights_vox = labelweights_vox[1:].astype(np.float32)/np.sum(labelweights_vox[1:].astype(np.float32))
    #caliweights = np.array([0.388,0.357,0.038,0.033,0.017,0.02,0.016,0.025,0.002,0.002,0.002,0.007,0.006,0.022,0.004,0.0004,0.003,0.002,0.024,0.029])
    #They hard-coded the fact there were 20 classes even here...
    #Maybe someone could try to reimplement it...?
    #log_string('eval point calibrated average acc: %f' % (np.average(np.array(total_correct_class[1:])/(np.array(total_seen_class[1:],dtype=np.float)+1e-6),weights=caliweights)))
    log_string('eval point calibrated average acc: %f' % (np.average(np.array(total_correct_class[1:])/(np.array(total_seen_class[1:],dtype=np.float)+1e-6))))
    per_class_str = 'vox based --------'
    for l in range(1,NUM_CLASSES):
        per_class_str += 'class %d weight: %f, acc: %f; ' % (l,labelweights_vox[l-1],total_correct_class[l]/float(total_seen_class[l]))
    log_string(per_class_str)
    EPOCH_CNT += 1
    return total_correct/float(total_seen)

# evaluate on whole scenes to generate numbers provided in the paper
def eval_whole_scene_one_epoch(sess, ops, test_writer):
    """ ops: dict mapping from string to tf ops """
    global EPOCH_CNT
    is_training = False
    test_idxs = np.arange(0, len(TEST_DATASET_WHOLE_SCENE))
    num_batches = len(TEST_DATASET_WHOLE_SCENE)

    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]

    total_correct_vox = 0
    total_seen_vox = 0
    total_seen_class_vox = [0 for _ in range(NUM_CLASSES)]
    total_correct_class_vox = [0 for _ in range(NUM_CLASSES)]

    log_string(str(datetime.now()))
    log_string('---- EPOCH %03d EVALUATION WHOLE SCENE----'%(EPOCH_CNT))

    labelweights = np.zeros(NUM_CLASSES)
    labelweights_vox = np.zeros(NUM_CLASSES)
    is_continue_batch = False

    extra_batch_data = np.zeros((0,NUM_POINT,3))
    extra_batch_label = np.zeros((0,NUM_POINT))
    extra_batch_smpw = np.zeros((0,NUM_POINT))
    for batch_idx in range(num_batches):
        if not is_continue_batch:
            batch_data, batch_label, batch_smpw = TEST_DATASET_WHOLE_SCENE[batch_idx]
            batch_data = np.concatenate((batch_data,extra_batch_data),axis=0)
            batch_label = np.concatenate((batch_label,extra_batch_label),axis=0)
            batch_smpw = np.concatenate((batch_smpw,extra_batch_smpw),axis=0)
        else:
            batch_data_tmp, batch_label_tmp, batch_smpw_tmp = TEST_DATASET_WHOLE_SCENE[batch_idx]
            batch_data = np.concatenate((batch_data,batch_data_tmp),axis=0)
            batch_label = np.concatenate((batch_label,batch_label_tmp),axis=0)
            batch_smpw = np.concatenate((batch_smpw,batch_smpw_tmp),axis=0)
        if batch_data.shape[0]<BATCH_SIZE:
            is_continue_batch = True
            continue
        elif batch_data.shape[0]==BATCH_SIZE:
            is_continue_batch = False
            extra_batch_data = np.zeros((0,NUM_POINT,3))
            extra_batch_label = np.zeros((0,NUM_POINT))
            extra_batch_smpw = np.zeros((0,NUM_POINT))
        else:
            is_continue_batch = False
            extra_batch_data = batch_data[BATCH_SIZE:,:,:]
            extra_batch_label = batch_label[BATCH_SIZE:,:]
            extra_batch_smpw = batch_smpw[BATCH_SIZE:,:]
            batch_data = batch_data[:BATCH_SIZE,:,:]
            batch_label = batch_label[:BATCH_SIZE,:]
            batch_smpw = batch_smpw[:BATCH_SIZE,:]

        aug_data = batch_data
        feed_dict = {ops['pointclouds_pl']: aug_data,
                     ops['labels_pl']: batch_label,
                     ops['smpws_pl']: batch_smpw,
                     ops['is_training_pl']: is_training}
        summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
                                                      ops['loss'], ops['pred']], feed_dict=feed_dict)
        test_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 2) # BxN
        correct = np.sum((pred_val == batch_label) & (batch_label>0) & (batch_smpw>0)) # evaluate only on 20 categories but not unknown
        total_correct += correct
        total_seen += np.sum((batch_label>0) & (batch_smpw>0))
        loss_sum += loss_val
        tmp,_ = np.histogram(batch_label,range(NUM_CLASSES+1))
        labelweights += tmp
        for l in range(NUM_CLASSES):
            total_seen_class[l] += np.sum((batch_label==l) & (batch_smpw>0))
            total_correct_class[l] += np.sum((pred_val==l) & (batch_label==l) & (batch_smpw>0))

        for b in xrange(batch_label.shape[0]):
            _, uvlabel, _ = pc_util.point_cloud_label_to_surface_voxel_label_fast(aug_data[b,batch_smpw[b,:]>0,:], np.concatenate((np.expand_dims(batch_label[b,batch_smpw[b,:]>0],1),np.expand_dims(pred_val[b,batch_smpw[b,:]>0],1)),axis=1), res=0.02)
            total_correct_vox += np.sum((uvlabel[:,0]==uvlabel[:,1])&(uvlabel[:,0]>0))
            total_seen_vox += np.sum(uvlabel[:,0]>0)
            tmp,_ = np.histogram(uvlabel[:,0],range(NUM_CLASSES+1))
            labelweights_vox += tmp
            for l in range(NUM_CLASSES):
                total_seen_class_vox[l] += np.sum(uvlabel[:,0]==l)
                total_correct_class_vox[l] += np.sum((uvlabel[:,0]==l) & (uvlabel[:,1]==l))

    log_string('eval whole scene mean loss: %f' % (loss_sum / float(num_batches)))
    log_string('eval whole scene point accuracy vox: %f'% (total_correct_vox / float(total_seen_vox)))
    log_string('eval whole scene point avg class acc vox: %f' % (np.mean(np.array(total_correct_class_vox[1:])/(np.array(total_seen_class_vox[1:],dtype=np.float)+1e-6))))
    log_string('eval whole scene point accuracy: %f'% (total_correct / float(total_seen)))
    log_string('eval whole scene point avg class acc: %f' % (np.mean(np.array(total_correct_class[1:])/(np.array(total_seen_class[1:],dtype=np.float)+1e-6))))
    labelweights = labelweights[1:].astype(np.float32)/np.sum(labelweights[1:].astype(np.float32))
    labelweights_vox = labelweights_vox[1:].astype(np.float32)/np.sum(labelweights_vox[1:].astype(np.float32))
    #caliweights = np.array([0.388,0.357,0.038,0.033,0.017,0.02,0.016,0.025,0.002,0.002,0.002,0.007,0.006,0.022,0.004,0.0004,0.003,0.002,0.024,0.029])
    #caliacc = np.average(np.array(total_correct_class_vox[1:])/(np.array(total_seen_class_vox[1:],dtype=np.float)+1e-6),weights=caliweights)
    #see lines 330
    caliacc = np.average(np.array(total_correct_class_vox[1:])/(np.array(total_seen_class_vox[1:],dtype=np.float)+1e-6))    
    log_string('eval whole scene point calibrated average acc vox: %f' % caliacc)

    per_class_str = 'vox based --------'
    for l in range(1,NUM_CLASSES):
        per_class_str += 'class %d weight: %f, acc: %f; ' % (l,labelweights_vox[l-1],total_correct_class_vox[l]/float(total_seen_class_vox[l]))
    log_string(per_class_str)
    EPOCH_CNT += 1
    return caliacc


if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    train()
    LOG_FOUT.close()
