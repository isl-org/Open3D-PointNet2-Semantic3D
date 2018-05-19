import os.path
import sys
ROOT_DIR = os.path.abspath(os.path.pardir)
sys.path.append(ROOT_DIR)

import tensorflow as tf
import numpy as np
import utils.tf_util as tf_util
from utils.pointnet_util import pointnet_sa_module, pointnet_fp_module

def placeholder_inputs(batch_size, num_point, hyperparams):
    feature_size = 3*int(hyperparams['use_color'])+int(hyperparams['use_z_feature'])
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3+feature_size))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size, num_point))
    smpws_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point))
    return pointclouds_pl, labels_pl, smpws_pl


def get_model(point_cloud, is_training, num_class, hyperparams, bn_decay=None):
    """ Semantic segmentation PointNet, input is BxNx3, output Bxnum_class """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}

    if hyperparams['use_color'] or hyperparams['use_z_feature']:
        feature_size = 3*int(hyperparams['use_color'])+int(hyperparams['use_z_feature'])
        l0_xyz = tf.slice(point_cloud, [0,0,0], [-1,-1,3])
        l0_points = tf.slice(point_cloud, [0,0,3], [-1,-1,feature_size])
    else:
        l0_xyz = point_cloud
        l0_points = None
    end_points['l0_xyz'] = l0_xyz

    # Layer 1
    l1_xyz, l1_points, l1_indices = pointnet_sa_module(l0_xyz, l0_points, npoint=hyperparams['l1_npoint'], radius=hyperparams['l1_radius'], nsample=hyperparams['l1_nsample'], mlp=[32,32,64], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer1')
    l2_xyz, l2_points, l2_indices = pointnet_sa_module(l1_xyz, l1_points, npoint=hyperparams['l2_npoint'], radius=hyperparams['l2_radius'], nsample=hyperparams['l2_nsample'], mlp=[64,64,128], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer2')
    l3_xyz, l3_points, l3_indices = pointnet_sa_module(l2_xyz, l2_points, npoint=hyperparams['l3_npoint'], radius=hyperparams['l3_radius'], nsample=hyperparams['l3_nsample'], mlp=[128,128,256], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer3')
    l4_xyz, l4_points, l4_indices = pointnet_sa_module(l3_xyz, l3_points, npoint=hyperparams['l4_npoint'], radius=hyperparams['l4_radius'], nsample=hyperparams['l4_nsample'], mlp=[256,256,512], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer4')

    # Feature Propagation layers
    l3_points = pointnet_fp_module(l3_xyz, l4_xyz, l3_points, l4_points, [256,256], is_training, bn_decay, scope='fa_layer1')
    l2_points = pointnet_fp_module(l2_xyz, l3_xyz, l2_points, l3_points, [256,256], is_training, bn_decay, scope='fa_layer2')
    l1_points = pointnet_fp_module(l1_xyz, l2_xyz, l1_points, l2_points, [256,128], is_training, bn_decay, scope='fa_layer3')
    l0_points = pointnet_fp_module(l0_xyz, l1_xyz, l0_points, l1_points, [128,128,128], is_training, bn_decay, scope='fa_layer4')

    # FC layers
    net = tf_util.conv1d(l0_points, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
    end_points['feats'] = net 
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp1')
    net = tf_util.conv1d(net, num_class, 1, padding='VALID', activation_fn=None, scope='fc2')

    return net, end_points


#For get_loss I added the end_points parameter. Like in pointnet2_cls_ssg.py, it's not used in the function.
def get_loss(pred, label, smpw, end_points): 
    """ pred: BxNxC, #one score per class per batch element (N is the nb of points)
        label: BxN,  #one label per batch element
	smpw: BxN """
    classify_loss = tf.losses.sparse_softmax_cross_entropy(labels=label, logits=pred, weights=smpw)
    tf.summary.scalar('classify loss', classify_loss)
    tf.add_to_collection('losses', classify_loss)
    return classify_loss

