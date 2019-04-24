import math
from collections import namedtuple

import numpy as np
import tensorflow as tf


from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops

import tf_extended as tfe
from tf_extended import tensors as tfe_tensors
from nets import custom_layers
from nets import ssd_common

slim = tf.contrib.slim


# =========================================================================== #
# combine class definition.
# =========================================================================== #
combine_Params = namedtuple('SSDParameters', ['img_shape',
                                         'num_classes',
                                         'no_annotation_label',
                                         'feat_layers',
                                         'feat_shapes',
                                         'allowed_borders',
                                         'anchor_sizes',
                                         'anchor_ratios',
                                         'anchor_steps',
                                         'anchor_offset',
                                         'prior_scaling'
                                         ])


class combine_Net(object):
    """Implementation of the combine VGG-based network.

    The default features layers with 320x320 image input are:
      conv4 ==> 40 x 40
      conv5 ==> 20 x 20
      conv6 ==> 10 x 10
      conv7 ==> 5 x 5

    The default image size used to train this network is 320x320.
    """
    default_params = combine_Params(
        img_shape=(320, 320),
        num_classes=21,
        no_annotation_label=21,
        feat_layers=['block7','block6', 'block5', 'block4'],
        feat_shapes=[(5, 5), (10, 10), (20, 20), (40, 40)],
        allowed_borders = [32, 16, 8, 4],
        anchor_sizes=[(224., 256.),
                      (160., 192.),
                      (96., 128.),
                      (32., 64.)],
        anchor_ratios=[[1, 2, 3, 1./2, 1./3],
                       [1, 2, 3, 1./2, 1./3],
                       [1, 2, 3, 1./2, 1./3],
                       [1, 2, 3, 1./2, 1./3]],
        anchor_steps=[64, 32, 16, 8],
        anchor_offset=0.5,
        prior_scaling=[0.1, 0.1, 0.2, 0.2]#[1., 1., 1., 1.]#
        )

    def __init__(self, params=None):
        """Init the combine net with some parameters. Use the default ones
        if none provided.
        """
        if isinstance(params, combine_Params):
            self.params = params
        else:
            self.params = combine_Net.default_params

    # ======================================================================= #
    def net(self, inputs,
            is_training=True,
            dropout_keep_prob=0.5,
            prediction_fn=slim.softmax,
            reuse=None,
            scope='combine'):
        r = combine_reducedfc(inputs,
                    num_classes=self.params.num_classes,
                    feat_layers=self.params.feat_layers,
                    anchor_sizes=self.params.anchor_sizes,
                    anchor_ratios=self.params.anchor_ratios,
                    is_training=is_training,
                    dropout_keep_prob=dropout_keep_prob,
                    prediction_fn=prediction_fn,
                    reuse=reuse,
                    scope=scope)
        return r

    def arg_scope(self, weight_decay=0.0005, is_training=True, data_format='NHWC'):
        """Network arg_scope.
        """
        return combine_arg_scope(weight_decay, is_training=is_training, data_format=data_format)

    # ======================================================================= #
    def anchors(self, img_shape, dtype=np.float32):
        """Compute the default anchor boxes, given an image shape.
        """
        return combine_anchors_all_layers(img_shape,
                                      self.params.feat_shapes,
                                      self.params.anchor_sizes,
                                      self.params.anchor_ratios,
                                      self.params.anchor_steps,
                                      self.params.anchor_offset,
                                      dtype)

    def bboxes_encode(self, labels, bboxes, anchors, positive_threshold=0.5, ignore_threshold=0.3,
                      scope=None):
        """Encode labels and bounding boxes.
        """
        return ssd_common.tf_ssd_bboxes_encode(
            labels, bboxes, anchors,
            self.params.num_classes,
            self.params.img_shape,
            self.params.allowed_borders,
            self.params.no_annotation_label,
            positive_threshold = positive_threshold,
            ignore_threshold = ignore_threshold,
            prior_scaling = self.params.prior_scaling,
            scope=scope)

    def bboxes_decode(self, feat_localizations, anchors,
                      scope='ssd_bboxes_decode'):
        """Encode labels and bounding boxes.
        """
        return ssd_common.tf_ssd_bboxes_decode(
            feat_localizations, anchors,
            prior_scaling=self.params.prior_scaling,
            scope=scope)
    def bboxes_filter_min(self, scores, bboxes, top_k, minsize=0.03, scope=None):
        """Sort bounding boxes by decreasing order and keep only the top_k.
        If inputs are dictionnaries, assume every key is a different class.
        Assume a batch-type input.

        Args:
          scores: Batch x N Tensor/Dictionary containing float scores.
          bboxes: Batch x N x 4 Tensor/Dictionary containing boxes coordinates.
          top_k: Top_k boxes to keep.
        Return:
          scores, bboxes: Sorted Tensors/Dictionaries of shape Batch x Top_k x 1|4.
        """
        # Dictionaries as inputs.
        if isinstance(scores, dict) or isinstance(bboxes, dict):
            with tf.name_scope(scope, 'bboxes_sort_dict'):
                d_scores = {}
                d_bboxes = {}
                for c in scores.keys():
                    s, b = self.bboxes_filter_min(scores[c], bboxes[c], top_k, minsize=minsize)
                    d_scores[c] = s
                    d_bboxes[c] = b
                return d_scores, d_bboxes

        # Tensors inputs.
        with tf.name_scope(scope, 'bboxes_filter_min', [scores, bboxes]):
            scores, bboxes = tf.squeeze(scores, 0), tf.squeeze(bboxes, 0)
            h = (bboxes[:, 2] - bboxes[:, 0])
            w = (bboxes[:, 3] - bboxes[:, 1])
            mask = tf.greater(w, minsize)
            mask = tf.logical_and(mask, tf.greater(h, minsize))
            # Boolean masking...
            scores = tf.boolean_mask(scores, mask)
            bboxes = tf.boolean_mask(bboxes, mask)

            scores = tfe_tensors.pad_axis(scores, 0, top_k, axis=0)
            bboxes = tfe_tensors.pad_axis(bboxes, 0, top_k, axis=0)

            return tf.expand_dims(scores, axis = 0), tf.expand_dims(bboxes, axis = 0)
    def detected_bboxes(self, predictions, localisations,
                        select_threshold=None, nms_threshold=0.5,
                        clipping_bbox=None, top_k=400, keep_top_k=200):
        # Get the detected bounding boxes from the combine network output.
        # Select top_k bboxes from predictions, and clip
        rscores, rbboxes = \
            ssd_common.tf_ssd_bboxes_select(predictions, localisations,
                                            select_threshold=select_threshold,
                                            num_classes=self.params.num_classes)

        if clipping_bbox is not None:
            rbboxes = tfe.bboxes_clip(clipping_bbox, rbboxes)
        rscores, rbboxes = self.bboxes_filter_min(rscores, rbboxes, top_k)
        rscores, rbboxes = \
            tfe.bboxes_sort(rscores, rbboxes, top_k=top_k)
        # Apply NMS algorithm.
        rscores, rbboxes = \
            tfe.bboxes_nms_batch(rscores, rbboxes,
                                 nms_threshold=nms_threshold,
                                 keep_top_k=keep_top_k)

        return rscores, rbboxes

    def losses(self, logits, localisations, objness_logits, objness_pred,
               gclasses, glocalisations, gscores,
               match_threshold=0.5,
               neg_threshold = 0.3,
               objness_threshold = 0.03,
               negative_ratio=3.,
               alpha=1./3,
               beta=1./3,
               label_smoothing=0.,
               scope='combine_losses'):
        """Define the combine network losses.
        """
        return combine_losses(logits, localisations, objness_logits, objness_pred,
                          gclasses, glocalisations, gscores,
                          match_threshold = match_threshold,
                          neg_threshold = neg_threshold,
                          objness_threshold = objness_threshold,
                          negative_ratio=negative_ratio,
                          alpha=alpha,
                          beta=beta,
                          label_smoothing=label_smoothing,
                          scope=scope)

def combine_anchor_one_layer(img_shape,
                         feat_shape,
                         sizes,
                         ratios,
                         step,
                         offset=0.5,
                         dtype=np.float32):
    """Computer combine default anchor boxes for one feature layer.

    Determine the relative position grid of the centers, and the relative
    width and height.

    Arguments:
      feat_shape: Feature shape, used for computing relative position grids;
      size: Absolute reference sizes;
      ratios: Ratios to use on these features;
      img_shape: Image shape, used for computing height, width relatively to the
        former;
      offset: Grid offset.

    Return:
      y, x, h, w: Relative x and y grids, and height and width.
    """
    y, x = np.mgrid[0:feat_shape[0], 0:feat_shape[1]]
    y = ((y.astype(dtype) + offset) * step) / img_shape[0]
    x = ((x.astype(dtype) + offset) * step) / img_shape[1]

    # Expand dims to support easy broadcasting.
    y = np.expand_dims(y, axis=-1)
    x = np.expand_dims(x, axis=-1)

    # Compute relative height and width.
    # Tries to follow the original implementation of combine for the order.
    num_anchors = len(sizes) * len(ratios)
    h = np.zeros((num_anchors, ), dtype=dtype)
    w = np.zeros((num_anchors, ), dtype=dtype)

    for i, r in enumerate(ratios):
        for j, s in enumerate(sizes):
            h[i*len(sizes) + j] = s / img_shape[0] / math.sqrt(r)
            w[i*len(sizes) + j] = s / img_shape[1] * math.sqrt(r)

    return y, x, h, w


def combine_anchors_all_layers(img_shape,
                           layers_shape,
                           anchor_sizes,
                           anchor_ratios,
                           anchor_steps,
                           offset=0.5,
                           dtype=np.float32):

    layers_anchors = []
    for i, s in enumerate(layers_shape):
        anchor_bboxes = combine_anchor_one_layer(img_shape, s,
                                             anchor_sizes[i],
                                             anchor_ratios[i],
                                             anchor_steps[i],
                                             offset=offset, dtype=dtype)
        layers_anchors.append(anchor_bboxes)
    return layers_anchors


# =========================================================================== #
# Functional definition of VGG-based combine net.
# =========================================================================== #
def tensor_shape(x, rank=3):
    """Returns the dimensions of a tensor.
    Args:
      image: A N-D Tensor of shape.
    Returns:
      A list of dimensions. Dimensions that are statically known are python
        integers,otherwise they are integer scalar tensors.
    """
    if x.get_shape().is_fully_defined():
        return x.get_shape().as_list()
    else:
        static_shape = x.get_shape().with_rank(rank).as_list()
        dynamic_shape = tf.unstack(tf.shape(x), rank)
        return [s if s is not None else d
                for s, d in zip(static_shape, dynamic_shape)]


def pred_cls_module(net_input, var_scope, num_anchors, num_classes):
  with tf.variable_scope(var_scope + '_inception1'):
    with tf.variable_scope('Branch_0'):
      branch_0 = slim.conv2d(net_input, 512, [3, 3], normalizer_fn = None, activation_fn = None, scope='Conv2d_3x3')
    with tf.variable_scope('Branch_1'):
      branch_1 = slim.conv2d(net_input, 512, [1, 1], normalizer_fn = None, activation_fn = None, scope='Conv2d_1x1')

    net_input = array_ops.concat([branch_0, branch_1], 3)
    # only activation after concat
    net_input = slim.batch_norm(net_input, activation_fn=tf.nn.relu)

  with tf.variable_scope(var_scope + '_inception2'):
    with tf.variable_scope('Branch_0'):
      branch_0 = slim.conv2d(net_input, 512, [3, 3], normalizer_fn = None, activation_fn = None, scope='Conv2d_3x3')
    with tf.variable_scope('Branch_1'):
      branch_1 = slim.conv2d(net_input, 512, [1, 1], normalizer_fn = None, activation_fn = None, scope='Conv2d_1x1')

    net_input = array_ops.concat([branch_0, branch_1], 3)
    # only activation after concat
    net_input = slim.batch_norm(net_input, activation_fn=tf.nn.relu)

    cls_pred = slim.conv2d(net_input, num_anchors * num_classes, [3, 3], activation_fn=None, scope='Conv2d_pred_3x3')

  cls_pred = custom_layers.channel_to_last(cls_pred)
  cls_pred = tf.reshape(cls_pred, tensor_shape(cls_pred, 4)[:-1]+[num_anchors, num_classes])

  return cls_pred

def reg_bbox_module(net_input, var_scope, num_anchors):# = 'reg_bbox_@4'
  with tf.variable_scope(var_scope):
    net_input = slim.conv2d(net_input, 512, [3, 3], normalizer_fn = slim.batch_norm, scope='Conv2d_0_3x3')

    loc_pred = slim.conv2d(net_input, 4 * num_anchors, [3, 3], activation_fn=None, scope='Conv2d_1_3x3')

    loc_pred = custom_layers.channel_to_last(loc_pred)
    loc_pred = tf.reshape(loc_pred, tensor_shape(loc_pred, 4)[:-1] + [num_anchors, 4])

  return loc_pred

# it seem's that no matter how many channals ref_map has, 512 will be used after deconv
def reverse_connection_module_with_pred(left_input, right_input, num_classes, num_anchors, var_scope):
  if right_input is None:
      ref_map = slim.conv2d(left_input, 512, [2, 2], stride=2, normalizer_fn = slim.batch_norm, scope = var_scope + '_conv_left')
  else:
      left_conv = slim.conv2d(left_input, 512, [3, 3], normalizer_fn = slim.batch_norm, scope = var_scope + '_conv_left')
      # remove BN for deconv, but leave Relu
      upsampling = slim.conv2d_transpose(right_input, 512, [2, 2], stride=2, normalizer_fn = None, scope = var_scope + '_deconv_right')
      ref_map = tf.nn.relu(left_conv + upsampling)

  objness_ref_map = slim.conv2d(ref_map, 512, [3, 3], normalizer_fn = slim.batch_norm, scope= var_scope + '_objectness')
  objectness_logits = tf.reshape(slim.conv2d(objness_ref_map, 2 * num_anchors, [3, 3], activation_fn = None, scope= var_scope + '_objectness_score'), tensor_shape(objness_ref_map, 4)[:-1]+[num_anchors, 2])

  # objectness_logits = tf.reshape(slim.conv2d(ref_map, 2 * num_anchors, [3, 3], activation_fn = None, scope= var_scope + '_objectness'), tensor_shape(ref_map, 4)[:-1]+[num_anchors, 2])

  return ref_map, objectness_logits, pred_cls_module(ref_map, var_scope, num_anchors, num_classes), reg_bbox_module(ref_map, var_scope, num_anchors)

def combine_net(inputs,
            num_classes=combine_Net.default_params.num_classes,
            feat_layers=combine_Net.default_params.feat_layers,
            anchor_sizes=combine_Net.default_params.anchor_sizes,
            anchor_ratios=combine_Net.default_params.anchor_ratios,
            is_training=True,
            dropout_keep_prob=0.5,
            prediction_fn=slim.softmax,
            reuse=None,
            scope='combine'):

    # End_points collect relevant activations for external use.
    end_points = {}
    with tf.variable_scope(scope, 'combine', [inputs], reuse=reuse):

        # Original VGG-16 blocks.
        net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
        end_points['block1'] = net
        net = slim.max_pool2d(net, [2, 2], scope='pool1')
        # Block 2.
        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
        end_points['block2'] = net
        net = slim.max_pool2d(net, [2, 2], scope='pool2')
        # Block 3.
        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
        end_points['block3'] = net
        net = slim.max_pool2d(net, [2, 2], scope='pool3')
        # Block 4.
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
        end_points['block4'] = net
        net = slim.max_pool2d(net, [2, 2], scope='pool4')
        # Block 5.
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
        end_points['block5'] = net
        # different betweent SSD here
        net = slim.max_pool2d(net, [2, 2], scope='pool5')
        # Block 6
        net = slim.conv2d(net, 4096, [7, 7], scope='fc6')
        end_points['block6'] = net
        # Block 7: 1x1 conv, no padding.
        net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
        end_points['block7'] = net

        # Prediction and localisations layers.
        predictions = []
        logits = []
        localisations = []
        objness_pred = []
        objness_logits = []
        cur_ref_map = None
        for i, layer in enumerate(feat_layers):
            with tf.variable_scope('reverse_module'):
                cur_ref_map, objness, cls_pred, bbox_reg = reverse_connection_module_with_pred(end_points[layer], cur_ref_map, num_classes,\
                                              len(anchor_sizes[i]) * len(anchor_ratios[i]), var_scope = layer + '_reverse')
                predictions.append(prediction_fn(cls_pred))
                logits.append(cls_pred)
                obj_pred_neg_pos = prediction_fn(objness)
                objness_pred.append(tf.slice(obj_pred_neg_pos, [0, 0, 0, 0, 1], [-1, -1, -1, -1, 1]))
                objness_logits.append(objness)
                localisations.append(bbox_reg)

        return predictions, logits, objness_pred, objness_logits, localisations, end_points

def combine_reducedfc(inputs,
            num_classes=combine_Net.default_params.num_classes,
            feat_layers=combine_Net.default_params.feat_layers,
            anchor_sizes=combine_Net.default_params.anchor_sizes,
            anchor_ratios=combine_Net.default_params.anchor_ratios,
            is_training=True,
            dropout_keep_prob=0.5,
            prediction_fn=slim.softmax,
            reuse=None,
            scope='combine'):
    #combine net definition.

    # End_points collect relevant activations for external use.
    end_points = {}
    with tf.variable_scope(scope, 'combine', [inputs], reuse=reuse):

        # Original VGG-16 blocks.
        net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
        end_points['block1'] = net
        net = slim.max_pool2d(net, [2, 2], scope='pool1')
        # Block 2.
        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
        end_points['block2'] = net
        net = slim.max_pool2d(net, [2, 2], scope='pool2')
        # Block 3.
        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
        end_points['block3'] = net
        net = slim.max_pool2d(net, [2, 2], scope='pool3')
        # Block 4.
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
        end_points['block4'] = net
        net = slim.max_pool2d(net, [2, 2], scope='pool4')
        # Block 5.
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
        end_points['block5'] = net
        # different betweent SSD here
        net = slim.max_pool2d(net, [2, 2], scope='pool5')
        # Block 6
         # Use conv2d instead of fully_connected layers.
        net = slim.conv2d(net, 1024, [3, 3], stride=1, rate=3, padding='SAME', scope='fc6')
        end_points['block6'] = net
        net = slim.conv2d(net, 1024, [1, 1], stride=1, rate=1, padding='SAME', scope='fc7')
        end_points['block7'] = net

        # Prediction and localisations layers.
        predictions = []
        logits = []
        localisations = []
        objness_pred = []
        objness_logits = []
        cur_ref_map = None
        for i, layer in enumerate(feat_layers):
            with tf.variable_scope('reverse_module'):
                cur_ref_map, objness, cls_pred, bbox_reg = reverse_connection_module_with_pred(end_points[layer], cur_ref_map, num_classes, len(anchor_sizes[i]) * len(anchor_ratios[i]), var_scope = layer + '_reverse')
                predictions.append(prediction_fn(cls_pred))
                logits.append(cls_pred)
                obj_pred_neg_pos = prediction_fn(objness)
                #objness_pred.append(tf.ones_like(tf.slice(obj_pred_neg_pos, [0, 0,0,0,1], [-1, -1,-1,-1,1])))
                objness_pred.append(tf.slice(obj_pred_neg_pos, [0, 0, 0, 0, 1], [-1, -1, -1, -1, 1]))
                objness_logits.append(objness)
                localisations.append(bbox_reg)

        return predictions, logits, objness_pred, objness_logits, localisations, end_points

combine_net.default_image_size = 320


def truncated_normal_001_initializer():
  def _initializer(shape, dtype=tf.float32, partition_info=None):
    #Initializer function
    if not dtype.is_floating:
      raise TypeError('Cannot create initializer for non-floating point type.')
    return tf.truncated_normal(shape, 0.0, 0.01, dtype, seed=None)
  return _initializer

def combine_arg_scope(weight_decay=0.0005, is_training=True, data_format='NHWC'):
    """Defines the VGG arg scope.

    Args:
      weight_decay: The l2 regularization coefficient.

    Returns:
      An arg_scope.
    """
    with slim.arg_scope([slim.conv2d_transpose, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        weights_initializer=tf.contrib.layers.xavier_initializer(),
                        biases_initializer=tf.zeros_initializer()):
      with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu,
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        weights_initializer=tf.contrib.layers.xavier_initializer(),#truncated_normal_001_initializer(),
                        biases_initializer=tf.zeros_initializer()):
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose, slim.max_pool2d],
                            padding='SAME',
                            data_format=data_format):
            with slim.arg_scope([slim.batch_norm],
                            # default no activation_fn for BN
                            activation_fn=None,
                            decay=0.997,
                            epsilon=1e-5,
                            scale=True,
                            fused=True,
                            is_training = is_training,
                            data_format=data_format):
                with slim.arg_scope([custom_layers.pad2d,
                                     custom_layers.l2_normalization,
                                     custom_layers.channel_to_last],
                                    data_format=data_format) as sc:
                    return sc


# =========================================================================== #
# combine loss function.
# =========================================================================== #
def combine_losses(logits, localisations, objness_logits, objness_pred,
               gclasses, glocalisations, gscores,
               match_threshold = 0.5,
               neg_threshold = 0.3,
               objness_threshold = 0.03,
               negative_ratio=3.,
               alpha=1./3,
               beta=1./3,
               label_smoothing=0.,
               device='/cpu:0',
               scope=None):
    with tf.name_scope(scope, 'loss'):
        logits_shape = tfe.get_shape(logits[0], 5)
        num_classes = logits_shape[-1]
        batch_size = logits_shape[0]

        # Flatten out all vectors
        flogits = []
        fobjness_logits = []
        fobjness_pred = []
        fgclasses = []
        fgscores = []
        flocalisations = []
        fglocalisations = []
        for i in range(len(logits)):
            flogits.append(tf.reshape(logits[i], [-1, num_classes]))
            fobjness_logits.append(tf.reshape(objness_logits[i], [-1, 2]))
            fobjness_pred.append(tf.reshape(objness_pred[i], [-1]))
            fgclasses.append(tf.reshape(gclasses[i], [-1]))
            fgscores.append(tf.reshape(gscores[i], [-1]))
            flocalisations.append(tf.reshape(localisations[i], [-1, 4]))
            fglocalisations.append(tf.reshape(glocalisations[i], [-1, 4]))
        # concat along different feature map (from last to front: layer7->layer4)
        logits = tf.concat(flogits, axis=0)
        objness_logits = tf.concat(fobjness_logits, axis=0)
        objness_pred = tf.concat(fobjness_pred, axis=0)
        gclasses = tf.concat(fgclasses, axis=0)
        gscores = tf.concat(fgscores, axis=0)
        localisations = tf.concat(flocalisations, axis=0)
        glocalisations = tf.concat(fglocalisations, axis=0)
        dtype = logits.dtype

        positive_mask = gclasses > 0

        fpositive_mask = tf.cast(positive_mask, dtype)
        n_positives = tf.reduce_sum(fpositive_mask)
        negtive_mask = tf.equal(gclasses, 0) #(gclasses == 0)
        #negtive_mask = tf.logical_and(gscores < neg_threshold, tf.logical_not(positive_mask))
        fnegtive_mask = tf.cast(negtive_mask, dtype)
        n_negtives = tf.reduce_sum(fnegtive_mask)

        # random select hard negtive for objectness
        n_neg_to_select = tf.cast(negative_ratio * n_positives, tf.int32)
        n_neg_to_select = tf.minimum(n_neg_to_select, tf.cast(n_negtives, tf.int32))

        rand_neg_mask = tf.random_uniform(tfe.get_shape(gscores, 1), minval=0, maxval=1.) < tfe.safe_divide(tf.cast(n_neg_to_select, dtype), n_negtives, name='rand_select_objness')
        # include both random_select negtive and all positive examples
        final_neg_mask_objness = tf.stop_gradient(tf.logical_or(tf.logical_and(negtive_mask, rand_neg_mask), positive_mask))
        total_examples_for_objness = tf.reduce_sum(tf.cast(final_neg_mask_objness, dtype))
        # the label for objectness is all the positive
        objness_pred_label = tf.stop_gradient(tf.cast(positive_mask, tf.int32))
        objness_pred_in_positive = tf.cast(positive_mask, dtype) * objness_pred
        # max objectness score in all positive positions
        max_objness_in_positive = tf.reduce_max(objness_pred_in_positive)
        # the position of max objectness score in all positive positions
        max_objness_mask = tf.equal(objness_pred_in_positive, max_objness_in_positive)


        # objectness mask for select real positive for detection
        objectness_mask = objness_pred > objness_threshold
        cls_positive_mask = tf.stop_gradient(tf.logical_and(positive_mask, objectness_mask))
        cls_negtive_mask = tf.logical_and(objectness_mask, negtive_mask)
        #cls_negtive_mask = tf.logical_and(objectness_mask, tf.logical_not(cls_positive_mask))

        n_cls_negtives = tf.reduce_sum(tf.cast(cls_negtive_mask, dtype))

        fcls_positive_mask = tf.cast(cls_positive_mask, dtype)
        n_cls_positives = tf.reduce_sum(fcls_positive_mask)
        n_cls_neg_to_select = tf.cast(negative_ratio * n_cls_positives, tf.int32)
        n_cls_neg_to_select = tf.minimum(n_cls_neg_to_select, tf.cast(n_cls_negtives, tf.int32))
        # random selected negtive mask
        rand_cls_neg_mask = tf.random_uniform(tfe.get_shape(gscores, 1), minval=0, maxval=1.) < tfe.safe_divide(tf.cast(n_cls_neg_to_select, dtype), n_cls_negtives, name='rand_select_cls')
        # include both random_select negtive and all positive(positive is filtered by objectness)
        final_cls_neg_mask_objness = tf.stop_gradient(tf.logical_or(tf.logical_and(cls_negtive_mask, rand_cls_neg_mask), cls_positive_mask))
        total_examples_for_cls = tf.reduce_sum(tf.cast(final_cls_neg_mask_objness, dtype))

        # Add cross-entropy loss.
        with tf.name_scope('cross_entropy_pos'):
            #weights = (1. - alpha - beta) * tf.cast(final_cls_neg_mask_objness, dtype)
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.stop_gradient(tf.clip_by_value(gclasses, 0, num_classes)))

            loss = tf.cond(n_positives > 0., lambda: (1. - alpha - beta) * tf.reduce_mean(tf.boolean_mask(loss, final_cls_neg_mask_objness)), lambda: 0.)
            tf.losses.add_loss(loss)

        with tf.name_scope('cross_entropy_objectness'):
            #weights = alpha * tf.cast(final_neg_mask_objness, dtype)
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=objness_logits, labels=objness_pred_label)
            loss = tf.cond(n_positives > 0., lambda: alpha * tf.reduce_mean(tf.boolean_mask(loss, final_neg_mask_objness)), lambda: 0.)
            tf.losses.add_loss(loss)

        # Add localization loss
        with tf.name_scope('localization'):
            loss = custom_layers.modified_smooth_l1(localisations, tf.stop_gradient(glocalisations), sigma = 3.)
            loss = tf.cond(n_cls_positives > 0., lambda: beta * tf.reduce_mean(tf.boolean_mask(tf.reduce_sum(loss, axis=-1), tf.stop_gradient(cls_positive_mask))), lambda: 0.)
            tf.losses.add_loss(loss)

