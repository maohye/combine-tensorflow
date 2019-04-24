
# ==============================================================================
"""TF Extended: additional bounding boxes methods.
"""
import numpy as np
import tensorflow as tf

from tf_extended import tensors as tfe_tensors
from tf_extended import math as tfe_math


# =========================================================================== #
# Standard boxes algorithms.
# =========================================================================== #
def bboxes_sort_all_classes(classes, scores, bboxes, top_k=400, scope=None):
    """Sort bounding boxes by decreasing order and keep only the top_k.
    Assume the input Tensors mix-up objects with different classes.
    Assume a batch-type input.

    Args:
      classes: Batch x N Tensor containing integer classes.
      scores: Batch x N Tensor containing float scores.
      bboxes: Batch x N x 4 Tensor containing boxes coordinates.
      top_k: Top_k boxes to keep.
    Return:
      classes, scores, bboxes: Sorted tensors of shape Batch x Top_k.
    """
    with tf.name_scope(scope, 'bboxes_sort', [classes, scores, bboxes]):
        scores, idxes = tf.nn.top_k(scores, k=top_k, sorted=True)

        # Trick to be able to use tf.gather: map for each element in the batch.
        def fn_gather(classes, bboxes, idxes):
            cl = tf.gather(classes, idxes)
            bb = tf.gather(bboxes, idxes)
            return [cl, bb]
        r = tf.map_fn(lambda x: fn_gather(x[0], x[1], x[2]),
                      [classes, bboxes, idxes],
                      dtype=[classes.dtype, bboxes.dtype],
                      parallel_iterations=10,
                      back_prop=False,
                      swap_memory=False,
                      infer_shape=True)
        classes = r[0]
        bboxes = r[1]
        return classes, scores, bboxes


def bboxes_sort(scores, bboxes, top_k=400, scope=None):
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
                s, b = bboxes_sort(scores[c], bboxes[c], top_k=top_k)
                d_scores[c] = s
                d_bboxes[c] = b
            return d_scores, d_bboxes

    # Tensors inputs.
    with tf.name_scope(scope, 'bboxes_sort', [scores, bboxes]):
        # Sort scores...
        scores, idxes = tf.nn.top_k(scores, k=top_k, sorted=True)

        # Trick to be able to use tf.gather: map for each element in the first dim.
        def fn_gather(bboxes, idxes):
            bb = tf.gather(bboxes, idxes)
            return [bb]
        r = tf.map_fn(lambda x: fn_gather(x[0], x[1]),
                      [bboxes, idxes],
                      dtype=[bboxes.dtype],
                      parallel_iterations=10,
                      back_prop=False,
                      swap_memory=False,
                      infer_shape=True)
        bboxes = r[0]

        return tfe_tensors.pad_axis(scores, 0, top_k, axis=1), tfe_tensors.pad_axis(bboxes, 0, top_k, axis=1)
        #return scores, bboxes


def bboxes_clip(bbox_ref, bboxes, scope=None):
    """Clip bounding boxes to a reference box.
    Batch-compatible if the first dimension of `bbox_ref` and `bboxes`
    can be broadcasted.

    Args:
      bbox_ref: Reference bounding box. Nx4 or 4 shaped-Tensor;
      bboxes: Bounding boxes to clip. Nx4 or 4 shaped-Tensor or dictionary.
    Return:
      Clipped bboxes.
    """
    # Bboxes is dictionary.
    if isinstance(bboxes, dict):
        with tf.name_scope(scope, 'bboxes_clip_dict'):
            d_bboxes = {}
            for c in bboxes.keys():
                d_bboxes[c] = bboxes_clip(bbox_ref, bboxes[c])
            return d_bboxes

    # Tensors inputs.
    with tf.name_scope(scope, 'bboxes_clip'):
        # Easier with transposed bboxes. Especially for broadcasting.
        bbox_ref = tf.transpose(bbox_ref)
        bboxes = tf.transpose(bboxes)
        # Intersection bboxes and reference bbox.
        ymin = tf.maximum(bboxes[0], bbox_ref[0])
        xmin = tf.maximum(bboxes[1], bbox_ref[1])
        ymax = tf.minimum(bboxes[2], bbox_ref[2])
        xmax = tf.minimum(bboxes[3], bbox_ref[3])
        ymin = tf.minimum(ymin, ymax)
        xmin = tf.minimum(xmin, xmax)
        bboxes = tf.transpose(tf.stack([ymin, xmin, ymax, xmax], axis=0))
        return bboxes


def bboxes_resize(bbox_ref, bboxes, name=None):
    """Resize bounding boxes based on a reference bounding box,
    assuming that the latter is [0, 0, 1, 1] after transform. Useful for
    updating a collection of boxes after cropping an image.
    """
    # Bboxes is dictionary.
    if isinstance(bboxes, dict):
        with tf.name_scope(name, 'bboxes_resize_dict'):
            d_bboxes = {}
            for c in bboxes.keys():
                d_bboxes[c] = bboxes_resize(bbox_ref, bboxes[c])
            return d_bboxes

    # Tensors inputs.
    with tf.name_scope(name, 'bboxes_resize'):
        # Translate.
        v = tf.stack([bbox_ref[0], bbox_ref[1], bbox_ref[0], bbox_ref[1]])
        bboxes = bboxes - v
        # Scale.
        s = tf.stack([bbox_ref[2] - bbox_ref[0],
                      bbox_ref[3] - bbox_ref[1],
                      bbox_ref[2] - bbox_ref[0],
                      bbox_ref[3] - bbox_ref[1]])
        bboxes = bboxes / s
        return bboxes

def bboxes_nms(scores, bboxes, nms_threshold = 0.5, keep_top_k = 200, mode = 'min', scope=None):
    with tf.name_scope(scope, 'tf_bboxes_nms', [scores, bboxes]):
        # get the cls_score for the most-likely class
        num_anchors = tf.shape(scores)[0]
        def nms_proc(scores, bboxes):
            # sort all the bboxes
            scores, idxes = tf.nn.top_k(scores, k = num_anchors, sorted = True)
            bboxes = tf.gather(bboxes, idxes)

            ymin = bboxes[:, 0]
            xmin = bboxes[:, 1]
            ymax = bboxes[:, 2]
            xmax = bboxes[:, 3]

            vol_anchors = (xmax - xmin) * (ymax - ymin)

            nms_mask = tf.cast(tf.ones_like(scores, dtype=tf.int8), tf.bool)
            keep_mask = tf.cast(tf.zeros_like(scores, dtype=tf.int8), tf.bool)

            def safe_divide(numerator, denominator):
                return tf.where(tf.greater(denominator, 0), tf.divide(numerator, denominator), tf.zeros_like(denominator))

            def get_scores(bbox, nms_mask):
                # the inner square
                inner_ymin = tf.maximum(ymin, bbox[0])
                inner_xmin = tf.maximum(xmin, bbox[1])
                inner_ymax = tf.minimum(ymax, bbox[2])
                inner_xmax = tf.minimum(xmax, bbox[3])
                h = tf.maximum(inner_ymax - inner_ymin, 0.)
                w = tf.maximum(inner_xmax - inner_xmin, 0.)
                inner_vol = h * w
                this_vol = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                if mode == 'union':
                    union_vol = vol_anchors - inner_vol  + this_vol
                elif mode == 'min':
                    union_vol = tf.minimum(vol_anchors, this_vol)
                else:
                    raise ValueError('unknown mode to use for nms.')
                return safe_divide(inner_vol, union_vol) * tf.cast(nms_mask, tf.float32)

            def condition(index, nms_mask, keep_mask):
                return tf.logical_and(tf.reduce_sum(tf.cast(nms_mask, tf.int32)) > 0, tf.less(index, keep_top_k))

            def body(index, nms_mask, keep_mask):
                # at least one True in nms_mask
                indices = tf.where(nms_mask)[0][0]
                bbox = bboxes[indices]
                this_mask = tf.one_hot(indices, num_anchors, on_value=False, off_value=True, dtype=tf.bool)
                keep_mask = tf.logical_or(keep_mask, tf.logical_not(this_mask))
                nms_mask = tf.logical_and(nms_mask, this_mask)

                nms_scores = get_scores(bbox, nms_mask)

                nms_mask = tf.logical_and(nms_mask, nms_scores < nms_threshold)
                return [index+1, nms_mask, keep_mask]

            index = 0
            [index, nms_mask, keep_mask] = tf.while_loop(condition, body, [index, nms_mask, keep_mask])

            return tfe_tensors.pad_axis(tf.boolean_mask(scores, keep_mask), 0, keep_top_k, axis=0), tfe_tensors.pad_axis(tf.boolean_mask(bboxes, keep_mask), 0, keep_top_k, axis=0)

        return tf.cond(tf.less(num_anchors, 1), lambda: (scores, bboxes), lambda: nms_proc(scores, bboxes))

def bboxes_nms1(scores, bboxes, nms_threshold=0.5, keep_top_k=200, scope=None):
    with tf.name_scope(scope, 'bboxes_nms_single', [scores, bboxes]):
        # Apply NMS algorithm.
        idxes = tf.image.non_max_suppression(bboxes, scores,
                                             keep_top_k, nms_threshold)
        scores = tf.gather(scores, idxes)
        bboxes = tf.gather(bboxes, idxes)
        # Pad results.
        scores = tfe_tensors.pad_axis(scores, 0, keep_top_k, axis=0)
        bboxes = tfe_tensors.pad_axis(bboxes, 0, keep_top_k, axis=0)
        return scores, bboxes


def bboxes_nms_batch(scores, bboxes, nms_threshold=0.5, keep_top_k=200,
                     scope=None):
    """Apply non-maximum selection to bounding boxes. In comparison to TF
    implementation, use classes information for matching.
    Use only on batched-inputs. Use zero-padding in order to batch output
    results.

    Args:
      scores: Batch x N Tensor/Dictionary containing float scores.
      bboxes: Batch x N x 4 Tensor/Dictionary containing boxes coordinates.
      nms_threshold: Matching threshold in NMS algorithm;
      keep_top_k: Number of total object to keep after NMS.
    Return:
      scores, bboxes Tensors/Dictionaries, sorted by score.
        Padded with zero if necessary.
    """
    # Dictionaries as inputs.
    if isinstance(scores, dict) or isinstance(bboxes, dict):
        with tf.name_scope(scope, 'bboxes_nms_batch_dict'):
            d_scores = {}
            d_bboxes = {}
            for c in scores.keys():
                s, b = bboxes_nms_batch(scores[c], bboxes[c],
                                        nms_threshold=nms_threshold,
                                        keep_top_k=keep_top_k)
                d_scores[c] = s
                d_bboxes[c] = b
            return d_scores, d_bboxes

    # Tensors inputs.
    with tf.name_scope(scope, 'bboxes_nms_batch'):
        r = tf.map_fn(lambda x: bboxes_nms(x[0], x[1],
                                           nms_threshold, keep_top_k),
                      (scores, bboxes),
                      dtype=(scores.dtype, bboxes.dtype),
                      parallel_iterations=10,
                      back_prop=False,
                      swap_memory=False,
                      infer_shape=True)
        scores, bboxes = r
        return scores, bboxes


# def bboxes_fast_nms(classes, scores, bboxes,
#                     nms_threshold=0.5, eta=3., num_classes=21,
#                     pad_output=True, scope=None):
#     with tf.name_scope(scope, 'bboxes_fast_nms',
#                        [classes, scores, bboxes]):

#         nms_classes = tf.zeros((0,), dtype=classes.dtype)
#         nms_scores = tf.zeros((0,), dtype=scores.dtype)
#         nms_bboxes = tf.zeros((0, 4), dtype=bboxes.dtype)


def bboxes_matching(label, scores, bboxes,
                    glabels, gbboxes, gdifficults,
                    matching_threshold=0.5, scope=None):
    """Matching a collection of detected boxes with groundtruth values.
    Does not accept batched-inputs.
    The algorithm goes as follows: for every detected box, check
    if one grountruth box is matching. If none, then considered as False Positive.
    If the grountruth box is already matched with another one, it also counts
    as a False Positive. We refer the Pascal VOC documentation for the details.

    Args:
      rclasses, rscores, rbboxes: N(x4) Tensors. Detected objects, sorted by score;
      glabels, gbboxes: Groundtruth bounding boxes. May be zero padded, hence
        zero-class objects are ignored.
      matching_threshold: Threshold for a positive match.
    Return: Tuple of:
       n_gbboxes: Scalar Tensor with number of groundtruth boxes (may difer from
         size because of zero padding).
       tp_match: (N,)-shaped boolean Tensor containing with True Positives.
       fp_match: (N,)-shaped boolean Tensor containing with False Positives.
    """
    with tf.name_scope(scope, 'bboxes_matching_single',
                       [scores, bboxes, glabels, gbboxes]):
        rsize = tf.size(scores)
        rshape = tf.shape(scores)
        rlabel = tf.cast(label, glabels.dtype)
        # Number of groundtruth boxes.
        gdifficults = tf.cast(gdifficults, tf.bool)
        n_gbboxes = tf.count_nonzero(tf.logical_and(tf.equal(glabels, label),
                                                    tf.logical_not(gdifficults)))
        # Grountruth matching arrays.
        gmatch = tf.zeros(tf.shape(glabels), dtype=tf.bool)
        grange = tf.range(tf.size(glabels), dtype=tf.int32)
        # True/False positive matching TensorArrays.
        sdtype = tf.bool
        ta_tp_bool = tf.TensorArray(sdtype, size=rsize, dynamic_size=False, infer_shape=True)
        ta_fp_bool = tf.TensorArray(sdtype, size=rsize, dynamic_size=False, infer_shape=True)

        # Loop over returned objects.
        def m_condition(i, ta_tp, ta_fp, gmatch):
            r = tf.less(i, rsize)
            return r

        def m_body(i, ta_tp, ta_fp, gmatch):
            # Jaccard score with groundtruth bboxes.
            rbbox = bboxes[i]
            jaccard = bboxes_jaccard(rbbox, gbboxes)
            jaccard = jaccard * tf.cast(tf.equal(glabels, rlabel), dtype=jaccard.dtype)

            # Best fit, checking it's above threshold.
            idxmax = tf.cast(tf.argmax(jaccard, axis=0), tf.int32)
            jcdmax = jaccard[idxmax]
            match = jcdmax > matching_threshold
            existing_match = gmatch[idxmax]
            not_difficult = tf.logical_not(gdifficults[idxmax])

            # TP: match & no previous match and FP: previous match | no match.
            # If difficult: no record, i.e FP=False and TP=False.
            tp = tf.logical_and(not_difficult,
                                tf.logical_and(match, tf.logical_not(existing_match)))
            ta_tp = ta_tp.write(i, tp)
            fp = tf.logical_and(not_difficult,
                                tf.logical_or(existing_match, tf.logical_not(match)))
            ta_fp = ta_fp.write(i, fp)
            # Update grountruth match.
            mask = tf.logical_and(tf.equal(grange, idxmax),
                                  tf.logical_and(not_difficult, match))
            gmatch = tf.logical_or(gmatch, mask)

            return [i+1, ta_tp, ta_fp, gmatch]
        # Main loop definition.
        i = 0
        [i, ta_tp_bool, ta_fp_bool, gmatch] = \
            tf.while_loop(m_condition, m_body,
                          [i, ta_tp_bool, ta_fp_bool, gmatch],
                          parallel_iterations=1,
                          back_prop=False)
        # TensorArrays to Tensors and reshape.
        tp_match = tf.reshape(ta_tp_bool.stack(), rshape)
        fp_match = tf.reshape(ta_fp_bool.stack(), rshape)
        return n_gbboxes, tp_match, fp_match


def bboxes_matching_batch(labels, scores, bboxes,
                          glabels, gbboxes, gdifficults,
                          matching_threshold=0.5, scope=None):
    # Dictionaries as inputs.
    if isinstance(scores, dict) or isinstance(bboxes, dict):
        with tf.name_scope(scope, 'bboxes_matching_batch_dict'):
            d_n_gbboxes = {}
            d_tp = {}
            d_fp = {}
            for c in labels:
                n, tp, fp, _ = bboxes_matching_batch(c, scores[c], bboxes[c],
                                                     glabels, gbboxes, gdifficults,
                                                     matching_threshold)
                d_n_gbboxes[c] = n
                d_tp[c] = tp
                d_fp[c] = fp
            return d_n_gbboxes, d_tp, d_fp, scores

    with tf.name_scope(scope, 'bboxes_matching_batch',
                       [scores, bboxes, glabels, gbboxes]):
        r = tf.map_fn(lambda x: bboxes_matching(labels, x[0], x[1],
                                                x[2], x[3], x[4],
                                                matching_threshold),
                      (scores, bboxes, glabels, gbboxes, gdifficults),
                      dtype=(tf.int64, tf.bool, tf.bool),
                      parallel_iterations=10,
                      back_prop=False,
                      swap_memory=True,
                      infer_shape=True)
        return r[0], r[1], r[2], scores


# =========================================================================== #
# Some filteting methods.
# =========================================================================== #
def bboxes_filter_center(labels, bboxes, margins=[0., 0., 0., 0.],
                         scope=None):
    with tf.name_scope(scope, 'bboxes_filter', [labels, bboxes]):
        cy = (bboxes[:, 0] + bboxes[:, 2]) / 2.
        cx = (bboxes[:, 1] + bboxes[:, 3]) / 2.
        mask = tf.greater(cy, margins[0])
        mask = tf.logical_and(mask, tf.greater(cx, margins[1]))
        mask = tf.logical_and(mask, tf.less(cx, 1. + margins[2]))
        mask = tf.logical_and(mask, tf.less(cx, 1. + margins[3]))
        # Boolean masking...
        labels = tf.boolean_mask(labels, mask)
        bboxes = tf.boolean_mask(bboxes, mask)
        return labels, bboxes


def bboxes_filter_overlap(labels, bboxes,
                          threshold=0.5, assign_negative=False,
                          scope=None):
    with tf.name_scope(scope, 'bboxes_filter', [labels, bboxes]):
        scores = bboxes_intersection(tf.constant([0, 0, 1, 1], bboxes.dtype),
                                     bboxes)
        # always keep al least one
        max_intersection = tf.reduce_max(scores)
        max_intersection_mask = tf.equal(scores, max_intersection)

        mask = tf.logical_or(scores > threshold, max_intersection_mask)

        if assign_negative:
            labels = tf.where(mask, labels, -labels)
            # bboxes = tf.where(mask, bboxes, bboxes)
        else:
            labels = tf.boolean_mask(labels, mask)
            bboxes = tf.boolean_mask(bboxes, mask)
        return labels, bboxes


def bboxes_filter_labels(labels, bboxes,
                         out_labels=[], num_classes=np.inf,
                         scope=None):
    """Filter out labels from a collection. Typically used to get
    of DontCare elements. Also remove elements based on the number of classes.

    Return:
      labels, bboxes: Filtered elements.
    """
    with tf.name_scope(scope, 'bboxes_filter_labels', [labels, bboxes]):
        mask = tf.greater_equal(labels, num_classes)
        for l in labels:
            mask = tf.logical_and(mask, tf.not_equal(labels, l))
        labels = tf.boolean_mask(labels, mask)
        bboxes = tf.boolean_mask(bboxes, mask)
        return labels, bboxes


# =========================================================================== #
# Standard boxes computation.
# =========================================================================== #
def bboxes_jaccard(bbox_ref, bboxes, name=None):
    with tf.name_scope(name, 'bboxes_jaccard'):
        # Should be more efficient to first transpose.
        bboxes = tf.transpose(bboxes)
        bbox_ref = tf.transpose(bbox_ref)
        # Intersection bbox and volume.
        int_ymin = tf.maximum(bboxes[0], bbox_ref[0])
        int_xmin = tf.maximum(bboxes[1], bbox_ref[1])
        int_ymax = tf.minimum(bboxes[2], bbox_ref[2])
        int_xmax = tf.minimum(bboxes[3], bbox_ref[3])
        h = tf.maximum(int_ymax - int_ymin, 0.)
        w = tf.maximum(int_xmax - int_xmin, 0.)
        # Volumes.
        inter_vol = h * w
        union_vol = -inter_vol \
            + (bboxes[2] - bboxes[0]) * (bboxes[3] - bboxes[1]) \
            + (bbox_ref[2] - bbox_ref[0]) * (bbox_ref[3] - bbox_ref[1])
        jaccard = tfe_math.safe_divide(inter_vol, union_vol, 'jaccard')
        return jaccard


def bboxes_intersection(bbox_ref, bboxes, name=None):
    with tf.name_scope(name, 'bboxes_intersection'):
        # Should be more efficient to first transpose.
        bboxes = tf.transpose(bboxes)
        bbox_ref = tf.transpose(bbox_ref)
        # Intersection bbox and volume.
        int_ymin = tf.maximum(bboxes[0], bbox_ref[0])
        int_xmin = tf.maximum(bboxes[1], bbox_ref[1])
        int_ymax = tf.minimum(bboxes[2], bbox_ref[2])
        int_xmax = tf.minimum(bboxes[3], bbox_ref[3])
        h = tf.maximum(int_ymax - int_ymin, 0.)
        w = tf.maximum(int_xmax - int_xmin, 0.)
        # Volumes.
        inter_vol = h * w
        bboxes_vol = (bboxes[2] - bboxes[0]) * (bboxes[3] - bboxes[1])
        scores = tfe_math.safe_divide(inter_vol, bboxes_vol, 'intersection')
        return scores
