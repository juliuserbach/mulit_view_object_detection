import os
import random
import datetime
import re
import math
import logging
from collections import OrderedDict
import multiprocessing
import numpy as np
from scipy.spatial.transform import Rotation as R
import imageio
import tensorflow as tf
import keras
import keras.backend as K
import keras.layers as KL
import keras.engine as KE
import keras.models as KM

class MVNet(object):
    def __init__(self,
                 vmin,
                 vmax,
                 vox_bs,
                 im_bs,
                 grid_size,
                 im_h,
                 im_w,
                 mode="TRAIN",
                 norm="IN"):
        self.batch_size = vox_bs
        self.im_batch = im_bs
        self.nvox = grid_size
        self.im_h = im_h
        self.im_w = im_w
        self.vmin = vmin
        self.vmax = vmax
        self.vsize = float(self.vmax - self.vmin) / self.nvox
        self.mode = mode
        self.norm = norm

    @property
    def vox_tensor_shape(self):
        return [self.batch_size, self.nvox, self.nvox, self.nvox, 1]

    @property
    def vfp_vox_tensor_shape(self):
        return [
            self.batch_size, self.im_batch, self.nvox, self.nvox, self.nvox, 1
        ]

    @property
    def im_tensor_shape(self):
        return [self.batch_size, self.im_batch, self.im_h, self.im_w, 3]

    @property
    def depth_tensor_shape(self):
        return [self.batch_size, self.im_batch, self.im_h, self.im_w, 1]

    @property
    def K_tensor_shape(self):
        return [self.batch_size, self.im_batch, 3, 3]

    @property
    def R_tensor_shape(self):
        return [self.batch_size, self.im_batch, 3, 4]

    @property
    def quat_tensor_shape(self):
        return [self.batch_size, self.im_batch, 4]

    @property
    def total_ims_per_batch(self):
        return self.batch_size * self.im_batch

    # def print_net(self):
    #     if hasattr(self, 'im_net'):
    #         print('\n')
    #         pretty_line('Image Encoder')
    #         for k, v in sorted(self.im_net.iteritems()):
    #             print( k + '\t' + str(v.get_shape().as_list()))

    #     if hasattr(self, 'grid_net'):
    #         print( '\n')
    #         pretty_line('Grid Net')
    #         for k, v in sorted(self.grid_net.iteritems()):
    #             print(k + '\t' + str(v.get_shape().as_list()))

    #     if hasattr(self, 'depth_net'):
    #         print( '\n')
    #         pretty_line('Depth Net')
    #         for k, v in sorted(self.depth_net.iteritems()):
    #             print( k + '\t' + str(v.get_shape().as_list()))

    #     if hasattr(self, 'encoder'):
    #         print( '\n')
    #         pretty_line('Encoder')
    #         for k, v in sorted(self.encoder.iteritems()):
    #             print( k + '\t' + str(v.get_shape().as_list()))

    #     if hasattr(self, 'decoder'):
    #         print( '\n')
    #         pretty_line('Decoder')
    #         for k, v in sorted(self.decoder.iteritems()):
    #             print( k + '\t' + str(v.get_shape().as_list()))

        return
# def load_image_gt(dataset, config, image_id):
#     image = dataset.load_image(image_id)
#     mask, class_ids = dataset.load_mask(image_id)
    
# def data_generator(dataset, config, shuffle=True, batch_size=1)
#     b = 0  # batch item index
#     image_index = -1
#     image_ids = np.copy(dataset.image_ids)
    
#     while True:
#         image_index = (image_index + 1) % len(image_ids)
#         if shuffle and image_index == 0:
#             np.random.shuffle(image_ids)

#         # Get GT bounding boxes and masks for image.
#         image_id = image_ids[image_index]
        
#         image, image_meta, gt_class_ids, gt_boxes, gt_masks = \
#                     load_image_gt(dataset, config, image_id)
    
class ProjectionNet():
    def __init__(self, config):
        self.config = config
        input_image = KL.Input(shape=[config.NUM_VIEWS] + list(config.IMAGE_SHAPE),
                               name="input_image")
        input_R = KL.Input(shape=[config.NUM_VIEWS, 3, 4], 
                              name="input_R")
        input_Kmat = KL.Input(shape=[3, 3],
                              name="input_Kmat")

        x, grid_pos = unproj_feat(name="unproj_feat",config=self.config)([input_image, input_R, input_Kmat])# x = [Ibilin, position of grid]
        x = grid_reas(name="grid_reas", config=config)(x)
        out = proj_grid(name="proj_grid",config=self.config)([x, grid_pos, input_R, input_Kmat])
        
        inputs = [input_image, input_R, input_Kmat]
        outputs = [out, x]
        self.keras_model = KM.Model(inputs, outputs)
        
    def run(self, inputs):
        grid = self.keras_model.predict(
            inputs, batch_size=1, verbose=0)
        K.clear_session()
        return grid
    
    
class unproj_feat(KL.Layer):
    """ unproject feature map into 3D grid"""
    def __init__(self, config, **kwargs):
        self.config = config
        super(unproj_feat, self).__init__(**kwargs)
    def call(self, inputs):
        feats, Rcam, Kmat = inputs
        config = self.config
        
        # Construct [R^{T}|-R^{T}t]
        Rcam_old = Rcam
        Rt = tf.matrix_transpose(Rcam[:, :, :, :3])
        tr = tf.expand_dims(Rcam[:, :, :, 3], axis=-1)
        # repeat Kmat 
        Kmat = repeat_tensor(Kmat, config.NUM_VIEWS, rep_dim=1)
        Rcam = tf.concat([Rt, -tf.matmul(Rt, tr)], axis=3)
        Rcam = collapse_dims(Rcam)
        Kmat = collapse_dims(Kmat)


        KRcam = tf.linalg.matmul(Kmat, Rcam)
            
        feats = collapse_dims(feats)
        nR, fh, fw, fdim = feats.get_shape().as_list()
        nR = config.NUM_VIEWS

        rsz_h = float(fh) / config.IMAGE_SHAPE[0]   # image height
        rsz_w = float(fw) / config.IMAGE_SHAPE[1]   # image width

        # Create Voxel grid 
        # !! change coordinates, grid has to be rotated, P-C?!!
        grid_range = tf.range(config.vmin + config.vsize / 2.0, config.vmax,
                                      config.vsize)
        
        # calculate position of grid in world coordinate frame
        grid_position = K.dot(tf.reshape(Rcam_old[0,0,:,:], [3,4]), tf.constant([0.0, 0.0, 0.9375, 1.0], shape=[4,1]) )
        grid_position = tf.reshape(grid_position, [3])

        # adjust grid coordinates to world frame
        grid = tf.stack(
            tf.meshgrid(grid_range + grid_position[0] , grid_range + grid_position[1], grid_range + grid_position[2]))# set z-offset from camera to grid to 1
        rs_grid = tf.reshape(grid, [3, -1])
        nV = rs_grid.get_shape()[1].value
        rs_grid = tf.concat([rs_grid, tf.ones([1, nV])], axis=0)

        # Project grid
        im_p = tf.matmul(tf.reshape(KRcam, [-1, 4]), rs_grid) 
        im_x, im_y, im_z = im_p[::3, :], im_p[1::3, :], im_p[2::3, :]
        im_x = (im_x / im_z) * rsz_w
        im_y = (im_y / im_z) * rsz_h
        

        # Bilinear interpolation
        im_x = tf.clip_by_value(im_x, 0, fw - 1)
        im_y = tf.clip_by_value(im_y, 0, fh - 1)
        im_x0 = tf.cast(tf.floor(im_x), 'int32')
        im_x1 = im_x0 + 1
        im_y0 = tf.cast(tf.floor(im_y), 'int32')
        im_y1 = im_y0 + 1
        im_x0_f, im_x1_f = tf.to_float(im_x0), tf.to_float(im_x1)
        im_y0_f, im_y1_f = tf.to_float(im_y0), tf.to_float(im_y1)

        ind_grid = tf.range(0, nR)
        ind_grid = tf.expand_dims(ind_grid, 1)
        im_ind = tf.tile(ind_grid, [1, nV])

        def _get_gather_inds(x, y):
            indices = tf.reshape(tf.stack([im_ind, y, x], axis=2), [-1, 3])
            print_op = tf.print("indices: ", [indices])
            with tf.control_dependencies([print_op]):
                indices = indices *1
            return indices

        # Gather  values
        Ia = tf.gather_nd(feats, _get_gather_inds(im_x0, im_y0))
        Ib = tf.gather_nd(feats, _get_gather_inds(im_x0, im_y1))
        Ic = tf.gather_nd(feats, _get_gather_inds(im_x1, im_y0))
        Id = tf.gather_nd(feats, _get_gather_inds(im_x1, im_y1))
        # Calculate bilinear weights
        wa = (im_x1_f - im_x) * (im_y1_f - im_y)
        wb = (im_x1_f - im_x) * (im_y - im_y0_f)
        wc = (im_x - im_x0_f) * (im_y1_f - im_y)
        wd = (im_x - im_x0_f) * (im_y - im_y0_f)
        wa, wb = tf.reshape(wa, [-1, 1]), tf.reshape(wb, [-1, 1])
        wc, wd = tf.reshape(wc, [-1, 1]), tf.reshape(wd, [-1, 1])
        Ibilin = tf.add_n([wa * Ia, wb * Ib, wc * Ic, wd * Id])

        fdim = Ibilin.get_shape()[-1].value
        Ibilin = tf.reshape(Ibilin, [
            config.vox_bs, config.NUM_VIEWS, config.nvox, config.nvox, config.nvox,
            fdim
        ])
        Ibilin = tf.transpose(Ibilin, [0, 1, 3, 2, 4, 5])

        return [Ibilin, grid_position]


class proj_grid(KL.Layer):
    def __init__(self, config, **kwargs):
        self.config = config
        super(proj_grid, self).__init__(**kwargs)
    def call(self, inputs):
        grid, grid_pos, Rcam, Kmat = inputs
        config = self.config
        proj_size = config.PROJ_SIZE
        rsz_factor = float(proj_size) / config.IMAGE_SHAPE[0]   # image height
        
        print_op = tf.print("resize_factor: ", rsz_factor, {2: rsz_factor})
        with tf.control_dependencies([print_op]):
            Kmat = Kmat * rsz_factor

        Kmat = repeat_tensor(Kmat, config.NUM_VIEWS, rep_dim=1)
        K_shape = Kmat.get_shape().as_list()
        print("K_shape: {}".format(K_shape))
        bs, im_bs, h, w, d, ch = grid.get_shape().as_list()
        print("grid_shape: {}".format(grid.get_shape().as_list()))
        npix = proj_size**2
        with tf.variable_scope('ProjSlice'):
            # Setup dimensions
            with tf.name_scope('PixelCenters'):
                # Setup image grids to unproject along rays
                im_range = tf.range(0.5, proj_size, 1)
                im_grid = tf.stack(tf.meshgrid(im_range, im_range))
                rs_grid = tf.reshape(im_grid, [2, -1])
                # Append rsz_factor to ensure that
                rs_grid = tf.concat(
                    [rs_grid, tf.ones((1, npix)) * rsz_factor], axis=0)
                rs_grid = tf.reshape(rs_grid, [1, 1, 3, npix])
                rs_grid = tf.tile(rs_grid, [bs, im_bs, 1, 1])

            with tf.name_scope('Im2Cam'):
                # Compute Xc - points in camera frame
                Xc = tf.matrix_triangular_solve(
                    Kmat, rs_grid, lower=False, name='KinvX')

                print("Xc_shape: {}".format(Xc.get_shape().as_list()))
                Xc = tf.concat(
                    [Xc, tf.ones([bs, im_bs, 1, npix])],
                    axis=-2)
            with tf.name_scope('Cam2World'):
#                 Rcam = repeat_tensor(Rcam, config.samples, rep_dim=2)
                Xw = tf.matmul(Rcam, Xc)
                # Transform world points to grid locations to sample from
                grid_pos = repeat_tensor(grid_pos, npix, rep_dim=-1)
                Xw = (Xw - grid_pos - config.vmin)
                Xw = (Xw / (config.vmax - config.vmin)) * config.nvox
                
                # bs, im_bs, samples, npix, 3
                
                Xw = tf.transpose(Xw, [0, 1, 3, 2])
            with tf.name_scope('Interp'):
                sample_grid = collapse_dims(grid)
                sample_locs = collapse_dims(Xw)
   
                lshape = [bs * im_bs] + tf_static_shape(sample_locs)[1:] #tf_static_shape(sample_locs)
                vox_idx = tf.range(lshape[0])
#                 vox_idx = repeat_tensor(vox_idx, lshape[1], rep_dim=1)
                vox_idx = tf.reshape(vox_idx, [-1, 1])
                vox_idx = repeat_tensor(vox_idx, 1 * npix, rep_dim=1)
                vox_idx = tf.reshape(vox_idx, [-1, 1])
                sample_idx = tf.concat(
                    [tf.to_float(vox_idx),
                    tf.reshape(sample_locs, [-1, 3])],
                    axis=1)

                g_val = nearest3(sample_grid, sample_idx)
                g_val = tf.reshape(g_val, [
                    bs, im_bs, proj_size, proj_size, -1
                ])
                ray_slices = tf.transpose(g_val, [0, 1, 2, 3, 4])
                print("ray_slices_shape: {}".format(ray_slices.get_shape().as_list()))
                return ray_slices

class grid_reas(KL.Layer):
    def __init__(self, config, **kwargs):
        self.config = config
        super(grid_reas, self).__init__(**kwargs)
    def call(self, inputs):
        grid = inputs

        if self.config.CONV:
            grid = tf.transpose(grid, [0, 2, 3, 4, 5, 1])
            grid_shape = tf_static_shape(grid)
            print("grid_reas_shape1: {}".format(tf_static_shape(grid)))

            grid = tf.reshape(grid, grid_shape[0:4] + [grid_shape[4]*grid_shape[5]])
            print("grid_reas_shape2: {}".format(tf_static_shape(grid)))
            grid = KL.Conv3D(filters=3, kernel_size=(3,3,3), padding='same')(grid)
            grid = repeat_tensor(grid, self.config.NUM_VIEWS, rep_dim=1)

        else:
#             grid = K.sum(grid, axis=1)
            grid = grid[:,0,:,:,:,:] 
            grid = repeat_tensor(grid, self.config.NUM_VIEWS, rep_dim=1)
        print("grid_reas_shape: {}".format(tf_static_shape(grid)))
        return grid
        
            
            
            
def repeat_tensor(T, nrep, rep_dim=1):
        repT = tf.expand_dims(T, rep_dim)
        tile_dim = [1] * len(tf_static_shape(repT))
        tile_dim[rep_dim] = nrep
        repT = tf.tile(repT, tile_dim)
        return repT

def collapse_dims(T):
    shape = tf_static_shape(T)
    return tf.reshape(T, [-1] + shape[2:])

def tf_static_shape(T):
    return T.get_shape().as_list()

def nearest3(grid, idx, clip=False):
    with tf.variable_scope('NearestInterp'):
        _, h, w, d, f = grid.get_shape().as_list()
        x, y, z = idx[:, 1], idx[:, 2], idx[:, 3]
        g_val = tf.gather_nd(grid, tf.cast(tf.round(idx), 'int32'))
        print_op = tf.print("g_val, idx: ", [g_val[0:10], idx[1:10]])
        with tf.control_dependencies([print_op]):
            g_val = g_val * 1
        if clip:
            x_inv = tf.logical_or(x < 0, x > h - 1)
            y_inv = tf.logical_or(y < 0, y > w - 1)
            z_inv = tf.logical_or(z < 0, x > d - 1)
            valid_idx = 1 - \
                tf.to_float(tf.logical_or(tf.logical_or(x_inv, y_inv), z_inv))
            g_val = g_val * valid_idx[tf.newaxis, ...]
        return g_val
    
    
def quat2rot(q):
    '''q = [w, x, y, z]
    https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion'''
    eps = 1e-5
    w, x, y, z = q
    n = np.linalg.norm(q)
    s = (0 if n < eps else 2.0 / n)
    wx = s * w * x
    wy = s * w * y
    wz = s * w * z
    xx = s * x * x
    xy = s * x * y
    xz = s * x * z
    yy = s * y * y
    yz = s * y * z
    zz = s * z * z
    R = np.array([[1 - (yy + zz), xy - wz,
                   xz + wy], [xy + wz, 1 - (xx + zz), yz - wx],
                  [xz - wy, yz + wx, 1 - (xx + yy)]])
    return R
