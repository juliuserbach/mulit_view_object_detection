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

reused_resnet_lay ={}

class ProjectionNet():
    def __init__(self, config):
        self.config = config
        input_image = KL.Input(shape=[config.NUM_VIEWS] + list(config.IMAGE_SHAPE),
                               name="input_image")
        input_R = KL.Input(shape=[config.NUM_VIEWS, 3, 4], 
                              name="input_R")
        input_Kmat = KL.Input(shape=[3, 3],
                              name="input_Kmat")
        
        x, grid_pos = KL.Lambda(lambda x: unproj_feat(x, config))([input_image, input_R, input_Kmat])
        x = KL.Lambda(lambda x: grid_reas(x, 'grid1', config, kernel=(3, 3, 3), filters=256))(x)

        out = KL.Lambda(lambda x: proj_grid(x, config, 100))([x, grid_pos, input_R, input_Kmat])
        
        inputs = [input_image, input_R, input_Kmat]
        outputs = [out, x]
        self.keras_model = KM.Model(inputs, outputs)
        
    def run(self, inputs):
        grid = self.keras_model.predict(
            inputs, batch_size=1, verbose=0)
        K.clear_session()
        return grid
    
    
def unproj_feat(inputs, config):
    feats, Rcam, Kmat = inputs

    # Construct [R^{T}|-R^{T}t]
    Rcam_old = Rcam
    Rt = tf.matrix_transpose(Rcam[:, :, :, :3])
    tr = tf.expand_dims(Rcam[:, :, :, 3], axis=-1)
    # repeat Kmat 
    num_views = tf.shape(feats)[1]
    Kmat = gather_repeat(Kmat, num_views, axis=1)

    Rcam = tf.concat([Rt, -tf.matmul(Rt, tr)], axis=3)
    print("shape of Kmat: {}".format(Kmat.get_shape().as_list()))
    Rcam = collapse_dims(Rcam)
    print("shape of Rcam: {}".format(Rcam.get_shape().as_list()))
    Kmat = collapse_dims(Kmat)


    KRcam = tf.linalg.matmul(Kmat, Rcam)
    
    feats = collapse_dims(feats)
    feats_shape = tf.shape(feats)
    nR = feats_shape[0]
    _, fh, fw, fdim = tf_static_shape(feats)
    nR = num_views * config.BATCH_SIZE

    rsz_h = float(fh) / config.IMAGE_SHAPE[0]   # image height
    rsz_w = float(fw) / config.IMAGE_SHAPE[1]   # image width

    # Create Voxel grid 
    # !! change coordinates, grid has to be rotated, P-C?!!
    grid_range = tf.range(config.vmin + config.vsize / 2.0, config.vmax,
                                  config.vsize)
    grid_range_z = tf.range(-(config.nvox_z-1)*0.5*config.vsize, (config.nvox_z-1)*0.5*config.vsize+config.vsize/2, config.vsize)
    grid_range = tf.expand_dims(grid_range, 0)
    grid_range_z = tf.expand_dims(grid_range_z, 0)
    grid_range = tf.tile(grid_range, [config.BATCH_SIZE, 1])
    grid_range_z = tf.tile(grid_range_z, [config.BATCH_SIZE, 1])
    # calculate position of grid in world coordinate frame
    if hasattr(config, 'GRID_DIST'):
        grid_dist =config.GRID_DIST
    else: 
        grid_dist = 600/320 * config.vmax
    grid_position = K.dot(tf.reshape(Rcam_old[:,0,:,:], [-1, 3, 4]), tf.constant([0.0, 0.0, grid_dist, 1.0], shape=[4,1]) )
    grid_position = tf.reshape(grid_position, [config.BATCH_SIZE, 3])
    print("grid position shape: {}".format(grid_position.get_shape().as_list()))
    # adjust grid coordinates to world frame
    grid = tf.stack(
        tf.meshgrid(grid_range + grid_position[:,0, tf.newaxis],
                    grid_range + grid_position[:,1, tf.newaxis],
                    grid_range_z + grid_position[:,2, tf.newaxis]))# set z-offset from camera so that the grid side length equals the visible width
    rs_grid = tf.reshape(grid, [3, -1])
    nV = tf.shape(rs_grid)[1]
    rs_grid = tf.concat([rs_grid, tf.ones([1, nV])], axis=0)
    print("grid shape: {}".format(rs_grid.get_shape().as_list()))

    # Project grid/
    im_p = tf.matmul(tf.reshape(KRcam, [-1, 4]), rs_grid) 
    im_x, im_y, im_z = im_p[::3, :], im_p[1::3, :], im_p[2::3, :]
    im_x = (im_x / im_z) * rsz_w
    im_y = (im_y / im_z) * rsz_h


    # Bilinear interpolation
#     im_x = tf.clip_by_value(im_x, 0, fw - 1)
#     im_y = tf.clip_by_value(im_y, 0, fh - 1)
#     mask = tf.math.logical_and(im_x > 0,  im_x < fw - 1)
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

    fdim = Ibilin.get_shape().as_list()[-1]
    Ibilin = tf.reshape(Ibilin, [
        config.BATCH_SIZE, num_views, config.nvox, config.nvox, config.nvox_z,
        fdim
    ])
    Ibilin = tf.transpose(Ibilin, [0, 1, 3, 2, 4, 5])

    return [Ibilin, grid_position]
    
    
# def unproj_feat(inputs, config):
#     feats, Rcam, Kmat = inputs

#     # Construct [R^{T}|-R^{T}t]
#     Rcam_old = Rcam
#     Rt = tf.matrix_transpose(Rcam[:, :, :, :3])
#     tr = tf.expand_dims(Rcam[:, :, :, 3], axis=-1)
#     # repeat Kmat 
#     Kmat = repeat_tensor(Kmat, config.NUM_VIEWS, rep_dim=1)
#     Rcam = tf.concat([Rt, -tf.matmul(Rt, tr)], axis=3)
#     Rcam = collapse_dims(Rcam)
#     Kmat = collapse_dims(Kmat)


#     KRcam = tf.linalg.matmul(Kmat, Rcam)

#     feats = collapse_dims(feats)
#     nR, fh, fw, fdim = feats.get_shape().as_list()
#     nR = config.NUM_VIEWS * config.BATCH_SIZE
# #         fh = feats.shape[1].value
# #         fw = feats.shape[2].value
# #         fdim = feats.shape[3].value
#     print("nR: {}".format(nR))
#     rsz_h = float(fh) / config.IMAGE_SHAPE[0]   # image height
#     rsz_w = float(fw) / config.IMAGE_SHAPE[1]   # image width

#     # Create Voxel grid 
#     # !! change coordinates, grid has to be rotated, P-C?!!
#     grid_range = tf.range(config.vmin + config.vsize / 2.0, config.vmax,
#                                   config.vsize)
#     grid_range_z = tf.range(-(config.nvox_z-1)*0.5*config.vsize, (config.nvox_z-1)*0.5*config.vsize+config.vsize/2, config.vsize)
#     print("grid_range_z:  {}".format(grid_range_z.get_shape().as_list()))

#     # calculate position of grid in world coordinate frame
#     grid_dist = 600/320 * config.vmax
#     print(grid_dist)
#     grid_position = K.dot(tf.reshape(Rcam_old[0,0,:,:], [3,4]), tf.constant([0.0, 0.0, config.grid_dist, 1.0], shape=[4,1]) )
#     grid_position = tf.reshape(grid_position, [3])

#     # adjust grid coordinates to world frame
#     grid = tf.stack(
#         tf.meshgrid(grid_range + grid_position[0] , grid_range + grid_position[1], grid_range_z + grid_position[2]))# set z-offset from camera to grid to 1
#     rs_grid = tf.reshape(grid, [3, -1])
#     nV = rs_grid.get_shape()[1].value
#     print("nV:  {}".format(nV))
#     print("rs_grid:  {}".format(rs_grid.get_shape().as_list()))
#     rs_grid = tf.concat([rs_grid, tf.ones([1, nV])], axis=0)

#     # Project grid
#     im_p = tf.matmul(tf.reshape(KRcam, [-1, 4]), rs_grid) 
#     im_x, im_y, im_z = im_p[::3, :], im_p[1::3, :], im_p[2::3, :]
#     im_x = (im_x / im_z) * rsz_w
#     im_y = (im_y / im_z) * rsz_h


#     # Bilinear interpolation
# #     im_x = tf.clip_by_value(im_x, 0, fw - 1)
# #     im_y = tf.clip_by_value(im_y, 0, fh - 1)
#     print_op = tf.print("at clip:", {1: tf.math.reduce_sum(tf.where(tf.math.logical_and(im_x<0, im_x>fw-1)))})
# #             with tf.control_dependencies([print_op]):
# #                 mrcnn_bbox = KL.Lambda(lambda x: x*1.0)(mrcnn_bbox)
#     im_x0 = tf.cast(tf.floor(im_x), 'int32')
#     im_x1 = im_x0 + 1
#     im_y0 = tf.cast(tf.floor(im_y), 'int32')
#     im_y1 = im_y0 + 1
#     im_x0_f, im_x1_f = tf.to_float(im_x0), tf.to_float(im_x1)
#     im_y0_f, im_y1_f = tf.to_float(im_y0), tf.to_float(im_y1)

#     ind_grid = tf.range(0, nR)
#     ind_grid = tf.expand_dims(ind_grid, 1)
#     im_ind = tf.tile(ind_grid, [1, nV])

#     def _get_gather_inds(x, y):
#         indices = tf.reshape(tf.stack([im_ind, y, x], axis=2), [-1, 3])
#         return indices

#     # Gather  values
#     Ia = tf.gather_nd(feats, _get_gather_inds(im_x0, im_y0))
#     Ib = tf.gather_nd(feats, _get_gather_inds(im_x0, im_y1))
#     Ic = tf.gather_nd(feats, _get_gather_inds(im_x1, im_y0))
#     Id = tf.gather_nd(feats, _get_gather_inds(im_x1, im_y1))
#     # Calculate bilinear weights
#     wa = (im_x1_f - im_x) * (im_y1_f - im_y)
#     wb = (im_x1_f - im_x) * (im_y - im_y0_f)
#     wc = (im_x - im_x0_f) * (im_y1_f - im_y)
#     wd = (im_x - im_x0_f) * (im_y - im_y0_f)
#     wa, wb = tf.reshape(wa, [-1, 1]), tf.reshape(wb, [-1, 1])
#     wc, wd = tf.reshape(wc, [-1, 1]), tf.reshape(wd, [-1, 1])
#     Ibilin = tf.add_n([wa * Ia, wb * Ib, wc * Ic, wd * Id])

#     fdim = Ibilin.get_shape()[-1].value
#     Ibilin = tf.reshape(Ibilin, [
#         config.BATCH_SIZE, config.NUM_VIEWS, config.nvox, config.nvox, config.nvox_z,
#         fdim
#     ])
#     Ibilin = tf.transpose(Ibilin, [0, 1, 3, 2, 4, 5])

#     return [Ibilin, grid_position]

def proj_grid(inputs, config, proj_size):
    """ projects the 3D feature grid back into a 2D image plane with equal side lengths of proj_size
    """
    grid, grid_pos, Rcam, Kmat = inputs
    rsz_factor = float(proj_size) / config.IMAGE_SHAPE[0]   # image height
    Kmat = Kmat * rsz_factor

    Kmat = tf.expand_dims(Kmat, axis=1)
   
    bs, h, w, d, ch = grid.get_shape().as_list()
#     im_bs = Rcam.get_shape().as_list()[1]

    Rcam = tf.reshape(Rcam[:,0,:,:], [bs, 1, 3, 4])
    
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
            rs_grid = tf.tile(rs_grid, [bs, 1, 1, 1])#[bs, im_bs, 1, 1]

        with tf.name_scope('Im2Cam'):
            # Compute Xc - points in camera frame
            Xc = tf.matrix_triangular_solve(
                Kmat, rs_grid, lower=False, name='KinvX')
            
            # Define z values of samples along ray
            if hasattr(config, 'GRID_DIST'):
                grid_dist =config.GRID_DIST
            else: 
                grid_dist = 600/320 * config.vmax
            z_samples = tf.linspace(grid_dist - config.vmax*0.8, grid_dist + config.vmax*0.8, config.samples)

            # Transform Xc to Xw using transpose of rotation matrix
            Xc = repeat_tensor(Xc, config.samples, rep_dim=2)
            
            Xc = Xc * z_samples[tf.newaxis, tf.newaxis, :, tf.newaxis,
                                tf.newaxis]
            
            Xc = tf.concat(
                [Xc, tf.ones([bs, 1, config.samples, 1, npix])],#[bs, im_bs, 1, 1]
                axis=-2)
            
        with tf.name_scope('Cam2World'):
            Rcam = repeat_tensor(Rcam, config.samples, rep_dim=2)
            Xw = tf.matmul(Rcam, Xc)
            # Transform world points to grid locations to sample from
            grid_pos = repeat_tensor(grid_pos, npix, rep_dim=-1)
            # take non symmetric grid into account
            vmin = tf.reshape(tf.constant([config.vmin, config.vmin, -config.nvox_z*0.5*config.vsize]), [1, 1, 3, 1])
            vmax = tf.reshape(tf.constant([config.vmax, config.vmax, config.nvox_z*0.5*config.vsize]), [1, 1, 3, 1])
            nvox = tf.reshape(tf.constant([config.nvox*1.0, config.nvox*1.0, config.nvox_z*1.0]), [1, 1, 3, 1]) 
            Xw = (Xw - grid_pos - vmin)
            Xw = (Xw / (vmax - vmin)) * nvox

            # bs, im_bs, samples, npix, 3
            Xw = tf.transpose(Xw, [0, 1, 2, 4, 3])

        with tf.name_scope('Interp'):
#                 sample_grid = collapse_dims(grid)
            sample_grid = grid
            sample_locs = collapse_dims(Xw)

            lshape = [bs * 1] + tf_static_shape(sample_locs)[1:] #tf_static_shape(sample_locs) [bs * 1]
            vox_idx = tf.range(lshape[0])
            vox_idx = repeat_tensor(vox_idx, lshape[1], rep_dim=1)
            vox_idx = tf.reshape(vox_idx, [-1, 1])
            vox_idx = repeat_tensor(vox_idx, 1 * npix, rep_dim=1)
            vox_idx = tf.reshape(vox_idx, [-1, 1])
            sample_idx = tf.concat(
                [tf.to_float(vox_idx),
                tf.reshape(sample_locs, [-1, 3])],
                axis=1)

            g_val = nearest3(sample_grid, sample_idx)
            g_val = tf.reshape(g_val, [
                bs, config.samples, proj_size, proj_size, -1    #bs, im_bs, proj_size, proj_size, -1
            ])
            ray_slices = tf.transpose(g_val, [0, 1, 2, 3, 4])
            return ray_slices

            
def repeat_tensor_tf(T, nrep, rep_dim=1):
        repT = tf.expand_dims(T, rep_dim)
        tile_dim = [1] * len(tf_static_shape(repT))
        tile_dim[rep_dim] = nrep
        repT = tf.tile(repT, tile_dim)
        return repT

def repeat_tensor(T, nrep, rep_dim=1):
        repT = tf.expand_dims(T, rep_dim)
        tile_dim = tf.Variable(tf.ones(shape=tf.rank(repT), dtype=tf.int32))
        tile_dim = tile_dim[rep_dim].assign(nrep)
        repT = tf.tile(repT, tile_dim)
        return repT
    
def gather_repeat(values, repeats, axis=1):
    values = tf.expand_dims(values, axis)
    indices = tf.tile(tf.zeros(shape=1, dtype=tf.int32), repeats[None])
    print(indices)
    return tf.gather(values, indices, axis=axis)
    
def collapse_dims(T):
    shape = tf_static_shape(T)
    shape = tf.concat([tf.constant(-1)[None], shape[2:]], axis=0) 
    return tf.reshape(T, shape)


def tf_static_shape(T):
    return T.get_shape().as_list()

def tf_dynamic_shape(T):
    return tf.shape(T)


# def proj_grid(inputs, config, proj_size):
#     """ projects the 3D feature grid back into a 2D image plane with equal side lengths of proj_size
#     """
#     grid, grid_pos, Rcam, Kmat = inputs
#     rsz_factor = float(proj_size) / config.IMAGE_SHAPE[0]   # image height
#     Kmat = Kmat * rsz_factor

#     Kmat = repeat_tensor(Kmat, 1, rep_dim=1)
#     K_shape = Kmat.get_shape().as_list()
   
#     bs, h, w, d, ch = grid.get_shape().as_list()
# #     im_bs = Rcam.get_shape().as_list()[1]

#     Rcam = tf.reshape(Rcam[:,0,:,:], [bs, 1, 3, 4])
    
#     npix = proj_size**2
#     with tf.variable_scope('ProjSlice'):
#         # Setup dimensions
#         with tf.name_scope('PixelCenters'):
#             # Setup image grids to unproject along rays
#             im_range = tf.range(0.5, proj_size, 1)
#             im_grid = tf.stack(tf.meshgrid(im_range, im_range))
#             rs_grid = tf.reshape(im_grid, [2, -1])
#             # Append rsz_factor to ensure that
#             rs_grid = tf.concat(
#                 [rs_grid, tf.ones((1, npix)) * rsz_factor], axis=0)
#             rs_grid = tf.reshape(rs_grid, [1, 1, 3, npix])
#             rs_grid = tf.tile(rs_grid, [bs, 1, 1, 1])#[bs, im_bs, 1, 1]

#         with tf.name_scope('Im2Cam'):
#             # Compute Xc - points in camera frame
#             Xc = tf.matrix_triangular_solve(
#                 Kmat, rs_grid, lower=False, name='KinvX')
            
            
#             # Define z values of samples along ray
#             grid_dist = 600/320 * config.vmax 
#             grid_dist = config.grid_dist
#             z_samples = tf.linspace(grid_dist - config.vmax, grid_dist + config.vmax, config.samples)

#             # Transform Xc to Xw using transpose of rotation matrix
#             Xc = repeat_tensor(Xc, config.samples, rep_dim=2)
            
#             Xc = Xc * z_samples[tf.newaxis, tf.newaxis, :, tf.newaxis,
#                                 tf.newaxis]
            
#             Xc = tf.concat(
#                 [Xc, tf.ones([bs, 1, config.samples, 1, npix])],#[bs, im_bs, 1, 1]
#                 axis=-2)
            
#         with tf.name_scope('Cam2World'):
#             Rcam = repeat_tensor(Rcam, config.samples, rep_dim=2)
#             Xw = tf.matmul(Rcam, Xc)
#             # Transform world points to grid locations to sample from
#             grid_pos = repeat_tensor(grid_pos, npix, rep_dim=-1)
#             # take non symmetric grid into account
#             vmin = tf.reshape(tf.constant([config.vmin, config.vmin, -config.nvox_z*0.5*config.vsize]), [1, 1, 3, 1])
#             vmax = tf.reshape(tf.constant([config.vmax, config.vmax, config.nvox_z*0.5*config.vsize]), [1, 1, 3, 1])
#             nvox = tf.reshape(tf.constant([config.nvox*1.0, config.nvox*1.0, config.nvox_z*1.0]), [1, 1, 3, 1]) 
#             Xw = (Xw - grid_pos - vmin)
#             Xw = (Xw / (vmax - vmin)) * nvox

#             # bs, im_bs, samples, npix, 3
#             Xw = tf.transpose(Xw, [0, 1, 2, 4, 3])

#         with tf.name_scope('Interp'):
# #                 sample_grid = collapse_dims(grid)
#             sample_grid = grid
#             sample_locs = collapse_dims(Xw)

#             lshape = [bs * 1] + tf_static_shape(sample_locs)[1:] #tf_static_shape(sample_locs) [bs * 1]
#             vox_idx = tf.range(lshape[0])
#             vox_idx = repeat_tensor(vox_idx, lshape[1], rep_dim=1)
#             vox_idx = tf.reshape(vox_idx, [-1, 1])
#             vox_idx = repeat_tensor(vox_idx, 1 * npix, rep_dim=1)
#             vox_idx = tf.reshape(vox_idx, [-1, 1])
#             sample_idx = tf.concat(
#                 [tf.to_float(vox_idx),
#                 tf.reshape(sample_locs, [-1, 3])],
#                 axis=1)

#             g_val = nearest3(sample_grid, sample_idx)
#             g_val = tf.reshape(g_val, [
#                 bs, config.samples, proj_size, proj_size, -1    #bs, im_bs, proj_size, proj_size, -1
#             ])
#             ray_slices = tf.transpose(g_val, [0, 1, 2, 3, 4])
#             return ray_slices

            
# def repeat_tensor(T, nrep, rep_dim=1):
#         repT = tf.expand_dims(T, rep_dim)
#         tile_dim = [1] * len(tf_static_shape(repT))
#         tile_dim[rep_dim] = nrep
#         repT = tf.tile(repT, tile_dim)
#         return repT

# def collapse_dims(T):
#     shape = tf_static_shape(T)
#     return tf.reshape(T, [-1] + shape[2:])

# def tf_static_shape(T):
#     return T.get_shape().as_list()

def nearest3(grid, idx, clip=False):
    with tf.variable_scope('NearestInterp'):
        _, h, w, d, f = grid.get_shape().as_list()
        x, y, z = idx[:, 1], idx[:, 2], idx[:, 3]
        g_val = tf.gather_nd(grid, tf.cast(tf.round(idx), 'int32'))
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


def grid_reas(inputs, scope, config, kernel=(3, 3, 3), filters=256):

    x = inputs
    grid_shape = tf_static_shape(x)
    print("Grid_shape grid_reas: {}".format(grid_shape))
#     x = KL.Lambda(lambda x: tf.transpose(x, [0, 5, 2, 3, 4, 1]))(x)
    print("Grid_shape grid_reas2: {}".format(tf_static_shape(x)))

#     x = KL.Lambda(lambda x: tf.reshape(x, grid_shape[0:4] + [grid_shape[4]*grid_shape[5]]))(x)
    name_conv = scope + '/3D_conv'
    name_bn = scope + '/batch_norm'
    if config.GRID_REAS == 'mean':
        x = KL.Lambda(lambda x: K.mean(x, axis=-1)[:, :, :, :, :, None])(x)
        x = KL.Lambda(lambda x: tf.transpose(x, [0, 5, 2, 3, 4, 1]))(x)
        x = KL.Lambda(lambda x: tf.reshape(x, [grid_shape[0]] + grid_shape[2:]))(x)
    elif config.GRID_REAS == 'conv3d':
        if name_conv not in reused_resnet_lay:
            reused_resnet_lay[name_conv] = KL.Conv3D(filters=1, kernel_size=(3,3,3), padding='same', name=name_conv)
        x = KL.TimeDistributed(reused_resnet_lay[name_conv], name=name_conv)(x)
        print("weights_shape: {}".format(reused_resnet_lay[name_conv].weights[0].shape))
        x = KL.Lambda(lambda x: tf.transpose(x, [0, 5, 2, 3, 4, 1]))(x)
        x = KL.Lambda(lambda x: tf.reshape(x, [grid_shape[0]] + grid_shape[2:]))(x)
    elif config.GRID_REAS == 'ident':
        x_shape = x.shape.as_list()
        x = KL.Lambda(lambda x: tf.transpose(x, [0, 2, 3, 4, 1, 5]))(x)
        x = KL.Lambda(lambda x: tf.reshape(x, [x_shape[0]]+x_shape[2:-1]+[config.NUM_VIEWS*x_shape[-1]]))(x)
#         x_con = x[:,0,:,:,:,:]
#         for i in range(config.NUM_VIEWS-1):
#             x_con = KL.Lambda(lambda x: tf.concat([x_con[:,i,:,:,:,:], x[:,i+1,:,:,:,:]], axis=-1))(x)
        print("Grid_shape after reshape: {}".format(x.get_shape().as_list()))
        x = KL.Lambda(lambda x: x[:, :, :, :, 3:])(x)
    
#     x = KL.BatchNormalization(name=name_bn)(x, training=config.TRAIN_BN)
    print("Grid_shape grid_reas_end: {}".format(tf_static_shape(x)))
    x = KL.Activation('relu')(x)
    return x