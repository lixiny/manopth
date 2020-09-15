import os
from jax import numpy as np
from jax import jit
import pickle

from manopth.anchorutils import anchor_load


class AnchorLayer:
    def __init__(self, anchor_root):
        face_vert_idx, anchor_weight, merged_vertex_assignment, anchor_mapping = anchor_load(anchor_root)
        self.face_vert_idx = np.expand_dims(np.asarray(face_vert_idx), 0)
        self.anchor_weight = np.expand_dims(np.asarray(anchor_weight), 0)
        self.merged_vertex_assignment = np.asarray(merged_vertex_assignment)
        self.anchor_mapping = anchor_mapping

    @staticmethod
    def _recov_anchor_batch(vertices, idx, weights):
        # vertices = ARRAY[NBATCH, 778, 3]
        # idx = ARRAY[1, 32, 3]
        # weights = ARRAY[1, 32, 2]
        batch_size = vertices.shape[0]
        batch_idx = np.arange(batch_size)[:, None, None]  # ARRAY[NBATCH, 1, 1]
        indexed_vertices = vertices[batch_idx, idx, :]  # ARRAY[NBATCH, 32, 3, 3]
        base_vec_1 = indexed_vertices[:, :, 1, :] - indexed_vertices[:, :, 0, :]  # ARRAY[NBATCH, 32, 3]
        base_vec_2 = indexed_vertices[:, :, 2, :] - indexed_vertices[:, :, 0, :]  # ARRAY[NBATCH, 32, 3]
        weights_1 = weights[:, :, 0:1]  # ARRAY[1, 32, 1]
        weights_2 = weights[:, :, 1:2]  # ARRAY[1, 32, 1]
        rebuilt_anchors = weights_1 * base_vec_1 + weights_2 * base_vec_2  # ARRAY[NBATCH, 32, 3]
        origins = indexed_vertices[:, :, 0, :]  # ARRAY[NBATCH, 32, 3]
        rebuilt_anchors = rebuilt_anchors + origins
        return rebuilt_anchors

    def __call__(self, vertices):
        """
        vertices: ARRAY[N_BATCH, 778, 3]
        """
        anchor_pos = self._recov_anchor_batch(vertices, self.face_vert_idx, self.anchor_weight)
        return anchor_pos

