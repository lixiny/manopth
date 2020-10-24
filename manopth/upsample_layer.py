import torch
import numpy as np
from torch.nn import Module

from manopth.manolayer import ManoLayer
import open3d as o3d


class UpSampleLayer(Module):
    def __init__(self):
        super().__init__()

    def forward(self, vertices, faces):
        """
           *
          / \
         /   \
        * --- *
           |
           *
          /|\
         / o \
        * --- *
        """
        device = vertices.device
        new_verts = torch.mean(vertices[faces], dim=-2)
        new_idx_head = torch.cat([faces[..., [0, 1]], faces[..., [1, 2]], faces[..., [2, 0]]], dim=0)
        new_idx_tail = torch.arange(vertices.shape[-2], vertices.shape[-2] + faces.shape[-2]).repeat(3).to(device)

        new_faces = torch.cat([new_idx_head, new_idx_tail[:, None]], dim=-1)
        new_verts = torch.cat([vertices, new_verts], dim=-2)
        return new_verts, new_faces


if __name__ == "__main__":
    ncomps = 6

    mano_layer = ManoLayer(mano_root="data/mano", use_pca=True, ncomps=ncomps, flat_hand_mean=False)

    random_shape = torch.rand(1, 10)
    random_pose = torch.rand(1, ncomps + 3)

    # Forward pass through MANO layer
    hand_verts, _ = mano_layer(random_pose, random_shape)
    hand_faces = mano_layer.th_faces

    print(hand_verts.shape, hand_faces.shape)

    hand_mesh = o3d.geometry.TriangleMesh()
    hand_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(hand_faces))
    hand_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(hand_verts.squeeze(0)))
    hand_mesh.compute_vertex_normals()

    vis_pred = o3d.visualization.Visualizer()
    vis_pred.create_window(window_name="Predicted Hand", width=1080, height=1080)
    vis_pred.add_geometry(hand_mesh)

    up_sample_layer = UpSampleLayer()
    hand_verts = hand_verts.squeeze(0)
    up_verts, up_faces = hand_verts, hand_faces
    for _ in range(5):
        up_verts, up_faces = up_sample_layer(up_verts, up_faces)

    print(up_verts.shape)
    print(up_faces.shape)

    hand_up = o3d.geometry.TriangleMesh()
    np.random.shuffle(np.asarray(up_faces))
    hand_up.triangles = o3d.utility.Vector3iVector(up_faces[:50000])
    hand_up.vertices = o3d.utility.Vector3dVector(np.asarray(up_verts))
    hand_up.compute_vertex_normals()

    vis_up = o3d.visualization.Visualizer()
    vis_up.create_window(window_name="Up Hand", width=1080, height=1080)
    vis_up.add_geometry(hand_up)

    while True:
        vis_pred.update_geometry(hand_mesh)
        vis_up.update_geometry(hand_up)
        vis_up.update_renderer()
        vis_pred.update_renderer()

        if not vis_pred.poll_events() or not vis_up.poll_events():
            break
