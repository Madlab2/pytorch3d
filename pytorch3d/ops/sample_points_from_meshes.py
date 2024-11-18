# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


"""
This module implements utility functions for sampling points from
batches of meshes.
"""
import sys
from typing import Tuple, Union

import torch
from pytorch3d.ops.mesh_face_areas_normals import mesh_face_areas_normals
from pytorch3d.ops.packed_to_padded import packed_to_padded
from pytorch3d.renderer.mesh.rasterizer import Fragments as MeshFragments
from pytorch3d.ops import cot_laplacian
from pytorch3d.structures import Meshes

def compute_mean_curvature(verts_packed, faces_packed, vert_normals, signed=True):
    """
    Compute mean curvature at each vertex using PyTorch3D.
    
    Args:
        verts (torch.Tensor): Vertex coordinates of shape (V, 3), where V is the number of vertices.
        faces (torch.Tensor): Face indices of shape (F, 3), where F is the number of faces.
    
    Returns:
        torch.Tensor: Mean curvature for each vertex of shape (V,).
    """
    with torch.no_grad():
        L, inv_areas = cot_laplacian(verts_packed, faces_packed)  # Shape: (V, V)
    
        # Apply the Laplacian to the vertex positions to get curvature approximation
        mean_curvature_normals = torch.sparse.mm(L.double(), verts_packed.double())  # Shape: (V, 3)
        
        # Compute mean curvature as the magnitude of the mean curvature normal vector
        mean_curvature = mean_curvature_normals.norm(dim=1)  # Shape: (V,)
    
    if signed:
        # Compute the signed mean curvature by projecting onto the normal
        signs = (mean_curvature_normals * vert_normals).sum(dim=1)
        signed_mean_curvature = signs * mean_curvature
        # Standard-normalize the mean curvature values
        mean = signed_mean_curvature.mean()
        std = signed_mean_curvature.std()
        return (signed_mean_curvature - mean) / std
    else:
        return mean_curvature

def sample_points_from_meshes(
    meshes,
    num_samples: int = 10000,
    return_normals: bool = False,
    return_textures: bool = False,
    interpolate_features: str =  None,
    use_centroids : bool = False,
    return_curvature : bool = False,
    sample_face_idxs_raw : torch.Tensor = None
) -> Union[
    Tuple[torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
]:
    """
    Convert a batch of meshes to a batch of pointclouds by uniformly sampling
    points on the surface of the mesh with probability proportional to the
    face area.

    Args:
        meshes: A Meshes object with a batch of N meshes.
        num_samples: Integer giving the number of point samples per mesh.
        return_normals: If True, return normals for the sampled points.
        return_textures: If True, return textures for the sampled points.
        interpolate_features: If 'barycentric', use barycentric coordinates of
            the sampled surface points to interpolate vertex features. If
            'majority', use majority voting for the class of the sampled
            surface points. If 'nearest', assign class of vertex with highest
            barycentric weight to sampled vertex.

    Returns:
        4-element tuple containing

        - **samples**: FloatTensor of shape (N, num_samples, 3) giving the
          coordinates of sampled points for each mesh in the batch. For empty
          meshes the corresponding row in the samples array will be filled with 0.
        - **normals**: FloatTensor of shape (N, num_samples, 3) giving a normal vector
          to each sampled point. Only returned if return_normals is True.
          For empty meshes the corresponding row in the normals array will
          be filled with 0.
        - **textures**: FloatTensor of shape (N, num_samples, C) giving a C-dimensional
          texture vector to each sampled point. Only returned if return_textures is True.
          For empty meshes the corresponding row in the textures array will
          be filled with 0.
        - **sample_features**: Tensor of shape (N, num_samples, D) giving
        D-dimensional features per point interpolated from the vertex features
        of the mesh.

        Note that in a future releases, we will replace the 3-element tuple output
        with a `Pointclouds` datastructure, as follows

        .. code-block:: python

            Pointclouds(samples, normals=normals, features=textures)
    """
    if meshes.isempty():
        raise ValueError("Meshes are empty.")

    verts = meshes.verts_packed()
    features = meshes.verts_features_packed()
    if not torch.isfinite(verts).all():
        raise ValueError("Meshes contain nan or inf.")

    if return_textures and meshes.textures is None:
        raise ValueError("Meshes do not contain textures.")

    if (interpolate_features is not None and features is None):
        raise ValueError("Meshes do not contain vertex features.")

    # hard coded
    num_samples = 100000 # use 50.000 when no replacement sampling

    faces = meshes.faces_packed()
    mesh_to_face = meshes.mesh_to_faces_packed_first_idx()
    num_meshes = len(meshes)
    num_valid_meshes = torch.sum(meshes.valid)  # Non empty meshes.

    # Initialize samples tensor with fill value 0 for empty meshes.
    samples = torch.zeros((num_meshes, num_samples, 3), device=meshes.device)
    
    if sample_face_idxs_raw is None:
        # Only compute samples for non empty meshes
        with torch.no_grad():
            areas, _ = mesh_face_areas_normals(verts, faces)  # Face areas can be zero.
            max_faces = meshes.num_faces_per_mesh().max().item()
            areas_padded = packed_to_padded(
                areas, mesh_to_face[meshes.valid], max_faces
            )  # (N, F)

            # TODO (gkioxari) Confirm multinomial bug is not present with real data.
            sample_face_idxs_raw = areas_padded.multinomial(
                num_samples, replacement=True
            )  # (N, num_samples)
    
    sample_face_idxs = sample_face_idxs_raw + mesh_to_face[meshes.valid].view(num_valid_meshes, 1)
    print("min, max, shape from prediction-mesh samples:", sample_face_idxs.min(), sample_face_idxs.max(), sample_face_idxs.shape)
    print("faces shape:", faces.shape)
    # Get the vertex coordinates of the sampled faces.
    face_verts = verts[faces]
    v0, v1, v2 = face_verts[:, 0], face_verts[:, 1], face_verts[:, 2]

    # Randomly generate barycentric coords.
    w0, w1, w2 = _rand_barycentric_coords(
        num_valid_meshes, num_samples, verts.dtype, verts.device
    ) # [1, 100000] in loss, [2, 100000] in target

    # Use the barycentric coords to get a point on each sampled face.
    a = v0[sample_face_idxs]  # (N, num_samples, 3)
    b = v1[sample_face_idxs]  # (1, 100000, 3) in loss, but (2, 100000, 3) in target. Apparently totally okay!
    c = v2[sample_face_idxs]
    if use_centroids:
        # no random point on face but use the barycentric centroid
        samples[meshes.valid] = 1/3 *a + 1/3 * b + 1/3 * c
    else:
        samples[meshes.valid] = w0[:, :, None] * a + w1[:, :, None] * b + w2[:, :, None] * c
    

    if return_normals:
        # Initialize normals tensor with fill value 0 for empty meshes.
        # Normals for the sampled points are face normals computed from
        # the vertices of the face in which the sampled point lies.
        normals = torch.zeros((num_meshes, num_samples, 3), device=meshes.device)
        vert_normals_all = (v1 - v0).cross(v2 - v1, dim=1)
        vert_normals_all = vert_normals_all / vert_normals_all.norm(dim=1, p=2, keepdim=True).clamp(
            min=sys.float_info.epsilon
        )
        vert_normals = vert_normals_all[sample_face_idxs]
        normals[meshes.valid] = vert_normals
    
    if return_curvature:
        #assert num_meshes == 1
        #compute curvature per vertex and then use mean of three vertex curvatures for each sample
        vertex_normals = meshes.verts_normals_packed()
        vertex_curvatures = compute_mean_curvature(verts, faces, vertex_normals, signed=True).half()
        curvatures_v0 = vertex_curvatures[faces[sample_face_idxs][:, :, 0]]
        curvatures_v1 = vertex_curvatures[faces[sample_face_idxs][:, :, 1]]
        curvatures_v2 = vertex_curvatures[faces[sample_face_idxs][:, :, 2]]
        sampled_face_curvatures = (curvatures_v0 + curvatures_v1 + curvatures_v2) / 3  # (N, num_samples)

    if return_textures:
        # fragment data are of shape NxHxWxK. Here H=S, W=1 & K=1.
        pix_to_face = sample_face_idxs.view(len(meshes), num_samples, 1, 1)  # NxSx1x1
        bary = torch.stack((w0, w1, w2), dim=2).unsqueeze(2).unsqueeze(2)  # NxSx1x1x3
        # zbuf and dists are not used in `sample_textures` so we initialize them with dummy
        dummy = torch.zeros(
            (len(meshes), num_samples, 1, 1), device=meshes.device, dtype=torch.float32
        )  # NxSx1x1
        fragments = MeshFragments(
            pix_to_face=pix_to_face, zbuf=dummy, bary_coords=bary, dists=dummy
        )
        textures = meshes.sample_textures(fragments)  # NxSx1x1xC
        textures = textures[:, :, 0, 0, :]  # NxSxC

    if interpolate_features is not None:
        D = features.shape[-1] # feature-dim
        # Features for the sampled points are features computed from
        # the vertices of the face in which the sampled point lies.
        # Initialize features tensor with fill value -1 for empty meshes.
        sample_features = -1 * torch.ones((num_meshes, num_samples, D), device=meshes.device)
        face_features = features[faces]
        if interpolate_features == 'barycentric':
            f0, f1, f2 = face_features[:, 0], face_features[:, 1], face_features[:, 2]
            # Use the barycentric coords to interpolate features
            a = f0[sample_face_idxs]  # (N, num_samples, D)
            b = f1[sample_face_idxs]
            c = f2[sample_face_idxs]
            sample_features[meshes.valid] = w0[:, :, None] * a + w1[:, :, None] * b + w2[:, :, None] * c

        elif interpolate_features == 'majority':
            face_features_sampled = face_features[sample_face_idxs]
            sample_features = torch.mode(face_features_sampled, dim=2)[0]

        elif interpolate_features == 'nearest':
            face_features_sampled = face_features[sample_face_idxs]
            # Nearest = highest barycentric weight
            nearest_idx = torch.argmax(torch.stack([w0, w1, w2]), dim=0)
            sample_features = torch.gather(
                face_features_sampled.squeeze(-1),
                2,
                nearest_idx.unsqueeze(-1)
            )

        else:
            raise ValueError("Unknown interpolation %s", interpolate_features)

    # return
    # TODO(gkioxari) consider returning a Pointclouds instance [breaking]
    # TODO(fabibo3): Pointclouds do not support features and textures at the
    # same time
    if return_normals and return_textures and interpolate_features:
        # pyre-fixme[61]: `sample_features` may not be initialized here.
        # pyre-fixme[61]: `normals` may not be initialized here.
        # pyre-fixme[61]: `textures` may not be initialized here.
        return samples, normals, textures, sample_features, sample_face_idxs
    if return_normals and return_textures:
        # pyre-fixme[61]: `normals` may not be initialized here.
        # pyre-fixme[61]: `textures` may not be initialized here.
        return samples, normals, textures, sample_face_idxs
    
    # if return_normals and return_curvature and interpolate_features:
    #     return samples, normals, sample_features, sampled_face_curvatures, sample_face_idxs
    
    if return_normals and interpolate_features:  # return_textures is False
        # pyre-fixme[61]: `sample_features` may not be initialized here.
        # pyre-fixme[61]: `normals` may not be initialized here.
        return samples, normals, sample_features, sample_face_idxs
    
    if return_normals and return_curvature:
        return samples, normals, sampled_face_curvatures, sample_face_idxs
    
    if return_textures and interpolate_features:  # return_normals is False
        # pyre-fixme[61]: `sample_features` may not be initialized here.
        # pyre-fixme[61]: `textures` may not be initialized here.
        return samples, textures, sample_features, sample_face_idxs
    if interpolate_features:
        # pyre-fixme[61]: `sample_features` may not be initialized here.
        return samples, sample_features, sample_face_idxs
    if return_normals:  # return_textures is False
        # pyre-fixme[61]: `normals` may not be initialized here.
        return samples, normals, sample_face_idxs
    if return_textures:  # return_normals is False
        # pyre-fixme[61]: `textures` may not be initialized here.
        return samples, textures, sample_face_idxs
    return samples, sample_face_idxs


def _rand_barycentric_coords(
    size1, size2, dtype: torch.dtype, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Helper function to generate random barycentric coordinates which are uniformly
    distributed over a triangle.

    Args:
        size1, size2: The number of coordinates generated will be size1*size2.
                      Output tensors will each be of shape (size1, size2).
        dtype: Datatype to generate.
        device: A torch.device object on which the outputs will be allocated.

    Returns:
        w0, w1, w2: Tensors of shape (size1, size2) giving random barycentric
            coordinates
    """
    uv = torch.rand(2, size1, size2, dtype=dtype, device=device)
    u, v = uv[0], uv[1]
    u_sqrt = u.sqrt()
    w0 = 1.0 - u_sqrt
    w1 = u_sqrt * (1.0 - v)
    w2 = u_sqrt * v
    # pyre-fixme[7]: Expected `Tuple[torch.Tensor, torch.Tensor, torch.Tensor]` but
    #  got `Tuple[float, typing.Any, typing.Any]`.
    return w0, w1, w2
