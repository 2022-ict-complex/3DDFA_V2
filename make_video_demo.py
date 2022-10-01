import os
import sys
import torch
import pytorch3d
import argparse
import matplotlib.pyplot as plt

# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes, load_obj

# Data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    TexturesUV,
    TexturesVertex,
    TexturesAtlas
)
from pytorch3d.renderer.blending import hard_rgb_blend, BlendParams
from trimesh.visual import texture, TextureVisuals
from trimesh import Trimesh

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    device_numb = 0
else:
    device = torch.device("cpu")
    device_numb = None

import sys
import os

import numpy as np
from PIL import Image

import trimesh

from tqdm import tqdm

import cv2

class SimpleShader(torch.nn.Module):
    def __init__(self, device="cpu", blend_params=None):
        super().__init__()
        self.blend_params = blend_params if blend_params is not None else BlendParams()

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        blend_params = kwargs.get("blend_params", self.blend_params)
        texels = meshes.sample_textures(fragments)
        images = hard_rgb_blend(texels, fragments, blend_params)
        return images  # (N, H, W, 3) RGBA image


def load_colored_obj(filename):
    trimesh_mesh = trimesh.load_mesh(filename, process=False)

    if hasattr(trimesh_mesh.visual, 'vertex_colors'):
        faces = np.array(trimesh_mesh.faces, dtype=np.float32)
        verts = np.array(trimesh_mesh.vertices, dtype=np.float32)

        # faces /= 100
    #     verts /= (verts.max()*10)

    #     verts -= verts.mean()
    #     verts /= verts.std()

        verts -= verts.min()
        verts /= verts.max()
        verts -= 0.5 # -0.5 ~ 0.5
        verts = verts / 5 * 2 
        # print(verts.min(), verts.max())

        verts, faces = torch.from_numpy(verts).float(), torch.from_numpy(faces)

        verts_rgb = torch.from_numpy(trimesh_mesh.visual.vertex_colors[None,:,:3]).float()
        verts_rgb /= 255
        textures = TexturesVertex(verts_features=verts_rgb.to(device))

        mesh = Meshes(
            verts=[verts.to(device)],   
            faces=[faces.to(device)],
            textures=textures
        )
    else:
        verts, faces, aux = load_obj(
            filename,
            device=device_numb,
            load_textures=True,
            create_texture_atlas=True,
            # texture_atlas_size=4,
            # texture_wrap="repeat"
            )
        atlas = aux.texture_atlas

        # verts -= verts.mean()
        # # verts /= (verts.abs().max()*10)
        # verts /= (verts.abs().max()*5)

        verts -= verts.min()
        verts /= verts.max()
        verts -= 0.5 # -0.5 ~ 0.5
        verts = verts / 5 * 2 
        # print(verts.min(), verts.max())

        mesh = Meshes(
            verts=[verts],
            faces=[faces.verts_idx],
            textures=TexturesAtlas(atlas=[atlas]),
        )

        # # Render the plotly figure
        # fig = plot_scene({
        #     "subplot1": {
        #         "cow_mesh": mesh
        #     }
        # })
        # fig.show()

    return mesh


def mesh2png_with_rotation(mesh, pitch=0.0, yaw=0.0, output_size=512):
    # R, T = look_at_view_transform(1, pitch, -7+yaw)
    # T[0][0] += 0.01
    # T[0][1] -= 0.01
    # cameras = FoVPerspectiveCameras(device=device, R=R, T=T, fov=12)

    R, T = look_at_view_transform(1.2, -3+pitch, -6+yaw)
    T[0][0] += 0.01
    T[0][1] -= 0.01
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T, fov=23)

    # R, T = look_at_view_transform(1.8, pitch, -3+yaw)
    # T[0][0] += 0.1
    # T[0][1] -= 0.1
    # cameras = FoVPerspectiveCameras(device=device, R=R, T=T, fov=60)

    raster_settings = RasterizationSettings(
        image_size=output_size, 
        # blur_radius=1e-6,
    )

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings
        ),
        shader=SimpleShader(
            device=device, 
        )
    )

    images = renderer(mesh).cpu().numpy()[0]
    images *= 255
    images = images.astype(np.uint8)
                
    return images


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-i', '--input', default='3d_src_auto/input', type=str,
        help='path to the input directory, where input images are stored.'
    )
    parser.add_argument(
        '-o', '--output', default='3d_src_auto/3d_result', type=str,
        help='path to the output directory, where results(obj,txt files) will be stored.'
    )
    parser.add_argument(
        '--output-size', default=1024, type=int
    )
    args = parser.parse_args()

    obj_target_path = args.input
    obj_filename_list = [
        os.path.join(obj_target_path, filename) 
            for filename in os.listdir(obj_target_path) 
                if filename.endswith('.obj')
    ]

    src_result_base_path = args.output
    os.makedirs(src_result_base_path, exist_ok=True)

    for obj_filename in tqdm(obj_filename_list):
        basename = os.path.basename(obj_filename)
        name = os.path.splitext(basename)[0]

        mesh = load_colored_obj(obj_filename)


        src_result_path = os.path.join(src_result_base_path, f'{name}_3DDFAV2.mp4')
        video_fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            src_result_path,
            video_fourcc, 
            30, 
            (args.output_size,args.output_size)
        )

        temp_pitch_range = list(range(30, -30-1, -10))
        for pitch in temp_pitch_range:
            temp_yaw_range = list(range(0, -61, -3))
            temp_yaw_range += list(range(-60, 60+1, 3))
            temp_yaw_range += list(range(60, -1, -3))
            for yaw in temp_yaw_range:
                images = mesh2png_with_rotation(
                    mesh, 
                    pitch=pitch, 
                    yaw=yaw,
                    output_size=args.output_size
                )
                pil_image = Image.fromarray(images)
                # new_image = pil_image
                new_image = Image.new("RGBA", pil_image.size, 'BLACK') # Create a white rgba background
                # new_image = Image.new("RGBA", pil_image.size, (0,0,0,200)) # Create a white rgba background
                new_image.paste(pil_image, (0, 0), pil_image)

                new_image = np.array(new_image)[:,:,:3]
                new_image = new_image[:,:,::-1]

                out.write(new_image)
            #     break
            # break


        out.release()