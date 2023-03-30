import torch
import torchvision

import kornia

from PIL                    import Image
from utils.video            import Video
from utils.limit_subdivide  import LimitSubdivide
from utils.helpers          import cosine_avg, create_scene
from utils.camera           import CameraBatch, get_camera_params
from utils.resize_right     import resize, cubic, linear, lanczos2, lanczos3
from nvdiffmodeling.src     import util
from nvdiffmodeling.src     import obj
from nvdiffmodeling.src     import mesh
from nvdiffmodeling.src     import render
from nvdiffmodeling.src     import texture
from nvdiffmodeling.src     import regularizer
import trimesh

import nvdiffrast.torch as dr



def nvdiff_meshes(cfg, device=0):

    # store Mesh objects
    # meshes=["/home/hyoshida/git/CLIP-Mesh/primitives/spot.obj"]
    meshes = [] # store Mesh objects
    subdiv = [] # store per mesh limit subdivison
    train_params = [] # store all trainable paramters
    vert_trains = []

    for idx, m in enumerate(cfg["meshes"]): 
        # os.path.dirname(meshes_pathes[0]) 
        load_mesh = obj.load_obj(m)

        if cfg["unit"][idx]: # If mesh is to be unit sized
            try:
                load_mesh = mesh.unit_size(load_mesh)
            except:
                from nvdiffmodeling.src import mesh
                load_mesh = mesh.unit_size(load_mesh)

        v_pos = torch.tensor(cfg["scales"][idx]).to(load_mesh.v_pos.device) * load_mesh.v_pos.clone().detach()
        v_pos = torch.tensor(cfg["offsets"][idx]).to(v_pos.device) + v_pos.clone().detach()

        # Final mesh after all adjustments
        load_mesh = mesh.Mesh(v_pos, base=load_mesh)
        # If true is in train_mesh_idx[mesh_idx] then we initialize
        # all textures else we start with textures already on mesh
        if True in cfg["train_mesh_idx"][idx]:

            # vertices 
            vertices = load_mesh.v_pos.clone().detach().requires_grad_(True)

            # faces
            faces = load_mesh.t_pos_idx.clone().detach()

            # texture map
            texture_map = texture.create_trainable(np.random.uniform(size=[cfg["texture_resolution"]]*2 + [cfg["channels"]], low=0.0, high=1.0), [cfg["texture_resolution"]]*2, True)

            # normal map
            normal_map = texture.create_trainable(np.array([0, 0, 1]), [cfg["texture_resolution"]]*2, True)

            # specular map
            specular_map = texture.create_trainable(np.array([0, 0, 0]), [cfg["texture_resolution"]]*2, True)

        else:

            # vertices 
            vertices = load_mesh.v_pos.clone().detach().requires_grad_(True)

            # faces
            faces = load_mesh.t_pos_idx.clone().detach()

            # get existing texture and specular maps
            kd_ = load_mesh.material['kd'].data.permute(0, 3, 1, 2)
            ks_ = load_mesh.material['ks'].data.permute(0, 3, 1, 2)

            # if there is a normal map load it or initial a plain one
            try:
                nrml_ = load_mesh.material['normal'].data.permute(0, 3, 1, 2)
            except:
                nrml_ = torch.zeros( (1, 3, cfg["texture_resolution"], cfg["texture_resolution"]) ).to(device)
                nrml_[:, 2, :, :] = 1.0

            # convert all texture maps to trainable tensors
            texture_map  = texture.create_trainable( resize(kd_, out_shape=(cfg["texture_resolution"], cfg["texture_resolution"])).permute(0, 2, 3, 1), [cfg["texture_resolution"]]*2, True)
            specular_map = texture.create_trainable( resize(ks_, out_shape=(cfg["texture_resolution"], cfg["texture_resolution"])).permute(0, 2, 3, 1), [cfg["texture_resolution"]]*2, True)
            normal_map   = texture.create_trainable( resize(nrml_, out_shape=(cfg["texture_resolution"], cfg["texture_resolution"])).permute(0, 2, 3, 1), [cfg["texture_resolution"]]*2, True)


        # Training parameters
        if "verts" in cfg["train_mesh_idx"][idx]:
            train_params += [vertices]
            vert_train = True
            vert_trains.append(vert_train)
        if "texture" in cfg["train_mesh_idx"][idx]:
            train_params += texture_map.getMips()
        if "normal" in cfg["train_mesh_idx"][idx]:
            train_params += normal_map.getMips()
        if "specular" in cfg["train_mesh_idx"][idx]:
            train_params += specular_map.getMips()


        # finally load mesh
        load_mesh = mesh.Mesh(
            vertices,
            faces,
            material={
                'bsdf': cfg['bsdf'],
                'kd': texture_map,
                'ks': specular_map,
                'normal': normal_map,
            },
            base=load_mesh # Get UVs from original loaded mesh
        )
        meshes.append( load_mesh )

        if "verts" in cfg["train_mesh_idx"][idx]:
            subdiv.append( LimitSubdivide(
                load_mesh.v_pos.clone().detach(),
                load_mesh.t_pos_idx.clone().detach(),
            ) )
        else:
            subdiv.append( None )

    return meshes, subdiv, train_params, vert_trains

def create_cams(cfg):

    # Dataset to get random camera parameters
    cams_data = CameraBatch(
        cfg["train_res"],
        [cfg["dist_min"], cfg["dist_max"]],
        [cfg["azim_min"], cfg["azim_max"]],
        [cfg["elev_alpha"], cfg["elev_beta"], cfg["elev_max"]],
        [cfg["fov_min"], cfg["fov_max"]],
        cfg["aug_loc"],
        cfg["aug_light"],
        cfg["aug_bkg"],
        cfg["batch_size"]
    )

    cams = torch.utils.data.DataLoader(
        cams_data,
        cfg["batch_size"],
        num_workers=0,
        pin_memory=True
    )

    return cams


def nvdiff_rendermeshes(meshes, subdiv, cfg, device=0):

    render_meshes = []          # store meshes with texture that will be rendered

    lapl_funcs    = []          # store laplacian for each mesh

    for i, m in enumerate(meshes):

    # Limit subdivide vertices if needed
        if subdiv[i] != None:
            n_vert = subdiv[i].get_limit(
                m.v_pos.to('cpu').double()
            ).to(device)

        else:
            n_vert = m.v_pos


        # Low pass filter for textures

        ready_texture = texture.Texture2D(
            kornia.filters.gaussian_blur2d(
                m.material['kd'].data.permute(0, 3, 1, 2),
                kernel_size=(cfg["kernel_size"], cfg["kernel_size"]),
                sigma=(cfg["blur_sigma"], cfg["blur_sigma"]),
            ).permute(0, 2, 3, 1).contiguous()
        )

        ready_specular = texture.Texture2D(
            kornia.filters.gaussian_blur2d(
                m.material['ks'].data.permute(0, 3, 1, 2),
                kernel_size=(cfg["kernel_size"], cfg["kernel_size"]),
                sigma=(cfg["blur_sigma"], cfg["blur_sigma"]),
            ).permute(0, 2, 3, 1).contiguous()
        )

        ready_normal = texture.Texture2D(
            kornia.filters.gaussian_blur2d(
                m.material['normal'].data.permute(0, 3, 1, 2),
                kernel_size=(cfg["kernel_size"], cfg["kernel_size"]),
                sigma=(cfg["blur_sigma"], cfg["blur_sigma"]),
            ).permute(0, 2, 3, 1).contiguous()
        )
            
        # Final mesh with vertices and textures
        load_mesh = mesh.Mesh(
            n_vert,
            m.t_pos_idx,
            material={
                'bsdf': cfg['bsdf'],
                'kd': ready_texture,
                'ks': ready_specular,
                'normal': ready_normal,
            },
            base=m # gets uvs etc from here
        )


        render_meshes.append(load_mesh.eval())

        if subdiv[i] != None:
            lapl_funcs.append(regularizer.laplace_regularizer_const(m))
        else:
            lapl_funcs.append(None)

    return render_meshes, lapl_funcs

def const_scene_camparams(render_meshes, cams, device=0, ):

    # Create a scene with the textures and another without textures
    complete_scene = create_scene(render_meshes, sz=cfg["texture_resolution"])
    complete_scene = mesh.auto_normals(complete_scene)
    complete_scene = mesh.compute_tangents(complete_scene)


    # Render scene for training
    params_camera = next(iter(cams))
    # print("print cams", cams.shape)

    # broadcast the first pose
    params_camera["campos"][:,] = params_camera["campos"][0]
    # print(params_camera["campos"])

    for key in params_camera:
        params_camera[key] = params_camera[key].to(device)

    return complete_scene, params_camera

    # return params_camera

def const_trainrender(complete_scene, params_camera, cfg):

    # Render with only textured meshes
    glctx = dr.RasterizeGLContext()

    params = {
        'mvp': params_camera['mvp'],
        'lightpos': params_camera['lightpos'],
        'campos': params_camera['campos'],
        'resolution': [cfg["train_res"], cfg["train_res"]]
    }

    train_render = render.render_mesh(
        glctx,
        complete_scene.eval(params),
        params["mvp"],
        params["campos"],
        params["lightpos"],
        cfg["light_power"],
        cfg["train_res"],
        spp=1, # no upscale here / render at any resolution then use resize_right to downscale
        num_layers=cfg["layers"],
        msaa=False,
        background=params_camera["bkgs"],
    ).permute(0, 3, 1, 2) # switch to B, C, H, W

    if cfg["resize_method"] == "cubic":

        train_render = resize(
            train_render,
            out_shape=(224, 224), # resize to clip
            interp_method=cubic
        )
        
    # Render with only textured meshes

    return train_render


def const_cfg(offset=[0,0,0]):
        

    cfg = {
            # Parameters
            "epochs": 2000,
            "batch_size": 12,
            "train_res": 356, 
            "resize_method": "cubic", 
            "bsdf": "diffuse", 
            "texture_resolution": 512, 
            "kernel_size": 7,
            "blur_sigma": 3,
            "shape_imgs_frac": 0.5 ,
            "aug_light": "true" , 
            "aug_bkg": "true" ,
            "layers": 2,

            # Camera Parameters
            "fov_min": 89.0,             # Minimum camera field of view angle during renders 
            "fov_max": 90.0, #90.0,            # Maximum camera field of view angle during renders 
            "dist_min": 5.0,            # Minimum distance of camera from mesh during renders
            "dist_max": 5.0, #5.0,            # Maximum distance of camera from mesh during renders
            "light_power": 5.0,         # Light intensity
            "elev_alpha": 1.0,          # Alpha parameter for Beta distribution for elevation sampling
            "elev_beta": 0.01, #5.0,           # Beta parameter for Beta distribution for elevation sampling
            "elev_max": 10.0,           # Maximum elevation angle
            "azim_min": 0.0, #-360.0,         # Minimum azimuth angle
            "azim_max": 0.0,          # Maximum azimuth angle
            "aug_loc": "false",            # Offset mesh from center of image?
            
            "meshes": [
                "/home/itoh/ClipMesh/data/face/face.obj", 
                "/home/itoh/ClipMesh/data/silk_hat/silk_hat.obj"] ,

            "unit": "true",
            "train_mesh_idx": [
                # ["verts", "texture", "normal", "false"], 
                # ["verts", "texture", "normal", "false"]
                ["texture", "normal", "true"], 
                ["texture", "normal", "true"],
                ],
            "scales": [1.0, 1.0],
            "offsets": [[0.0, -1.0, 0.0], offset],


        }

    return cfg



cfg = const_cfg()


meshes, subdiv, _, _ = nvdiff_meshes(cfg)
cams = create_cams(cfg)


render_meshes, _ =  nvdiff_rendermeshes(meshes, subdiv, cfg)

# import ipdb; ipdb.set_trace()

complete_scene, params_camera  = const_scene_camparams(render_meshes, cams)

train_render = const_trainrender(complete_scene, params_camera, cfg)

s_log = train_render[torch.randint(low=0, high=cfg["batch_size"], size=(5 if cfg["batch_size"] > 5 else cfg["batch_size"], )) , :, :, :]
s_log = torchvision.utils.make_grid(s_log)

# Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
ndarr = s_log.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
im = Image.fromarray(ndarr)
im.save("nvdiff_test.png")

