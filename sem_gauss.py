import argparse
import os
import shutil
import sys
import time
from importlib.machinery import SourceFileLoader

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, _BASE_DIR)

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from tqdm import tqdm
from datasets.gradslam_datasets import (
    load_dataset_config,
    ReplicaDataset,
    ScannetDataset,
    TUMDataset,
)
from utils.common_utils import seed_everything, save_params_ckpt, save_params
from utils.eval_utils import report_progress, eval, decode_segmap
from utils.keyframe_selection import keyframe_selection_overlap
from utils.recon_utils import setup_camera
from utils.slam_helpers import (
    transformed_params2rendervar,transform_to_frame, l1_loss_v1, matrix_to_quaternion
)
from utils.slam_external import calc_ssim, build_rotation, prune_gaussians
from utils.dinov2_seg import Segmentation

from diff_gaussian_rasterization import GaussianRasterizer as Renderer

def get_dataset(config_dict, basedir, sequence, **kwargs):
    if config_dict["dataset_name"].lower() in ["replica"]:
        return ReplicaDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["scannet"]:
        return ScannetDataset(config_dict, basedir, sequence, **kwargs)
    else:
        raise ValueError(f"Unknown dataset name {config_dict['dataset_name']}")


def get_pointcloud(color, depth, sem_feature, intrinsics, w2c, transform_pts=True,
                   mask=None, compute_mean_sq_dist=False, mean_sq_dist_method="projective"):
    width, height = color.shape[2], color.shape[1]
    CX = intrinsics[0][2]
    CY = intrinsics[1][2]
    FX = intrinsics[0][0]
    FY = intrinsics[1][1]

    # Compute indices of pixels
    x_grid, y_grid = torch.meshgrid(torch.arange(width).cuda().float(), 
                                    torch.arange(height).cuda().float(),
                                    indexing='xy')
    xx = (x_grid - CX)/FX
    yy = (y_grid - CY)/FY
    xx = xx.reshape(-1)
    yy = yy.reshape(-1)
    depth_z = depth[0].reshape(-1)

    # Initialize point cloud
    pts_cam = torch.stack((xx * depth_z, yy * depth_z, depth_z), dim=-1)
    if transform_pts:
        pix_ones = torch.ones(height * width, 1).cuda().float()
        pts4 = torch.cat((pts_cam, pix_ones), dim=1)
        c2w = torch.inverse(w2c)
        pts = (c2w @ pts4.T).T[:, :3]
    else:
        pts = pts_cam

    # Compute mean squared distance for initializing the scale of the Gaussians
    if compute_mean_sq_dist:
        if mean_sq_dist_method == "projective":
            # Projective Geometry (this is fast, farther -> larger radius)
            scale_gaussian = depth_z / ((FX + FY)/2)
            mean3_sq_dist = scale_gaussian**2
        else:
            raise ValueError(f"Unknown mean_sq_dist_method {mean_sq_dist_method}")

    cols = torch.permute(color, (1, 2, 0)).reshape(-1, 3) # (C, H, W) -> (H, W, C) -> (H * W, C)
    feat_dim = sem_feature.shape[0]
    semantic = torch.permute(sem_feature, (1, 2, 0)).reshape(-1, feat_dim)
    point_cld = torch.cat((pts, cols, semantic), -1)

    # Select points based on mask
    if mask is not None:
        point_cld = point_cld[mask]
        if compute_mean_sq_dist:
            mean3_sq_dist = mean3_sq_dist[mask]

    if compute_mean_sq_dist:
        return point_cld, mean3_sq_dist
    else:
        return point_cld


def initialize_params(init_pt_cld, num_frames, mean3_sq_dist):
    num_pts = init_pt_cld.shape[0]
    means3D = init_pt_cld[:, :3] # [num_gaussians, 3]
    unnorm_rots = np.tile([1, 0, 0, 0], (num_pts, 1)) # [num_gaussians, 3]
    logit_opacities = torch.zeros((num_pts, 1), dtype=torch.float, device="cuda")

    params = {
        'means3D': means3D,
        'rgb_colors': init_pt_cld[:, 3:6],
        'sem_labels' : init_pt_cld[:, 6:],
        'unnorm_rotations': unnorm_rots,
        'logit_opacities': logit_opacities,
        'log_scales': torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 1)),
    }

    cam_rots = np.tile([1, 0, 0, 0], (1, 1))
    cam_rots = np.tile(cam_rots[:, :, None], (1, 1, num_frames))
    params['cam_unnorm_rots'] = cam_rots
    params['cam_trans'] = np.zeros((1, 3, num_frames))

    for k, v in params.items():
        # Check if value is already a torch tensor
        if not isinstance(v, torch.Tensor):
            params[k] = torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True))
        else:
            params[k] = torch.nn.Parameter(v.cuda().float().contiguous().requires_grad_(True))

    variables = {'max_2D_radius': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'means2D_gradient_accum': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'denom': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'timestep': torch.zeros(params['means3D'].shape[0]).cuda().float()}

    return params, variables


def initialize_optimizer(params, lrs_dict, tracking):
    lrs = lrs_dict
    param_groups = [{'params': [v], 'name': k, 'lr': lrs[k]} for k, v in params.items()]
    if tracking:
        return torch.optim.Adam(param_groups)
    else:
        return torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)

def initialize_first_timestep(dataset, num_frames, scene_radius_depth_ratio, mean_sq_dist_method, seg_net):
    # Get RGB-D Data & semantic & Camera Parameters
    color, depth, semantic, intrinsics, pose = dataset[0]


    # get semantic feature
    rgb_input = color.permute(2, 0, 1).unsqueeze(0).cuda()/255.0  # [1,3,H,W]
    seg_net.set_mode_get_feature()
    sem_feature = seg_net.cnn(rgb_input)    # [1,16,H,W]    #TODO: sem_feature和color都在gpu上(cuda:0)
    sem_feature = sem_feature.squeeze(0).detach() # [16,H,W]

    # Process RGB-D Data
    depth = depth.permute(2, 0, 1) # (H, W, C) -> (C, H, W)
    color = color.permute(2, 0, 1) / 255

    # Process Camera Parameters
    intrinsics = intrinsics[:3, :3]
    w2c = torch.linalg.inv(pose)

    # Setup Camera
    cam = setup_camera(color.shape[2], color.shape[1], intrinsics.cpu().numpy(), w2c.detach().cpu().numpy())

    # Get Initial Point Cloud (PyTorch CUDA Tensor)
    mask = (depth > 0) # Mask out invalid depth values
    mask = mask.reshape(-1)
    init_pt_cld, mean3_sq_dist = get_pointcloud(color, depth, sem_feature, intrinsics, w2c,
                                                mask=mask, compute_mean_sq_dist=True, 
                                                mean_sq_dist_method=mean_sq_dist_method)

    # Initialize Parameters
    params, variables = initialize_params(init_pt_cld, num_frames, mean3_sq_dist)

    # Initialize an estimate of scene radius for Gaussian-Splatting Densification
    variables['scene_radius'] = torch.max(depth)/scene_radius_depth_ratio

    return params, variables, intrinsics, w2c, cam

def get_loss(params, curr_data, variables, iter_time_idx, loss_weights, use_sil_for_loss,
             sil_thres, use_l1,ignore_outlier_depth_loss, seg_net, tracking=False,
             mapping=False, BA=False, plot_dir=None, visualize_tracking_loss=False):
    # Initialize Loss Dictionary
    losses = {}

    if tracking:
        # Get current frame Gaussians, where only the camera pose gets gradient
        transformed_pts = transform_to_frame(params, iter_time_idx, 
                                             gaussians_grad=False,
                                             camera_grad=True)
    elif mapping:
        # Get current frame Gaussians, where only the Gaussians get gradient
        transformed_pts = transform_to_frame(params, iter_time_idx,
                                                 gaussians_grad=True,
                                                 camera_grad=False)
    elif BA:
        # Get current frame Gaussians, where both the Gaussians and the camera pose get gradient
        transformed_pts = transform_to_frame(params, iter_time_idx,
                                             gaussians_grad=True,
                                             camera_grad=True)
    else:
        # Get current frame Gaussians, where only the Gaussians get gradient
        transformed_pts = transform_to_frame(params, iter_time_idx,
                                             gaussians_grad=True,
                                             camera_grad=False)

    # Initialize Render Variables
    rendervar = transformed_params2rendervar(params, curr_data['w2c'], transformed_pts)
    # RGB & depth & semantic Rendering
    rendervar['means2D'].retain_grad()
    im_dep, radius, _, semantics = Renderer(raster_settings=curr_data['cam'])(**rendervar)
    #semantics here is feature
    variables['means2D'] = rendervar['means2D']

    im = im_dep[0:3, :, :]
    depth_sil = im_dep[3:, :, :]
    # process semantic feature
    seg_net.set_mode_classification()
    semantics = semantics.unsqueeze(0)
    out_sem = seg_net.cnn(semantics)
    if BA:   #if BA get_loss return
        return im, depth_sil[0, :, :], semantics

    # Depth & Silhouette Rendering
    depth = depth_sil[0, :, :].unsqueeze(0)
    silhouette = depth_sil[1, :, :]
    presence_sil_mask = (silhouette > sil_thres)
    depth_sq = depth_sil[2, :, :].unsqueeze(0)
    uncertainty = depth_sq - depth**2
    uncertainty = uncertainty.detach()

    # Mask with valid depth values (accounts for outlier depth values)
    nan_mask = (~torch.isnan(depth)) & (~torch.isnan(uncertainty))
    if ignore_outlier_depth_loss:
        depth_error = torch.abs(curr_data['depth'] - depth) * (curr_data['depth'] > 0)
        mask = (depth_error < 10*depth_error.median())
        mask = mask & (curr_data['depth'] > 0)
    else:
        mask = (curr_data['depth'] > 0)
    mask = mask & nan_mask
    # Mask with presence silhouette mask (accounts for empty space)
    if tracking and use_sil_for_loss:
        mask = mask & presence_sil_mask

    # Depth loss
    if use_l1:
        mask = mask.detach()
        if tracking:
            losses['depth'] = torch.abs(curr_data['depth'] - depth)[mask].sum()
        else:
            losses['depth'] = torch.abs(curr_data['depth'] - depth)[mask].mean()

    lossF = torch.nn.CrossEntropyLoss()
    # RGB Loss
    if tracking and (use_sil_for_loss or ignore_outlier_depth_loss):
        color_mask = torch.tile(mask, (3, 1, 1))
        color_mask = color_mask.detach()#(3,h,w)
        semantic_mask = mask.detach()#(1,h,w)
        semantic_mask2 = torch.tile(mask, (out_sem.detach().size(1), 1, 1))  #(n_classes,h,w)
        semantic_mask2 = semantic_mask2.unsqueeze(0).detach()#(1,n_classes,h,w)

        losses['im'] = torch.abs(curr_data['im'] - im)[color_mask].sum()
        gt_tracking = torch.max(curr_data['se'], 1).indices
        losses['se'] = torch.abs(lossF(out_sem*semantic_mask2, gt_tracking*semantic_mask)).sum()
        losses['se_fe'] = torch.abs(curr_data['se_fe'] - semantics).sum()

    elif tracking:
        losses['im'] = torch.abs(curr_data['im'] - im).sum()
        gt_tracking = torch.max(curr_data['se'], 1).indices
        losses['se'] = torch.abs(lossF(out_sem, gt_tracking)).sum()
        losses['se_fe'] = torch.abs(curr_data['se_fe'] - semantics).sum()

    else:
        losses['im'] = 0.8 * l1_loss_v1(im, curr_data['im']) + 0.2 * (1.0 - calc_ssim(im, curr_data['im']))
        gt = torch.max(curr_data['se'], 1).indices
        losses['se'] = torch.abs(lossF(out_sem, gt)).mean()
        losses['se_fe'] = torch.abs(curr_data['se_fe'] - semantics).mean()

    # Visualize the Diff Images
    if tracking and visualize_tracking_loss and iter_time_idx%1000 == 0:
        fig, ax = plt.subplots(3, 4, figsize=(12, 6))
        weighted_render_im = im * color_mask
        weighted_im = curr_data['im'] * color_mask
        weighted_render_depth = depth * mask
        weighted_depth = curr_data['depth'] * mask

        diff_rgb = torch.abs(weighted_render_im - weighted_im).mean(dim=0).detach().cpu()
        diff_depth = torch.abs(weighted_render_depth - weighted_depth).mean(dim=0).detach().cpu()

        viz_img = torch.clip(weighted_im.permute(1, 2, 0).detach().cpu(), 0, 1)
        ax[0, 0].imshow(viz_img)
        ax[0, 0].set_title("Weighted GT RGB")
        viz_render_img = torch.clip(weighted_render_im.permute(1, 2, 0).detach().cpu(), 0, 1)
        ax[1, 0].imshow(viz_render_img)
        ax[1, 0].set_title("Weighted Rendered RGB")
        ax[0, 1].imshow(weighted_depth[0].detach().cpu(), cmap="jet", vmin=0, vmax=6)
        ax[0, 1].set_title("Weighted GT Depth")
        ax[1, 1].imshow(weighted_render_depth[0].detach().cpu(), cmap="jet", vmin=0, vmax=6)
        ax[1, 1].set_title("Weighted Rendered Depth")
        ax[0, 2].imshow(diff_rgb, cmap="jet", vmin=0, vmax=0.8)
        ax[0, 2].set_title(f"Diff RGB, Loss: {torch.round(losses['im'])}")
        ax[1, 2].imshow(diff_depth, cmap="jet", vmin=0, vmax=0.8)
        ax[1, 2].set_title(f"Diff Depth, Loss: {torch.round(losses['depth'])}")
        ax[0, 3].imshow(presence_sil_mask.detach().cpu(), cmap="gray")
        ax[0, 3].set_title("Silhouette Mask")
        ax[1, 3].imshow(mask[0].detach().cpu(), cmap="gray")
        ax[1, 3].set_title("Loss Mask")

        currgt = torch.max(curr_data['se'], 1).indices.detach().squeeze()
        viz_img_se = decode_segmap(currgt.cpu(), seg_net.n_classes)
        ax[2, 0].imshow(viz_img_se)
        ax[2, 0].set_title("Weighted GT Semantic")

        currre = torch.max(out_sem, 1).indices.detach().squeeze()
        viz_render_img_se = decode_segmap(currre.cpu(), seg_net.n_classes)
        ax[2, 1].imshow(viz_render_img_se)
        ax[2, 1].set_title("Weighted Rendered SEMANTIC")

        # Turn off axis
        for i in range(3):
            for j in range(4):
                ax[i, j].axis('off')
        # Set Title
        fig.suptitle(f"iter_time_idx: {iter_time_idx}", fontsize=16)
        # Figure Tight Layout
        fig.tight_layout()
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir, f"tmp{iter_time_idx}.png"), bbox_inches='tight')
        plt.close()

    weighted_losses = {k: v * loss_weights[k] for k, v in losses.items()}
    loss = sum(weighted_losses.values())

    seen = radius > 0
    variables['max_2D_radius'][seen] = torch.max(radius[seen], variables['max_2D_radius'][seen])
    variables['seen'] = seen
    weighted_losses['loss'] = loss

    return loss, variables, weighted_losses

def initialize_new_params(new_pt_cld, mean3_sq_dist):
    num_pts = new_pt_cld.shape[0]
    means3D = new_pt_cld[:, :3] # [num_gaussians, 3]
    unnorm_rots = np.tile([1, 0, 0, 0], (num_pts, 1)) # [num_gaussians, 3]
    logit_opacities = torch.zeros((num_pts, 1), dtype=torch.float, device="cuda")
    params = {
        'means3D': means3D,
        'rgb_colors': new_pt_cld[:, 3:6],
        'sem_labels': new_pt_cld[:, 6:],
        'unnorm_rotations': unnorm_rots,
        'logit_opacities': logit_opacities,
        'log_scales': torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 1)),
    }
    for k, v in params.items():
        # Check if value is already a torch tensor
        if not isinstance(v, torch.Tensor):
            params[k] = torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True))
        else:
            params[k] = torch.nn.Parameter(v.cuda().float().contiguous().requires_grad_(True))

    return params

def add_new_gaussians(params, variables, curr_data, sil_thres, time_idx, mean_sq_dist_method, sem_feature):
    # Silhouette Rendering
    transformed_pts = transform_to_frame(params, time_idx, gaussians_grad=False, camera_grad=False)
    # semantic
    rendervar = transformed_params2rendervar(params, curr_data['w2c'], transformed_pts)
    im_dep, _, _, _ = Renderer(raster_settings=curr_data['cam'])(**rendervar)
    depth_sil = im_dep[3:, :, :]

    silhouette = depth_sil[1, :, :]
    non_presence_sil_mask = (silhouette < sil_thres)
    # Check for new foreground objects by using GT depth
    gt_depth = curr_data['depth'][0, :, :]
    render_depth = depth_sil[0, :, :]
    depth_error = torch.abs(gt_depth - render_depth) * (gt_depth > 0)
    non_presence_depth_mask = (render_depth > gt_depth) * (depth_error > 50*depth_error.median())
    # Determine non-presence mask
    non_presence_mask = non_presence_sil_mask | non_presence_depth_mask
    # Flatten mask
    non_presence_mask = non_presence_mask.reshape(-1)

    # Get the new frame Gaussians based on the Silhouette
    if torch.sum(non_presence_mask) > 0:
        # Get the new pointcloud in the world frame
        curr_cam_rot = torch.nn.functional.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
        curr_cam_tran = params['cam_trans'][..., time_idx].detach()
        curr_w2c = torch.eye(4).cuda().float()
        curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
        curr_w2c[:3, 3] = curr_cam_tran
        valid_depth_mask = (curr_data['depth'][0, :, :] > 0)
        non_presence_mask = non_presence_mask & valid_depth_mask.reshape(-1)
        new_pt_cld, mean3_sq_dist = get_pointcloud(curr_data['im'], curr_data['depth'], sem_feature, curr_data['intrinsics'],
                                    curr_w2c, mask=non_presence_mask, compute_mean_sq_dist=True,
                                    mean_sq_dist_method=mean_sq_dist_method)

        new_params = initialize_new_params(new_pt_cld, mean3_sq_dist)
        for k, v in new_params.items():
            params[k] = torch.nn.Parameter(torch.cat((params[k], v), dim=0).requires_grad_(True))
        num_pts = params['means3D'].shape[0]
        variables['means2D_gradient_accum'] = torch.zeros(num_pts, device="cuda").float()
        variables['denom'] = torch.zeros(num_pts, device="cuda").float()
        variables['max_2D_radius'] = torch.zeros(num_pts, device="cuda").float()
        new_timestep = time_idx*torch.ones(new_pt_cld.shape[0],device="cuda").float()
        variables['timestep'] = torch.cat((variables['timestep'],new_timestep),dim=0)

    return params, variables

def initialize_camera_pose(params, curr_time_idx, forward_prop):
    with torch.no_grad():
        if curr_time_idx > 1 and forward_prop:
            # Initialize the camera pose for the current frame based on a constant velocity model
            # Rotation
            prev_rot1 = F.normalize(params['cam_unnorm_rots'][..., curr_time_idx-1].detach())
            prev_rot2 = F.normalize(params['cam_unnorm_rots'][..., curr_time_idx-2].detach())
            new_rot = F.normalize(prev_rot1 + (prev_rot1 - prev_rot2))
            params['cam_unnorm_rots'][..., curr_time_idx] = new_rot.detach()
            # Translation
            prev_tran1 = params['cam_trans'][..., curr_time_idx-1].detach()
            prev_tran2 = params['cam_trans'][..., curr_time_idx-2].detach()
            new_tran = prev_tran1 + (prev_tran1 - prev_tran2)
            params['cam_trans'][..., curr_time_idx] = new_tran.detach()
        else:
            # Initialize the camera pose for the current frame
            params['cam_unnorm_rots'][..., curr_time_idx] = params['cam_unnorm_rots'][..., curr_time_idx-1].detach()
            params['cam_trans'][..., curr_time_idx] = params['cam_trans'][..., curr_time_idx-1].detach()
    
    return params

def dense_semantic_slam(config: dict):
    # Loading Config
    print("Loading Config:")
    if "use_depth_loss_thres" not in config['tracking']:
        config['tracking']['use_depth_loss_thres'] = False
        config['tracking']['depth_loss_thres'] = 100000
    if "visualize_tracking_loss" not in config['tracking']:
        config['tracking']['visualize_tracking_loss'] = False

    # Create Output Directories
    output_dir = os.path.join(config["workdir"], config["run_name"])
    eval_dir = os.path.join(output_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)
    process_dir = os.path.join(output_dir, "process")
    os.makedirs(process_dir, exist_ok=True)

    # semantic segmentation
    seg_net = Segmentation(config)

    # Get Device
    device = torch.device(config["primary_device"])

    # Load Dataset
    print("Loading Dataset ...")
    dataset_config = config["data"]
    if "gradslam_data_cfg" not in dataset_config:
        gradslam_data_cfg = {}
        gradslam_data_cfg["dataset_name"] = dataset_config["dataset_name"]
    else:
        gradslam_data_cfg = load_dataset_config(dataset_config["gradslam_data_cfg"])
    dataset = get_dataset(
        config_dict=gradslam_data_cfg,
        basedir=dataset_config["basedir"],
        sequence=os.path.basename(dataset_config["sequence"]),
        start=dataset_config["start"],
        end=dataset_config["end"],
        stride=dataset_config["stride"],
        desired_height=dataset_config["desired_image_height"],
        desired_width=dataset_config["desired_image_width"],
        device=device,
        relative_pose=True,
    )
    num_frames = dataset_config["num_frames"]
    if num_frames == -1:
        num_frames = len(dataset)

    # Initialize Parameters & Canoncial Camera parameters
    params, variables, intrinsics, first_frame_w2c, cam = initialize_first_timestep(dataset, num_frames,
                                                                                        config['scene_radius_depth_ratio'],
                                                                                        config['mean_sq_dist_method'], seg_net)

    # Initialize list to keep track of Keyframes
    keyframe_list = []
    keyframe_time_indices = []

    # Init Variables to keep track of ground truth poses and runtimes
    gt_w2c_all_frames = []

    tracking_iter_time_sum = 0
    tracking_iter_time_count = 0
    mapping_iter_time_sum = 0
    mapping_iter_time_count = 0
    tracking_frame_time_sum = 0
    tracking_frame_time_count = 0
    mapping_frame_time_sum = 0
    mapping_frame_time_count = 0
    time_use_sum = 0

    # Load Checkpoint
    if config['load_checkpoint']:
        checkpoint_time_idx = config['checkpoint_time_idx']
        print(f"Loading Checkpoint for Frame {checkpoint_time_idx}")
        ckpt_path = os.path.join(config['workdir'], config['run_name'], f"params{checkpoint_time_idx}.npz")
        params = dict(np.load(ckpt_path, allow_pickle=True))
        params = {k: torch.tensor(params[k]).cuda().float().requires_grad_(True) for k in params.keys()}
        variables['max_2D_radius'] = torch.zeros(params['means3D'].shape[0]).cuda().float()
        variables['means2D_gradient_accum'] = torch.zeros(params['means3D'].shape[0]).cuda().float()
        variables['denom'] = torch.zeros(params['means3D'].shape[0]).cuda().float()
        variables['timestep'] = torch.zeros(params['means3D'].shape[0]).cuda().float()
        # Load the keyframe time idx list
        keyframe_time_indices = np.load(
            os.path.join(config['workdir'], config['run_name'], f"keyframe_time_indices{checkpoint_time_idx}.npy"))
        keyframe_time_indices = keyframe_time_indices.tolist()

        # Loading the saved pose and representation of scene
        for time_idx in range(checkpoint_time_idx):
            # Load frames incrementally instead of all frames
            color, depth, semantic, _, gt_pose = dataset[time_idx]
            # get gt_semantic: process to (1,n_classes,h,w)
            semantic = semantic.squeeze()
            semantic = semantic.unsqueeze(0).unsqueeze(0)
            semantic_gt = torch.zeros_like(semantic, dtype=torch.float32)
            semantic_gt = torch.tile(semantic_gt, (1, config["model"]["n_classes"], 1, 1))
            for channel in range(config["model"]["n_classes"]):
                channel1 = channel * 1.0
                semantic_gt[0, channel, :, :] = semantic * (semantic[0, 0, :, :] == channel1)
            # Process poses
            gt_w2c = torch.linalg.inv(gt_pose)
            gt_w2c_all_frames.append(gt_w2c)
            # Initialize Keyframe List
            if time_idx in keyframe_time_indices:
                # Get the estimated rotation & translation
                curr_cam_rot = F.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
                curr_cam_tran = params['cam_trans'][..., time_idx].detach()
                curr_w2c = torch.eye(4).cuda().float()
                curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
                curr_w2c[:3, 3] = curr_cam_tran
                # Initialize Keyframe Info
                rgb_input = color.permute(2, 0, 1).unsqueeze(0).cuda() / 255.0  # [1,3,H,W]
                # get segmentation semantic
                seg_net.set_mode_get_feature()
                sem_feature = seg_net.cnn(rgb_input)  # [1,16,H,W]    #TODO: sem_feature和color都在gpu上(cuda:0)
                sem_feature = sem_feature.squeeze(0).detach()  # [16,H,W]
                seg_net.set_mode_get_semantic()
                sem_out = seg_net.cnn(rgb_input)  # [1,n_classes,H,W]
                sem_out = sem_out.detach()

                color = color.permute(2, 0, 1) / 255
                depth = depth.permute(2, 0, 1)

                if config["use_gt_semantic"]:#This will put semantic_gt into keyframe_list as supervision
                    curr_keyframe = {'cam': cam, 'id': time_idx, 'est_w2c': curr_w2c, 'color': color.cpu(),
                                    'depth': depth.cpu(), 'se': semantic_gt.cpu(), 'se_fe': sem_feature.unsqueeze(0).cpu()}
                else:#This will put the segmentation semantic into keyframe_list as supervision
                    curr_keyframe = {'cam': cam, 'id': time_idx, 'est_w2c': curr_w2c, 'color': color.cpu(),
                                     'depth': depth.cpu(), 'se': sem_out.cpu(), 'se_fe': sem_feature.unsqueeze(0).cpu()}
                # Add to keyframe list
                keyframe_list.append(curr_keyframe)
    else:
        checkpoint_time_idx = 0

    progress_bar = tqdm(range(checkpoint_time_idx, num_frames), desc="Processing Frames")
    # Iterate frame by frame
    for time_idx in progress_bar:
        # Load frames incrementally instead of all frames
        color, depth, semantic, _, gt_pose = dataset[time_idx]

        #get gt_semantic: process to (1,n_classes,h,w)
        semantic = semantic.squeeze()
        semantic = semantic.unsqueeze(0).unsqueeze(0)
        semantic_gt = torch.zeros_like(semantic, dtype=torch.float32)
        semantic_gt = torch.tile(semantic_gt,(1, config["model"]["n_classes"], 1 ,1))
        for channel in range(config["model"]["n_classes"]):
            channel1 = channel*1.0
            semantic_gt[0, channel, :, :] = semantic * (semantic[0, 0, :, :] == channel1)

        # Process poses
        gt_w2c = torch.linalg.inv(gt_pose)

        # sget segmentation semantic and semantic feature
        rgb_input = color.permute(2, 0, 1).unsqueeze(0).cuda()/255.0
        seg_net.set_mode_get_feature()
        sem_feature = seg_net.cnn(rgb_input)
        sem_feature = sem_feature.squeeze(0).detach()
        seg_net.set_mode_get_semantic()
        sem_out = seg_net.cnn(rgb_input)  # [1,n_classes,H,W]
        sem_out = sem_out.detach()

        # Process RGB-D Data
        depth = depth.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
        color = color.permute(2, 0, 1) / 255

        gt_w2c_all_frames.append(gt_w2c)
        curr_gt_w2c = gt_w2c_all_frames

        # Optimize only current time step for tracking
        iter_time_idx = time_idx

        # Initialize Data
        if config['use_gt_semantic']:#use gt_semantic as input to supervise
            curr_data = {'cam': cam, 'im': color, 'depth': depth, 'se': semantic_gt, 'se_fe': sem_feature.unsqueeze(0), 'id': iter_time_idx, 'intrinsics': intrinsics,
                     'w2c': first_frame_w2c, 'iter_gt_w2c_list': curr_gt_w2c}
        else:   #use the result of segmentation as input to supervise
            curr_data = {'cam': cam, 'im': color, 'depth': depth, 'se': sem_out, 'se_fe': sem_feature.unsqueeze(0), 'id': iter_time_idx,'intrinsics': intrinsics,
                     'w2c': first_frame_w2c, 'iter_gt_w2c_list': curr_gt_w2c}

        # Initialize Data for Tracking
        tracking_curr_data = curr_data

        # Optimization Iterations
        num_iters_mapping = config['mapping']['num_iters']
        
        # Initialize the camera pose for the current frame
        if time_idx > 0:
            params = initialize_camera_pose(params, time_idx, forward_prop=config['tracking']['forward_prop'])

        # Tracking
        tracking_start_time = time.time()
        if time_idx > 0 and not config['tracking']['use_gt_poses']:
            # Reset Optimizer & Learning Rates for tracking
            optimizer = initialize_optimizer(params, config['tracking']['lrs'], tracking=True)
            # Keep Track of Best Candidate Rotation & Translation
            candidate_cam_unnorm_rot = params['cam_unnorm_rots'][..., time_idx].detach().clone()
            candidate_cam_tran = params['cam_trans'][..., time_idx].detach().clone()
            current_min_loss = float(1e20)
            # Tracking Optimization
            iter = 0
            do_continue_slam = False
            num_iters_tracking = config['tracking']['num_iters']
            # progress_bar = tqdm(range(num_iters_tracking), desc=f"Tracking Time Step: {time_idx}")
            while True:
                iter_start_time = time.time()
                # Loss for current frame
                loss, variables, losses = get_loss(params, tracking_curr_data, variables, iter_time_idx, config['tracking']['loss_weights'],
                                                   config['tracking']['use_sil_for_loss'], config['tracking']['sil_thres'],
                                                   config['tracking']['use_l1'], config['tracking']['ignore_outlier_depth_loss'],
                                                   seg_net=seg_net, tracking=True,
                                                   plot_dir=process_dir, visualize_tracking_loss=config['tracking']['visualize_tracking_loss'],
                                                   )
                # Backprop
                loss.backward()
                # Optimizer Update
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    # Save the best candidate rotation & translation
                    if loss < current_min_loss:
                        current_min_loss = loss
                        candidate_cam_unnorm_rot = params['cam_unnorm_rots'][..., time_idx].detach().clone()
                        candidate_cam_tran = params['cam_trans'][..., time_idx].detach().clone()
                    # Report Progress
                    if config['report_iter_progress']:
                        report_progress(params, tracking_curr_data, iter+1, progress_bar, iter_time_idx, sil_thres=config['tracking']['sil_thres'], tracking=True)
                    else:
                        progress_bar.update(1)
                # Update the runtime numbers
                iter_end_time = time.time()
                tracking_iter_time_sum += iter_end_time - iter_start_time
                tracking_iter_time_count += 1
                # Check if we should stop tracking
                iter += 1
                if iter == num_iters_tracking:
                    if losses['depth'] < config['tracking']['depth_loss_thres'] and config['tracking']['use_depth_loss_thres']:
                        break
                    elif config['tracking']['use_depth_loss_thres'] and not do_continue_slam:
                        do_continue_slam = True
                        progress_bar = tqdm(range(num_iters_tracking), desc=f"Tracking Time Step: {time_idx}")
                        num_iters_tracking = 2*num_iters_tracking
                    else:
                        break

            progress_bar.close()
            # Copy over the best candidate rotation & translation
            with torch.no_grad():
                params['cam_unnorm_rots'][..., time_idx] = candidate_cam_unnorm_rot
                params['cam_trans'][..., time_idx] = candidate_cam_tran

        elif time_idx > 0 and config['tracking']['use_gt_poses']:
            with torch.no_grad():
                # Get the ground truth pose relative to frame 0
                rel_w2c = curr_gt_w2c[-1]
                rel_w2c_rot = rel_w2c[:3, :3].unsqueeze(0).detach()
                rel_w2c_rot_quat = matrix_to_quaternion(rel_w2c_rot)
                rel_w2c_tran = rel_w2c[:3, 3].detach()
                # Update the camera parameters
                params['cam_unnorm_rots'][..., time_idx] = rel_w2c_rot_quat
                params['cam_trans'][..., time_idx] = rel_w2c_tran
        # Update the runtime numbers
        tracking_end_time = time.time()
        tracking_frame_time_sum += tracking_end_time - tracking_start_time
        tracking_frame_time_count += 1

        if time_idx == 0 or (time_idx+1) % config['report_global_progress_every'] == 0:
            try:
                # Report Final Tracking Progress
                progress_bar = tqdm(range(1), desc=f"Tracking Result Time Step: {time_idx}")
                with torch.no_grad():
                    report_progress(params, tracking_curr_data, 1, progress_bar, iter_time_idx, sil_thres=config['tracking']['sil_thres'], tracking=True)
                progress_bar.close()
            except:
                ckpt_output_dir = os.path.join(config["workdir"], config["run_name"])
                save_params_ckpt(params, ckpt_output_dir, time_idx)
                print('Failed to evaluate trajectory.')

        # Densification & KeyFrame-based Mapping
        if time_idx == 0 or (time_idx+1) % config['map_every'] == 0:
            # Densification
            if config['mapping']['add_new_gaussians'] and time_idx > 0:
                # Setup Data for Densification
                densify_curr_data = curr_data

                # Add new Gaussians to the scene based on the Silhouette
                params, variables = add_new_gaussians(params, variables, densify_curr_data, 
                                                      config['mapping']['sil_thres'], time_idx,
                                                      config['mean_sq_dist_method'], sem_feature)

            with torch.no_grad():
                # Get the current estimated rotation & translation
                curr_cam_rot = F.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
                curr_cam_tran = params['cam_trans'][..., time_idx].detach()
                curr_w2c = torch.eye(4).cuda().float()
                curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
                curr_w2c[:3, 3] = curr_cam_tran
                # Select Keyframes for Mapping
                num_keyframes = config['mapping_window_size']-2
                selected_keyframes = keyframe_selection_overlap(depth, curr_w2c, intrinsics, keyframe_list[:-1], num_keyframes)
                selected_time_idx = [keyframe_list[frame_idx]['id'] for frame_idx in selected_keyframes]
                if len(keyframe_list) > 0:
                    # Add last keyframe to the selected keyframes
                    selected_time_idx.append(keyframe_list[-1]['id'])
                    selected_keyframes.append(len(keyframe_list)-1)
                # Add current frame to the selected keyframes
                selected_time_idx.append(time_idx)
                selected_keyframes.append(-1)
                # Print the selected keyframes
                # print(f"\nSelected Keyframes at Frame {time_idx}: {selected_time_idx}")

            # Reset Optimizer & Learning Rates for Full Map Optimization
            optimizer = initialize_optimizer(params, config['mapping']['lrs'], tracking=False)

            # Mapping
            mapping_start_time = time.time()
            # if num_iters_mapping > 0:
            #     progress_bar = tqdm(range(num_iters_mapping), desc=f"Mapping Time Step: {time_idx}")
            for iter in range(num_iters_mapping):
                iter_start_time = time.time()
                time1 = time.time()
                # Randomly select a frame until current time step amongst keyframes
                rand_idx = np.random.randint(0, len(selected_keyframes))
                selected_rand_keyframe_idx = selected_keyframes[rand_idx]
                if selected_rand_keyframe_idx == -1:
                    # Use Current Frame Data
                    iter_time_idx = time_idx
                    iter_color = color
                    iter_depth = depth
                    if config['use_gt_semantic']:
                        iter_semantic = semantic_gt
                    else:
                        iter_semantic = sem_out
                    iter_sem_feature = sem_feature.unsqueeze(0)
                else:
                    # Use Keyframe Data
                    iter_time_idx = keyframe_list[selected_rand_keyframe_idx]['id']
                    iter_color = keyframe_list[selected_rand_keyframe_idx]['color']
                    iter_depth = keyframe_list[selected_rand_keyframe_idx]['depth']
                    iter_semantic = keyframe_list[selected_rand_keyframe_idx]['se']
                    iter_sem_feature = keyframe_list[selected_rand_keyframe_idx]['se_fe']

                iter_gt_w2c = gt_w2c_all_frames[:iter_time_idx+1]
                iter_data = {'cam': cam, 'im': iter_color.to('cuda',non_blocking=True), 'depth': iter_depth.to('cuda',non_blocking=True),
                             'se': iter_semantic.to('cuda',non_blocking=True), 'se_fe': iter_sem_feature.to('cuda',non_blocking=True),
                             'id': iter_time_idx, 'intrinsics': intrinsics, 'w2c': first_frame_w2c, 'iter_gt_w2c_list': iter_gt_w2c}

                # Loss for current frame
                loss, variables, losses = get_loss(params, iter_data, variables, iter_time_idx, config['mapping']['loss_weights'],
                                                config['mapping']['use_sil_for_loss'], config['mapping']['sil_thres'],
                                                config['mapping']['use_l1'], config['mapping']['ignore_outlier_depth_loss'],
                                                seg_net=seg_net, mapping=True)
                # Backprop
                loss.backward()
                with torch.no_grad():
                    # Prune Gaussians
                    if config['mapping']['prune_gaussians']:
                        params, variables = prune_gaussians(params, variables, optimizer, iter, config['mapping']['pruning_dict'])
                    # Optimizer Update
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    time2 = time.time()
                    time_use_sum += time2 - time1
                    # Report Progress
                    if config['report_iter_progress']:
                        report_progress(params, iter_data, iter+1, progress_bar, iter_time_idx, sil_thres=config['mapping']['sil_thres'],
                                            mapping=True, online_time_idx=time_idx)
                    else:
                        progress_bar.update(1)
                # Update the runtime numbers
                iter_end_time = time.time()
                mapping_iter_time_sum += iter_end_time - iter_start_time
                mapping_iter_time_count += 1
            if num_iters_mapping > 0:
                progress_bar.close()
            # Update the runtime numbers
            mapping_end_time = time.time()
            mapping_frame_time_sum += mapping_end_time - mapping_start_time
            mapping_frame_time_count += 1

            if time_idx == 0 or (time_idx+1) % config['report_global_progress_every'] == 0:
                try:
                    # Report Mapping Progress
                    progress_bar = tqdm(range(1), desc=f"Mapping Result Time Step: {time_idx}")
                    with torch.no_grad():
                        report_progress(params, curr_data, 1, progress_bar, time_idx, sil_thres=config['mapping']['sil_thres'],
                                            mapping=True, online_time_idx=time_idx)
                    progress_bar.close()
                except:
                    ckpt_output_dir = os.path.join(config["workdir"], config["run_name"])
                    save_params_ckpt(params, ckpt_output_dir, time_idx)
                    print('Failed to evaluate trajectory.')
        # BA
        if ((time_idx + 1) % config['BA_every'] == 0):
            BA_idx = []
            # set the num of BA frames
            BA_t = config['BA_iters']
            for BA_iters in range(BA_t + 1):
                idx = (config['BA_every'] // config['BA_iters']) * BA_iters + time_idx - config['BA_every']
                BA_idx.append(idx)

            # init optimizer
            BA_optimizer = initialize_optimizer(params, config['BA']['lrs'], tracking=False)
            currBA_data = {
                'w2c': first_frame_w2c,
                'cam': cam
            }
            #BA
            for num_iter in range(config['BA']['num_iters']):
                BA_loss_sum = 0
                for BA_iters in range(BA_t):
                    BA_id = BA_idx[BA_iters]
                    BA_id_2 = BA_idx[BA_iters + 1]

                    BA_im_re, BA_depth_re, BA_semantic_re = get_loss(params, currBA_data, variables, BA_id,
                                                                     config['tracking']['loss_weights'],
                                                                     config['tracking']['use_sil_for_loss'],
                                                                     config['tracking']['sil_thres'],
                                                                     config['tracking']['use_l1'],
                                                                     config['tracking'][
                                                                         'ignore_outlier_depth_loss'],
                                                                     seg_net=seg_net, BA=True,
                                                                     plot_dir=process_dir,
                                                                     )
                    # next frame for BA
                    BA_im_re_2, BA_depth_re_2, BA_semantic_re_2 = get_loss(params, currBA_data, variables,
                                                                           BA_id_2,
                                                                           config['tracking']['loss_weights'],
                                                                           config['tracking'][
                                                                               'use_sil_for_loss'],
                                                                           config['tracking']['sil_thres'],
                                                                           config['tracking']['use_l1'],
                                                                           config['tracking'][
                                                                               'ignore_outlier_depth_loss'],
                                                                           seg_net=seg_net, BA=True,
                                                                           plot_dir=process_dir,
                                                                           )
                    h = color.shape[1]
                    w = color.shape[2]
                    BA_num = color.view(3, -1)
                    ones = torch.ones(1, BA_num.shape[1]).cuda().float()
                    zeros_d = torch.zeros(2, BA_num.shape[1]).cuda().float()
                    ones_d = torch.ones(1, BA_num.shape[1]).cuda().float()
                    zeros_ones_d = torch.cat((zeros_d, ones_d), dim=0)

                    #get pose of two connected frame
                    BA_curr_cam_rot = F.normalize(params['cam_unnorm_rots'][..., BA_id].detach())
                    BA_curr_cam_tran = params['cam_trans'][..., BA_id].detach()
                    BA_curr_w2c = torch.eye(4).cuda().float()
                    BA_curr_w2c[:3, :3] = build_rotation(BA_curr_cam_rot)
                    BA_curr_w2c[:3, 3] = BA_curr_cam_tran

                    BA_curr_cam_rot_2 = F.normalize(params['cam_unnorm_rots'][..., BA_id_2].detach())
                    BA_curr_cam_tran_2 = params['cam_trans'][..., BA_id_2].detach()
                    BA_curr_w2c_2 = torch.eye(4).cuda().float()
                    BA_curr_w2c_2[:3, :3] = build_rotation(BA_curr_cam_rot_2)
                    BA_curr_w2c_2[:3, 3] = BA_curr_cam_tran_2

                    T = BA_curr_w2c_2 * BA_curr_w2c.inverse()

                    BA_color_pre = (T @ torch.cat((BA_im_re.view(3, -1), ones), dim=0))[:3, :].view((3, h, w))
                    BA_depth_pre = (T @ torch.cat((BA_depth_re.view(1, -1).view(1, -1), zeros_ones_d), dim=0))[:1, :].view((1, h, w))

                    T2 = torch.cat((T, T), dim=1)
                    T3 = torch.cat((T2, T2), dim=0)  # feature dim = 8
                    T4 = torch.cat((T3, T3), dim=1)
                    T5 = torch.cat((T4, T4), dim=0)  # feature dim = 16
                    T6 = torch.cat((T5, T5), dim=0)
                    T7 = torch.cat((T6, T6), dim=1)  # feature dim = 32
                    BA_semantic_pre = (T5 @ BA_semantic_re.squeeze(0).view(config['model']['c_dim'], -1)).view((config['model']['c_dim'], h, w))

                    BA_losses = {}
                    BA_losses['im'] = (torch.abs(BA_im_re_2.detach() - BA_color_pre).sum())
                    BA_losses['depth'] = (torch.abs(BA_depth_re_2.detach() - BA_depth_pre).sum())
                    BA_losses['se'] = (torch.abs(BA_semantic_re_2.detach() - BA_semantic_pre).sum())

                    BA_loss_weights = config['BA']['loss_weights']

                    BA_loss_sum = (BA_loss_sum + BA_losses['im'] * BA_loss_weights['im']
                                   + BA_losses['depth'] * BA_loss_weights['depth']
                                   + BA_losses['se'] * BA_loss_weights['se']
                                   )
                    torch.cuda.empty_cache()
                    del BA_im_re_2, BA_depth_re_2, BA_semantic_re_2
                # print(BA_loss_sum)
                BA_loss_sum.backward()
                BA_optimizer.step()
                BA_optimizer.zero_grad(set_to_none=True)

            del BA_im_re, BA_depth_re, BA_semantic_re, BA_loss_sum, BA_losses
            del BA_color_pre, BA_depth_pre, BA_semantic_pre
            del T, T2, T3, T4, T5
            BA_idx.clear()
            torch.cuda.empty_cache()

        # Add frame to keyframe list
        if ((time_idx == 0) or ((time_idx+1) % config['keyframe_every'] == 0) or \
                    (time_idx == num_frames-2)) and (not torch.isinf(curr_gt_w2c[-1]).any()) and (not torch.isnan(curr_gt_w2c[-1]).any()):
            with torch.no_grad():
                # Get the current estimated rotation & translation
                curr_cam_rot = F.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
                curr_cam_tran = params['cam_trans'][..., time_idx].detach()
                curr_w2c = torch.eye(4).cuda().float()
                curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
                curr_w2c[:3, 3] = curr_cam_tran

                if config['use_gt_semantic']:
                    curr_keyframe = {'cam': cam, 'id': time_idx, 'est_w2c': curr_w2c, 'color': color.cpu(), 'depth': depth.cpu(),
                                     'se': semantic_gt.cpu(), 'se_fe':sem_feature.unsqueeze(0).cpu()}
                else:
                    curr_keyframe = {'cam': cam, 'id': time_idx, 'est_w2c': curr_w2c, 'color': color.cpu(), 'depth': depth.cpu(),
                                     'se': sem_out.cpu(), 'se_fe':sem_feature.unsqueeze(0).cpu()}
                # Add to keyframe list
                keyframe_list.append(curr_keyframe)
                keyframe_time_indices.append(time_idx)
        
        # Checkpoint settled iteration
        if (time_idx+1) % config["checkpoint_interval"] == 0 and config['save_checkpoints']:
            ckpt_output_dir = os.path.join(config["workdir"], config["run_name"])
            save_params_ckpt(params, ckpt_output_dir, time_idx)
            np.save(os.path.join(ckpt_output_dir, f"keyframe_time_indices{time_idx}.npy"), np.array(keyframe_time_indices))

        #torch.cuda.empty_cache()

    # Compute Average Runtimes
    if tracking_iter_time_count == 0:
        tracking_iter_time_count = 1
        tracking_frame_time_count = 1
    if mapping_iter_time_count == 0:
        mapping_iter_time_count = 1
        mapping_frame_time_count = 1
    tracking_iter_time_avg = tracking_iter_time_sum / tracking_iter_time_count
    tracking_frame_time_avg = tracking_frame_time_sum / (tracking_frame_time_count-1)
    mapping_iter_time_avg = mapping_iter_time_sum / mapping_iter_time_count
    mapping_frame_time_avg = mapping_frame_time_sum / mapping_frame_time_count
    time_use_sum_avg = time_use_sum / mapping_iter_time_count
    print(f"\nAverage Tracking/Iteration Time: {tracking_iter_time_avg*1000} ms")
    print(f"Average Tracking/Frame Time: {tracking_frame_time_avg} s")
    print(f"Average Mapping/Iteration Time: {mapping_iter_time_avg*1000} ms")
    print(f"Average Mapping/Frame Time: {mapping_frame_time_avg} s")
    print(f"Average Our_Mapping/Iteration Time: {time_use_sum_avg*1000} ms")
    
    # Evaluate Final Parameters
    with torch.no_grad():
        eval(config, dataset, params, num_frames, eval_dir, sil_thres=config['mapping']['sil_thres'],
                 mapping_iters=config['mapping']['num_iters'], add_new_gaussians=config['mapping']['add_new_gaussians'],
                 eval_every=config['eval_every'])

    # Add Camera Parameters to Save them
    params['timestep'] = variables['timestep']
    params['intrinsics'] = intrinsics.detach().cpu().numpy()
    params['w2c'] = first_frame_w2c.detach().cpu().numpy()
    params['org_width'] = dataset_config["desired_image_width"]
    params['org_height'] = dataset_config["desired_image_height"]
    params['gt_w2c_all_frames'] = []
    for gt_w2c_tensor in gt_w2c_all_frames:
        params['gt_w2c_all_frames'].append(gt_w2c_tensor.detach().cpu().numpy())
    params['gt_w2c_all_frames'] = np.stack(params['gt_w2c_all_frames'], axis=0)
    params['keyframe_time_indices'] = np.array(keyframe_time_indices)
    
    # Save Parameters
    save_params(params, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=str, help="Path to experiment file")
    args = parser.parse_args()

    experiment = SourceFileLoader(
        os.path.basename(args.experiment), args.experiment
    ).load_module()
    print(experiment)

    # Set Experiment Seed
    seed_everything(seed=experiment.config['seed'])
    
    # Create Results Directory and Copy Config
    results_dir = os.path.join(
        experiment.config["workdir"], experiment.config["run_name"]
    )
    if not experiment.config['load_checkpoint']:
        os.makedirs(results_dir, exist_ok=True)
        shutil.copy(args.experiment, os.path.join(results_dir, "config.py"))

    dense_semantic_slam(experiment.config)