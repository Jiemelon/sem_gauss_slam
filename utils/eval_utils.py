import cv2
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

from datasets.gradslam_datasets.geometryutils import relative_transformation
from datasets.gradslam_datasets import load_dataset_config
from utils.recon_utils import setup_camera
from utils.slam_external import build_rotation,calc_psnr
from utils.slam_helpers import transform_to_frame, transformed_params2rendervar
from utils.segmentationMetric import SegmentationMetric
from utils.dinov2_seg import Segmentation
from diff_gaussian_rasterization import GaussianRasterizer as Renderer

from pytorch_msssim import ms_ssim
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
loss_fn_alex = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).cuda()

def align(model, data):
    """Align two trajectories using the method of Horn (closed-form).

    Args:
        model -- first trajectory (3xn)
        data -- second trajectory (3xn)

    Returns:
        rot -- rotation matrix (3x3)
        trans -- translation vector (3x1)
        trans_error -- translational error per point (1xn)

    """
    np.set_printoptions(precision=3, suppress=True)
    model_zerocentered = model - model.mean(1).reshape((3,-1))
    data_zerocentered = data - data.mean(1).reshape((3,-1))

    W = np.zeros((3, 3))
    for column in range(model.shape[1]):
        W += np.outer(model_zerocentered[:,
                         column], data_zerocentered[:, column])
    U, d, Vh = np.linalg.linalg.svd(W.transpose())
    S = np.matrix(np.identity(3))
    if (np.linalg.det(U) * np.linalg.det(Vh) < 0):
        S[2, 2] = -1
    rot = U*S*Vh
    trans = data.mean(1).reshape((3,-1)) - rot * model.mean(1).reshape((3,-1))

    model_aligned = rot * model + trans
    alignment_error = model_aligned - data

    trans_error = np.sqrt(np.sum(np.multiply(
        alignment_error, alignment_error), 0)).A[0]

    return rot, trans, trans_error


def evaluate_ate(gt_traj, est_traj):
    """
    Input : 
        gt_traj: list of 4x4 matrices 
        est_traj: list of 4x4 matrices
        len(gt_traj) == len(est_traj)
    """
    gt_traj_pts = [gt_traj[idx][:3,3] for idx in range(len(gt_traj))]
    est_traj_pts = [est_traj[idx][:3,3] for idx in range(len(est_traj))]

    gt_traj_pts  = torch.stack(gt_traj_pts).detach().cpu().numpy().T
    est_traj_pts = torch.stack(est_traj_pts).detach().cpu().numpy().T

    _, _, trans_error = align(gt_traj_pts, est_traj_pts)

    avg_trans_error = trans_error.mean()

    return avg_trans_error

def report_progress(params, data, i, progress_bar, iter_time_idx, sil_thres, every_i=1, qual_every_i=1, 
                    tracking=False, mapping=False, online_time_idx=None):
    if i % every_i == 0 or i == 1:
        if tracking:
            # Get list of gt poses
            gt_w2c_list = data['iter_gt_w2c_list']
            valid_gt_w2c_list = []
            
            # Get latest trajectory
            latest_est_w2c = data['w2c']
            latest_est_w2c_list = []
            latest_est_w2c_list.append(latest_est_w2c)
            valid_gt_w2c_list.append(gt_w2c_list[0])
            for idx in range(1, iter_time_idx+1):
                # Check if gt pose is not nan for this time step
                if torch.isnan(gt_w2c_list[idx]).sum() > 0:
                    continue
                interm_cam_rot = F.normalize(params['cam_unnorm_rots'][..., idx].detach())
                interm_cam_trans = params['cam_trans'][..., idx].detach()
                intermrel_w2c = torch.eye(4).cuda().float()
                intermrel_w2c[:3, :3] = build_rotation(interm_cam_rot)
                intermrel_w2c[:3, 3] = interm_cam_trans
                latest_est_w2c = intermrel_w2c
                latest_est_w2c_list.append(latest_est_w2c)
                valid_gt_w2c_list.append(gt_w2c_list[idx])

            # Get latest gt pose
            gt_w2c_list = valid_gt_w2c_list
            iter_gt_w2c = gt_w2c_list[-1]
            # Get euclidean distance error between latest and gt pose
            iter_pt_error = torch.sqrt((latest_est_w2c[0,3] - iter_gt_w2c[0,3])**2 + (latest_est_w2c[1,3] - iter_gt_w2c[1,3])**2 + (latest_est_w2c[2,3] - iter_gt_w2c[2,3])**2)
            if iter_time_idx > 0:
                # Calculate relative pose error
                rel_gt_w2c = relative_transformation(gt_w2c_list[-2], gt_w2c_list[-1])
                rel_est_w2c = relative_transformation(latest_est_w2c_list[-2], latest_est_w2c_list[-1])
                rel_pt_error = torch.sqrt((rel_gt_w2c[0,3] - rel_est_w2c[0,3])**2 + (rel_gt_w2c[1,3] - rel_est_w2c[1,3])**2 + (rel_gt_w2c[2,3] - rel_est_w2c[2,3])**2)
            else:
                rel_pt_error = torch.zeros(1).float()
            
            # Calculate ATE RMSE
            ate_rmse = evaluate_ate(gt_w2c_list, latest_est_w2c_list)
            ate_rmse = np.round(ate_rmse, decimals=6)

        # Get current frame Gaussians
        transformed_pts = transform_to_frame(params, iter_time_idx, 
                                             gaussians_grad=False,
                                             camera_grad=False)

        # Initialize Render Variables
        # semantic
        rendervar = transformed_params2rendervar(params, data['w2c'], transformed_pts)
        im_dep, radius, _, semantics = Renderer(raster_settings=data['cam'])(**rendervar)
        im = im_dep[0:3, :, :]
        depth_sil = im_dep[3:, :, :]
        rastered_depth = depth_sil[0, :, :].unsqueeze(0)
        valid_depth_mask = (data['depth'] > 0)
        silhouette = depth_sil[1, :, :]
        presence_sil_mask = (silhouette > sil_thres)
        #TODO: 语义评估，代码放这里

        print("report_progress")
        for k, v in params.items():
            print("k: ",k)

        if tracking:
            psnr = calc_psnr(im * presence_sil_mask, data['im'] * presence_sil_mask).mean()
        else:
            psnr = calc_psnr(im, data['im']).mean()

        if tracking:
            diff_depth_rmse = torch.sqrt((((rastered_depth - data['depth']) * presence_sil_mask) ** 2))
            diff_depth_rmse = diff_depth_rmse * valid_depth_mask
            rmse = diff_depth_rmse.sum() / valid_depth_mask.sum()
            diff_depth_l1 = torch.abs((rastered_depth - data['depth']) * presence_sil_mask)
            diff_depth_l1 = diff_depth_l1 * valid_depth_mask
            depth_l1 = diff_depth_l1.sum() / valid_depth_mask.sum()
        else:
            diff_depth_rmse = torch.sqrt((((rastered_depth - data['depth'])) ** 2))
            diff_depth_rmse = diff_depth_rmse * valid_depth_mask
            rmse = diff_depth_rmse.sum() / valid_depth_mask.sum()
            diff_depth_l1 = torch.abs((rastered_depth - data['depth']))
            diff_depth_l1 = diff_depth_l1 * valid_depth_mask
            depth_l1 = diff_depth_l1.sum() / valid_depth_mask.sum()

        if not (tracking or mapping):
            progress_bar.set_postfix({f"Time-Step: {iter_time_idx} | PSNR: {psnr:.{7}} | Depth RMSE: {rmse:.{7}} | L1": f"{depth_l1:.{7}}"})
            progress_bar.update(every_i)
        elif tracking:
            progress_bar.set_postfix({f"Time-Step: {iter_time_idx} | Rel Pose Error: {rel_pt_error.item():.{7}} | Pose Error: {iter_pt_error.item():.{7}} | ATE RMSE": f"{ate_rmse.item():.{7}}"})
            progress_bar.update(every_i)
        elif mapping:
            progress_bar.set_postfix({f"Time-Step: {online_time_idx} | Frame {data['id']} | PSNR: {psnr:.{7}} | Depth RMSE: {rmse:.{7}} | L1": f"{depth_l1:.{7}}"})
            progress_bar.update(every_i)

def eval(config, dataset, final_params, num_frames, eval_dir, sil_thres,
         mapping_iters, add_new_gaussians, eval_every=1, save_frames=True):
    print("Evaluating Final Parameters ...")
    psnr_list = []
    rmse_list = []
    l1_list = []
    lpips_list = []
    ssim_list = []
    mIou_list = []
    miou_eval_list = []
    pixAcc_list = []
    pixAcc_eval_list = []

    if save_frames:
        render_rgb_dir = os.path.join(eval_dir, "rendered_rgb")
        os.makedirs(render_rgb_dir, exist_ok=True)
        render_depth_dir = os.path.join(eval_dir, "rendered_depth")
        os.makedirs(render_depth_dir, exist_ok=True)
        render_semantic_dir = os.path.join(eval_dir, "rendered_semantic")
        os.makedirs(render_semantic_dir, exist_ok=True)
        rgb_dir = os.path.join(eval_dir, "gt_rgb")
        os.makedirs(rgb_dir, exist_ok=True)
        depth_dir = os.path.join(eval_dir, "gt_depth")
        os.makedirs(depth_dir, exist_ok=True)
        semantic_dir = os.path.join(eval_dir, "gt_semantic")
        os.makedirs(semantic_dir, exist_ok=True)
        feature_dir = os.path.join(eval_dir, "feature")
        os.makedirs(feature_dir, exist_ok=True)
        mesh_dir = os.path.join(eval_dir, "mesh")
        os.makedirs(mesh_dir, exist_ok=True)


    gt_w2c_list = []
    seg_net = Segmentation(config)

    ##pre for mesh
    pose_folder = os.path.join(config['data']['basedir'], config['data']['sequence'])
    pose_path = os.path.join(pose_folder, "traj.txt")
    with open(pose_path, "r") as f:
        pose_lines = f.readlines()
    first_pose_line = pose_lines[0]
    first_pose_c2w = np.array(list(map(float, first_pose_line.split()))).reshape(4, 4)
    first_pose_c2w = torch.from_numpy(first_pose_c2w).float()
    first_pose_w2c = np.linalg.inv(first_pose_c2w.cpu().numpy())

    # _, _, _, _, first_gt_pose = dataset[0]
    # first_pose_w2c = np.linalg.inv(first_gt_pose.cpu().numpy())
    # print("herh2", first_pose_w2c)

    cam_cfg = load_dataset_config(config["data"]["gradslam_data_cfg"])
    W = cam_cfg["camera_params"]["image_width"] - 2 * cam_cfg["camera_params"]["crop_edge"]
    H = cam_cfg["camera_params"]["image_height"]- 2 * cam_cfg["camera_params"]["crop_edge"]
    fx = cam_cfg["camera_params"]["fx"]
    fy = cam_cfg["camera_params"]["fy"]
    cx = cam_cfg["camera_params"]["cx"]
    cy = cam_cfg["camera_params"]["cy"]

    scale = 1.0
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=5.0 * scale / 512.0,
        sdf_trunc=0.04 * scale,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    volume_se = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=5.0 * scale / 512.0,
        sdf_trunc=0.04 * scale,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    compensate_vector = (-0.0 * scale / 512.0, 2.5 *
                         scale / 512.0, -2.5 * scale / 512.0)

    #create folder to save mesh
    mesh_name = f'pred_mesh.ply'
    mesh_out_file = f'{mesh_dir}/{mesh_name}'
    mesh_name_se = f'pred_mesh_se.ply'
    mesh_out_file_se = f'{mesh_dir}/{mesh_name_se}'
    os.makedirs(f'{mesh_dir}/mid_mesh', exist_ok=True)

    #eval starting
    for time_idx in tqdm(range(num_frames)):
        # Get RGB-D & semantic & Camera Parameters
        color, depth, semantic,intrinsics, pose = dataset[time_idx]
        gt_w2c = torch.linalg.inv(pose)
        gt_w2c_list.append(gt_w2c)
        intrinsics = intrinsics[:3, :3]

        #get ge_semantic
        semantic = semantic.squeeze()
        semantic = semantic.unsqueeze(0).unsqueeze(0)
        semantic_gt = torch.zeros_like(semantic, dtype=torch.float32)
        semantic_gt = torch.tile(semantic_gt, (1, config["model"]["n_classes"], 1, 1))
        for channel in range(config["model"]["n_classes"]):
            channel1 = channel * 1.0
            semantic_gt[0, channel, :, :] = semantic * (semantic[0, 0, :, :] == channel1)

        rgb_input = color.permute(2, 0, 1).unsqueeze(0).cuda()/255.0  # [1,3,H,W]
        seg_net.set_mode_get_semantic()
        sem_out = seg_net.cnn(rgb_input)
        sem_out = sem_out.detach()

        # Process RGB-D Data
        color = color.permute(2, 0, 1) / 255 # (H, W, C) -> (C, H, W)
        depth = depth.permute(2, 0, 1) # (H, W, C) -> (C, H, W)

        if time_idx == 0:
            # Process Camera Parameters
            first_frame_w2c = torch.linalg.inv(pose)
            # Setup Camera
            cam = setup_camera(color.shape[2], color.shape[1], intrinsics.cpu().numpy(), first_frame_w2c.detach().cpu().numpy())
        
        # Skip frames if not eval_every
        if time_idx != 0 and (time_idx+1) % eval_every != 0:
            continue

        # Get current frame Gaussians
        transformed_pts = transform_to_frame(final_params, time_idx,
                                             gaussians_grad=False,
                                             camera_grad=False)
 
        # Define current frame data
        curr_data = {'cam': cam, 'im': color, 'se':sem_out, 'depth': depth, 'id': time_idx, 'intrinsics': intrinsics, 'w2c': first_frame_w2c}

        # Initialize Render Variables
        rendervar = transformed_params2rendervar(final_params, curr_data['w2c'], transformed_pts)
        # RGB & depth & semantic Rendering
        im_dep, radius, _, semantics = Renderer(raster_settings=curr_data['cam'])(**rendervar)
        im = im_dep[0:3, :, :]
        depth_sil = im_dep[3:, :, :]

        seg_net.set_mode_classification()
        semantics = semantics.unsqueeze(0)
        out_sem = seg_net.cnn(semantics)  # TODO 注意维数对应
        out_sem = out_sem.detach()

        # Render Depth & Silhouette
        rastered_depth = depth_sil[0, :, :].unsqueeze(0)
        # Mask invalid depth in GT
        valid_depth_mask = (curr_data['depth'] > 0)
        rastered_depth_viz = rastered_depth.detach()
        rastered_depth = rastered_depth * valid_depth_mask
        silhouette = depth_sil[1, :, :]
        presence_sil_mask = (silhouette > sil_thres)

        if mapping_iters==0 and not add_new_gaussians:
            weighted_im = im * presence_sil_mask * valid_depth_mask
            weighted_gt_im = curr_data['im'] * presence_sil_mask * valid_depth_mask
        else:
            weighted_im = im * valid_depth_mask
            weighted_gt_im = curr_data['im'] * valid_depth_mask

        psnr = calc_psnr(weighted_im, weighted_gt_im).mean()
        ssim = ms_ssim(weighted_im.unsqueeze(0).cpu(), weighted_gt_im.unsqueeze(0).cpu(), 
                        data_range=1.0, size_average=True)
        lpips_score = loss_fn_alex(torch.clamp(weighted_im.unsqueeze(0), 0.0, 1.0),
                                    torch.clamp(weighted_gt_im.unsqueeze(0), 0.0, 1.0)).item()

        #evaluate the rendered semantic
        metric = SegmentationMetric(out_sem.size(1))
        metric.update(out_sem, semantic_gt)
        pixAcc, mIou = metric.get()

        # evaluate the segmentation semantic (to compare with the rendered semantic)
        metric2 = SegmentationMetric(curr_data['se'].size(1))
        metric2.update(curr_data['se'], semantic_gt)
        pixAcc_eval, miou_eval = metric2.get()

        psnr_list.append(psnr.cpu().numpy())
        ssim_list.append(ssim.cpu().numpy())
        lpips_list.append(lpips_score)
        pixAcc_list.append(pixAcc)
        mIou_list.append(mIou)
        miou_eval_list.append(miou_eval)
        pixAcc_eval_list.append(pixAcc_eval)

        # Compute Depth RMSE
        if mapping_iters==0 and not add_new_gaussians:
            diff_depth_rmse = torch.sqrt((((rastered_depth - curr_data['depth']) * presence_sil_mask) ** 2))
            diff_depth_rmse = diff_depth_rmse * valid_depth_mask
            rmse = diff_depth_rmse.sum() / valid_depth_mask.sum()
            diff_depth_l1 = torch.abs((rastered_depth - curr_data['depth']) * presence_sil_mask)
            diff_depth_l1 = diff_depth_l1 * valid_depth_mask
            depth_l1 = diff_depth_l1.sum() / valid_depth_mask.sum()
        else:
            diff_depth_rmse = torch.sqrt((((rastered_depth - curr_data['depth'])) ** 2))
            diff_depth_rmse = diff_depth_rmse * valid_depth_mask
            rmse = diff_depth_rmse.sum() / valid_depth_mask.sum()
            diff_depth_l1 = torch.abs((rastered_depth - curr_data['depth']))
            diff_depth_l1 = diff_depth_l1 * valid_depth_mask
            depth_l1 = diff_depth_l1.sum() / valid_depth_mask.sum()
        rmse_list.append(rmse.cpu().numpy())
        l1_list.append(depth_l1.cpu().numpy())

        #### generate mesh
        with torch.no_grad():
            # Get the current estimated rotation & translation
            curr_cam_rot = F.normalize(final_params['cam_unnorm_rots'][..., time_idx].detach())
            curr_cam_tran = final_params['cam_trans'][..., time_idx].detach()
            curr_w2c = torch.eye(4).cuda().float()
            curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
            curr_w2c[:3, 3] = curr_cam_tran

        w2c = curr_w2c.cpu().numpy() @ first_pose_w2c

        re_depth = rastered_depth_viz[0].detach().cpu().numpy()
        depth_mesh = np.clip(re_depth, 0., 10.)
        depth_mesh = o3d.geometry.Image(depth_mesh.astype(np.float32))
        re_color = im.detach().cpu().permute(1, 2, 0).numpy()
        color_mesh = (np.clip(re_color, 0.0, 1.0) * 255.0).astype(np.uint8)
        color_mesh = o3d.geometry.Image(np.ascontiguousarray(color_mesh))
        re_seman = torch.from_numpy(decode_segmap(torch.max(out_sem, 1).indices.squeeze().cpu(), seg_net.n_classes))
        seman_mesh = (re_seman.detach().cpu().numpy() * 255.0).astype(np.uint8)
        seman_mesh = o3d.geometry.Image(np.ascontiguousarray(seman_mesh))

        #rgb mesh
        intrinsic = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_mesh,
        depth_mesh,
        depth_scale=1.0,
        depth_trunc=30,
        convert_rgb_to_intensity=False)
        volume.integrate(rgbd, intrinsic, w2c)
        # semantic mesh
        rgbd_se = o3d.geometry.RGBDImage.create_from_color_and_depth(
        seman_mesh,
        depth_mesh,
        depth_scale=1.0,
        depth_trunc=30,
        convert_rgb_to_intensity=False)
        volume_se.integrate(rgbd_se, intrinsic, w2c)

        #save_frames
        if save_frames:
            # Save Rendered RGB and Depth
            viz_render_im = torch.clamp(im, 0, 1)
            viz_render_im = viz_render_im.detach().cpu().permute(1, 2, 0).numpy()
            vmin = 0
            vmax = 6
            viz_render_depth = rastered_depth_viz[0].detach().cpu().numpy()
            normalized_depth = np.clip((viz_render_depth - vmin) / (vmax - vmin), 0, 1)
            depth_colormap = cv2.applyColorMap((normalized_depth * 255).astype(np.uint8), cv2.COLORMAP_JET)
            #change1
            curr_re = torch.max(out_sem, 1).indices.squeeze()
            viz_render_se_pre = decode_segmap(curr_re.cpu(), seg_net.n_classes)
            #print("viz_render_se_pre:", viz_render_se_pre.shape)  # (680,1200,3)
            viz_render_se = torch.from_numpy(viz_render_se_pre)
            viz_render_se = viz_render_se.detach().cpu().numpy()
            cv2.imwrite(os.path.join(render_rgb_dir, "gs_{:04d}.png".format(time_idx)), cv2.cvtColor(viz_render_im*255, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(render_depth_dir, "gs_{:04d}.png".format(time_idx)), viz_render_depth * 1000.0)
            cv2.imwrite(os.path.join(render_semantic_dir, "gs_{:04d}.png".format(time_idx)),cv2.cvtColor(viz_render_se * 255, cv2.COLOR_RGB2BGR))
            """
            # Save GT RGB and Depth
            viz_gt_im = torch.clamp(curr_data['im'], 0, 1)
            viz_gt_im = viz_gt_im.detach().cpu().permute(1, 2, 0).numpy()
            viz_gt_depth = curr_data['depth'][0].detach().cpu().numpy()
            normalized_depth = np.clip((viz_gt_depth - vmin) / (vmax - vmin), 0, 1)
            depth_colormap = cv2.applyColorMap((normalized_depth * 255).astype(np.uint8), cv2.COLORMAP_JET)
            #change1
            curr_gt = torch.max(semantic_gt, 1).indices.squeeze()
            viz_gt_se_pre = decode_segmap(curr_gt.cpu(), seg_net.n_classes)
            viz_gt_se = torch.from_numpy(viz_gt_se_pre)
            viz_gt_se = viz_gt_se.detach().cpu().numpy()
            cv2.imwrite(os.path.join(rgb_dir, "gt_{:04d}.png".format(time_idx)), cv2.cvtColor(viz_gt_im*255, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(depth_dir, "gt_{:04d}.png".format(time_idx)), depth_colormap)
            cv2.imwrite(os.path.join(semantic_dir, "gs_{:04d}.png".format(time_idx)),cv2.cvtColor(viz_gt_se * 255, cv2.COLOR_RGB2BGR))
            """

    #save mesh
    o3d_mesh = volume.extract_triangle_mesh()
    o3d_mesh_se = volume_se.extract_triangle_mesh()
    np.save(os.path.join(f'{mesh_dir}',
                         'vertices_pos.npy'), np.asarray(o3d_mesh.vertices))
    np.save(os.path.join(f'{mesh_dir}',
                         'vertices_pos_se.npy'), np.asarray(o3d_mesh_se.vertices))
    o3d_mesh = o3d_mesh.translate(compensate_vector)
    o3d_mesh_se = o3d_mesh_se.translate(compensate_vector)
    o3d.io.write_triangle_mesh(mesh_out_file, o3d_mesh)
    o3d.io.write_triangle_mesh(mesh_out_file_se, o3d_mesh_se)
    print('🕹️ Meshing finished.')

    try:
        # Compute the final ATE RMSE
        # Get the final camera trajectory
        num_frames = final_params['cam_unnorm_rots'].shape[-1]
        latest_est_w2c = first_frame_w2c
        latest_est_w2c_list = []
        latest_est_w2c_list.append(latest_est_w2c)
        valid_gt_w2c_list = []
        valid_gt_w2c_list.append(gt_w2c_list[0])
        for idx in range(1, num_frames):
            # Check if gt pose is not nan for this time step
            if torch.isnan(gt_w2c_list[idx]).sum() > 0:
                continue
            interm_cam_rot = F.normalize(final_params['cam_unnorm_rots'][..., idx].detach())
            interm_cam_trans = final_params['cam_trans'][..., idx].detach()
            intermrel_w2c = torch.eye(4).cuda().float()
            intermrel_w2c[:3, :3] = build_rotation(interm_cam_rot)
            intermrel_w2c[:3, 3] = interm_cam_trans
            latest_est_w2c = intermrel_w2c
            latest_est_w2c_list.append(latest_est_w2c)
            valid_gt_w2c_list.append(gt_w2c_list[idx])
        gt_w2c_list = valid_gt_w2c_list
        # Calculate ATE RMSE
        ate_rmse = evaluate_ate(gt_w2c_list, latest_est_w2c_list)
        print("Final Average ATE RMSE: {:.2f} cm".format(ate_rmse*100))
    except:
        ate_rmse = 100.0
        print('Failed to evaluate trajectory with alignment.')
    
    # Compute Average Metrics
    psnr_list = np.array(psnr_list)
    rmse_list = np.array(rmse_list)
    l1_list = np.array(l1_list)
    ssim_list = np.array(ssim_list)
    lpips_list = np.array(lpips_list)
    pixAcc_list = np.array(pixAcc_list)
    mIou_list = np.array(mIou_list)
    pixAcc_eval_list = np.array(pixAcc_eval_list)
    miou_eval_list = np.array(miou_eval_list)

    avg_psnr = psnr_list.mean()
    avg_rmse = rmse_list.mean()
    avg_l1 = l1_list.mean()
    avg_ssim = ssim_list.mean()
    avg_lpips = lpips_list.mean()
    avg_pixAcc = pixAcc_list.mean()
    avg_mIou = mIou_list.mean()
    avg_pixAcc_eval = pixAcc_eval_list.mean()
    avg_miou_eval = miou_eval_list.mean()

    Average_list = []
    Average_list.append(ate_rmse*100)
    Average_list.append(avg_psnr)
    Average_list.append(avg_rmse*100)
    Average_list.append(avg_l1*100)
    Average_list.append(avg_ssim)
    Average_list.append(avg_lpips)
    Average_list.append(avg_pixAcc)
    Average_list.append(avg_mIou)
    Average_list.append(avg_pixAcc_eval)
    Average_list.append(avg_miou_eval)

    Average_list = np.array(Average_list)
    np.savetxt(os.path.join(eval_dir, "Average.txt"), Average_list)

    print("Average PSNR: {:.2f}".format(avg_psnr))
    print("Average Depth RMSE: {:.2f} cm".format(avg_rmse*100))
    print("Average Depth L1: {:.2f} cm".format(avg_l1*100))
    print("Average MS-SSIM: {:.3f}".format(avg_ssim))
    print("Average LPIPS: {:.3f}".format(avg_lpips))
    print("Average pixAcc: {:.3f}".format(avg_pixAcc))
    print("Average mIou: {:.3f}".format(avg_mIou))
    print("Average pixAcc_eval: {:.3f}".format(avg_pixAcc_eval))
    print("Average miou_eval: {:.3f}".format(avg_miou_eval))

    # Save metric lists as text files
    np.savetxt(os.path.join(eval_dir, "psnr.txt"), psnr_list)
    np.savetxt(os.path.join(eval_dir, "rmse.txt"), rmse_list)
    np.savetxt(os.path.join(eval_dir, "l1.txt"), l1_list)
    np.savetxt(os.path.join(eval_dir, "ssim.txt"), ssim_list)
    np.savetxt(os.path.join(eval_dir, "lpips.txt"), lpips_list)
    np.savetxt(os.path.join(eval_dir, "miou.txt"), mIou_list)
    np.savetxt(os.path.join(eval_dir, "pixacc.txt"), pixAcc_list)
    np.savetxt(os.path.join(eval_dir, "miou_eval.txt"), miou_eval_list)
    np.savetxt(os.path.join(eval_dir, "pixacc_eval.txt"), pixAcc_eval_list)

    # Plot PSNR & L1 as line plots
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    axs[0].plot(np.arange(len(psnr_list)), psnr_list)
    axs[0].set_title("RGB PSNR")
    axs[0].set_xlabel("Time Step")
    axs[0].set_ylabel("PSNR")
    axs[1].plot(np.arange(len(l1_list)), l1_list*100)
    axs[1].set_title("Depth L1")
    axs[1].set_xlabel("Time Step")
    axs[1].set_ylabel("L1 (cm)")
    fig.suptitle("Average PSNR: {:.2f}, Average Depth L1: {:.2f} cm, ATE RMSE: {:.2f} cm".format(avg_psnr, avg_l1*100, ate_rmse*100), y=1.05, fontsize=16)
    plt.savefig(os.path.join(eval_dir, "metrics.png"), bbox_inches='tight')

    plt.close()

def decode_segmap(image, nc=25):
    #viz the semantic result

    label_colors = np.array([(0, 0, 0),  # 0=background
                         (174, 199, 232), (152, 223, 138), (31, 119, 180), (255, 187, 120), (188, 189, 34),
                         (140, 86, 75), (255, 152, 150), (214, 39, 40), (197, 176, 213), (148, 103, 189),
                         (196, 156, 148), (23, 190, 207), (178, 76, 76), (247, 182, 210), (66, 188, 102),
                         (219, 219, 141), (140, 57, 197), (202, 185, 52), (51, 176, 203), (200, 54, 131),
                         (92, 193, 61), (78, 71, 183), (172, 114, 82), (255, 127, 14), (91, 163, 138),
                         (153, 98, 156), (140, 153, 101), (158, 218, 229), (100, 125, 154), (178, 127, 135),
                         (120, 185, 128), (146, 111, 194), (44, 160, 44), (112, 128, 144), (96, 207, 209),
                         (227, 119, 194), (213, 92, 176), (94, 106, 211), (82, 84, 163), (100, 85, 144),
                         (100, 218, 200),
                         (255, 179, 0),    (144, 238, 144),  (135, 206, 235), (255, 105, 180), (106, 90, 205),
                         (255, 165, 0),    (72, 209, 204),   (199, 21, 133),  (70, 130, 180),  (255, 99, 71),
                         (147, 112, 219),  (60, 179, 113),   (220, 20, 60)
                        ])
    """

    label_colors = np.array([(0, 0, 0),  # 0=background
                                 (174, 199, 232), (152, 223, 138), (31, 119, 180), (255, 187, 120), (188, 189, 34),
                                 (140, 86, 75), (255, 152, 150), (214, 39, 40), (197, 176, 213), (148, 103, 189),
                                 (196, 156, 148), (23, 190, 207), (178, 76, 76), (247, 182, 210), (66, 188, 102),
                                 (219, 219, 141), (140, 57, 197), (202, 185, 52), (51, 176, 203), (200, 54, 131),
                                 (92, 193, 61), (78, 71, 183), (172, 114, 82), (255, 127, 14), (91, 163, 138),
                                 (153, 98, 156), (140, 153, 101), (158, 218, 229), (100, 125, 154), (178, 127, 135),
                                 (120, 185, 128), (146, 111, 194), (44, 160, 44), (112, 128, 144), (96, 207, 209),
                                 (227, 119, 194), (213, 92, 176), (94, 106, 211), (82, 84, 163), (100, 85, 144),
                                 (100, 218, 200)])
    """

    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)

    for l in range(0, nc):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]
    rgb = np.stack([r, g, b], axis=2)
    return rgb.astype(np.float32) / 255.0