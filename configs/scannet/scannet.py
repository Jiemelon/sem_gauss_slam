import os
from os.path import join as p_join

primary_device = "cuda:0"

scenes = ["scene0000_00", "scene0059_00", "scene0106_00", 
          "scene0169_00", "scene0181_00", "scene0207_00"]

seed = int(2024)
scene_name = scenes[int(5)]

map_every = 8
keyframe_every = 5
mapping_window_size = 10
tracking_iters = 100
mapping_iters = 30
scene_radius_depth_ratio = 3

group_name = "ScanNet"
run_name = f"{scene_name}_seed{seed}"

config = dict(
    workdir=f"./experiments/{group_name}",
    group_name=group_name,
    run_name=run_name,
    seed=seed,
    primary_device=primary_device,
    map_every=map_every, # Mapping every nth frame
    keyframe_every=keyframe_every, # Keyframe every nth frame
    BA_every=32,  # change15
    BA_iters=15,
    mapping_window_size=mapping_window_size, # Mapping window size
    report_global_progress_every=1000, # Report Global Progress every nth frame
    eval_every=5, # Evaluate every nth frame (at end of SLAM)
    scene_radius_depth_ratio=scene_radius_depth_ratio, # Max First Frame Depth to Scene Radius Ratio (For Pruning/Densification)
    mean_sq_dist_method="projective", # ["projective", "knn"] (Type of Mean Squared Distance Calculation for Scale of Gaussians)
    report_iter_progress=False,
    load_checkpoint=False,
    checkpoint_time_idx=0,
    save_checkpoints=True, # Save Checkpoints
    checkpoint_interval=300, # Checkpoint Interval
    use_gt_semantic=False,  # change9
    model=dict(
        c_dim=16,    # feature dimension
        pretrained_model_path=f"/data0/3dg/splatam/segmentation/scannet/dinov2_{scene_name}.pth",
        n_classes=28, # number of nlasses (需要修改，这个是room0的)
        # 相机的参数
        crop_edge=10,
        H=480,
        W=640,
    ),
    data=dict(
        basedir="/data0/scannet",
        gradslam_data_cfg="./configs/data/scannet.yaml",
        sequence=scene_name,
        desired_image_height=480,
        desired_image_width=640,
        start=0,
        end=-1,
        stride=1,
        num_frames=-1,
    ),
    BA=dict(
        use_gt_poses=False, # Use GT Poses for Tracking
        forward_prop=True, # Forward Propagate Poses
        num_iters=40,
        use_sil_for_loss=True,
        sil_thres=0.99,
        use_l1=True,
        ignore_outlier_depth_loss=False,
        # semantic: for visualization
        visualize_tracking_loss=False,
        loss_weights=dict(
            im=0.5,
            depth=1.0,
            # semantic
            se=0.004,
        ),
        lrs=dict(
            #change17
            means3D=0.000001,
            rgb_colors=0.000025,
            # semantic
            sem_labels=0.000025,
            unnorm_rotations=0.00001,
            logit_opacities=0.0005,
            log_scales=0.00001,
            #######
            cam_unnorm_rots=0.0000005,
            cam_trans=0.0000005,
        ),
    ),
    tracking=dict(
        use_gt_poses=False, # Use GT Poses for Tracking
        forward_prop=True, # Forward Propagate Poses
        num_iters=tracking_iters,
        use_sil_for_loss=True,
        sil_thres=0.99,
        use_l1=True,
        ignore_outlier_depth_loss=False,
        use_uncertainty_for_loss_mask=False,
        use_uncertainty_for_loss=False,
        use_chamfer=False,
        visualize_tracking_loss=True,
        loss_weights=dict(
            im=0.5,
            depth=1.0,
            # semantic
            #se=20000,
            #se_fe=0.002,
            se=0,
            se_fe=0,
        ),
        lrs=dict(
            means3D=0.0,
            rgb_colors=0.0,
            # semantic
            sem_labels=0.0,
            unnorm_rotations=0.0,
            logit_opacities=0.0,
            log_scales=0.0,
            cam_unnorm_rots=0.0005,
            cam_trans=0.0005,
        ),
    ),
    mapping=dict(
        num_iters=mapping_iters,
        add_new_gaussians=True,
        sil_thres=0.5, # For Addition of new Gaussians
        use_l1=True,
        use_sil_for_loss=False,
        ignore_outlier_depth_loss=False,
        use_uncertainty_for_loss_mask=False,
        use_uncertainty_for_loss=False,
        use_chamfer=False,
        loss_weights=dict(
            im=0.5,
            depth=1.0,
            se=0.14,  # use_F
            se_fe=0.01,
        ),
        lrs=dict(
            means3D=0.0001,
            rgb_colors=0.0025,
            # semantic
            sem_labels=0.0025,
            unnorm_rotations=0.001,
            logit_opacities=0.05,
            log_scales=0.001,
            cam_unnorm_rots=0.0000,
            cam_trans=0.0000,
        ),
        prune_gaussians=True, # Prune Gaussians during Mapping
        pruning_dict=dict( # Needs to be updated based on the number of mapping iterations
            start_after=0,
            remove_big_after=0,
            stop_after=20,
            prune_every=20,
            removal_opacity_threshold=0.005,
            final_removal_opacity_threshold=0.005,
            reset_opacities=False,
            reset_opacities_every=500, # Doesn't consider iter 0
        ),
    ),
)