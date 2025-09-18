import os
from os.path import join as p_join

scenes = ["room_0", "room_1", "room_2",
          "office_0", "office_1", "office_2",
          "office_3", "office_4"]

primary_device="cuda:0"
seed = 2027
scene_name = scenes[5]

map_every = 8
keyframe_every = 5

mapping_window_size = 24
tracking_iters = 40
mapping_iters = 60

group_name = "Replica"
run_name = f"{scene_name}_{seed}"

config = dict(
    workdir=f"./experiments/{group_name}",
    group_name=group_name,
    run_name=run_name,
    seed=seed,
    primary_device=primary_device,
    map_every=map_every, # Mapping every nth frame
    keyframe_every=keyframe_every, # Keyframe every nth frame
    BA_every=32,
    BA_iters=15,
    mapping_window_size=mapping_window_size, # Mapping window size
    report_global_progress_every=1000, # Report Global Progress every nth frame
    eval_every=5, # Evaluate every nth frame (at end of SLAM)
    scene_radius_depth_ratio=3, # Max First Frame Depth to Scene Radius Ratio (For Pruning/Densification)
    mean_sq_dist_method="projective", # ["projective", "knn"] (Type of Mean Squared Distance Calculation for Scale of Gaussians)
    report_iter_progress=False,
    load_checkpoint=False,
    checkpoint_time_idx=1499,
    save_checkpoints=True, # Save Checkpoints
    checkpoint_interval=300, # Checkpoint Interval
    use_gt_semantic=False,
    model=dict(
        c_dim=16,    # feature dimension
        pretrained_model_path=f"/data0/3dg/splatam/segmentation/replica/dinov2_replica.pth",
        n_classes=52, # number of nlasses (需要修改，这个是room0的)
        # 相机的参数
        crop_edge=0,
        H=680,
        W=1200,
    ),
    data=dict(
        basedir="/data0/replica",
        gradslam_data_cfg="./configs/data/replica.yaml",
        sequence=scene_name,
        desired_image_height=680,
        desired_image_width=1200,
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
            cam_unnorm_rots=0.0000004,
            cam_trans=0.000002,
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
        # semantic: for visualization
        visualize_tracking_loss=True,
        loss_weights=dict(
            im=0.5,
            depth=1.0,
            # semantic
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
            cam_unnorm_rots=0.0004,
            cam_trans=0.002,
        ),
    ),
    mapping=dict(
        num_iters=mapping_iters,
        add_new_gaussians=True,
        sil_thres=0.5, # For Addition of new Gaussians
        use_l1=True,
        use_sil_for_loss=False,
        ignore_outlier_depth_loss=False,
        loss_weights=dict(
            im=0.5, #0.1 /1
            depth=1.0,  #0.5 /2
            se=0.01,  #use_F    #0.005 / 0.02
            se_fe=0.01,    #use_F #0.005 / 0.02
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