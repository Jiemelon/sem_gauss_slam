import subprocess




#mesh_out_file = f'./experiments/Replica_V2/office_4_107/eval_nvs/mesh/pred_mesh.ply'
mesh_out_file = f'./experiments/Replica/office_2_2027/eval/mesh/pred_mesh.ply'
# mesh_out_file = f'./experiments/our_mesh/pred_mesh.ply'

print('Evaluating 3D reconstruction')
result_recon_obj = subprocess.run(['python', '-u', './eval_mesh/eval_recon.py', '--rec_mesh',
                                    mesh_out_file,
                                    '--gt_mesh', f'./experiments/eval_mesh/office2_mesh.ply',
                                    '-3d'],
                                    text=True, check=True, capture_output=True)
result_recon1 = result_recon_obj.stdout
result_recon2 = result_recon_obj.stderr
print(result_recon1)
print(result_recon2)

print('✨ Successfully evaluated 3D reconstruction.')

