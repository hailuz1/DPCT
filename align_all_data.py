import os
import glob
import open3d as o3d

from copy import deepcopy
from Holo_code.utils.align_frame import align_frame
from Holo_code.utils.draw_registration import draw_registration_result
from Holo_code.utils.pcd_segmentation import get_centers
from Holo_code.utils.convert_pc_mesh import convert_point_cloud_to_mesh , sample_random_point_cloud
import train_point_corr_holo
import numpy as np

if __name__ == "__main__":
    orbit_pdc = o3d.io.read_point_cloud("Holo_code/data/ply data/orbit.ply")
    if not(os.path.exists('data/datasets/hololens_off/1.off')):
        #orbit_pdc_mesh = o3d.io.read_triangle_mesh("Holo_code/data/ply data/orbit.ply")
        #orbit_pdc_mesh_sampeled = orbit_pdc_mesh.sample_points_uniformly(number_of_points=1024) # Downsample for the DPC model
        #orbit_pdc_mesh_sampeled_temp = convert_point_cloud_to_mesh(orbit_pdc_mesh_sampeled)
        orbit_pc_sampeled = sample_random_point_cloud(orbit_pdc)
        orbit_pdc_mesh_sampeled = convert_point_cloud_to_mesh(orbit_pc_sampeled)
        o3d.io.write_triangle_mesh("data/datasets/hololens_off/1.off", orbit_pdc_mesh_sampeled) # For DPC Model


    orbit_pdc = orbit_pdc.uniform_down_sample(every_k_points=5)
    orbit_extracted_pdc = o3d.io.read_point_cloud("Holo_code/data/ply data/orbit-extracted.ply")
    orbit_landmarks_centers = get_centers(json_path='Holo_code/data/orbit interest points/orbit centers.json')

    for pcd_path in glob.glob("Holo_code/data/**/*-cam_space.ply", recursive=True):
        filename = os.path.basename(pcd_path)
        room_world_pcd = o3d.io.read_point_cloud(pcd_path.replace('-cam_space.ply', '.ply'))
        room_camera_pcd = o3d.io.read_point_cloud(pcd_path)

        orbit_model_pdc, room_world_pcd, reg_p2p = align_frame(deepcopy(orbit_pdc), deepcopy(orbit_extracted_pdc),
                                                               deepcopy(orbit_landmarks_centers), deepcopy(room_world_pcd),
                                                               room_camera_pcd) or (None, None, None)

        if reg_p2p:
            #o3d.visualization.draw_geometries([room_world_pcd])
            #o3d.visualization.draw_geometries([orbit_model_pdc])
            #draw_registration_result(room_world_pcd, orbit_model_pdc, reg_p2p.transformation)

            

            orbit_model_pdc_sampled = sample_random_point_cloud(orbit_model_pdc)
            source_temp = deepcopy(room_world_pcd)
            """ Do translation on the source point cloud """
            source_temp.transform(reg_p2p.transformation)
            source_temp_sampeled = sample_random_point_cloud(source_temp) # maybe need uniform ??
            # ratio_pc = len(source_temp.points)/1024
            # source_temp_sampeled = source_temp.random_down_sample(1/ratio_pc)
            mesh_source_temp = convert_point_cloud_to_mesh(source_temp_sampeled)
            if os.path.exists('data/datasets/hololens_off/2.off'):
                os.remove('data/datasets/hololens_off/2.off')
            o3d.io.write_triangle_mesh("data/datasets/hololens_off/2.off", mesh_source_temp) # For DPC Model


            """ TODO: Implement the DPC model for non-rigid alignment
            
             P = DPC(source,target)
             argmax_idx = 1
             scalar_maps_i = P[i].argmax(dim=argmax_idx).detach().cpu().numpy() if len(P.shape) == 3 else P
             target.[:] = source[scalar_maps_i] 
             o3d.visualization.draw_geometries([source])
             o3d.visualization.draw_geometries([target])
             o3d.visualization.draw_geometries([source,target])
             
             """
            test_out, model = train_point_corr_holo.main()
            inx_vec = np.zeros([1,len(test_out[0])]).astype(int)
            for ind,val in enumerate(list(test_out[0].items())):
                inx_vec[0,ind] = int(val[1])

            arr_points = np.array(source_temp_sampeled.points)[inx_vec,:] 
            orbit_model_pdc_sampled.points = o3d.utility.Vector3dVector(arr_points.squeeze())
            #arr_points = np.array(source_temp_sampeled.points)
            #xcv = arr_points[inx_vec,:]
            #source_temp_sampeled.points = o3d.utility.Vector3dVector(xcv.squeeze())
            o3d.io.write_point_cloud('data/datasets/hololens_off/results4.ply',orbit_model_pdc_sampled)
            o3d.visualization.draw_geometries([source_temp_sampeled])     














