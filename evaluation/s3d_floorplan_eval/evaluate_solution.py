import copy
import functools
import numpy as np
import os

from Evaluator.Evaluator import Evaluator
from options import MCSSOptions
from DataRW.S3DRW import S3DRW
from DataRW.wrong_annotatios import wrong_s3d_annotations_list
from planar_graph_utils import get_regions_from_pg


room_polys_def = [np.array([[191, 150],
       [191,  70],
       [222,  70],
       [222, 150],
       [191, 150]]), np.array([[232,  65],
       [232,  11],
       [202,  11],
       [202,  65],
       [232,  65]]), np.array([[ 47,  50],
       [ 47, 150],
       [ 24, 150],
       [ 24,  50],
       [ 47,  50]]), np.array([[199, 156],
       [199, 234],
       [146, 234],
       [146, 156],
       [199, 156]]), np.array([[109, 184],
       [120, 184],
       [120, 156],
       [ 50, 156],
       [ 50, 234],
       [109, 234],
       [109, 184]]), np.array([[110, 234],
       [144, 234],
       [144, 187],
       [110, 187],
       [110, 234]]), np.array([[ 50,  50],
       [ 50, 150],
       [123, 150],
       [123, 184],
       [144, 184],
       [144, 150],
       [190, 150],
       [190,  70],
       [108,  70],
       [108,  50],
       [ 50,  50]])]

pg_base = '../npyoutput/Structure3d'
# pg_base='E:/FCR/heat-master/heat-master/results/npy_polygraph_s3d_256'

options = MCSSOptions()
opts = options.parse()

if __name__ == '__main__':

    # data_rw = FloorNetRW(opts)
    corner_num=0.0
    if opts.scene_id == "val":

        opts.scene_id = "scene_03250" # Temp. value
        data_rw = S3DRW(opts)
        scene_list = data_rw.loader.scenes_list

        quant_result_dict = None
        quant_result_maskrcnn_dict = None
        scene_counter = 0
        cut_points=0.0
        for scene_ind, scene in enumerate(scene_list):
            if int(scene[6:]) in wrong_s3d_annotations_list:
                continue

            # print("------------")
            curr_opts = copy.deepcopy(opts)
            curr_opts.scene_id = scene
            curr_data_rw = S3DRW(curr_opts)
            # print("Running Evaluation for scene %s" % scene)

            evaluator = Evaluator(curr_data_rw, curr_opts)

            # TODO load your room polygons into room_polys, list of polygons (n x 2)
            # room_polys = np.array([[[0,0], [200, 0], [200, 200]]]) # Placeholder

            pg_path = os.path.join(pg_base, scene[6:] + '.npy')
            # pg_path = os.path.join(pg_base, str(int(scene[6:])-3250) + '_recon.npy')
            example_pg = np.load(pg_path, allow_pickle=True).tolist()
            corner_num+=len(example_pg['corners'])
            regions = get_regions_from_pg(example_pg, corner_sorted=True)
            ori_corners=[tuple(p) for p in example_pg['corners']]
            remove_list=[]
            regions_corner=[]
            for r in regions:
                for p in r:
                    regions_corner.append(tuple(p))
            for p in ori_corners:
                if not p in regions_corner:
                    remove_list.append(p)
            room_polys = regions
            cut_points+=len(remove_list)
            # room_polys = room_polys_def # Placeholder


            quant_result_dict_scene =\
                evaluator.evaluate_scene(room_polys=room_polys,remove_corners=remove_list)

            if quant_result_dict is None:
                quant_result_dict = quant_result_dict_scene
            else:
                for k in quant_result_dict.keys():
                    quant_result_dict[k] += quant_result_dict_scene[k]

            scene_counter += 1

            # break

        # for k in quant_result_dict.keys():
        #     quant_result_dict[k] /= float(scene_counter)
        room_metric=quant_result_dict['room_metric']
        room_iou_metric=quant_result_dict['room_iou_metric']
        pred_polys=quant_result_dict['pred_polys']
        gt_polys=quant_result_dict['gt_polys']
        corner_metric=quant_result_dict['corner_metric']
        gt_corners_n=quant_result_dict['gt_corners_n']
        pred_corners_n=quant_result_dict['pred_corners_n']
        angle_metric=quant_result_dict['angle_metric']
        angle_angle_metric=quant_result_dict['angle_angle_metric']
        
        room_overlap=quant_result_dict['room_overlap']
        
        print('corners - precision: %.3f recall: %.3f' % (corner_metric/ pred_corners_n,corner_metric/ gt_corners_n))
        print('regions - precision: %.3f recall: %.3f ' % (room_metric/ pred_polys,room_metric/ gt_polys))
        print('angles - precision: %.3f recall: %.3f' % (angle_metric/ pred_corners_n, angle_metric/ gt_corners_n))
        print('structral - S_cons: %.3f  MAnE: %.3f S_comp: %.3f ' % (10*(cut_points+room_overlap+1)/(room_metric+corner_metric+1),
                                                                      (angle_angle_metric+(gt_corners_n-angle_metric)*10)/(gt_corners_n+ 1e-8),
                                                                      room_metric/ (corner_num + 1e-8)))
