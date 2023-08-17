import os
import numpy as np
import pickle
import cv2
from planar_graph_utils import render

corner_metric_thresh = 10
angle_metric_thresh = 5

class Metric():
    def calc(self, gt_data, conv_data, thresh=10.0, iou_thresh=0.7):
        ### compute corners precision/recall
        gts = gt_data['corners']
        dets = conv_data['corners']

        per_sample_corner_tp = 0.0
        per_sample_corner_fp = 0.0
        per_sample_corner_length = gts.shape[0]
        found = [False] * gts.shape[0]
        per_dist=[20] * gts.shape[0]
        c_det_annot = {}


        # for each corner detection
        for i, det in enumerate(dets):
            # get closest gt
            near_gt = [0, 999999.0, (0.0, 0.0)]
            for k, gt in enumerate(gts):
                dist = np.linalg.norm(gt - det)
                if dist < near_gt[1]:
                    near_gt = [k, dist, gt]
            if not found[near_gt[0]]:
                per_dist[near_gt[0]]=near_gt[1]
            if near_gt[1] <= thresh and not found[near_gt[0]]:
                per_sample_corner_tp += 1.0
                found[near_gt[0]] = True
                c_det_annot[i] = near_gt[0]
            else:
                per_sample_corner_fp += 1.0

        per_corner_score = {
            'recall': per_sample_corner_tp / gts.shape[0],
            'precision': per_sample_corner_tp / (per_sample_corner_tp + per_sample_corner_fp + 1e-8),
        }

        ### compute edges precision/recall
        per_sample_edge_tp = 0.0
        per_sample_edge_fp = 0.0
        edge_corner_annots = gt_data['edges']
        per_sample_edge_length = edge_corner_annots.shape[0]

        false_edge_ids = []
        match_gt_ids = set()

        for l, e_det in enumerate(conv_data['edges']):
            c1, c2 = e_det

            # check if corners are mapped
            if (c1 not in c_det_annot.keys()) or (c2 not in c_det_annot.keys()):
                per_sample_edge_fp += 1.0
                false_edge_ids.append(l)
                continue
            # check hit
            c1_prime = c_det_annot[c1]
            c2_prime = c_det_annot[c2]
            is_hit = False

            for k, e_annot in enumerate(edge_corner_annots):
                c3, c4 = e_annot
                if ((c1_prime == c3) and (c2_prime == c4)) or ((c1_prime == c4) and (c2_prime == c3)):
                    is_hit = True
                    match_gt_ids.add(k)
                    break

            # hit
            if is_hit:
                per_sample_edge_tp += 1.0
            else:
                per_sample_edge_fp += 1.0
                false_edge_ids.append(l)

        per_edge_score = {
            'recall': per_sample_edge_tp / edge_corner_annots.shape[0],
            'precision': per_sample_edge_tp / (per_sample_edge_tp + per_sample_edge_fp + 1e-8)
        }

        # computer regions precision/recall
        conv_mask = render(corners=conv_data['corners'], edges=conv_data['edges'],render_pad=0, edge_linewidth=1)[0]
        conv_mask = 1 - conv_mask
        conv_mask = conv_mask.astype(np.uint8)
        labels, region_mask = cv2.connectedComponents(conv_mask, connectivity=4)

        #cv2.imwrite('mask-pred.png', region_mask.astype(np.uint8) * 20)

        background_label = region_mask[0, 0]
        all_conv_masks = []
        all_conv_polys=[]
        for region_i in range(1, labels):
            if region_i == background_label:
                continue
            the_region = region_mask == region_i
            # if the_region.sum() < 20:
            #     continue
            the_poly=self.polygonize_mask(the_region)
            
            all_conv_polys.append(the_poly)        
            all_conv_masks.append(the_region)

        gt_mask = render(corners=gt_data['corners'], edges=gt_data['edges'], render_pad=0, edge_linewidth=1)[0]
        gt_mask = 1 - gt_mask
        gt_mask = gt_mask.astype(np.uint8)
        labels, region_mask = cv2.connectedComponents(gt_mask, connectivity=4)

        #cv2.imwrite('mask-gt.png', region_mask.astype(np.uint8) * 20)

        background_label = region_mask[0, 0]
        all_gt_masks = []
        all_gt_polys=[]
        for region_i in range(1, labels):
            if region_i == background_label:
                continue
            the_region = region_mask == region_i
            # if the_region.sum() < 20:
            #     continue
            the_poly=self.polygonize_mask(the_region) 
            all_gt_polys.append(the_poly)  
            all_gt_masks.append(the_region)

        per_sample_region_tp = 0.0
        per_sample_region_fp = 0.0
        per_sample_region_length = len(all_gt_masks)
        found = [False] * len(all_gt_masks)
        per_iou=[0] * len(all_gt_masks)
        for i, r_det in enumerate(all_conv_masks):
            # gt closest gt
            near_gt = [0, 0, None]
            for k, r_gt in enumerate(all_gt_masks):
                iou = np.logical_and(r_gt, r_det).sum() / float(np.logical_or(r_gt, r_det).sum())
                if iou > near_gt[1]:
                    near_gt = [k, iou, r_gt]
            if near_gt[1] >= iou_thresh and not found[near_gt[0]]:
                per_sample_region_tp += 1.0
                found[near_gt[0]] = True
                per_iou[near_gt[0]] = near_gt[1]
            else:
                per_sample_region_fp += 1.0

        per_region_score = {
            'recall': per_sample_region_tp / len(all_gt_masks),
            'precision': per_sample_region_tp / (per_sample_region_tp + per_sample_region_fp + 1e-8)
        }

        per_sample_angle_tp,per_angle,room_overlap,all_gt_angle,all_pred_angle=self.cal_angle_metric(all_conv_polys,all_gt_polys)
        
        return {
            'per_dist':sum(per_dist),
            'corner_tp': per_sample_corner_tp,
            'corner_fp': per_sample_corner_fp,
            'corner_length': per_sample_corner_length,
            'edge_tp': per_sample_edge_tp,
            'edge_fp': per_sample_edge_fp,
            'edge_length': per_sample_edge_length,
            'per_iou':sum(per_iou),
            'region_tp': per_sample_region_tp,
            'region_fp': per_sample_region_fp,
            'region_length': per_sample_region_length,
            'angle_tp': per_sample_angle_tp,
            'per_angle':per_angle,
            'gt_angle_length':all_gt_angle,
            'pred_angle_length': all_pred_angle,
            'corner': per_corner_score,
            'edge': per_edge_score,
            'region': per_region_score,
            'room_overlap':room_overlap
        }

    def cal_angle_metric(self,pred_polys,gt_polys,masks_list=None,ignore_mask_region=0):
        def get_room_metric():
            pred_overlaps = [False] * len(pred_room_map_list)

            for pred_ind1 in range(len(pred_room_map_list) - 1):
                pred_map1 = pred_room_map_list[pred_ind1]

                for pred_ind2 in range(pred_ind1 + 1, len(pred_room_map_list)):
                    pred_map2 = pred_room_map_list[pred_ind2]

                    # if dataset_type == "s3d":
                    kernel = np.ones((5, 5), np.uint8)
                    # else:
                    #     kernel = np.ones((3, 3), np.uint8)

                    # todo: for our method, the rooms share corners and edges, need to check here
                    pred_map1_er = cv2.erode(pred_map1, kernel)
                    pred_map2_er = cv2.erode(pred_map2, kernel)

                    # intersection = (pred_map1_er + pred_map2_er) == 2
                    intersection = (pred_map1_er + pred_map2_er) == 2

                    intersection_area = np.sum(intersection)

                    if intersection_area >= 1:
                        pred_overlaps[pred_ind1] = True
                        pred_overlaps[pred_ind2] = True
            room_overlap=pred_overlaps
            room_metric = [np.bool((1 - pred_overlaps[ind]) * pred2gt_exists[ind]) for ind in range(len(pred_polys))]
            return room_metric,room_overlap
        
        def get_line_vector(p1, p2):
            p1 = np.concatenate((p1, np.array([1])))
            p2 = np.concatenate((p2, np.array([1])))

            line_vector = -np.cross(p1, p2)

            return line_vector

        def get_poly_orientation(my_poly):
            angles_sum = 0
            for v_ind, _ in enumerate(my_poly):
                if v_ind < len(my_poly) - 1:
                    v_sides = my_poly[[v_ind - 1, v_ind, v_ind, v_ind + 1], :]
                else:
                    v_sides = my_poly[[v_ind - 1, v_ind, v_ind, 0], :]

                v1_vector = get_line_vector(v_sides[0], v_sides[1])
                v1_vector = v1_vector / (np.linalg.norm(v1_vector, ord=2) + 1e-4)
                v2_vector = get_line_vector(v_sides[2], v_sides[3])
                v2_vector = v2_vector / (np.linalg.norm(v2_vector, ord=2) + 1e-4)

                orientation = (v_sides[1, 1] - v_sides[0, 1]) * (v_sides[3, 0] - v_sides[1, 0]) - (
                        v_sides[3, 1] - v_sides[1, 1]) * (
                                        v_sides[1, 0] - v_sides[0, 0])

                v1_vector_2d = v1_vector[:2] / (v1_vector[2] + 1e-4)
                v2_vector_2d = v2_vector[:2] / (v2_vector[2] + 1e-4)

                v1_vector_2d = v1_vector_2d / (np.linalg.norm(v1_vector_2d, ord=2) + 1e-4)
                v2_vector_2d = v2_vector_2d / (np.linalg.norm(v2_vector_2d, ord=2) + 1e-4)

                angle_cos = v1_vector_2d.dot(v2_vector_2d)
                angle_cos = np.clip(angle_cos, -1, 1)

                # G.T. has clockwise orientation, remove minus in the equation

                angle = np.sign(orientation) * np.abs(np.arccos(angle_cos))
                angle_degree = angle * 180 / np.pi

                angles_sum += angle_degree

            return np.sign(angles_sum)

        def get_angle_v_sides(inp_v_sides, poly_orient):
            v1_vector = get_line_vector(inp_v_sides[0], inp_v_sides[1])
            v1_vector = v1_vector / (np.linalg.norm(v1_vector, ord=2) + 1e-4)
            v2_vector = get_line_vector(inp_v_sides[2], inp_v_sides[3])
            v2_vector = v2_vector / (np.linalg.norm(v2_vector, ord=2) + 1e-4)

            orientation = (inp_v_sides[1, 1] - inp_v_sides[0, 1]) * (inp_v_sides[3, 0] - inp_v_sides[1, 0]) - (
                    inp_v_sides[3, 1] - inp_v_sides[1, 1]) * (
                                    inp_v_sides[1, 0] - inp_v_sides[0, 0])

            v1_vector_2d = v1_vector[:2] / (v1_vector[2]+ 1e-4)
            v2_vector_2d = v2_vector[:2] / (v2_vector[2]+ 1e-4)

            v1_vector_2d = v1_vector_2d / (np.linalg.norm(v1_vector_2d, ord=2) + 1e-4)
            v2_vector_2d = v2_vector_2d / (np.linalg.norm(v2_vector_2d, ord=2) + 1e-4)

            angle_cos = v1_vector_2d.dot(v2_vector_2d)
            angle_cos = np.clip(angle_cos, -1, 1)

            angle = poly_orient * np.sign(orientation) * np.arccos(angle_cos)
            angle_degree = angle * 180 / np.pi

            return angle_degree

        def poly_map_sort_key(x):
            return np.sum(x[1])
        
        h, w = (256,256)
        gt_room_map_list = []
        for room_ind, poly in enumerate(gt_polys):
            room_map = np.zeros((h, w))
            cv2.fillPoly(room_map, [poly], color=1.)

            gt_room_map_list.append(room_map)

        gt_polys_sorted_indcs = [i[0] for i in sorted(enumerate(gt_room_map_list), key=poly_map_sort_key, reverse=True)]

        gt_polys = [gt_polys[ind] for ind in gt_polys_sorted_indcs]
        gt_room_map_list = [gt_room_map_list[ind] for ind in gt_polys_sorted_indcs]

        if pred_polys is not None:
            pred_room_map_list = []
            for room_ind, poly in enumerate(pred_polys):
                room_map = np.zeros((h, w))
                cv2.fillPoly(room_map, [poly], color=1.)

                pred_room_map_list.append(room_map)
        else:
            pred_room_map_list = masks_list

        gt2pred_indices = [-1] * len(gt_polys)
        gt2pred_exists = [False] * len(gt_polys)
        # gt2pred_ious = [0] * len(gt_polys)
        for gt_ind, gt_map in enumerate(gt_room_map_list):

            best_iou = 0.
            best_ind = -1
            for pred_ind, pred_map in enumerate(pred_room_map_list):

                intersection = (1 - ignore_mask_region) * ((pred_map + gt_map) == 2)
                union = (1 - ignore_mask_region) * ((pred_map + gt_map) >= 1)

                iou = np.sum(intersection) / (np.sum(union) + 1)

                if iou > best_iou and iou > 0.5:
                    best_iou = iou
                    best_ind = pred_ind

            #         plt.figure()
            #         plt.subplot(121)
            #         plt.imshow(pred_map)
            #         plt.subplot(122)
            #         plt.imshow(gt_map)
            #         plt.show()
            # if best_ind == -1:
            #     plt.figure()
            #     plt.imshow(gt_map)
            #     plt.show()

            gt2pred_indices[gt_ind] = best_ind
            gt2pred_exists[gt_ind] = best_ind != -1
            # if best_ind == -1:
            #     plt.figure()
            #     plt.imshow(gt_map)
            #     plt.show()

        pred2gt_exists = [True if pred_ind in gt2pred_indices else False for pred_ind, _ in enumerate(pred_polys)]
        pred2gt_indices = [gt2pred_indices.index(pred_ind) if pred_ind in gt2pred_indices else -1 for pred_ind, _ in enumerate(pred_polys)]
        # print(gt2pred_indices)
        # print(pred2gt_indices)
        # assert False

        # import pdb; pdb.set_trace()
        room_metric,room_overlap= get_room_metric()

        room_angles_metric = []
        room_angles_angle_metric=[]
        false_point=0
        for pred_poly_ind, gt_poly_ind in enumerate(pred2gt_indices):
            p_poly = pred_polys[pred_poly_ind][:-1] # Last vertex = First vertex

            p_poly_angle_metrics = [False] * p_poly.shape[0]
            gt_poly = gt_polys[gt_poly_ind][:-1]
            p_dist_angle_metrics = [0] * gt_poly.shape[0]
            if not room_metric[pred_poly_ind]:
                room_angles_metric += p_poly_angle_metrics
                room_angles_angle_metric+=p_dist_angle_metrics
                continue
            # for v in p_poly:
            #     v_dists = np.linalg.norm(v[None,:] - gt_poly, axis=1, ord=2)
            #     v_min_dist = np.min(v_dists)
            #
            #     v_tp = v_min_dist <= 10
            #     room_corners_metric.append(v_tp)

            gt_poly_orient = get_poly_orientation(gt_poly)
            p_poly_orient = get_poly_orientation(p_poly)

            for v_gt_ind, v in enumerate(gt_poly):
                v_dists = np.linalg.norm(v[None,:] - p_poly, axis=1, ord=2)
                v_ind = np.argmin(v_dists)
                v_min_dist = v_dists[v_ind]

                if v_min_dist > corner_metric_thresh:
                    false_point+=1
                    # room_angles_metric.append(False)
                    continue

                if v_ind < len(p_poly) - 1:
                    v_sides = p_poly[[v_ind - 1, v_ind, v_ind, v_ind + 1], :]
                else:
                    v_sides = p_poly[[v_ind - 1, v_ind, v_ind, 0], :]

                v_sides = v_sides.reshape((4,2))
                pred_angle_degree = get_angle_v_sides(v_sides, p_poly_orient)

                # Note: replacing some variables with values from the g.t. poly

                if v_gt_ind < len(gt_poly) - 1:
                    v_sides = gt_poly[[v_gt_ind - 1, v_gt_ind, v_gt_ind, v_gt_ind + 1], :]
                else:
                    v_sides = gt_poly[[v_gt_ind - 1, v_gt_ind, v_gt_ind, 0], :]

                v_sides = v_sides.reshape((4, 2))
                gt_angle_degree = get_angle_v_sides(v_sides, gt_poly_orient)

                angle_metric = np.abs(pred_angle_degree - gt_angle_degree)

                # room_angles_metric.append(angle_metric < 5)
                p_poly_angle_metrics[v_ind] = angle_metric <= angle_metric_thresh
                if p_poly_angle_metrics[v_ind]:
                    p_dist_angle_metrics[v_gt_ind]=angle_metric
                # if angle_metric > 5:
                #     print(v_gt_ind, angle_metric)
                #     print(pred_angle_degree, gt_angle_degree)
                #     input("?")


            room_angles_metric += p_poly_angle_metrics
            room_angles_angle_metric+=p_dist_angle_metrics

        pred_corners_n = sum([poly.shape[0] - 1 for poly in pred_polys])
        gt_corners_n = sum([poly.shape[0] - 1 for poly in gt_polys])
        # for am, cm in zip(room_angles_metric, corner_metric):
        #     assert not (cm == False and am == True), "cm: %d am: %d" %(cm, am)

        return sum(room_angles_metric),sum(room_angles_angle_metric),sum(room_overlap),gt_corners_n,pred_corners_n

    def polygonize_mask(self, mask, degree=0.01, return_mask=False):
        h, w = mask.shape[0], mask.shape[1]
        mask = mask

        room_mask = 255 * (mask == 1)
        room_mask = room_mask.astype(np.uint8)
        room_mask_inv = 255 - room_mask

        ret, thresh = cv2.threshold(room_mask_inv, 250, 255, cv2.THRESH_BINARY_INV)

        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

        cnt = contours[0]
        max_area = cv2.contourArea(cnt)

        for cont in contours:
            if cv2.contourArea(cont) > max_area:
                cnt = cont
                max_area = cv2.contourArea(cont)

        perimeter = cv2.arcLength(cnt, True)
        # epsilon = 0.01 * cv2.arcLength(cnt, True)
        epsilon = degree * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # approx = np.concatenate([approx, approx[0][None]], axis=0)
        approx = approx.astype(np.int32).reshape((-1, 2))

        # approx_tensor = torch.tensor(approx, device=self.device)

        # return approx_tensor
        if return_mask:
            room_filled_map = np.zeros((h, w))
            cv2.fillPoly(room_filled_map, [approx], color=1.)

            return approx, room_filled_map
        else:
            return approx

    
def compute_metrics(gt_data, pred_data):
    metric = Metric()
    score = metric.calc(gt_data, pred_data)
    return score


def get_recall_and_precision(tp, fp, length):
    recall = tp / (length + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    return recall, precision


if __name__ == '__main__':
    base_path = './'
    gt_datapath = '../data/cities_dataset/annot'
    metric = Metric()
    corner_tp = 0.0
    corner_fp = 0.0
    corner_length = 0.0
    edge_tp = 0.0
    edge_fp = 0.0
    edge_length = 0.0
    region_tp = 0.0
    region_fp = 0.0
    region_length = 0.0
    for file_name in os.listdir(base_path):
        if len(file_name) < 10:
            continue
        f = open(os.path.join(base_path, file_name), 'rb')
        gt_data = np.load(os.path.join(gt_datapath, file_name + '.npy'), allow_pickle=True).tolist()
        candidate = pickle.load(f)
        conv_corners = candidate.graph.getCornersArray()
        conv_edges = candidate.graph.getEdgesArray()
        conv_data = {'corners': conv_corners, 'edges': conv_edges}
        score = metric.calc(gt_data, conv_data)
        corner_tp += score['corner_tp']
        corner_fp += score['corner_fp']
        corner_length += score['corner_length']
        edge_tp += score['edge_tp']
        edge_fp += score['edge_fp']
        edge_length += score['edge_length']
        region_tp += score['region_tp']
        region_fp += score['region_fp']
        region_length += score['region_length']

    f = open(os.path.join(base_path, 'score.txt'), 'w')
    # corner
    recall, precision = get_recall_and_precision(corner_tp, corner_fp, corner_length)
    f_score = 2.0 * precision * recall / (recall + precision + 1e-8)
    print('corners - precision: %.3f recall: %.3f f_score: %.3f' % (precision, recall, f_score))
    f.write('corners - precision: %.3f recall: %.3f f_score: %.3f\n' % (precision, recall, f_score))

    # edge
    recall, precision = get_recall_and_precision(edge_tp, edge_fp, edge_length)
    f_score = 2.0 * precision * recall / (recall + precision + 1e-8)
    print('edges - precision: %.3f recall: %.3f f_score: %.3f' % (precision, recall, f_score))
    f.write('edges - precision: %.3f recall: %.3f f_score: %.3f\n' % (precision, recall, f_score))

    ##region
    recall, precision = get_recall_and_precision(region_tp, region_fp, region_length)
    f_score = 2.0 * precision * recall / (recall + precision + 1e-8)
    print('regions - precision: %.3f recall: %.3f f_score: %.3f' % (precision, recall, f_score))
    f.write('regions - precision: %.3f recall: %.3f f_score: %.3f\n' % (precision, recall, f_score))

    f.close()
