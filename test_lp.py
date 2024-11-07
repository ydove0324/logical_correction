#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import re
import math
import json
import random
import cvxpy as cp

from .utils import extract_bbx, extract_pos

M = 1e6  # A large number, should be chosen carefully depending on the context
EPSILON = 1e-2
PADDING_SCALE = 1.0 # 1.1

def get_half_bbx(bbx, vars, padding=1.0):
    x_size = bbx[0]
    y_size = bbx[1]
    _, _, rotate_0, rotate_90, rotate_180, rotate_270 = vars
    half_width = (cp.multiply(0.5 * x_size, rotate_0 + rotate_180) +
                   cp.multiply(0.5 * y_size, rotate_90 + rotate_270))
    half_height = (cp.multiply(0.5 * y_size, rotate_0 + rotate_180) +
                    cp.multiply(0.5 * x_size, rotate_90 + rotate_270))
    return half_width * padding, half_height * padding

def check_two_item_overlap(p1, l1, p2, l2):
    # p1, p2为矩形中心点坐标，坐标形式为(x, y)
    # l1, l2为矩形的长度，形式为(width, height)
    
    # 矩形1的边界
    x1_min = p1[0] - l1[0] / 2.0
    x1_max = p1[0] + l1[0] / 2.0
    y1_min = p1[1] - l1[1] / 2.0
    y1_max = p1[1] + l1[1] / 2.0

    # 矩形2的边界
    x2_min = p2[0] - l2[0] / 2.0
    x2_max = p2[0] + l2[0] / 2.0
    y2_min = p2[1] - l2[1] / 2.0
    y2_max = p2[1] + l2[1] / 2.0

    # 判断矩形是否重叠，增加10%的误差容忍度
    if x1_max < x2_min or x2_max < x1_min or y1_max < y2_min or y2_max < y1_min:
        return False
    else:
        return True

def not_overlap_constraints(cur_item_vars, cur_item_bbx, each_item_solution_vars, each_item_bbx):

    cx1 = cur_item_vars[0]
    cy1 = cur_item_vars[1]

    cx2 = each_item_solution_vars[0]
    cy2 = each_item_solution_vars[1]

    half_width1, half_height1 = get_half_bbx(cur_item_bbx, cur_item_vars, PADDING_SCALE)
    half_width2, half_height2 = get_half_bbx(each_item_bbx, each_item_solution_vars, PADDING_SCALE)

    # Binary variables to determine the relative positions
    left_of = cp.Variable(boolean=True)
    right_of = cp.Variable(boolean=True)
    above = cp.Variable(boolean=True)
    below = cp.Variable(boolean=True)

    # Constraints
    hard_constraints = [
        # Constraints for binary variable activation

        # object 1 is left of object 2
        cx2 - cx1 >= EPSILON + half_width1 + half_width2 - M * (1 - left_of),
        # object 1 is right of object 2
        cx1 - cx2 >= EPSILON + half_width1 + half_width2 - M * (1 - right_of),
        # object 1 is below object 2
        cy2 - cy1 >= EPSILON + half_height1 + half_height2 - M * (1 - below),
        # object 1 is above object 2
        cy1 - cy2 >= EPSILON + half_height1 + half_height2 - M * (1 - above),
        
        # Ensure that at least one of the binary variables must be True
        left_of + right_of + above + below >= 1,
    ]

    return hard_constraints

def inside_range_constraints(cur_item_vars, cur_item_bbx, parent_bbx):

    boundary_min_x, boundary_max_x, boundary_min_y, boundary_max_y = -parent_bbx[0] / 2.0, parent_bbx[0] / 2.0, -parent_bbx[1] / 2.0, parent_bbx[1] / 2.0
    half_width, half_height = get_half_bbx(cur_item_bbx, cur_item_vars)

    hard_constraints = [
        cur_item_vars[0] - half_width >= boundary_min_x,
        cur_item_vars[0] + half_width <= boundary_max_x,
        cur_item_vars[1] - half_height >= boundary_min_y,
        cur_item_vars[1] + half_height <= boundary_max_y,
    ]
    return hard_constraints

def relative_position_constraints(cur_item_vars, each_item_solution_vars, initial_item_pos, cur_item_bbx, each_item_bbx, placement_relationship_list, parent_bbx, same_area):

    hard_constraints = []
    soft_constraints = []

    x_relative_keep = cp.Variable(boolean=True)
    y_relative_keep = cp.Variable(boolean=True)

    # Here the padding_scale is 1.1; TODO: can change to 1.0 ? 
    half_width1, half_height1 = get_half_bbx(cur_item_bbx, cur_item_vars, 1.1)
    half_width2, half_height2 = get_half_bbx(each_item_bbx, each_item_solution_vars, 1.1)
    half_width = (half_width1 + half_width2) / 2.0
    half_height = (half_height1 + half_height2) / 2.0

    if placement_relationship_list:
        if "左" in placement_relationship_list:
            hard_constraints.extend([cur_item_vars[0] <= each_item_solution_vars[0] - half_width + M * (1 - x_relative_keep)])
        if "右" in placement_relationship_list:
            hard_constraints.extend([cur_item_vars[0] >= each_item_solution_vars[0] + half_width - M * (1 - x_relative_keep)])
        if "前" in placement_relationship_list:
            hard_constraints.extend([cur_item_vars[1] <= each_item_solution_vars[1] - half_height + M * (1 - y_relative_keep)])
        if "后" in placement_relationship_list:
            hard_constraints.extend([cur_item_vars[1] >= each_item_solution_vars[1] + half_height - M * (1 - y_relative_keep)])
        approach_factor = -5.0 if '紧挨' in placement_relationship_list else -0.001
        soft_constraints.extend([approach_factor / parent_bbx[0] * cp.square(cur_item_vars[0] - each_item_solution_vars[0]), approach_factor / parent_bbx[1] * cp.square(cur_item_vars[1] - each_item_solution_vars[1])])
        soft_constraints.extend([0.1 * x_relative_keep, 0.1 * y_relative_keep])

    # else:
    #     hard_constraints.extend([
    #         (cur_item_vars[0] - each_item_solution_vars[0]) * (initial_item_pos[0] - each_item_solution_vars[0]) >= 0 - M * (1 - x_relative_keep),
    #         (cur_item_vars[1] - each_item_solution_vars[1]) * (initial_item_pos[1] - each_item_solution_vars[1]) >= 0 - M * (1 - y_relative_keep),
    #     ])
    #     soft_constraints.extend([0.1 * x_relative_keep, 0.1 * y_relative_keep])

    if same_area:
        soft_constraints.extend([-1.0 / parent_bbx[0] * cp.square(cur_item_vars[0] - each_item_solution_vars[0]), -1.0 / parent_bbx[1] * cp.square(cur_item_vars[1] - each_item_solution_vars[1])])
    return hard_constraints, soft_constraints

def area_constraints(cur_item_vars, initial_item_pos, parent_bbx):
    hard_constraints = []
    soft_constraints = []
    soft_constraints.extend([-2.0 / parent_bbx[0] * cp.square(cur_item_vars[0] - initial_item_pos[0]), -2.0 / parent_bbx[1] * cp.square(cur_item_vars[1] - initial_item_pos[1])])

    return hard_constraints, soft_constraints

def edge_constraints(cur_item_vars, cur_item_bbx, cur_item_real_bbx, parent_bbx, edge_descrip, edge_factor, scene_info, orientation):

    boundary_min_x, boundary_max_x, boundary_min_y, boundary_max_y = -parent_bbx[0] / 2.0, parent_bbx[0] / 2.0, -parent_bbx[1] / 2.0, parent_bbx[1] / 2.0
    half_width, half_height = get_half_bbx(cur_item_bbx, cur_item_vars)

    x, y, rotate_0, rotate_90, rotate_180, rotate_270 = cur_item_vars

    hard_constraints = []
    soft_constraints = []
    edge_keeps = [cp.Variable(boolean=True) for i in range(4)]
    
    covered_length_by_wall = scene_info["covered_length_by_wall"]
    boundary_min_x += covered_length_by_wall[1]
    boundary_max_x -= covered_length_by_wall[0]
    boundary_min_y += covered_length_by_wall[2]
    boundary_max_y -= covered_length_by_wall[3]

    allowed_edge = scene_info["scene_boundary_info"]
    if not isinstance(allowed_edge, str):
        allowed_edge = ""
    allowed_edge_list = allowed_edge.split(',')
    if len(allowed_edge_list) < 4:
        allowed_edge_list.extend(['1']* (4-len(allowed_edge_list)))
    
    used_edge_keep_vars = []
    if "靠边" in edge_descrip and not "不靠边" in edge_descrip:
        if not '0' in allowed_edge_list[1]:
            # 靠左边
            hard_constraints.extend([x - boundary_min_x <= EPSILON + half_width + M * (1 - edge_keeps[0])])
            hard_constraints.extend([boundary_min_x - x <= EPSILON + half_width + M * (1 - edge_keeps[0])])
            hard_constraints.extend([rotate_270 <= 0 + M * (1 - edge_keeps[0])])
            if cur_item_real_bbx[0] / cur_item_real_bbx[1] >= 1.5:
                hard_constraints.extend([rotate_0 + rotate_180 <= 0 + M * (1 - edge_keeps[0])])
            if cur_item_real_bbx[1] / cur_item_real_bbx[0] >= 1.5:
                hard_constraints.extend([rotate_90 + rotate_270 <= 0 + M * (1 - edge_keeps[0])])
            used_edge_keep_vars.append(edge_keeps[0])

            if orientation == "西":
                hard_constraints.extend([rotate_0 >= 1 - M * (1 - edge_keeps[0])])
            elif orientation == "北":
                hard_constraints.extend([rotate_90 >= 1 - M * (1 - edge_keeps[0])])
            elif orientation == "东":
                hard_constraints.extend([rotate_180 >= 1 - M * (1 - edge_keeps[0])])
            elif orientation == "南":
                hard_constraints.extend([rotate_270 >= 1 - M * (1 - edge_keeps[0])])

        if not '0' in allowed_edge_list[0]:
            # 靠右边
            hard_constraints.extend([- x + boundary_max_x <= EPSILON + half_width + M * (1 - edge_keeps[1])])
            hard_constraints.extend([x - boundary_max_x <= EPSILON + half_width + M * (1 - edge_keeps[1])])
            hard_constraints.extend([rotate_90 <= 0 + M * (1 - edge_keeps[1])])
            if cur_item_real_bbx[0] / cur_item_real_bbx[1] >= 1.5:
                hard_constraints.extend([rotate_0 + rotate_180 <= 0 + M * (1 - edge_keeps[1])])
            if cur_item_real_bbx[1] / cur_item_real_bbx[0] >= 1.5:
                hard_constraints.extend([rotate_90 + rotate_270 <= 0 + M * (1 - edge_keeps[1])])
            used_edge_keep_vars.append(edge_keeps[1])

            if orientation == "东":
                hard_constraints.extend([rotate_0 >= 1 - M * (1 - edge_keeps[1])])
            elif orientation == "南":
                hard_constraints.extend([rotate_90 >= 1 - M * (1 - edge_keeps[1])])
            elif orientation == "西":
                hard_constraints.extend([rotate_180 >= 1 - M * (1 - edge_keeps[1])])
            elif orientation == "北":
                hard_constraints.extend([rotate_270 >= 1 - M * (1 - edge_keeps[1])])

        if not '0' in allowed_edge_list[2]:
            # 靠前边
            hard_constraints.extend([y - boundary_min_y <= EPSILON + half_height + M * (1 - edge_keeps[2])])
            hard_constraints.extend([- y + boundary_min_y <= EPSILON + half_height + M * (1 - edge_keeps[2])])
            hard_constraints.extend([rotate_0 <= 0 + M * (1 - edge_keeps[2])])
            if cur_item_real_bbx[0] / cur_item_real_bbx[1] >= 1.5:
                hard_constraints.extend([rotate_90 + rotate_270 <= 0 + M * (1 - edge_keeps[2])])
            if cur_item_real_bbx[1] / cur_item_real_bbx[0] >= 1.5:
                hard_constraints.extend([rotate_0 + rotate_180 <= 0 + M * (1 - edge_keeps[2])])
            used_edge_keep_vars.append(edge_keeps[2])

            if orientation == "南":
                hard_constraints.extend([rotate_0 >= 1 - M * (1 - edge_keeps[2])])
            elif orientation == "东":
                hard_constraints.extend([rotate_90 >= 1 - M * (1 - edge_keeps[2])])
            elif orientation == "北":
                hard_constraints.extend([rotate_180 >= 1 - M * (1 - edge_keeps[2])])
            elif orientation == "西":
                hard_constraints.extend([rotate_270 >= 1 - M * (1 - edge_keeps[2])])

        if not '0' in allowed_edge_list[3]:
            # 靠后边
            hard_constraints.extend([ -y + boundary_max_y <= EPSILON + half_height + M * (1 - edge_keeps[3])])
            hard_constraints.extend([ y - boundary_max_y <= EPSILON + half_height + M * (1 - edge_keeps[3]) ])
            hard_constraints.extend([rotate_180 <= 0 + M * (1 - edge_keeps[3])])
            if cur_item_real_bbx[0] / cur_item_real_bbx[1] >= 1.5:
                hard_constraints.extend([rotate_90 + rotate_270 <= 0 + M * (1 - edge_keeps[3])])
            if cur_item_real_bbx[1] / cur_item_real_bbx[0] >= 1.5:
                hard_constraints.extend([rotate_0 + rotate_180 <= 0 + M * (1 - edge_keeps[3])])
            used_edge_keep_vars.append(edge_keeps[3])

            if orientation == "北":
                hard_constraints.extend([rotate_0 >= 1 - M * (1 - edge_keeps[3])])
            elif orientation == "西":
                hard_constraints.extend([rotate_90 >= 1 - M * (1 - edge_keeps[3])])
            elif orientation == "南":
                hard_constraints.extend([rotate_180 >= 1 - M * (1 - edge_keeps[3])])
            elif orientation == "东":
                hard_constraints.extend([rotate_270 >= 1 - M * (1 - edge_keeps[3])])

        if len(hard_constraints):
            hard_constraints.extend([sum(used_edge_keep_vars) >= 1.0])
    else:
        return [], []

    return hard_constraints, soft_constraints

def alignment_constraints(cur_item_vars, each_item_solution_vars, each_item_bbx, placement_relationship_list, ratio=0.2, hard=False):
    soft_constraints = []
    hard_constraints = []
    x, y, rotate_0, rotate_90, rotate_180, rotate_270 = each_item_solution_vars

    align_keep = [cp.Variable(boolean=True) for _ in range(4)]

    if "左右居中对齐" in placement_relationship_list:
        hard_constraints.extend([
            cur_item_vars[0] - each_item_solution_vars[0] <= each_item_bbx[0] * ratio + M * (1 - align_keep[0]),
            each_item_solution_vars[0] - cur_item_vars[0] <= each_item_bbx[0] * ratio + M * (1 - align_keep[0]),
        ])
        hard_constraints.extend([
            cur_item_vars[1] - each_item_solution_vars[1] <= each_item_bbx[1] * ratio + M * (1 - align_keep[1]),
            each_item_solution_vars[1] - cur_item_vars[1] <= each_item_bbx[1] * ratio + M * (1 - align_keep[1]),
        ])
        soft_constraints.extend([
            align_keep[0] - (rotate_0 + rotate_180),
            align_keep[1] - (rotate_90 + rotate_270)
        ])
    if "前后居中对齐" in placement_relationship_list:
        hard_constraints.extend([
            cur_item_vars[0] - each_item_solution_vars[0] <= each_item_bbx[0] * ratio + M * (1 - align_keep[2]),
            each_item_solution_vars[0] - cur_item_vars[0] <= each_item_bbx[0] * ratio + M * (1 - align_keep[2]),
        ])
        hard_constraints.extend([
            cur_item_vars[1] - each_item_solution_vars[1] <= each_item_bbx[1] * ratio + M * (1 - align_keep[3]),
            each_item_solution_vars[1] - cur_item_vars[1] <= each_item_bbx[1] * ratio + M * (1 - align_keep[3]),
        ])
        soft_constraints.extend([
            align_keep[2] - (rotate_90 + rotate_270),
            align_keep[3] - (rotate_0 + rotate_180)
        ])

    if hard and len(soft_constraints):
        hard_constraints.extend([ each >= 0 for each in soft_constraints])
    return hard_constraints, soft_constraints

def face_to_constraints(cur_item_vars, cur_item_bbx, each_item_solution_vars, each_item_bbx):
    # Decision variables for object centers and rotations
    cx1, cy1, rotate_0_1, rotate_90_1, rotate_180_1, rotate_270_1 = cur_item_vars

    cx2, cy2 = each_item_solution_vars[:2]

    half_width1, half_height1 = get_half_bbx(cur_item_bbx, cur_item_vars, PADDING_SCALE)
    half_width2, half_height2 = get_half_bbx(each_item_bbx, each_item_solution_vars, PADDING_SCALE)

    # Binary variables to determine the relative positions
    left_of = cp.Variable(boolean=True)
    right_of = cp.Variable(boolean=True)
    above = cp.Variable(boolean=True)
    below = cp.Variable(boolean=True)

    # Constraints
    hard_constraints = [
        # Constraints for binary variable activation
        cx2 - cx1 >= EPSILON + half_width1 + half_width2 - M * (1 - left_of),
        cx1 - cx2 >= EPSILON + half_width1 + half_width2 - M * (1 - right_of),
        cy2 - cy1 >= EPSILON + half_height1 + half_height2 - M * (1 - below),
        cy1 - cy2 >= EPSILON + half_height1 + half_height2 - M * (1 - above),
        
        left_of + right_of + above + below == 1,
        rotate_90_1 + rotate_0_1 + rotate_180_1 <= M * (1 - right_of),
        rotate_270_1 + rotate_0_1 + rotate_180_1 <= M * (1 - left_of),
        rotate_180_1 + rotate_90_1 + rotate_270_1 <= M * (1 - above),
        rotate_0_1 + rotate_90_1 + rotate_270_1 <= M * (1 - below),

        # dont back to the target item
        # left_of + right_of + above + below >= 1,

        # # Directional constraints based on rotation
        # rotate_90_1 <= M * (1 - right_of),     # If right_of is true, cannot be rotated 90 degrees
        # rotate_270_1 <= M * (1 - left_of),   # If left_of is true, cannot be rotated 270 degrees
        # rotate_180_1 <= M * (1 - above),      # If above is true, cannot be rotated 180 degrees
        # rotate_0_1 <= M * (1 - below),        # If below is true, cannot be facing straight up (0 degrees)
    ]
    soft_constraints = []
    return hard_constraints, soft_constraints

def directional_constraints(cur_item_vars, cur_item_bbx, already_exists, orientation):
    hard_constraints = []
    soft_constraints = []
    item_names = [ each['scene_or_object_name'] for each in already_exists]
    _, _, rotate_0, rotate_90, rotate_180, rotate_270 = cur_item_vars

    if "与父物体使用方向一致" in orientation:
        hard_constraints.extend([rotate_0 == 1])
    elif orientation in item_names:
        target_item = already_exists[item_names.index(orientation)]
        target_hard_constraints, target_soft_constraints = face_to_constraints(cur_item_vars, cur_item_bbx, target_item['solution_vars'], extract_bbx(target_item['new_bounding_box']))
        hard_constraints.extend(target_hard_constraints)
        soft_constraints.extend(target_soft_constraints)
    else:
        hard_constraints.extend([])

    return hard_constraints, soft_constraints

def generate_all_constraints(dsl_result, each_children_scene_or_object, scene_info, already_exists, vars, ratio=1.0, retry_idx=0):

    _, _, rotate_0, rotate_90, rotate_180, rotate_270 = vars
    soft_constraints_list = []
    hard_constraints_list = [
        rotate_0 + rotate_90 + rotate_180 + rotate_270 == 1,
    ]

    cur_item_pos = extract_pos(dsl_result)
    cur_item_bbx = extract_bbx(each_children_scene_or_object['bounding_box'])
    cur_item_real_bbx = extract_bbx(each_children_scene_or_object['bounding_box_before_pad'])
    parent_bbx = extract_bbx(scene_info['bounding_box'])
    parent_name = scene_info['scene_or_object_name']
    placement_relationship_dict = each_children_scene_or_object['placement_relationship_dict'] if 'placement_relationship_dict' in each_children_scene_or_object else {}
    edge = each_children_scene_or_object['edge'] if 'edge' in each_children_scene_or_object else ''

    # Inside constraint
    hard_constraints_list.extend(inside_range_constraints(vars, cur_item_bbx, parent_bbx))

    _, soft_constraints = area_constraints(vars, cur_item_pos, parent_bbx)
    soft_constraints_list.extend(soft_constraints)

    # relative position for the first item
    if parent_name in placement_relationship_dict:
        hard_constraints, soft_constraints = relative_position_constraints(vars, [0,0,1,0,0,0], cur_item_pos, cur_item_bbx, parent_bbx, placement_relationship_dict[parent_name], parent_bbx, False)
        hard_constraints_list.extend(hard_constraints)
        soft_constraints_list.extend(soft_constraints)

        hard_constraints, soft_constraints = alignment_constraints(vars, [0,0,1,0,0,0], cur_item_bbx, placement_relationship_dict[parent_name], ratio, hard=retry_idx<1)
        hard_constraints_list.extend(hard_constraints)
        soft_constraints_list.extend(soft_constraints)

    # check if align exists in placement relation
    need_align = False
    for each_a_e in already_exists:
        item_name = each_a_e['scene_or_object_name']
        if item_name in placement_relationship_dict:
            if "对齐" in placement_relationship_dict[item_name]:
                need_align = True
                break
    
    orientation = each_children_scene_or_object.get('orientation', '前')
    # edge_factor = 1.0 if need_align else 2
    edge_factor = 2.0 if need_align else 2.0
    # edge constraint
    hard_edge_constraints, soft_edge_constraints = edge_constraints(vars, cur_item_bbx, cur_item_real_bbx, parent_bbx, edge, edge_factor, scene_info, orientation)
    if hard_edge_constraints:
        hard_constraints_list.extend(hard_edge_constraints)
    if soft_edge_constraints:
        soft_constraints_list.extend(soft_edge_constraints)
    
    # directional constraint
    hard_direction_constraints, soft_direction_constraints = directional_constraints(vars, cur_item_bbx, already_exists, orientation)
    hard_constraints_list.extend(hard_direction_constraints)
    soft_constraints_list.extend(soft_direction_constraints)
    
    for each_a_e in already_exists:
        each_item_pos = extract_pos(each_a_e['layout_function'])
        each_item_bbx = extract_pos(each_a_e['bounding_box'])
        if 'solution_vars' in each_a_e:
            each_item_solution_vars = each_a_e['solution_vars']
        else:
            if len(each_item_pos) > 2:
                each_item_pos = each_item_pos[:2]
            each_item_solution_vars = each_item_pos + [1,0,0,0]

        # overlap constraint
        overlap_constrains_add = True
        if ((cur_item_bbx[0] >= parent_bbx[0] * 2.0 / 3.0 and cur_item_bbx[1] >= parent_bbx[1] * 2.0 / 3.0) \
                or (each_item_bbx[0] >= parent_bbx[0] * 2.0 / 3.0 and each_item_bbx[1] >= parent_bbx[1] * 2.0 / 3.0)) \
                and each_item_bbx[2] < 1:
            overlap_constrains_add = False
        if cur_item_bbx[2] <= 0.01:
            overlap_constrains_add = False
        
        if overlap_constrains_add:
            hard_constraints_list.extend(not_overlap_constraints(vars, cur_item_bbx, each_item_solution_vars, each_item_bbx))

        # alignment constraint
        item_name = each_a_e['scene_or_object_name']
        placement_relationship_list = ""
        if item_name in placement_relationship_dict:
            placement_relationship_list = placement_relationship_dict[item_name]
            align_hard_constraints, align_soft_constraints = alignment_constraints(vars, each_item_solution_vars, each_item_bbx, placement_relationship_list, ratio, hard=retry_idx<1)
            soft_constraints_list.extend(align_soft_constraints)
            hard_constraints_list.extend(align_hard_constraints)
        
        if 'init_loc' in each_a_e:
            same_area = each_children_scene_or_object['init_loc'] == each_a_e['init_loc']
        else:
            same_area = False

        # relative position
        hard_constraints, soft_constraints = relative_position_constraints(vars, each_item_solution_vars, cur_item_pos, cur_item_bbx, each_item_bbx, placement_relationship_list, parent_bbx, same_area)
        hard_constraints_list.extend(hard_constraints)
        soft_constraints_list.extend(soft_constraints)

    return hard_constraints_list, soft_constraints_list

def result_not_valid(result):
    return result is None or math.isnan(result) or math.isinf(result)

def milp_for_one_item(dsl_result, each_children_scene_or_object, scene_info, already_exists):

    # x, y, rotate_0, rotate_90, rotate_180, rotate_270 Counterclockwise
    vars = [cp.Variable(), cp.Variable(), cp.Variable(boolean=True), cp.Variable(boolean=True), cp.Variable(boolean=True), cp.Variable(boolean=True)]
    retry_num = 2
    retry_idx = 0

    while retry_idx < retry_num:
        retry_idx += 1
        hard_constraints_list, soft_constraints_list = generate_all_constraints(dsl_result, each_children_scene_or_object, scene_info, already_exists, vars, 1.0, retry_idx)
        try:
            result = milp_solve(soft_constraints_list, hard_constraints_list)
        except Exception as e:
            print(f'milp solve error: {e}')
            continue
        if not result_not_valid(result):
            each_children_scene_or_object['solution_vars'] = [each.value.item() for each in vars]
            dsl_result = f"({vars[0].value.item():.2f},{vars[1].value.item():.2f})"
            return dsl_result
        print(f"{each_children_scene_or_object['scene_or_object_name']} retry No.{retry_idx}")
    print(f"Drop {each_children_scene_or_object['scene_or_object_name']}")
    return ""

def milp_solve(soft_constraints_list, hard_constraints_list, verbose=False):
    problem = cp.Problem(cp.Maximize(sum(soft_constraints_list)), hard_constraints_list)
    if verbose:
        print('solving milp using GUROBI ...')
    problem.solve(solver=cp.GUROBI, verbose=verbose)
    return problem.value

def milp_one_by_one(not_placed_items, already_exists, scene_info):
    for each_children_item in not_placed_items:
        if 'item_review' in each_children_item and not each_children_item['item_review']:
            continue
        same_plane_already_exists = [ each for each in already_exists if each['space_relation'] == each_children_item['space_relation'] and each.get('item_review', True)]
        llm_res = each_children_item.get('init_loc', '0,0')

        each_children_item["old_layout_function"] = each_children_item['space_relation'] + llm_res
        llm_res = milp_for_one_item(llm_res, each_children_item, scene_info, same_plane_already_exists)
        if not (llm_res and 'solution_vars' in each_children_item and 1 in each_children_item['solution_vars'][2:]):
            scene_info['eval_metrics']['fail_item_num'] += 1
            each_children_item["layout_function"] = ""
            continue
        each_children_item["layout_function"] = each_children_item['space_relation'] + llm_res
        each_children_item['orientation'] = each_children_item['solution_vars'][2:].index(1) * 90 + random.uniform(-2.5, 2.5)
        already_exists.append(each_children_item)
    return scene_info, already_exists

def milp_for_all_items(scene_info):
    
    children_items = scene_info['children_scene_or_objects']
    not_placed_items = [each for each in children_items if each['layout_function'] == '']

    vars_dict = {}
    already_exists = []
    hard_constraints_list = []
    soft_constraints_list = []
    for each in not_placed_items:
        # x, y, rotate_0, rotate_90, rotate_180, rotate_270 Counterclockwise
        vars_dict[each['scene_or_object_name']] = [cp.Variable(), cp.Variable(), cp.Variable(boolean=True), cp.Variable(boolean=True), cp.Variable(boolean=True), cp.Variable(boolean=True)]
        hard_constraints, soft_constraints = generate_all_constraints(each['init_loc'], each, scene_info, already_exists, vars_dict[each['scene_or_object_name']], 1.0, 0)
        hard_constraints_list.extend(hard_constraints)
        soft_constraints_list.extend(soft_constraints)
        each['solution_vars'] = vars_dict[each['scene_or_object_name']]
        already_exists.append(each)
    
    try:
        result = milp_solve(soft_constraints_list, hard_constraints_list)
    except Exception as e:
        print(f'milp solve error: {e}')
    
    if not result_not_valid(result):
        for name, vars in vars_dict.items():
            for idx in range(len(scene_info['children_scene_or_objects'])):
                cur_item = scene_info['children_scene_or_objects'][idx]
                if cur_item['scene_or_object_name'] != name:
                    continue
                cur_item["layout_function"] = cur_item['space_relation'] + f"({vars[0].value.item():.2f},{vars[1].value.item():.2f})"
                cur_item['solution_vars'] = [each.value.item() for each in vars]
                cur_item['orientation'] = cur_item['solution_vars'][2:].index(1) * 90 + random.uniform(-2.5, 2.5)
    return scene_info, [each for each in scene_info['children_scene_or_objects'] if each['layout_function']]
