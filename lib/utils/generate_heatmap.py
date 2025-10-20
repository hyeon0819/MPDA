from __future__ import division
import torch
import numpy as np
np.seterr(divide='ignore', invalid='ignore')

joints_prior = [[0,  1],
                [0,  3,  4,  5],
                [0,  9, 10, 11],
                [2,  6,  7,  8],
                [2, 12, 13, 14]]

def dist(a, b):  # 좌표 간 거리 구하는 공식
    if len(a) == 2:
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    elif len(a) == 3:
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2 + (a[2] - b[2])**2)
    else:
        print("wrong input")
        return
    
def compute_normal_vector(a, b, c):   # 법선 벡터 구하는 함수 (c 기준으로 a,b의 법선벡터)
    a, b, c = np.array(a), np.array(b), np.array(c)
    AB, AC = b - a, c - a
    normal_P = np.cross(AB, AC)  # 평면 P의 법선 벡터
    normal_ab = np.cross(AB, normal_P)  # 점 a,b를 지나는 평면의 법선 벡터
    return normal_ab

def enlarge_rectangle(a, b, c, d, proportion):  # 직사각형을 proportion 비율로 확장하는 함수
    a, b, c, d = np.array(a), np.array(b), np.array(c), np.array(d)
    centroid = (a + b + c + d) / 2.5
    a_enlarged = centroid + proportion * (a - centroid)
    b_enlarged = centroid + proportion * (b - centroid)
    c_enlarged = centroid + proportion * (c - centroid)
    d_enlarged = centroid + proportion * (d - centroid)    
    return a_enlarged, b_enlarged, c_enlarged, d_enlarged

# neck, hip을 중심으로 어깨 사이의 길이만큼의 폭을 가지는 직사각형 생성하는 함수 (torso 영역)
def create_torso_front(neck, hip, Lshld, Rshld):
    width = dist(Lshld, Rshld)
    neck, hip = np.array(neck), np.array(hip)
    Lshld, Rshld = np.array(Lshld), np.array(Rshld)
    vec_ab, vec_cd = hip - neck, Rshld - Lshld

    vec_perpendicular = np.cross(vec_ab, vec_cd)     # 선분 ab와 선분 cd의 외적을 구해서 수직 벡터 얻음
    vec_width = np.cross(vec_ab, vec_perpendicular)  # 선분 ab와 수직 벡터의 외적을 구해서 수직 벡터를 얻음

    unit_vec_width = vec_width / np.linalg.norm(vec_width) # 얻은 수직 벡터를 단위 벡터로 만들고 폭의 절반만큼 스케일링
    scaled_vec_width = unit_vec_width * width / 2

    p1 = neck + scaled_vec_width
    p2 = neck - scaled_vec_width
    p3 = hip + scaled_vec_width
    p4 = hip - scaled_vec_width

    return p1, p2, p4, p3

def create_torso_side(neck, hip, Lshld, Rshld):
    width = dist(Lshld, Rshld) / 2.5
    neck, hip = np.array(neck), np.array(hip)
    Lshld, Rshld = np.array(Lshld), np.array(Rshld)
    vec_ab, vec_cd = hip - neck, Rshld - Lshld

    vec_perpendicular = np.cross(vec_ab, vec_cd)     # 선분 ab와 선분 cd의 외적을 구해서 수직 벡터 얻음
    vec_width = np.cross(vec_ab, vec_perpendicular)  # 선분 ab와 수직 벡터의 외적을 구해서 수직 벡터를 얻음
    vec_width1 = np.cross(vec_ab, vec_width)

    unit_vec_width = vec_width1 / np.linalg.norm(vec_width1) # 얻은 수직 벡터를 단위 벡터로 만들고 폭의 절반만큼 스케일링
    scaled_vec_width = unit_vec_width * width / 2

    p1 = neck + scaled_vec_width
    p2 = neck - scaled_vec_width
    p3 = hip + scaled_vec_width
    p4 = hip - scaled_vec_width

    return p1, p2, p4, p3

# 두 관절을 중심으로 width만큼의 폭을 가지는 직사각형 생성하는 함수 (팔,다리 (limb) 영역)
# 카메라 방향으로 수직인 직사각형 생성
def create_limb(joint1, joint2, camera, width):
    joint1, joint2, camera = np.array(joint1), np.array(joint2), np.array(camera)
    vec_ab = joint2 - joint1
    vec_perpendicular = compute_normal_vector(joint1, joint2, camera)
    vec_width = np.cross(vec_ab, vec_perpendicular)  # 선분 ab와 수직 벡터의 외적을 구해서 수직 벡터를 얻음

    unit_vec_width = vec_width / np.linalg.norm(vec_width) # 얻은 수직 벡터를 단위 벡터로 만들고 폭의 절반만큼 스케일링
    scaled_vec_width = unit_vec_width * width / 2

    p1 = joint1 + scaled_vec_width
    p2 = joint1 - scaled_vec_width
    p3 = joint2 + scaled_vec_width
    p4 = joint2 - scaled_vec_width

    return p1, p2, p4, p3

# 양쪽 귀 관절을 중심으로 radius만큼의 폭을 가지는 num_points 의 꼭지점을 가지는 다각형을 생성하는 함수 (head)
# 카메라 방향으로 수직인 다각형 생성
def create_head(Lear, Rear, camera, radius=100, num_points=9):
    Lear, Rear, camera = np.array(Lear), np.array(Rear), np.array(camera)
    center = (Lear + Rear) / 2
    AB = center - camera

    if AB[0] == 0 and AB[1] == 0:
        V = np.cross(AB, [1, 0, 0])
    else:
        V = np.cross(AB, [0, 0, 1])

    # 수직 벡터 정규화
    V = V / np.linalg.norm(V)
    # 수직 벡터 2개의 외적을 사용하여 원의 평면에 있는 다른 법선 벡터 생성
    U = np.cross(AB, V)
    U = U / np.linalg.norm(U)
    # 원 주변의 점 계산
    theta = np.linspace(0, 2 * np.pi, num_points)
    points = center + radius * (np.outer(np.cos(theta), U) + np.outer(np.sin(theta), V))
    
    return points

# 모든 신체 평면 모델링하여 반환
def modeling_body_plane(pose, camera):
    body_planes = []
    body_planes.append(create_torso_front(pose[0], pose[2], pose[3], pose[9]))
    body_planes.append(create_torso_side(pose[0], pose[2], pose[3], pose[9]))
    body_planes.append(create_limb(pose[9], pose[10], camera, 80))
    body_planes.append(create_limb(pose[3], pose[4], camera, 80))
    body_planes.append(create_limb(pose[10], pose[11], camera, 60))
    body_planes.append(create_limb(pose[4], pose[5], camera, 60))
    body_planes.append(create_limb(pose[12], pose[13], camera, 120))
    body_planes.append(create_limb(pose[6], pose[7], camera, 120))
    body_planes.append(create_limb(pose[13], pose[14], camera, 100))
    body_planes.append(create_limb(pose[7], pose[8], camera, 100))
    body_planes.append(create_head(pose[16], pose[18], camera, radius=100))
    
    return body_planes

def modeling_body_plane_coco(pose, camera):
    body_planes = []
    body_planes.append(create_torso_front((pose[5]+pose[6])/2, (pose[11]+pose[12])/2, pose[5], pose[6]))
    body_planes.append(create_torso_side((pose[5]+pose[6])/2, (pose[11]+pose[12])/2, pose[5], pose[6]))
    body_planes.append(create_limb(pose[5], pose[7], camera, 80))
    body_planes.append(create_limb(pose[6], pose[8], camera, 80))
    body_planes.append(create_limb(pose[7], pose[9], camera, 60))
    body_planes.append(create_limb(pose[8], pose[10], camera, 60))
    body_planes.append(create_limb(pose[11], pose[13], camera, 120))
    body_planes.append(create_limb(pose[12], pose[14], camera, 120))
    body_planes.append(create_limb(pose[13], pose[15], camera, 100))
    body_planes.append(create_limb(pose[14], pose[16], camera, 100))
    body_planes.append(create_head(pose[4], pose[3], camera, radius=100))
    
    return body_planes


def find_plane(points):
    normal_vector = np.cross(points[1] - points[0], points[2] - points[0])
    D = -np.dot(normal_vector, points[0])
    return normal_vector, D

def is_point_inside_polygon(point, polygon_points):
    direction = np.cross(polygon_points[1] - polygon_points[0], polygon_points[2] - polygon_points[1])
    num_points = len(polygon_points)
    for i in range(num_points):
        if np.dot(direction, np.cross(polygon_points[(i + 1) % num_points] - polygon_points[i], point - polygon_points[i])) < 0:
            return False
    return True

# camera를 기준으로 해당 joint가 다각형 평면에 대해 occlusion 되었는지 여부를 판단하는 함수
def check_occlusion(target_joint, camera, polygon_points):
    plane_points = polygon_points[:3]
    target_joint = np.array(target_joint)
    camera = np.array(camera)
    polygon_points = [np.array(x) for x in polygon_points]

    normal_vector, D = find_plane(plane_points)

    # 선분과 평면의 교차 찾기
    t = (-D - np.dot(normal_vector, target_joint)) / np.dot(normal_vector, camera - target_joint)
    intersection = target_joint + t * (camera - target_joint)

    # 교차점이 다각형 내부에 있는지 확인
    if 0 <= t <= 1:
        if is_point_inside_polygon(intersection, polygon_points):
            return True
    return False

# joints_occ를 반환하는 함수 (가려지면 True, 아니면 False)
def check_occlusion_camera(target_poses, camera):
    num_joint = 15
    limb_joint_ids = [4,5,7,8, 10,11,13,14]
    bodies_planes = []
    for person in target_poses:    
        bodies_planes.append(modeling_body_plane(person, camera))

    joint_occlusions = []
    for person_id, person in enumerate(target_poses):
        joint_occlusion = np.full((num_joint,2), False)
        if np.any(person != 0):    
            for body_planes_id, body_planes in enumerate(bodies_planes):
                # Self occlusion에 대한 처리 (torso plane이 limb를 가릴 경우)
                if person_id == body_planes_id:
                    for torso_plane in body_planes[:2]:
                        for limb_joint_id in limb_joint_ids:
                            if check_occlusion(person[limb_joint_id], camera, torso_plane):
                                joint_occlusion[limb_joint_id] = [True, True]
                # Multi-person occlusion에 대한 처리
                else:
                    for body_plane in body_planes:
                        for target_id, target_joint in enumerate(person[:num_joint]):
                            if check_occlusion(target_joint, camera, body_plane):
                                joint_occlusion[target_id] = [True, True]
        joint_occlusions.append(joint_occlusion)
    
    return np.array(joint_occlusions)

# joints_occ를 반환하는 함수 (가려지면 True, 아니면 False)
def check_occlusion_camera_coco(target_poses, camera):
    num_joint = 17
    limb_joint_ids = [8,10,14,16, 7,9,13,15]
    bodies_planes = []
    for person in target_poses:    
        bodies_planes.append(modeling_body_plane_coco(person, camera))

    joint_occlusions = []
    for person_id, person in enumerate(target_poses):
        joint_occlusion = np.full((num_joint,2), False)
        if np.any(person != 0):    
            for body_planes_id, body_planes in enumerate(bodies_planes):
                # Self occlusion에 대한 처리 (torso plane이 limb를 가릴 경우)
                if person_id == body_planes_id:
                    for torso_plane in body_planes[:2]:
                        for limb_joint_id in limb_joint_ids:
                            if check_occlusion(person[limb_joint_id], camera, torso_plane):
                                joint_occlusion[limb_joint_id] = [True, True]
                # Multi-person occlusion에 대한 처리
                else:
                    for body_plane in body_planes:
                        for target_id, target_joint in enumerate(person[:num_joint]):
                            if check_occlusion(target_joint, camera, body_plane):
                                joint_occlusion[target_id] = [True, True]
        joint_occlusions.append(joint_occlusion)
    
    return np.array(joint_occlusions)

# joints freedom prior (joints_prior)을 기반으로 occlusion의 정도 (occlusion level)를 반환 
def occlusion_level(joints_occ, dataset='panoptic'):
    # prior (standard deviation of each joint's 3D position) 고려
    if dataset == 'shelf' or dataset == 'campus':
        joints_prior = [[0, 1, 3],
                        [0, 2, 4],                        
                        [5,  7,   9],
                        [6,  8,  10],
                        [11, 13, 15],
                        [12, 14, 16]]
        joints_prior_std = {0:82, 1:125, 2:125, 3:89, 4:89,
                            5:36, 6:36, 7:96, 8:96, 9:129, 10:129,
                            11:18, 12:18, 13:221, 14:221, 15:81, 16:81}
    else:
        joints_prior = [[0,  1],
                [0,  3,  4,  5],
                [0,  9, 10, 11],
                [2,  6,  7,  8],
                [2, 12, 13, 14]]
        joints_prior_std = {0:20, 1:78, 2:20, 3:32, 4:88, 5:93,
                            6:15, 7:213, 8:76, 9:32, 10:88, 11:93,
                            12:15, 13:213, 14:76}
    shape = joints_occ.shape
    joints_occ_level = np.zeros(shape)
    for joint_occ_idx, joint_occ in enumerate(joints_occ):
        if np.any(joint_occ):
            # root joint 부터 occlusion 여부 탐색
            for i in joints_prior:
                occ_level = 0
                for jp in i:
                    if joint_occ[jp][0]:
                        occ_level += joints_prior_std[jp]
                        joints_occ_level[joint_occ_idx][jp] = occ_level
                    else:
                        occ_level = 0
            # 거꾸로 탐색
            for i in joints_prior:
                occ_level = 1000
                for jp in reversed(i):
                    if not joint_occ[jp][0]:
                        occ_level = 0
                    else:
                        occ_level += joints_prior_std[jp]
                        if joints_occ_level[joint_occ_idx][jp][0] > occ_level:
                            joints_occ_level[joint_occ_idx][jp] = occ_level
                                
    return joints_occ_level