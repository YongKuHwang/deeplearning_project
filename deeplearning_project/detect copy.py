#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import argparse #명령행 인자를 파싱하기 위한 ARGPARSE 모듈을 임포트 합니다.
import os #운영 체제 관련 작업을 수행하기 위한 OS 모듈을 임포트 합니다.
import sys #시스템 관련 작업을 수행하기 위한 SYS 모듈을 임포트 합니다.
import os.path as osp #os.path 모듈을 osp별칭으로 임포트 합니다.

import torch #pytorch 라이브러리를 임포트 합니다.
import numpy as np
import cv2
import sys
from utils import ARUCO_DICT
import argparse
import time

ROOT = os.getcwd() #ROOT변수에 현재 작업 디렉토리의 경로를 저장합니다.
if str(ROOT) not in sys.path: #현재 작업 디렉토리 경로가 sys.path에 없다면 다음줄을 실행 합니다.
    sys.path.append(str(ROOT)) #현재 작업 디렉토리 경로를 sys.path에 추가 합니다.

from yolov6.utils.events import LOGGER #YOLOv6라이브러리 에서LOGGER를 가져옵니다.

import os
import cv2
import math
import torch
import numpy as np
import os.path as osp

from tqdm import tqdm

from yolov6.utils.events import LOGGER, load_yaml
from yolov6.layers.common import DetectBackend
from yolov6.data.data_augment import letterbox
from yolov6.data.datasets import LoadData
from yolov6.utils.nms import non_max_suppression

def pose_esitmation(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients):#함수는 입력된 프레임에서 아루코 마커를 검출하고 해당 마커의 포즈(위치와 자세)를 추정하는 함수 입니다.

    '''
    frame - Frame from the video stream 웹캠에서 받아온 영상 프레임
    matrix_coefficients - Intrinsic matrix of the calibrated camera 카메라의 내부 파라미터 행렬
    aruco_dict_type: 사용할 아루코 마커 사전의 종류
    distortion_coefficients - Distortion coefficients associated with your camera 왜곡계수

    return:-
    frame - The frame with the axis drawn on it
    '''
    value = None
    corners = None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)#입력 프레임을 흑백 이미지로 변환 합니다.
    cv2.aruco_dict = cv2.aruco.Dictionary_get(aruco_dict_type)#사용할 아루코마커 사전을 설정합니다. 별도의 지정이 없으면 DICT_ARUCO_ORIGINAL이 기본값으로 설정되어 있으므로 별도로 설정하지 않아도 된다.
    #만약 특별하게 다른 딕트를 이용하려면 aruco_dict_type = cv2.aruco.DICT_6X6_250 이런식으로 넣어주면 된다.
    parameters = cv2.aruco.DetectorParameters_create() #아루코마커 검출을 위한 파라미터 설정을 생성합니다.

    corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, cv2.aruco_dict, parameters=parameters) #아루코마커를 검출하고 마커의 코너 좌표(corners),id및 거부된 이미지 포인트를 반환합니다.
    print(f"ids : {ids}")
        # If markers are detected
    if len(corners) > 0: #마커가 검출되었는지 확인합니다.
        for i in range(0, len(ids)):# 검출된 각 마커에 대한 반복문 입니다. 모든 마커에 대해 아래 동작을 수행합니다.
            # Estimate pose of each marker and return the values rvec and tvec---(different from those of camera coefficients)
            rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.03, matrix_coefficients,#여기서 0.03은 한변의 길이를 나타냅니다.
                                                                       distortion_coefficients) #이함수를 사용하여 각 마커의 포즈(회전 벡터 rvec 및 변환 벡터 tvec)
            #이 함수는 검출된 마커의 코너좌표를 corners에 마커의 id를 ids에 거부된 이미지 포인트를 rejected_img_points에 반환합니다. 마커가 검출되지 않으면 corners, ids rejected_img_points는 빈리스트로 반환됩니다.
            # 회전 벡터를 회전 행렬로 변환
            r_matrix, _ = cv2.Rodrigues(rvec)
            # 마커의 실제 세계 좌표 계산
            real_world_coordinates3d = -np.matmul(np.linalg.inv(r_matrix), tvec[0, 0].reshape(-1, 1))
            image_coordinates2d, _ = cv2.projectPoints(real_world_coordinates3d, rvec, tvec, matrix_coefficients, distortion_coefficients)
            # print(f"image_coordinates2d{image_coordinates2d}")
            # print(f"real world coordinates3d: {real_world_coordinates3d}")
            # print(f"rotation vector : {rvec}")
            # print(f"translation vector : {tvec}\n")
            # print(markerPoints)
            value = round(tvec[0, 0, 2] * 100, 1) + 6
            print(value)


            # Draw a square around the markers
            # cv2.aruco.drawDetectedMarkers(frame, corners) #함수를 사용하여 입력 프레임에 검출된 아루코 마커 주위에 사각형을 그립니다. 이렇게 그려진 프레임은 마커가 감지된 것을 시각적으로 보여줍니다.

    return value, corners

def get_args_parser(add_help=True): #add_help인자를 기본값으로 받는 get_args_parser 함수를 정의합니다.
    parser = argparse.ArgumentParser(description='YOLOv6 PyTorch Inference.', add_help=add_help) #argparse를 사용하여 명령행 인자 파서를 생성하고 설명과 도움말 옵션을 추가 합니다.
    parser.add_argument('--weights', type=str, default='best_ckpt.pt', help='model path(s) for inference.')#모델 가중치 경로를 지정하는 명령행 인자를 추가 합니다.
    parser.add_argument('--source', type=str, default='data/images', help='the source path, e.g. image-file/dir.')#입력 소스 경로를 지정하는 명령행 인자를 추가합니다. 사진의 경우
    parser.add_argument('--webcam', action='store_true', default=True, help='whether to use webcam.') #웹캠 사용 여부를 나타내는 명령행 인자를 추가 합니다. action= 'store_true'의 뜻을 --webcam이라 입력하면 true 입력하지 않으면 False를 의미합니다. 
    parser.add_argument('--webcam-addr', type=str, default='/dev/video1', help='the web camera address, local camera or rtsp address.')#웹 캠 주소를 지정ㅇ하는 명령행 인자를 추가합니다.
    parser.add_argument('--yaml', type=str, default='data/dataset.yaml', help='data yaml file.') #데이터 yaml파일 경로를 지정하는 명령행 인자를 추가합니다.
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='the image-size(h,w) in inference size.') #이미지 크기를 지정하는 명령행 인자를 추가합니다.
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold for inference.')#신뢰도 임계값을 지정하는 명령행 인자를 추가합니다.
    parser.add_argument('--iou-thres', type=float, default=0.6, help='NMS IoU threshold for inference.')# nms iou임계값을 지정하는 명령행 인자를 추가합니다. 임계값보다 높은 경계상자중에 하나를 선택하고 나머지 겹치는 상자를 제거한다.
    parser.add_argument('--max-det', type=int, default=1000, help='maximal inferences per image.')#이미지당 최대 추론 개수를 지정하는 명령행 인자를 추가합니다.
    parser.add_argument('--device', default='0', help='device to run our model i.e. 0 or 0,1,2,3 or cpu.') #모델을 실행할 디바이스를 지정하는 명령행 인자를 추가합니다. 0은 gpu를 사용한다는 뜻
    parser.add_argument('--save-txt', action='store_true', default=False, help='save results to *.txt.') #결과를 텍스트 파일에 저장할지 여부를 나타내는 명령인자
    parser.add_argument('--save-img', action='store_true', default=True, help='do not save visuallized inference results.')#시각화된 결과를 저장할지 여부를 나타내는 명령행 인자를 추가합니다.
    parser.add_argument('--save-dir', type=str, default='/home/taen/dev_ws/YOLOv6/', help='directory to save predictions in. See --save-txt.') #예측 결과를 저장할 디렉토리 결로를 설정합니다. --save-txt 를 사용하여 텍스트 파일에 결과를 저장할 경우 텍스트 파일이 저장될 디렉토리 경로가 됩니다.
    parser.add_argument('--view-img', action='store_true', default=True, help='show inference results')#추론 결과를 화면에 표시할지 여부를 설정합니다.
    parser.add_argument('--classes', nargs='+', type=int, default=None, help='filter by classes, e.g. --classes 0, or --classes 0 2 3.') #특정 클래스 또는 클래스들로 필터링 할 지 여부를 설정합니다. 예를들어 --classes 0 또는 --classes 0 2 3 과 같이 클래스 번호를 지정할 수 있습니다.
    parser.add_argument('--agnostic-nms', action='store_true', default=False, help='class-agnostic NMS.') #클래스에 관계없이 nms(비최대 억제)를 수행할지 여부를 설정합니다. - 겹치는거 날리기
    parser.add_argument('--project', default='runs/inference', help='save inference results to project/name.')# 추론 결과를 저장할 프로젝트 경로를 설정합니다.
    parser.add_argument('--name', default='exp', help='save inference results to project/name.') #추론 결과를 저장할 프로젝트 경로에 추가적으로 이름을 지정합니다.
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels.')#시각화된 추론 결과 이미지에서 라벨을 숨길지 여부를 설정 합니다. 만약 이 옵션을 사용하면 라벨이 이미지에 표시되지 않습니다.
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences.')#시각화된 추론 결과 이미지에서 신뢰도(확률)를 숨길지 여부를 설정합니다. 이 옵션을 사용하면 객체의 확률정보가 이미지에 표시되지 않습니다.
    parser.add_argument('--half', action='store_true', default=False, help='whether to use FP16 half-precision inference.')#fp반정밀도(inference)를 사용할지 여부를 설정합니다. 이 옵션을 사용하면 모델 추론이 fp16 반정밀도에서 수행됩니다. fp16은 부동소수점 표현의 한 형태로, 모델의 메모리 사용량을 줄이고 속도를 높일 수 있습니다.

    args = parser.parse_args() #parser.parse_args() 함수는 명령행 인자를 파싱하고 파싱된 결과를 args변수에 저장 합니다. 이변수는 프로그램 내에서 명령행 인자의 값을 사용하기 위해 활용됩니다.
    LOGGER.info(args)#LOGGER는 로깅 기능을 수행하는 객체로, 이전에 정의된 로깅 설정에 따라 정보 메시지를 출력합니다. 예를 들어, 이 메세지는 프로그램 실행 중에 어떤 인자들이 설정되었는지 또는ㄴ 어떤 설정이 적용되었는지를 기록하기 위해 사용됩니다.
    return args #이 부분은 함수 get_args_parser의 반환 값을로 파싱된 ㅁ명령행 인자들을 반환합니다. 이렇게 하면 나중에ㅔ 이 함수를 호 출한 곳에서 파싱된 인자들을 사용할 수 있습니다.


@staticmethod # 이줄은 Python에서 정적 메서드를 정의하기 위한 데코레이터 입니다. 정적 메서드는 클래스의 인스턴스 없이 직접 클래스 이름으로 호출할 수 있는 메서드 입니다.
def generate_colors(i, bgr=False):#generate_colors함수를 정의합니다. 이 함수는 두개의 매개변수 i 와 bgr을 받습니다. i 색상 팔레트에서 사용할 색상의 인덱스를 나타내는 정수 매개변수 입니다. bgr 색상을 bgr 형식으로 반환할지 여부를 나타내는 부울 매개변수
    hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
            '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7') #hex변수에는 20개의 16진수 색상 코드가 포함된 튜플이 할당 됩니다. 이 색상 코드들은 색상 팔레트를 정의 합니다.
    palette = [] #빈 리스트 palette를 생성합니다. 이 리스트는 나중에 생성한 rgb색상들을 저장할 목적으로 사용됩니다.
    for iter in hex:#튜플의 각 항목에 대해 받복합니다.
        h = '#' + iter #현재 순회 중인 16진수 색상 코드 iter 앞에 # 문자를 추가하여 색상 코드를 완성합니다. 
        palette.append(tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))) #현재 순회중인 16진수 문자열에서 각각 r g b값을 나타내는 두 글자씩을 슬라이싱 합니다.
    num = len(palette)#생성된 팔레트에 있는 색상의 개수를 계산하여 num변수에 저장합니다.
    color = palette[int(i) % num] #i 값을 이용하여 팔레트에서 선택할 색상을 결정합니다. %연산자를 사용하여 i값을 팔레트 색상수로 나누어 나머지를 구하고 행당 나머지를 인덱스로 사용하여 팔레트에서 색상을 선택합니다.
    return (color[2], color[1], color[0]) if bgr else color #bgr 매개변수가 True인 경우 (B, G, R) 형식의 튜플로 색상을 반환하고, False인 경우 RGB 튜플로 색상을 반환합니다. 반환된 색상은 선택한 색상 팔레트에서의 색상입니다.이 함수는 주어진 인덱스 i에 해당하는 색상을 팔레트에서 선택하고, 선택한 색상을 RGB 또는 BGR 형식으로 반환하는 데 사용됩니다.

@torch.no_grad()
def run(weights,
        source,
        webcam,
        webcam_addr,
        yaml,
        img_size,
        conf_thres,
        iou_thres,
        max_det,
        device,
        save_txt,
        save_img,
        save_dir,
        view_img,
        classes,
        agnostic_nms,
        project,
        name,
        hide_labels,
        hide_conf,
        half
        ):
    """ Inference process, supporting inference on one image file or directory which containing images.
    Args:
        weights: The path of model.pt, e.g. yolov6s.pt
        source: Source path, supporting image files or dirs containing images.
        yaml: Data yaml file, .
        img_size: Inference image-size, e.g. 640
        conf_thres: Confidence threshold in inference, e.g. 0.25
        iou_thres: NMS IOU threshold in inference, e.g. 0.45
        max_det: Maximal detections per image, e.g. 1000
        device: Cuda device, e.e. 0, or 0,1,2,3 or cpu
        save_txt: Save results to *.txt
        not_save_img: Do not save visualized inference results
        classes: Filter by class: --class 0, or --class 0 2 3
        agnostic_nms: Class-agnostic NMS
        project: Save results to project/name
        name: Save results to project/name, e.g. 'exp'
        line_thickness: Bounding box thickness (pixels), e.g. 3
        hide_labels: Hide labels, e.g. False
        hide_conf: Hide confidences
        half: Use FP16 half-precision inference, e.g. False
    """



    #----------------------- aruco marker----------------------------------#

    ap = argparse.ArgumentParser()#argparse 라이브러리를 사용하여 명령줄 인수를 파싱하기 위한 ArgumentParser 객체를 생성합니다. ap는 이후에 명령줄 인수를 정의하고 처리하는데 사용됩니다.
    # ap.add_argument("-k", "--K_Matrix", default='calibration_matrix.npy', help="Path to calibration matrix (numpy file)")#이 옵션은 카메라 캘리브레이션 행렬의 경로를 받습니다. required=True는 이 옵션이 반드시 지정되어야 함을 나타냅니다.
    # ap.add_argument("-d", "--D_Coeff", default='distortion_coefficients.npy', help="Path to distortion coefficients (numpy file)")#이 옵션은 왜곡 계수(D계수)의 경로를 받습니다. 
    ap.add_argument("-t", "--type", type=str, default="DICT_5X5_100", help="Type of ArUCo tag to detect")#이 옵션은 아루코태그의 타입을 받습니다. type=str로 설정하여 문자열 형식의 값을 받도록 지정합니다. default="DICT_ARUCO_ORIGINAL"로 기본값을 설정합니다. 즉 이옵션이 지정되지 않을 겨우 DICT_ARUCO_ORIGINAL로 사용됩니다.
    args_pose = vars(ap.parse_args())#argparse를 사용하여 파싱한 명령줄 인수를 처리하고 결과를 딕셔너리 형태로 args변수에 저장합ㄴ디ㅏ. 딕셔너리의 키는 옵션의 이름이고 값은 해당 옵션에 단달된 값입니다.

    
    # if ARUCO_DICT.get(args_pose["type"], None) is None: #사용자가 명령줄 인수로 지정한 아루코 태그 유형(args["type"])이 지원되지 않을 경우를 확인 합니다.
    #     print(f"ArUCo tag type '{args['type']}' is not supported")#ARUCO_DICT딕셔너리에서 해당 태그 유형을 찾고 만약 찾을 수 없다면 NONE을 반환합니다. 따라서 반환된 값이 NONE인 경우 지원되지 않는 태그유형임을 나타내고 에러 메시지를 출력합니다.
    #     sys.exit(0)#sys.exit(0)을 사용하여 프로그램을 종료합ㄴ디ㅏ.

    aruco_dict_type = ARUCO_DICT[args_pose["type"]] #지원되는 아루코 태그 유형을 확인한 후, 해당 유형에 대한 딕셔너리 값을 가져와서 aruco_dict_type변수에 저장합니다.
    calibration_matrix_path = 'calibration_matrix.npy' #args["K_Matrix"] 카메라 캘리브레이션을 위한 행렬(K행렬)의 경로를 지정합니다.
    distortion_coefficients_path ='distortion_coefficients.npy' #args["D_Coeff"]카메라 왜곡 계수(D계수)의 경로를 지정합니다. 이역시 하드코딩 되어 있으며 args["D_Coeff"]옵션을 통해 받아 올 수 도 있습니다.
    
    k = np.load(calibration_matrix_path)#지정된 경로에서 카메라 캘리브레이션 행렬(K행렬)을 로드하고 k변수에 저장합니다. 이 행렬은 카메라의 내부 파라미터를 포함합니다.
    d = np.load(distortion_coefficients_path)#지정된 경로에서 왜곡계수를 로드하고 d 변수에 저장합니다. 이계수는 카메라 렌즈왜곡을 보정하는데 사용됩니다.

    

    #-----------------------yolo detecting----------------------------------#

    if save_dir is None:#save_dir 변수가 None일 경우를 확인합니다. save_dir은 결과 이미지 및 텍스트 파일이 저장될 디렉트로를 나타냅니다. - 만약 save_dir이 없다면
        save_dir = osp.join(project, name) #save_dir를 프로젝트및 이름을 결합한 디렉토리 경로로 설정합니다. osp.join 함수를 사용하여 경로를 결합합니다.
        save_txt_path = osp.join(save_dir, 'labels') # save_txt_path를 save_dir 디렉토리에 'labels' 서브 디렉토리로 설정 합니다.이 디렉토리는 텍스트 파일이 저장될 위치를 나타냅니다.
    else: #'save_dir가 none이 아닌경우 실행할 코드 블록을 지정합니다.
        save_txt_path = save_dir #save_dir이 이미 설정된 경우에는 save_txt_path를 그대로 save_dir로 설정 합니다.
    if (save_img or save_txt) and not osp.exists(save_dir):#이미지및 텍스트 파일을 저장해야하고 save_dir 디렉토리가 존재하지 않는 경우에 실행합니다. 이조건은 이미지 또는 텍스트 파일을 저장해야 하며 디렉토리가 존재하지 않을때 이조건이 참이 됩니다.
        os.makedirs(save_dir)#디렉토리가 존재하지 않을때 save_dir폴더를 만듭니다.
    else:#이미지및 텍스트 파일 저장 조건이 거짓인 경우 실행됩니다. 이 경우에는 이미 존재하는 디렉토리임을 로그에 경고합니다.
        LOGGER.warning('Save directory already existed')
    if save_txt:#텍스트 파일 저장 조건을 확인합ㄴ디ㅏ. 텍스트 파일을 저장해야할 경우 실행 됩니다.
        save_txt_path = osp.join(save_dir, 'labels')
        if not osp.exists(save_txt_path):
            os.makedirs(save_txt_path)

    cuda = device != 'cpu' and torch.cuda.is_available()
    device = torch.device(f'cuda:{device}' if cuda else 'cpu')
    model = DetectBackend(weights, device=device)#'DetectBackend'클래스를 사용하여 모델을 생성합니다. 'weights'는 모델 가중치를 나타내며, 'device'는 모델을 실행할 디바이스를 나타냅니다.
    files = LoadData(source, webcam, webcam_addr)#데이터를 로드하는 함수'LoadData'를 사용하여 'files' 변수에 데이터를 로드합니다. 'source', 'webcom', 'webcam_addr'는 함수에 전달 되는 매개변수입니다.
    stride = model.stride#'model'객체에서 stride '속성을 추출하여 stride변수에 저장 합니다. 'stride'는 모델의 스트라이드를 나타내며, 이미지 처리에 사용됩니다.
    img_size = check_img_size(img_size, s=stride)#이미지 크기를 확인하는 함수 'check_img_size를 호출하여 img_size 변수에 이미지 크기를 설정합니다. img_size는 스트라이드를 고려한 이미지 크기를 나타냅니다.
    class_names = load_yaml(yaml)['names']#YAML 파일에서 클래스의 이름을 로드하여 class_names 변수에 저장합니다. yaml은 YAML파일을 가리키며 파일에서 클래스 이름 정보를 추출 합니다.

    ''' Model Inference and results visualization '''
    windows = [] #비어있는 리스트 windows를 생성합니다. 이 리스트는 이후 결과 이미지를 저장 할 때 사용됩니다.
    for img_src, img_path, vid_cap in tqdm(files):#files 에서 이미지 소스 이미지경로 비디오 캡처 객체를 하나씩 반복하면서 이미지에 대한 추론을 수행합니다.
        
        value, corners = pose_esitmation(img_src, aruco_dict_type, k, d)

        img = letterbox(img_src, img_size, stride=stride)[0]#letterbox함수를 사용하여 이미지를 크기 조정합ㄴ디ㅏ. 이함수는 이미지를 원하는 크기와 스트라이드에 맞게 조정합ㄴ디ㅏ.
        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB 이미지의 차원 순서를 변경하여 HWV(높이, 너비, 채널)에서 CHW(채널, 높이, 너비)변경하고, BGR 색상순서를 RGB로 변경합니다.
        img = torch.from_numpy(np.ascontiguousarray(img))#넘파이 배열에서 파이토치 텐서로 이미지를 변환합니다.
        img = img.half() if half else img.float()  # uint8 to fp16/32  half변수가 true이면 이미지를 16비트 부동 소수점 형식으로 변환하고, 그렇지 않으면 32비트 부동 소수점으로 유지합니다. 이것은 모델의 데이터 타입과 일치 시키기 위해 수행됩니다.
        img /= 255  # 0 - 255 to 0.0 - 1.0 이미지의 픽셀 값 정규화 
        img = img.to(device)# 이미지를 목표 디바이스로 이동 시킨다. 이로써 모델을 사용하여 추론을 수행할 디바이스로 이미지가 전송됨
        if len(img.shape) == 3: #이미지의 차원을 확인하고 3차원 이미지인 경우에는 배치 차원을 추가하여 4차원 텐서로 만듭니다. 이것은 모델에 입력으로 공급하기 위해 필요한 작업입니다.
            img = img[None]
            # expand for batch dim
        pred_results = model(img)#모델을 사용하여 이미지에 대한 추론을 수행하고 추론 결과를 pred_results 변수에 저장합니다.
        det = non_max_suppression(pred_results, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)[0]#non_max_suppression 함수를 사용하여 예측 결과 에서 비최대 억제를 수행하고 해당결과를 det변수에 저장 합니다. 이때 매개변수를 사용하여 억제 작업을 설정 합니다. 결과는 주로 바운딩 박스에 관련된 것

        if webcam:#웹캠 변수가 트루인 경우 해당 조건문 실행 
            save_path = osp.join(save_dir, webcam_addr)#웹캠 모드인 경우 이미지의 저장경로 save_path를 save_dir과 webcam_addr을 결합하여 만듭니다.
            txt_path = osp.join(save_dir, webcam_addr)#txt_path도 save_dir과 webcam_addr을 결합하여 만듭니다.
        else:#웹캠 변수가 펄스인 경우에 해당 조건문 블록을 실행 합니다. 이것은 웹캠 모드가 아닌경우 정적 이미지를 처리하는 경우입니다.
            # Create output files in nested dirs that mirrors the structure of the images' dirs
            rel_path = osp.relpath(osp.dirname(img_path), osp.dirname(source))#입력 이미지의 디렉토리 경로에서 source 디렉토리까지의 상대 경로를 계산합니다. 이로써 입력 이미지의 디렉토리 구조를 유지하면서 결과를 저장합니다.
            save_path = osp.join(save_dir, rel_path, osp.basename(img_path))  # im.jpg 이미지를 정적 이미지 처리 모드로 처리하는 경우 이미지의 저장 경로 세이브 패스를 세이브디렉토리 상대경로 그리고 입력 이미지의 파일이름을 osp.basename(img_path)을 결합하여 만듭니다. 이로써 입력 이미지의 원래 디렉토리 구조를 따르면서 이미지를 저장합니다.
            txt_path = osp.join(save_dir, rel_path, 'labels', osp.splitext(osp.basename(img_path))[0])#레이블 파일을 저장하는 경로 txt_path는 세이브디렉토리 상대경로 라벨즈 그리고 입력 이미지의 파일 이름에서 확장자를 제외한 부분으로 결합하여 만듭니다. 이로써 레이블 파일도 입력 이미지의 구조를 따라 저장됩니다.
            os.makedirs(osp.join(save_dir, rel_path), exist_ok=True)# 결과를 저장할 디렉토리 경로를 생성합니다. exsit_ok=True를 사용하여 이미 디렉토리가 존재하는 경우에도 오류를 발생시키지 않도록 합니다.

        gn = torch.tensor(img_src.shape)[[1, 0, 1, 0]]  # normalization gain whwh 파이텐서로서 결과 저장
        img_ori = img_src.copy()#변수에 img_src의 복사본을 저장합니다. 이부분은 이미지를 변경하지 않고 원본 이미지를 유지하기 위해 사용됩니다. 원본 이미지와 복사본은 나중에 필요한 경우에 참조할 수 있도록 합니다.

        if len(det): #'det' 변수에 객체 탐지 결과가 있는 경우에만 이부분의 코드를 실행합니다. det 변수에는 객체 탐지 결과가 포함되어 있으먀 이격과는 이후에 처리 됩니다.
            '''Rescale the output to the original image shape''' #주석으로 객체 탐지 결과를 원본 이미지 크기로 다시 조정하는 부분을 설명 하고 있습니다.
            ratio = min(img.shape[2:][0] / img_src.shape[0], img.shape[2:][1] / img_src.shape[1]) #이미지 크기를 조정하기 위한 비율을 계산합니다. ratio는 원본이미지 크기와 탐지된 이미지 크기간의 비율중 더작은 값을 선택합니다.
            padding = (img.shape[2:][1] - img_src.shape[1] * ratio) / 2, (img.shape[2:][0] - img_src.shape[0] * ratio) / 2

            det[:, :4][:, [0, 2]] -= padding[0]
            det[:, :4][:, [1, 3]] -= padding[1]
            det[:, :4][:, :4] /= ratio

            det[:, :4][:, 0].clamp_(0, img_src.shape[1])  # x1
            det[:, :4][:, 1].clamp_(0, img_src.shape[0])  # y1
            det[:, :4][:, 2].clamp_(0, img_src.shape[1])  # x2
            det[:, :4][:, 3].clamp_(0, img_src.shape[0])  # y2
            coor = []    
            for *xyxy, conf, cls in reversed(det):
                if save_txt:  # Write to file
                    x = torch.tensor(xyxy).view(1, 4)
                    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
                    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
                    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
                    y[:, 2] = x[:, 2] - x[:, 0]  # width
                    y[:, 3] = x[:, 3] - x[:, 1]  # height

                    xywh = (y / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh, conf)
                    print()
                    with open(txt_path + '.txt', 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')
                    

                if save_img:
                    class_num = int(cls)  # integer class
                    label = None if hide_labels else (class_names[class_num] if hide_conf else f'{class_names[class_num]} {conf:.2f}')

                    # Add one xyxy box to image with label
                    lw = max(round(sum(img_ori.shape) / 2 * 0.003), 2)
                    color=generate_colors(class_num, True)
                    p1, p2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                    center_point = round((p1[0] + p2[0])/2), round((p1[1]+p2[1])/2)
                    cv2.circle(img_ori, center_point, 5,(0,255,0),2)
                    cv2.putText(img_ori,str(center_point),center_point,cv2.FONT_HERSHEY_PLAIN,2,(0,0,255))
                    cv2.rectangle(img_ori, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
                    if label:
                        tf = max(lw - 1, 1)  # font thickness
                        w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
                        outside = p1[1] - h - 3 >= 0  # label fits outside box
                        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
                        cv2.rectangle(img_ori, p1, p2, color, -1, cv2.LINE_AA)  # filled
                        cv2.putText(img_ori, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), cv2.FONT_HERSHEY_COMPLEX, lw / 3, (255, 255, 255),
                                    thickness=tf, lineType=cv2.LINE_AA)
                        print(p1,p2)

            img_src = np.asarray(img_ori)

            cv2.putText(img_src,'distance: '+str(value),(460,474), cv2.FONT_HERSHEY_COMPLEX, lw / 3, (255, 255, 255),
                                    thickness=tf, lineType=cv2.LINE_AA)
            
            cv2.aruco.drawDetectedMarkers(img_src, corners)
    
        if view_img:
            if img_path not in windows:
                windows.append(img_path)
                cv2.namedWindow(str(img_path), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                cv2.resizeWindow(str(img_path), img_src.shape[1], img_src.shape[0])
            cv2.imshow(str(img_path), img_src)
            key = cv2.waitKey(1)  # 1 millisecond
            if key == ord('q'):
                cv2.destroyAllWindows()
                break  # Exit the loop when 'q' is pressed



def check_img_size(img_size, s=32, floor=0):
    """Make sure image size is a multiple of stride s in each dimension, and return a new shape list of image."""
    if isinstance(img_size, int):  # integer i.e. img_size=640
        new_size = max(make_divisible(img_size, int(s)), floor)
    elif isinstance(img_size, list):  # list i.e. img_size=[640, 480]
        new_size = [max(make_divisible(x, int(s)), floor) for x in img_size]
    else:
        raise Exception(f"Unsupported type of img_size: {type(img_size)}")

    if new_size != img_size:
        print(f'WARNING: --img-size {img_size} must be multiple of max stride {s}, updating to {new_size}')
    return new_size if isinstance(img_size,list) else [new_size]*2

def make_divisible(x, divisor):
        # Upward revision the value x to make it evenly divisible by the divisor.
        return math.ceil(x / divisor) * divisor

def main(args):
    run(**vars(args))


if __name__ == "__main__":
    args = get_args_parser()
    main(args)