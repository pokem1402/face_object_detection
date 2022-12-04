# 운전자의 얼굴 인식을 통한 졸음 운전 탐지

## 0. Period

    2022.11.30 ~ 2022.12.05

## 1. Data

[졸음운전 예방을 위한 운전자 상태 정보 영상](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=173)

- 2022년 현재까지도 졸음 운전은 교통사고의 주요한 원인 중 하나
- [2022년 11월 고속도로 교통사고 사망자 중 90%는 졸음 운전이 원인](https://newsis.com/view/?id=NISX20221125_0002100808&cID=10401&pID=10400)
- 그래서 실시간에 가까운 졸음 운전 탐지 기술이 요구됨

## 2. Hypothesis

 (1) 운전자의 얼굴 구성요소 중에서 양 눈 및 입의 닫힘 상태를 탐지할 수 있다.
 (2) (1)에서 탐지된 구성요소를 통해 졸음 운전 여부를 파악할 수 있다.

## 3. preprocessing

### 3.1. Data Selection

 - 주어진 데이터는 약 35만 장.
 - 주어진 데이터를 모두 학습하는 것은 불가능에 가까움
 - 따라서 실제 도로 주행 데이터, 준 통제 환경, 통제 환경 데이터 중에 가장 표정 요소가 다양한 통제 환경 데이터 사용 
 - 통제 환경 데이터는 250명의 운전자에게 주어진 시나리오 대로 통제된 환경에서 촬영된 데이터로 가장 다양한 표정 데이터가 담김
 - 이러한 통제 환경의 데이터도 11만장이 넘는 방대한 데이터이기 때문에 기간 내에 학습하기 위해 순서대로 2만 2천장을 사용하여 학습함

### 3.2. Data Preprocessing
- 해당 프로젝트에서 사용한 Model인 [Yolo v7](https://github.com/WongKinYiu/yolov7)을 사용하기 위해서 image와 data annotation은 아래와 같은 형태로 데이터 파일에 존재해야함
```
preprocessed_dataset
└────train
|    └────images
|    |    └────── train1.jpg    
|    |    └────── train2.jpg
|    └────labels
|         └────── train1.txt    
|         └────── train1.txt
└────val
     └────images
     |    └────── val1.jpg    
     |    └────── val2.jpg
     └────labels
          └────── val1.txt    
          └────── val2.txt
```
- 그리고 각각의 label 파일에는 탐지할 객체마다 클래스 번호, 객체의 x 중앙값, y 중앙값, 너비, 높이 값이 한 줄로 표현되어야함. 객체는 하나 이상일 수 있음
- 이를 위해 JSON으로 주어진 label을 txt 형태로 바꾸고, 이미지 파일과 레이블 파일들을 데이터 디렉토리를 모델이 원하는 구조로 조직화함
  
## 4. Used Model : YOLO v.7

### 4.1 Why

 - 


## 5. Results

### 5.1. training summary
![confusion_matrix](Face_object_detection/yolov7_size640_epochs30_batch4/confusion_matrix.png)

## 6. Limitaition and To Do