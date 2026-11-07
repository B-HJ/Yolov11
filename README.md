YOLOv11(정식 표기: YOLO11)을 처음 공부하는 분도 이해할 수 있도록 개념, 설치, 코랩 실행, 데이터 준비, 모델 학습, 객체 탐지, 성능 평가까지 포함한 Markdown 문서로 정리했습니다.

아래 문서는 `YOLO11_초보자_학습가이드.md`라는 이름으로 저장하면 됩니다.

작성

# YOLO11 초보자 학습 가이드

> 구글 코랩(Google Colab)에서 YOLO11을 활용한 객체 탐지 및 인공지능 실습 안내서

---

# 목차

1. YOLO란 무엇인가?
2. YOLO11이란?
3. YOLO11의 주요 기능
4. YOLO11 모델 종류
5. YOLO11의 작동 원리
6. 구글 코랩 환경 설정
7. YOLO11 설치하기
8. 사전 학습 모델 불러오기
9. 이미지 객체 탐지하기
10. 동영상 객체 탐지하기
11. 나만의 데이터셋 만들기
12. YOLO11 학습하기
13. 학습 결과 확인하기
14. 모델 성능 평가하기
15. 학습된 모델 사용하기
16. 모델 내보내기
17. 자주 발생하는 오류
18. YOLO11 실습 프로젝트
19. 학습 로드맵
20. 참고 사이트

---

# 1\. YOLO란 무엇인가?

YOLO는 You Only Look Once의 약자입니다.

이미지나 동영상에 포함된 물체를 빠르게 찾아내는 실시간 객체 탐지(Object Detection) 기술입니다.

예를 들어 사진 한 장을 입력하면 다음과 같은 물체를 자동으로 찾아낼 수 있습니다.

- 사람
- 자동차
- 버스
- 자전거
- 강아지
- 고양이
- 가방
- 휴대전화

## 객체 탐지란?

객체 탐지는 이미지 안에 어떤 물체가 있는지 알아내고, 해당 물체의 위치를 사각형으로 표시하는 기술입니다.

객체 탐지 결과에는 일반적으로 다음 정보가 포함됩니다.

1. 물체의 종류(Class)
2. 물체의 위치(Bounding Box)
3. 예측 신뢰도(Confidence Score)

예를 들어 자동차를 탐지했다면 다음과 같이 결과가 나옵니다.

text

```text
객체: 자동차
신뢰도: 0.95
위치: x1, y1, x2, y2
```

신뢰도 0.95는 모델이 해당 객체를 자동차라고 판단한 점수가 95%라는 의미입니다. 반드시 실제 정확도가 95%라는 뜻은 아닙니다.

---

# 2\. YOLO11이란?

YOLO11은 Ultralytics에서 2024년 9월 27일 공개한 객체 탐지 및 컴퓨터 비전 모델입니다.

이전 YOLO 모델에 비해 모델 구조와 학습 방법을 개선하여 정확도와 효율성을 높였습니다.

## YOLO11의 특징

- 실시간 객체 탐지
- 빠른 이미지 처리
- 다양한 객체 인식
- 사용자 데이터로 모델 재학습 가능
- GPU를 활용한 학습 가능
- 이미지, 동영상 분석 가능
- 다양한 컴퓨터 비전 작업 지원

## YOLO11 활용 분야

| 분야 | 활용 사례 |
| --- | --- |
| 제조업 | 제품 불량 검사 |
| 교통 | 자동차 및 보행자 탐지 |
| 의료 영상 | 영상 내 관심 영역 분석 |
| 농업 | 작물 및 병해 탐지 |
| 보안 | 사람 및 차량 탐지 |
| 물류 | 물품 및 박스 탐지 |
| 스마트시티 | 교통량 분석 |

---

# 3\. YOLO11의 주요 기능

YOLO11은 객체 탐지 외에도 다양한 기능을 제공합니다.

| 기능 | 설명 | 모델 예시 |
| --- | --- | --- |
| Detection | 물체 위치와 종류 탐지 | yolo11n.pt |
| Segmentation | 물체의 영역을 픽셀 단위로 분리 | yolo11n-seg.pt |
| Classification | 이미지 전체의 종류 분류 | yolo11n-cls.pt |
| Pose Estimation | 사람의 관절 및 자세 추정 | yolo11n-pose.pt |
| OBB | 회전된 물체 탐지 | yolo11n-obb.pt |

## 3.1 객체 탐지(Detection)

이미지에서 물체를 찾고 사각형으로 표시합니다.

예:

- 사람 찾기
- 자동차 찾기
- 제품 찾기

## 3.2 객체 분할(Segmentation)

물체의 외곽선을 픽셀 단위로 구분합니다.

예:

- 자동차 영역 분리
- 사람 영역 분리
- 불량 제품 영역 표시

## 3.3 이미지 분류(Classification)

이미지 전체가 어떤 종류인지 분류합니다.

예:

- 정상 제품
- 불량 제품
- 고양이
- 강아지

## 3.4 자세 추정(Pose Estimation)

사람의 신체 주요 관절 위치를 추정합니다.

예:

- 운동 자세 분석
- 보행 분석
- 사람의 동작 분석

## 3.5 방향성 객체 탐지(OBB)

기울어져 있거나 회전된 물체를 회전된 사각형으로 탐지합니다.

예:

- 항공사진 속 선박
- 위성사진 속 자동차
- 공장 내 회전된 제품

---

# 4\. YOLO11 모델 종류

YOLO11은 모델 크기에 따라 여러 가지로 구분됩니다.

| 모델 | 특징 |
| --- | --- |
| YOLO11n | 가장 가벼운 모델 |
| YOLO11s | 작은 모델 |
| YOLO11m | 중간 크기 모델 |
| YOLO11l | 큰 모델 |
| YOLO11x | 가장 큰 모델 |

## 모델 선택 방법

- n: 빠른 실행과 가벼운 장치에 적합
- s: 속도와 정확도의 균형
- m: 비교적 높은 정확도가 필요한 경우
- l: 높은 성능이 필요한 경우
- x: 최대 수준의 모델 성능을 실험하는 경우

초보자는 `yolo11n.pt`부터 시작하는 것을 추천합니다.

---

# 5\. YOLO11의 작동 원리

YOLO11의 기본적인 처리 과정은 다음과 같습니다.

text

```text
이미지 입력
    ↓
이미지 전처리
    ↓
특징 추출
    ↓
객체 위치 및 종류 예측
    ↓
후처리
    ↓
객체 탐지 결과 출력
```

## 주요 용어

| 용어 | 의미 |
| --- | --- |
| Input | 입력 이미지 |
| Backbone | 이미지 특징 추출 |
| Neck | 다양한 크기의 특징 정보 결합 |
| Head | 객체 위치 및 종류 예측 |
| Bounding Box | 객체를 둘러싸는 사각형 |
| Confidence | 예측 신뢰도 |
| IoU | 두 영역이 겹치는 정도 |
| NMS | 중복 탐지 결과 제거에 사용하는 방법 |

---

# 6\. 구글 코랩 환경 설정

YOLO11은 구글 코랩에서 실습할 수 있습니다.

## 6.1 코랩 접속

[https://colab.research.google.com/](<https://colab.research.google.com/>)

1. 구글 코랩에 접속합니다.
2. 새 노트북을 만듭니다.
3. 상단 메뉴에서 런타임을 선택합니다.
4. 런타임 유형 변경을 클릭합니다.
5. GPU 사용이 가능한 경우 T4 GPU 등을 선택합니다.
6. 저장합니다.

GPU 사용 여부 확인:

python

```python
import torch

print("PyTorch 버전:", torch.__version__)
print("GPU 사용 가능:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU 이름:", torch.cuda.get_device_name(0))
```

---

# 7\. YOLO11 설치하기

코랩 코드 셀에 다음 명령어를 입력합니다.

python

```python
!pip install ultralytics
```

설치 확인:

python

```python
import ultralytics

print("Ultralytics 버전:", ultralytics.__version__)
```

YOLO11 라이브러리를 불러옵니다.

python

```python
from ultralytics import YOLO
```

---

# 8\. 사전 학습 모델 불러오기

YOLO11은 이미 학습된 모델을 제공하므로 처음부터 학습하지 않아도 객체 탐지를 실습할 수 있습니다.

python

```python
from ultralytics import YOLO

# YOLO11 사전 학습 모델 불러오기
model = YOLO("yolo11n.pt")

print("모델 불러오기 완료")
```

처음 실행할 때는 모델 가중치 파일이 자동으로 다운로드될 수 있습니다.

## 사전 학습 모델이란?

이미 대규모 데이터셋을 사용하여 학습된 모델입니다.

YOLO11 기본 Detection 모델은 COCO 데이터셋을 활용한 사전 학습 가중치를 제공합니다.

COCO 사전 학습 모델은 일반적인 80개 객체 클래스를 탐지할 수 있습니다.

---

# 9\. 이미지 객체 탐지하기

가장 기본적인 YOLO11 실습입니다.

python

```python
from ultralytics import YOLO

# 모델 불러오기
model = YOLO("yolo11n.pt")

# 이미지 객체 탐지
results = model(
    "https://ultralytics.com/images/bus.jpg"
)

# 탐지 결과 표시
results[0].show()
```

## 탐지 결과 저장하기

python

```python
results = model.predict(
    source="https://ultralytics.com/images/bus.jpg",
    save=True
)
```

결과는 일반적으로 다음 경로에 저장됩니다.

text

```text
runs/detect/predict/
```

## 신뢰도 설정하기

python

```python
results = model.predict(
    source="https://ultralytics.com/images/bus.jpg",
    conf=0.5
)
```

`conf=0.5`는 신뢰도 점수가 0.5 이상인 결과를 표시하도록 설정하는 것입니다.

---

# 10\. 동영상 객체 탐지하기

YOLO11은 동영상에서도 객체를 탐지할 수 있습니다.

python

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

results = model.predict(
    source="video.mp4",
    save=True,
    conf=0.5
)
```

동영상 파일을 코랩에 먼저 업로드해야 합니다.

웹캠이나 동영상 스트림도 환경에 맞게 연결할 수 있습니다.

---

# 11\. 나만의 데이터셋 만들기

내가 원하는 물체를 탐지하려면 직접 데이터셋을 준비하고 모델을 학습해야 합니다.

예를 들어 다음과 같은 프로젝트를 생각할 수 있습니다.

- 사과와 바나나 탐지
- 제품 불량 탐지
- 안전모 착용 여부 탐지
- 자동차 번호판 영역 탐지
- 공장 제품 종류 탐지

## 11.1 데이터셋 구성

YOLO Detection 데이터셋의 일반적인 구조는 다음과 같습니다.

text

```text
dataset/
│
├── images/
│   ├── train/
│   └── val/
│
├── labels/
│   ├── train/
│   └── val/
│
└── data.yaml
```

### 폴더 설명

| 폴더 | 설명 |
| --- | --- |
| images/train | 학습용 이미지 |
| images/val | 검증용 이미지 |
| labels/train | 학습용 정답 라벨 |
| labels/val | 검증용 정답 라벨 |
| data.yaml | 데이터셋 설정 파일 |

## 11.2 이미지 라벨링

객체 탐지를 위해서는 이미지 안에 있는 물체의 위치를 표시해야 합니다.

라벨링 도구 예시:

- CVAT
- LabelImg
- Roboflow

라벨은 일반적으로 이미지 파일과 동일한 이름의 `.txt` 파일로 저장합니다.

### YOLO 라벨 형식

text

```text
class_id x_center y_center width height
```

예:

text

```text
0 0.5 0.5 0.3 0.4
```

각 값은 이미지 크기에 대해 정규화된 값입니다.

- class\_id: 객체 클래스 번호
- x\_center: 박스 중심의 가로 좌표
- y\_center: 박스 중심의 세로 좌표
- width: 박스 너비
- height: 박스 높이

좌표와 너비, 높이는 일반적으로 0\~1 범위로 표현합니다.

---

# 12\. data.yaml 만들기

YOLO11이 데이터셋을 이해할 수 있도록 설정 파일을 만듭니다.

yaml

```yaml
path: /content/dataset

train: images/train
val: images/val

names:
  0: apple
  1: banana
  2: orange
```

위 예제는 사과, 바나나, 오렌지 세 가지 클래스를 사용하는 데이터셋입니다.

주의사항:

- `path`는 데이터셋의 기본 경로입니다.
- `train`은 학습 이미지 경로입니다.
- `val`은 검증 이미지 경로입니다.
- `names`는 클래스 번호와 이름입니다.
- 라벨의 class\_id와 names의 번호가 일치해야 합니다.

---

# 13\. YOLO11 모델 학습하기

데이터셋 준비가 완료되었다면 학습을 시작합니다.

python

```python
from ultralytics import YOLO

# 사전 학습 모델 불러오기
model = YOLO("yolo11n.pt")

# 모델 학습
results = model.train(
    data="/content/dataset/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    name="my_yolo11"
)
```

## 주요 학습 옵션

| 옵션 | 설명 |
| --- | --- |
| data | 데이터셋 설정 파일 |
| epochs | 전체 학습 반복 횟수 |
| imgsz | 입력 이미지 크기 |
| batch | 한 번에 학습하는 이미지 수 |
| name | 학습 결과 폴더 이름 |
| device | 학습 장치 설정 |
| workers | 데이터 로딩 작업자 수 |

### 초보자 추천 설정

python

```python
results = model.train(
    data="/content/dataset/data.yaml",
    epochs=50,
    imgsz=640,
    batch=8
)
```

GPU 메모리가 부족하면 batch 값을 줄여 보세요.

---

# 14\. 학습 결과 확인하기

학습이 끝나면 다음 폴더에 결과가 저장됩니다.

text

```text
runs/detect/my_yolo11/
```

대표적인 결과 파일:

| 파일 | 설명 |
| --- | --- |
| weights/best.pt | 검증 성능이 가장 좋았던 모델 |
| weights/last.pt | 마지막 학습 시점의 모델 |
| results.csv | 학습 과정별 지표 |
| results.png | 학습 결과 그래프 |
| confusion\_matrix.png | 혼동 행렬 |

일반적으로 실제 예측에는 `best.pt`를 우선 사용합니다.

---

# 15\. 모델 성능 평가하기

학습한 모델의 성능을 확인할 수 있습니다.

python

```python
from ultralytics import YOLO

model = YOLO(
    "runs/detect/my_yolo11/weights/best.pt"
)

metrics = model.val(
    data="/content/dataset/data.yaml"
)

print(metrics)
```

## 주요 평가 지표

| 지표 | 설명 |
| --- | --- |
| Precision | 탐지했다고 판단한 것 중 정답 비율 |
| Recall | 실제 정답 중 찾아낸 비율 |
| mAP50 | IoU 0.5 기준 평균 정밀도 |
| mAP50-95 | IoU 0.5\~0.95 기준 평균 정밀도 |

### Precision

잘못 탐지하는 결과가 적은지 확인하는 지표입니다.

### Recall

실제 존재하는 객체를 얼마나 놓치지 않고 찾았는지 확인하는 지표입니다.

### mAP

객체 탐지 모델의 전반적인 성능을 평가하는 대표적인 지표입니다.

---

# 16\. 학습된 모델 사용하기

학습한 모델을 불러와 새로운 이미지에 적용합니다.

python

```python
from ultralytics import YOLO

# 학습된 모델 불러오기
model = YOLO(
    "runs/detect/my_yolo11/weights/best.pt"
)

# 새로운 이미지 탐지
results = model.predict(
    source="/content/test.jpg",
    conf=0.5,
    save=True
)

# 결과 표시
results[0].show()
```

이제 내가 직접 학습한 객체 탐지 모델을 사용할 수 있습니다.

---

# 17\. 모델 내보내기(Export)

학습한 모델은 다른 환경에서도 사용할 수 있도록 다양한 형식으로 내보낼 수 있습니다.

예를 들어 ONNX 형식으로 변환할 수 있습니다.

python

```python
from ultralytics import YOLO

model = YOLO("best.pt")

model.export(format="onnx")
```

대표적인 내보내기 형식:

| 형식 | 용도 |
| --- | --- |
| PyTorch | Python 기반 모델 |
| ONNX | 다양한 실행 환경 |
| TensorRT | NVIDIA GPU 최적화 |
| OpenVINO | Intel 하드웨어 최적화 |
| CoreML | Apple 기기 활용 |

실제 배포 시에는 대상 장치와 지원 기능을 확인해야 합니다.

---

# 18\. 자주 발생하는 오류

## 오류 1. ultralytics 모듈이 없다고 나오는 경우

text

```text
ModuleNotFoundError: No module named 'ultralytics'
```

해결 방법:

python

```python
!pip install ultralytics
```

## 오류 2. GPU를 사용할 수 없는 경우

python

```python
import torch

print(torch.cuda.is_available())
```

False가 출력되면 코랩의 런타임 유형에서 GPU 사용 가능 여부를 확인합니다.

## 오류 3. 데이터셋 경로 오류

text

```text
Dataset not found
```

해결 방법:

- 데이터셋이 실제로 존재하는지 확인
- data.yaml의 path 확인
- train과 val 경로 확인
- 파일 이름 확인

## 오류 4. GPU 메모리 부족

해결 방법:

python

```python
results = model.train(
    data="/content/dataset/data.yaml",
    epochs=50,
    imgsz=640,
    batch=4
)
```

batch 값을 낮추거나 이미지 크기를 줄여 볼 수 있습니다.

## 오류 5. 학습 결과가 좋지 않은 경우

확인할 사항:

- 이미지 수가 충분한가?
- 라벨링이 정확한가?
- 클래스별 데이터 수가 균형적인가?
- 학습 데이터와 검증 데이터가 적절하게 분리되었는가?
- 학습 데이터와 실제 사용 환경이 비슷한가?

---

# 19\. YOLO11 실습 프로젝트

초보자는 다음 프로젝트를 순서대로 진행하는 것을 추천합니다.

## 프로젝트 1. 기본 객체 탐지

목표:

- 사전 학습 모델 사용
- 이미지 불러오기
- 사람과 자동차 탐지
- 탐지 결과 저장

## 프로젝트 2. 과일 탐지

목표:

- 사과, 바나나, 오렌지 데이터 수집
- 이미지 라벨링
- 데이터셋 구성
- YOLO11 학습
- 과일 탐지 테스트

## 프로젝트 3. 제품 불량 탐지

목표:

- 정상 제품과 불량 제품 이미지 수집
- 객체 또는 결함 영역 라벨링
- 모델 학습
- 새로운 제품 이미지 검사

## 프로젝트 4. 안전모 탐지

목표:

- 안전모 착용 이미지 수집
- 안전모 및 사람 객체 라벨링
- YOLO11 학습
- 안전모 착용 여부 탐지

실제 안전관리 시스템에 적용할 때는 별도의 성능 검증과 사람의 확인 절차가 필요합니다.

---

# 20\. YOLO11 학습 로드맵

| 단계 | 학습 내용 | 목표 |
| --- | --- | --- |
| 1단계 | Python 기초 | 코드 이해 |
| 2단계 | Google Colab | 실습 환경 익히기 |
| 3단계 | YOLO 개념 | 객체 탐지 이해 |
| 4단계 | 사전 학습 모델 | 기본 객체 탐지 |
| 5단계 | 데이터 라벨링 | 데이터셋 만들기 |
| 6단계 | 모델 학습 | 나만의 모델 만들기 |
| 7단계 | 성능 평가 | 모델 성능 분석 |
| 8단계 | 모델 배포 | 실제 프로젝트 활용 |

---

# 21\. 참고 사이트

1. Ultralytics 공식 YOLO11 문서\
   [https://docs.ultralytics.com/ko/models/yolo11/](<https://docs.ultralytics.com/ko/models/yolo11/>)
2. Ultralytics 공식 홈페이지\
   [https://www.ultralytics.com/](<https://www.ultralytics.com/>)
3. Ultralytics GitHub\
   [https://github.com/ultralytics/ultralytics](<https://github.com/ultralytics/ultralytics>)
4. Google Colab\
   [https://colab.research.google.com/](<https://colab.research.google.com/>)
5. CVAT 라벨링 도구\
   [https://www.cvat.ai/](<https://www.cvat.ai/>)
6. Roboflow\
   [https://roboflow.com/](<https://roboflow.com/>)

---

# 마무리

YOLO11은 이미지와 동영상에서 다양한 객체를 자동으로 탐지할 수 있는 컴퓨터 비전 모델입니다.

처음 공부할 때는 다음 순서를 추천합니다.

1. 코랩 환경 설정
2. YOLO11 설치
3. 사전 학습 모델 불러오기
4. 이미지 객체 탐지
5. 나만의 데이터셋 만들기
6. 모델 학습
7. 모델 성능 평가
8. 실제 프로젝트 적용

**초보자에게 가장 중요한 것은 처음부터 복잡한 모델을 학습하는 것이 아니라, 사전 학습된 YOLO11 모델로 객체 탐지를 직접 실행해 보는 것입니다.**

공식 문서: [https://docs.ultralytics.com/ko/models/yolo11/](<https://docs.ultralytics.com/ko/models/yolo11/>)

파일명: `YOLO11_초보자_학습가이드.md`

참고 자료는 Ultralytics 공식 문서를 기준으로 작성했습니다.

Ultralytics Docs

+2

다음 단계로는 이 문서를 바탕으로 구글 코랩에서 YOLO11을 직접 실행하고 나만의 객체 탐지 모델을 만드는 실습용 `.ipynb` 노트북을 만들면 좋습니다.
