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

YOLO 이미지 라벨링(Labeling) 완벽 가이드
YOLO 초보자를 위한 이미지 라벨링 개념, 필요성, 방법 및 Google Colab 학습 실습 가이드
1. 라벨링(Labeling)이란?

라벨링은 AI가 이미지 속 객체를 인식하고 학습할 수 있도록 사람이 정답 정보를 표시하는 작업이다.

예를 들어 자동차 이미지가 있다면 다음과 같이 표시한다.

자동차가 있는 위치
자동차의 종류
자동차의 영역
자동차의 클래스(Class)

AI는 처음부터 자동차나 사람, 제품의 불량 여부를 알지 못한다.

따라서 사람이 이미지에 정답을 표시하고, AI는 이 데이터를 이용해 학습한다.

라벨링은 AI에게 이미지 속 객체의 정답을 알려주는 작업이다.

2. 라벨링이 필요한 이유

기존에 학습된 YOLO 모델은 이미 학습한 객체를 탐지할 수 있다.

하지만 내가 원하는 새로운 객체를 탐지하려면 별도의 데이터 학습이 필요하다.

예를 들어 공장 제품의 불량을 검사하는 AI를 개발한다고 가정한다.

AI에게 다음과 같은 정보를 알려줘야 한다.

객체	클래스
정상 제품	normal
흠집	scratch
균열	crack
찍힘	dent

사람이 이미지에 해당 객체를 표시하면 YOLO가 이를 정답 데이터로 활용한다.

라벨링이 필요한 대표적인 분야
자율주행 자동차
공장 제품 불량 검사
사람 및 차량 탐지
의료 영상 분석
농작물 및 과일 분류
로봇 객체 인식
3. YOLO 라벨링의 종류

YOLO는 목적에 따라 여러 가지 라벨링 방식을 지원한다.

3.1 Detection (객체 탐지)

객체를 사각형(Bounding Box)으로 표시한다.

활용 사례:

자동차 탐지
사람 탐지
제품 위치 확인
안전모 착용 여부 확인

라벨 정보:

클래스 번호
중심 X 좌표
중심 Y 좌표
객체 너비
객체 높이
3.2 Segmentation (객체 분할)

객체의 외곽선을 따라 영역을 표시한다.

활용 사례:

제품의 정확한 형태 분석
금속 표면 흠집 검사
불량 영역의 면적 계산
객체별 영역 분리

Detection보다 객체의 경계를 세밀하게 표현할 수 있다.

3.3 Pose (자세 추정)

사람이나 동물의 주요 관절 위치를 표시한다.

활용 사례:

작업자 자세 분석
운동 동작 분석
사람의 행동 분석
3.4 OBB (Oriented Bounding Box)

회전된 사각형으로 객체를 표시한다.

활용 사례:

항공사진 속 자동차 탐지
위성사진 속 선박 탐지
회전된 제품 탐지
4. 라벨링 도구
4.1 Roboflow

웹브라우저에서 이미지 라벨링과 데이터셋 관리를 할 수 있는 도구다.

주요 기능:

이미지 업로드
Bounding Box 라벨링
Segmentation 라벨링
데이터셋 관리
YOLO 형식 데이터 내보내기

공식 홈페이지:

https://roboflow.com/

4.2 CVAT

이미지와 동영상에 객체를 표시할 수 있는 컴퓨터 비전 라벨링 도구다.

주요 기능:

Bounding Box
Polygon
Segmentation
동영상 객체 추적
다양한 데이터 형식 지원

공식 홈페이지:

https://www.cvat.ai/

초보자 추천

처음 시작한다면 Roboflow를 추천한다.

웹브라우저에서 작업할 수 있고, 라벨링한 데이터를 YOLO 학습 형식으로 쉽게 내보낼 수 있기 때문이다.

5. 실제 라벨링 작업 순서
STEP 1. 이미지 수집

탐지하려는 객체의 이미지를 준비한다.

예시:

자동차 이미지
사람 이미지
사과 이미지
바나나 이미지

처음에는 30~100장 정도의 소규모 데이터로 연습할 수 있다.

STEP 2. Roboflow 프로젝트 생성
Roboflow 홈페이지 접속
회원가입 및 로그인
Create New Project 선택
프로젝트 이름 입력
Object Detection 선택
이미지 업로드
STEP 3. 객체 라벨링

이미지에서 객체를 선택하고 사각형을 그린다.

예시 클래스:

객체	클래스 이름
사과	apple
바나나	banana
자동차	car
사람	person

모든 이미지에 동일한 기준으로 라벨링해야 한다.

STEP 4. 데이터셋 생성

라벨링이 끝나면 데이터셋을 생성한다.

일반적으로 다음과 같이 데이터를 분리한다.

Train: 모델 학습용
Validation: 학습 중 성능 확인용
Test: 최종 성능 평가용

데이터가 적을 때는 분리 비율을 적절하게 조정하되, 실제 평가에 사용할 Test 데이터는 별도로 보관하는 것이 좋다.

STEP 5. YOLO 형식으로 다운로드

Roboflow에서 데이터셋을 내보낼 때 YOLO 형식을 선택한다.

6. YOLO 라벨 파일 구조

YOLO Detection은 일반적으로 이미지 하나당 하나의 TXT 라벨 파일을 사용한다.

text
dataset/
│
├── images/
│   ├── train/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   │
│   ├── val/
│   │   └── image3.jpg
│   │
│   └── test/
│       └── image4.jpg
│
├── labels/
│   ├── train/
│   │   ├── image1.txt
│   │   └── image2.txt
│   │
│   ├── val/
│   │   └── image3.txt
│   │
│   └── test/
│       └── image4.txt
│
└── data.yaml

6.1 YOLO Detection TXT 파일

예시:

text
0 0.5 0.5 0.3 0.4
1 0.7 0.6 0.2 0.3


각 항목의 의미:

항목	의미
0	클래스 번호
0.5	객체 중심 X 좌표
0.5	객체 중심 Y 좌표
0.3	객체 너비
0.4	객체 높이

좌표와 크기는 이미지 크기에 대해 0~1 사이의 비율로 정규화한다.

6.2 data.yaml 파일
yaml
path: /content/dataset

train: images/train
val: images/val
test: images/test

names:
  0: apple
  1: banana


클래스 번호와 클래스 이름의 순서는 반드시 라벨 파일과 일치해야 한다.

7. Google Colab에서 YOLO11 학습하기

Roboflow에서 다운로드한 데이터셋을 Google Colab에 업로드한 후 학습할 수 있다.

STEP 1. Ultralytics 설치
python
!pip install ultralytics

STEP 2. 라이브러리 불러오기
python
from ultralytics import YOLO

STEP 3. 사전 학습 모델 불러오기
python
model = YOLO("yolo11n.pt")


yolo11n.pt는 YOLO11 모델 중 가벼운 Nano 모델이다.

STEP 4. 나만의 데이터셋 학습
python
model.train(
    data="/content/dataset/data.yaml",
    epochs=50,
    imgsz=640,
    batch=8
)


주요 설정:

항목	의미
data	데이터셋 YAML 경로
epochs	학습 반복 횟수
imgsz	입력 이미지 크기
batch	한 번에 학습하는 이미지 수
STEP 5. 새로운 이미지 탐지
python
model = YOLO("runs/detect/train/weights/best.pt")

results = model.predict(
    source="/content/test.jpg",
    save=True,
    conf=0.25
)


학습된 모델을 사용해 새로운 이미지의 객체를 탐지한다.

8. Detection과 Segmentation 비교
구분	Detection	Segmentation
표시 방법	사각형	객체 외곽선
탐지 위치	가능	가능
정확한 형태	제한적	상세 표현
면적 계산	부정확할 수 있음	가능
라벨링 난이도	낮음	높음
대표 모델	yolo11n.pt	yolo11n-seg.pt

공장 불량 검사에서는 불량의 위치만 찾는다면 Detection을 사용할 수 있다.

반면 흠집의 모양과 면적까지 분석하려면 Segmentation이 더 적합할 수 있다.

9. 라벨링 시 주의사항
9.1 정확한 객체 영역 표시

객체의 일부가 사각형 밖으로 벗어나지 않도록 한다.

9.2 일관된 클래스 이름 사용

동일한 객체를 서로 다른 이름으로 표시하지 않는다.

예를 들어 자동차를 car, vehicle, auto로 혼용하면 안 된다.

9.3 다양한 환경의 이미지 수집
밝은 환경
어두운 환경
다양한 각도
다양한 거리
다양한 배경
9.4 학습 데이터와 테스트 데이터 분리

학습에 사용한 이미지로 최종 성능을 평가하면 실제 성능을 제대로 확인하기 어렵다.

특히 같은 동영상에서 추출한 인접 프레임이 Train과 Test에 나뉘지 않도록 주의한다.

9.5 라벨 품질 확인

잘못된 라벨은 모델의 학습 품질을 떨어뜨린다.

라벨링이 끝난 후 반드시 이미지와 TXT 파일을 함께 확인한다.

10. Cosmos와 라벨링의 관계

NVIDIA Cosmos를 이용하면 기존 영상을 바탕으로 다양한 환경의 합성 영상을 생성할 수 있다.

예시:

공장 배경 변경
조명 변경
날씨 변경
주변 환경 변경

이렇게 생성된 영상은 YOLO 학습 데이터의 다양성을 높이는 데 활용할 수 있다.

단, 합성 영상은 실제 영상과 다를 수 있으므로 라벨을 검증해야 한다.

특히 객체의 위치나 모양이 바뀌었다면 기존 라벨을 그대로 사용해서는 안 된다.

권장 작업 순서:

실제 영상 수집
Cosmos로 영상 변환
합성 영상의 품질 검증
라벨 생성 및 수정
YOLO 학습
실제 영상으로 성능 평가

합성 데이터는 실제 데이터의 보조 수단으로 사용하는 것이 바람직하다.

11. 추천 실습 프로젝트
프로젝트 1. 사과와 바나나 탐지

목표: 사과와 바나나를 구분하는 AI 개발

작업 순서:

사과 이미지 수집
바나나 이미지 수집
Roboflow 라벨링
YOLO 데이터셋 생성
Google Colab 학습
새로운 이미지 탐지
프로젝트 2. 자동차와 사람 탐지

목표: 도로 영상에서 자동차와 사람 탐지

작업 순서:

도로 이미지 수집
자동차와 사람 라벨링
YOLO 학습
CCTV 영상 테스트
프로젝트 3. 공장 제품 불량 검사

목표: 금속 제품의 흠집 및 균열 탐지

작업 순서:

실제 제품 이미지 촬영
정상 및 불량 데이터 분류
Roboflow 또는 CVAT 라벨링
YOLO11-seg 학습
새로운 제품 영상으로 검증
12. 핵심 정리

라벨링은 AI가 이미지 속 객체를 학습할 수 있도록 정답을 제공하는 과정이다.

YOLO를 활용한 AI 개발은 다음과 같은 순서로 진행된다.

text
이미지 수집
    ↓
라벨링
    ↓
데이터셋 생성
    ↓
Google Colab 환경 구성
    ↓
YOLO 모델 학습
    ↓
새로운 이미지 탐지
    ↓
성능 평가 및 개선


기억해야 할 핵심

Detection은 객체를 사각형으로 표시한다.
Segmentation은 객체의 외곽선을 표시한다.
YOLO 라벨은 일반적으로 TXT 파일로 저장한다.
data.yaml에서 클래스 이름과 데이터 경로를 지정한다.
정확한 라벨링은 좋은 AI 모델을 만드는 중요한 요소다.
최종 성능은 학습에 사용하지 않은 실제 데이터로 평가해야 한다.
13. 공식 참고 사이트
YOLO 공식 문서: https://docs.ultralytics.com/
Roboflow: https://roboflow.com/
CVAT: https://www.cvat.ai/
Google Colab: https://colab.research.google.com/
NVIDIA Cosmos: https://www.nvidia.com/en-us/ai/cosmos/
