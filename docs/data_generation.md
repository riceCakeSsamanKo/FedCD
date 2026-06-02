# FedCD Baseline Data Generation and Layout

이 문서는 FedCD baseline 학습에서 사용하는 FL 데이터 생성 방식과 저장 구조를 정리한다. 핵심은 baseline 코드가 특정 생성 스크립트에 묶여 있지 않고, 아래의 공통 `.npz` 구조만 만족하면 같은 방식으로 데이터를 읽는다는 점이다.

## 1. 데이터 루트

FedCD-Baseline은 dataset 이름만 인자로 받고, 실제 데이터 루트는 `FedCD-Baseline/system/utils/data_utils.py`에서 찾는다.

검색 순서:

1. 환경변수 `FL_DATA_ROOT`
2. `C:\Users\mulso\Documents\GitHub\FedCD\fl_data`
3. `C:\Users\mulso\Documents\GitHub\fl_data`

현재 실험에서는 보통 sibling data root인 `C:\Users\mulso\Documents\GitHub\fl_data`를 사용한다.

PowerShell에서 명시하려면 다음처럼 둔다.

```powershell
$env:FL_DATA_ROOT = "C:\Users\mulso\Documents\GitHub\fl_data"
```

## 2. 공통 데이터 구조

모든 FedCD baseline 데이터셋은 아래 구조를 따른다.

```text
<FL_DATA_ROOT>/
  <dataset_name>/
    config.json
    train/
      0.npz
      1.npz
      ...
      <num_clients-1>.npz
    test/
      0.npz
      1.npz
      ...
      <num_clients-1>.npz
```

예시:

```text
C:\Users\mulso\Documents\GitHub\fl_data\Cifar10_pat_nc50\
C:\Users\mulso\Documents\GitHub\fl_data\Cifar10_splitgp_pat_rho0.2_nc50\
C:\Users\mulso\Documents\GitHub\fl_data\FashionMNIST_splitgp_pat_rho0.8_nc50\
```

각 client 파일은 compressed numpy archive이며, 내부 key는 `data` 하나다.

```python
archive = np.load("train/0.npz", allow_pickle=True)
data = archive["data"].tolist()
x = data["x"]
y = data["y"]
```

`data` dict 구조:

```text
data = {
  "x": numpy.ndarray,
  "y": numpy.ndarray
}
```

이미지 baseline에서 loader는 이를 다음처럼 변환한다.

```python
X = torch.Tensor(data["x"]).type(torch.float32)
y = torch.Tensor(data["y"]).type(torch.int64)
client_dataset = [(x_i, y_i), ...]
```

일반적인 shape:

```text
CIFAR-10/CIFAR-100: x.shape = (N, 3, 32, 32), y.shape = (N,)
FashionMNIST:       x.shape = (N, 1, 28, 28), y.shape = (N,)
```

`x`는 torchvision transform을 거친 float tensor를 numpy로 저장한 값이다. CIFAR 계열은 `(0.5, 0.5, 0.5)` mean/std normalization, FashionMNIST는 `[0.5]` normalization을 사용한다.

## 3. 기존 FedCD baseline용 pat/dir 데이터 생성

스크립트:

```text
C:\Users\mulso\Documents\GitHub\FedCD\tools\regenerate_fedcd_fl_data.py
```

기본 생성 명령:

```powershell
cd C:\Users\mulso\Documents\GitHub\FedCD
python tools\regenerate_fedcd_fl_data.py --fl-data-root ..\fl_data --delete-existing
```

선택 생성 예시:

```powershell
python tools\regenerate_fedcd_fl_data.py --fl-data-root ..\fl_data --datasets Cifar10 --num-clients 50 --scenarios pat --delete-existing
python tools\regenerate_fedcd_fl_data.py --fl-data-root ..\fl_data --datasets FashionMNIST --num-clients 20 50 --scenarios dir0.1 dir0.5 dir1.0 --delete-existing
```

기본 설정:

```text
seed = 1
non_iid = true
balance = true
batch_size = 10
scenarios = pat, dir0.1, dir0.5, dir1.0
num_clients = 20, 50
CIFAR-10 class_per_client = 2
CIFAR-100 class_per_client = 10
FashionMNIST class_per_client = 2
```

생성 방식은 dataset별로 다르다.

| Dataset | 생성 방식 | 설명 |
| --- | --- | --- |
| CIFAR-10 | FedCD original split | torchvision 원본 train 50000, test 10000을 유지하고 train/test를 각각 client별로 partition |
| CIFAR-100 | FedCD original split | CIFAR-10과 동일하게 원본 train/test 유지 후 각각 partition |
| FashionMNIST | FedCCM merged split | 원본 train 60000과 test 10000을 합친 70000개를 먼저 client별로 partition한 뒤, 각 client 데이터를 75/25 train/test로 분할 |

`pat`는 class-constrained partition이다. CIFAR-10/FashionMNIST에서 `class_per_client=2`이면 각 client는 두 개 class를 가진다. 현재 순차 class-pair 데이터에서는 NC50 기준 다음 구조가 된다.

```text
clients 0-9:   classes 0,1
clients 10-19: classes 2,3
clients 20-29: classes 4,5
clients 30-39: classes 6,7
clients 40-49: classes 8,9
```

`dir`는 class별 index를 client들에게 Dirichlet 분포로 나누는 방식이다. `dir0.1`, `dir0.5`, `dir1.0`의 숫자는 alpha 값이다. alpha가 작을수록 label 분포가 더 치우친다.

## 4. SplitGP rho 데이터 생성

### 4.0 실제 데이터 생성 코드

SplitGP rho 데이터 생성 코드는 아래 파일이다.

```text
C:\Users\mulso\Documents\GitHub\FedCD\tools\generate_splitgp_rho_data.py
```

코드 흐름은 다음과 같다.

| 단계 | 함수 | 역할 |
| --- | --- | --- |
| 1 | `parse_args()` | dataset, rho, client 수, 저장 경로, partition mode, client당 test sample 수를 인자로 받음 |
| 2 | `load_torchvision_split()` | CIFAR-10 또는 FashionMNIST의 원본 torchvision train/test split을 로드하고 normalize된 tensor를 numpy로 변환 |
| 3 | `make_splitgp_class_pair_train_partition()` | 현재 실험 기준 mode. `(0,1)`, `(2,3)`, `(4,5)`, `(6,7)`, `(8,9)` class pair를 만들고 NC50에서 pair마다 client 10명씩 배정 |
| 4 | `make_splitgp_train_partition()` | 선택 가능한 `shard_random` mode. 전체 train set을 class 기준 정렬 후 100개 shard로 나누고 client마다 shard 2개 할당 |
| 5 | `make_client_test_indices()` | client별 local test set을 고정 크기 1000개로 만들고, rho에 따라 main/OOD sample 수를 계산 |
| 6 | `write_client_npz()` | `<output-root>/<dataset>/train/<client_id>.npz`, `test/<client_id>.npz` 저장 |
| 7 | `config.json` 저장 | rho, partition mode, client별 train class, client별 test main/OOD count metadata 저장 |

현재 FedCCMV22와 FedCD baseline 비교에서 쓰는 기준 명령은 아래다.

```powershell
cd C:\Users\mulso\Documents\GitHub\FedCD
python tools\generate_splitgp_rho_data.py --output-root ..\fl_data --datasets cifar10 fashionmnist --rhos 0.0 0.2 0.4 0.6 0.8 --num-clients 50 --partition-mode class_pair --test-samples-per-client 1000 --force
```

rho 하나만 생성하려면 다음처럼 실행한다.

```powershell
python tools\generate_splitgp_rho_data.py --output-root ..\fl_data --datasets cifar10 --rhos 0.2 --num-clients 50 --partition-mode class_pair --test-samples-per-client 1000 --force
python tools\generate_splitgp_rho_data.py --output-root ..\fl_data --datasets fashionmnist --rhos 0.8 --num-clients 50 --partition-mode class_pair --test-samples-per-client 1000 --force
```

생성되는 dataset directory 이름은 다음 규칙을 따른다.

```text
<Dataset>_splitgp_pat_rho<rho>_nc<num_clients>
```

예시:

```text
Cifar10_splitgp_pat_rho0.2_nc50
FashionMNIST_splitgp_pat_rho0.8_nc50
```

### 4.1 rho별 데이터 생성 방식

rho 데이터에서 train 데이터는 rho에 따라 바뀌지 않는다. rho는 오직 client별 local test set 내부의 main/OOD 비율만 바꾼다.

현재 실험 기준:

```text
num_clients = 50
test_samples_per_client = 1000
partition_mode = class_pair
rho = #OOD test samples / #main test samples
```

class pair는 다음처럼 고정된다.

```text
clients 0-9:   classes 0,1
clients 10-19: classes 2,3
clients 20-29: classes 4,5
clients 30-39: classes 6,7
clients 40-49: classes 8,9
```

각 client의 train set은 rho와 무관하게 1000개다. 두 main class가 500개씩 들어간다.

```text
client train total = 1000
main class A = 500
main class B = 500
```

각 client의 test set도 rho와 무관하게 항상 1000개다. 다만 1000개 안에서 main class sample과 OOD class sample 수가 rho에 따라 달라진다.

계산식:

```text
OOD_count  = round(1000 * rho / (1 + rho))
main_count = 1000 - OOD_count
```

rho별 client당 test 구성은 다음과 같다.

| rho | main samples | OOD samples | total test samples | 해석 |
| --- | ---: | ---: | ---: | --- |
| 0.0 | 1000 | 0 | 1000 | client가 학습한 class만 test에 존재 |
| 0.2 | 833 | 167 | 1000 | main:OOD 비율이 약 1:0.2 |
| 0.4 | 714 | 286 | 1000 | main:OOD 비율이 약 1:0.4 |
| 0.6 | 625 | 375 | 1000 | main:OOD 비율이 약 1:0.6 |
| 0.8 | 556 | 444 | 1000 | main:OOD 비율이 약 1:0.8 |

예를 들어 client 0의 train class가 `[0, 1]`이면 다음처럼 구성된다.

| rho | client 0 test main | client 0 test OOD |
| --- | --- | --- |
| 0.0 | class 0/1에서 1000개 | 없음 |
| 0.2 | class 0/1에서 833개 | class 2-9에서 167개 |
| 0.4 | class 0/1에서 714개 | class 2-9에서 286개 |
| 0.6 | class 0/1에서 625개 | class 2-9에서 375개 |
| 0.8 | class 0/1에서 556개 | class 2-9에서 444개 |

main samples는 client의 두 train class에 거의 균등하게 나뉘고, OOD samples는 나머지 8개 class에 거의 균등하게 나뉜다. 예를 들어 rho 0.2에서 client 0은 main 833개를 class 0/1에 약 416/417개로 나누고, OOD 167개를 class 2-9에 약 20/21개씩 나눈다.

### 4.2 저장 구조

각 rho dataset은 기존 FedCD baseline loader가 그대로 읽을 수 있도록 같은 `.npz` 구조로 저장된다.

```text
<output-root>/<dataset-name>/config.json
<output-root>/<dataset-name>/train/0.npz
<output-root>/<dataset-name>/train/1.npz
...
<output-root>/<dataset-name>/test/0.npz
<output-root>/<dataset-name>/test/1.npz
...
```

각 `.npz` 내부는 다음 dict 하나다.

```python
data = {
    "x": numpy.ndarray,
    "y": numpy.ndarray,
}
```

config에는 다음 metadata가 들어간다.

```json
{
  "splitgp_rho": 0.2,
  "splitgp_partition_mode": "class_pair",
  "splitgp_test_samples_per_client": 1000,
  "client_train_classes": [[0, 1], [0, 1], ...],
  "client_test_counts": [
    {"client_id": 0, "main_samples": 833, "ood_samples": 167, "total_samples": 1000}
  ]
}
```

FedCD-Baseline은 `config.json`에 `splitgp_rho`가 있으면 rho dataset으로 판단하고 global test evaluation을 끈다. 따라서 rho 실험에서는 모든 client test set을 합친 global accuracy가 아니라 client별 local test accuracy의 평균을 사용한다.
## 5. config.json 구조

각 dataset directory의 `config.json`은 baseline 실행과 로그 정리에 필요한 metadata를 담는다.

기존 pat/dir 데이터의 주요 필드:

```json
{
  "num_clients": 50,
  "num_classes": 10,
  "non_iid": true,
  "balance": true,
  "partition": "pat",
  "Size of samples for labels in clients": [[[0, 500], [1, 500]], ...],
  "alpha": 0.1,
  "batch_size": 10,
  "use_original_test_split": true,
  "original_train_samples": 50000,
  "original_test_samples": 10000
}
```

SplitGP rho 데이터의 주요 필드:

```json
{
  "dataset_source": "Cifar10",
  "num_clients": 50,
  "num_classes": 10,
  "partition": "pat",
  "partition_detail": "splitgp_class_pair",
  "splitgp_rho": 0.2,
  "splitgp_partition_mode": "class_pair",
  "splitgp_test_samples_per_client": 1000,
  "client_train_classes": [[0, 1], [0, 1], ...],
  "client_test_counts": [
    {"client_id": 0, "main_samples": 833, "ood_samples": 167, "total_samples": 1000}
  ]
}
```

FedCD-Baseline `main.py`는 `config.json`에 `splitgp_rho`가 있으면 rho 데이터로 판단하고 `eval_common_global=False`로 설정한다. 따라서 rho 실험에서는 모든 client test set을 합친 global test accuracy를 사용하지 않고, 각 client local test accuracy의 평균을 기록한다.

## 6. FedCD baseline 평가 방식

일반 `pat/dir` 데이터:

```text
local_test_acc  = client별 local test sample 전체에 대한 weighted 평균
 global_test_acc = 모든 client test set을 합친 shared test set에서 각 client model을 평가한 평균
```

SplitGP rho 데이터:

```text
local_test_acc  = client별 local test accuracy의 macro average
 global_test_acc = 사용하지 않음, acc.csv에서 빈 값
```

rho 데이터에서 macro average를 쓰는 이유는 rho별로 각 client의 local test set을 동일하게 1000개로 고정하고, rho는 test 내부의 seen/OOD 비율만 바꾸기 때문이다. 이렇게 해야 FedCCMV22와 FedCD baseline이 같은 local evaluation 정의를 공유한다.

## 7. 검증 명령

데이터 구조 검증:

```powershell
cd C:\Users\mulso\Documents\GitHub\FedCD
python tools\verify_fl_data_setup.py --fl-data-root ..\fl_data
```

특정 client의 `.npz` 구조 확인:

```powershell
python -c "import numpy as np; p=r'..\fl_data\Cifar10_splitgp_pat_rho0.2_nc50\test\0.npz'; d=np.load(p, allow_pickle=True)['data'].tolist(); print(d['x'].shape, d['y'].shape); print(np.unique(d['y'], return_counts=True))"
```

rho config 확인:

```powershell
python -c "import json; p=r'..\fl_data\Cifar10_splitgp_pat_rho0.2_nc50\config.json'; c=json.load(open(p)); print(c['splitgp_rho'], c['splitgp_partition_mode'], c['client_train_classes'][0], c['client_test_counts'][0])"
```

## 8. Baseline 실행 시 dataset 인자

실행할 때는 full path가 아니라 dataset directory 이름을 넘긴다.

예시:

```powershell
cd C:\Users\mulso\Documents\GitHub\FedCD\FedCD-Baseline\system
$env:FL_DATA_ROOT = "C:\Users\mulso\Documents\GitHub\fl_data"
python main.py -data Cifar10_splitgp_pat_rho0.2_nc50 -m VGG8 -algo FedAvg -nc 50 -gr 100
```

정확한 알고리즘별 인자는 기존 shell script를 따른다. 데이터 쪽에서는 `<FL_DATA_ROOT>/<dataset_name>/train/<client_id>.npz`, `<FL_DATA_ROOT>/<dataset_name>/test/<client_id>.npz`, `config.json`만 올바르면 baseline 학습이 같은 loader로 동작한다.