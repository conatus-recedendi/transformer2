# 체크포인트 평가 사용법 가이드

## 1. 기본 평가 방법

### 방법 1: evaluate.py 스크립트 사용 (추천)

```bash
# 기본 평가 (validation 데이터)
python evaluate.py checkpoints/best_model.pth

# 학습 데이터로 평가 (overfitting 확인)
python evaluate.py checkpoints/best_model.pth --data_type train

# 빠른 평가 (20개 배치만)
python evaluate.py checkpoints/best_model.pth --max_batches 20

# 많은 샘플 분석
python evaluate.py checkpoints/best_model.pth --num_samples 10

# 샘플 분석 생략
python evaluate.py checkpoints/best_model.pth --no_samples

# 커스텀 출력 디렉토리
python evaluate.py checkpoints/best_model.pth --output_dir my_evaluation_results
```

### 방법 2: main.py를 통한 평가

```bash
# 기본 평가
python main.py --evaluate checkpoints/best_model.pth

# 학습 데이터로 평가
python main.py --evaluate checkpoints/best_model.pth --eval_data_type train

# 제한된 배치로 평가
python main.py --evaluate checkpoints/best_model.pth --eval_max_batches 50
```

## 2. 실제 사용 예시

### 먼저 모델 학습하여 체크포인트 생성

```bash
# tiny 모델로 100 스텝 학습
python main.py --config tiny --train_steps 100

# 학습 완료 후 생성되는 파일들:
# checkpoints_tiny_YYYYMMDD_HHMMSS/
#   ├── best_model.pth              # 최고 성능 모델
#   ├── checkpoint_step_100.pth     # 마지막 체크포인트
#   ├── training_results.json       # 학습 결과
#   └── training_curves.png         # 학습 곡선
```

### 생성된 체크포인트로 평가

```bash
# 최고 성능 모델 평가
python evaluate.py checkpoints_tiny_20241110_142345/best_model.pth

# 특정 스텝 체크포인트 평가
python evaluate.py checkpoints_tiny_20241110_142345/checkpoint_step_100.pth

# 여러 체크포인트 비교 평가
python evaluate.py checkpoints_tiny_20241110_142345/best_model.pth --output_dir eval_best
python evaluate.py checkpoints_tiny_20241110_142345/checkpoint_step_100.pth --output_dir eval_step100
```

## 3. 평가 결과 해석

### 콘솔 출력

```
EVALUATION RESULTS
============================================================
Average Loss: 4.2345          # 평균 손실 (낮을수록 좋음)
Perplexity: 68.42             # 퍼플렉서티 (낮을수록 좋음)
Total Batches: 45             # 평가된 배치 수
Total Tokens: 125,000         # 평가된 총 토큰 수
Loss Std: 0.234               # 손실의 표준편차
Loss Range: [3.8912, 4.7823] # 손실 범위
============================================================

--- Sample 1 ---
Source: Hello, how are you?
Target: 안녕하세요, 어떻게 지내세요?
Prediction: 안녕하세요, 어떻게 지내고 있나요?
Token Accuracy: 0.7500
```

### 생성되는 파일들

- `evaluation_results.json`: 상세한 평가 메트릭
- `loss_analysis.png`: 배치별 손실 분포 그래프

## 4. 고급 사용법

### 배치 처리 방법 이해

```bash
# 전체 데이터셋 평가 (시간이 오래 걸림)
python evaluate.py checkpoints/best_model.pth

# 빠른 평가 (처음 20개 배치만)
python evaluate.py checkpoints/best_model.pth --max_batches 20

# 매우 빠른 평가 (5개 배치만)
python evaluate.py checkpoints/best_model.pth --max_batches 5
```

### 데이터 타입별 평가

```bash
# Validation 데이터로 평가 (기본값)
python evaluate.py checkpoints/best_model.pth --data_type validation

# Training 데이터로 평가 (overfitting 확인)
python evaluate.py checkpoints/best_model.pth --data_type train
```

### 체크포인트 찾기

```bash
# 사용 가능한 체크포인트 확인
ls -la checkpoints_*/
ls -la checkpoints_*/*.pth

# 가장 최근 체크포인트 찾기
ls -t checkpoints_*/best_model.pth | head -1
```

## 5. 문제 해결

### 토크나이저를 찾을 수 없는 경우

```bash
# 학습 시 토크나이저가 저장되지 않은 경우
# 먼저 같은 설정으로 짧게 학습하여 토크나이저 생성
python main.py --config tiny --train_steps 10

# 그 후 평가 실행
python evaluate.py checkpoints/best_model.pth
```

### CUDA 메모리 부족 시

```bash
# 더 적은 배치로 평가
python evaluate.py checkpoints/best_model.pth --max_batches 10

# CPU로 평가 (느리지만 안전)
CUDA_VISIBLE_DEVICES="" python evaluate.py checkpoints/best_model.pth
```

## 6. 평가 스크립트 자동화

### 여러 체크포인트 일괄 평가

```bash
#!/bin/bash
# evaluate_all.sh

for checkpoint in checkpoints_*/best_model.pth; do
    echo "Evaluating $checkpoint"
    python evaluate.py "$checkpoint" --max_batches 20 --no_samples
done
```

### 성능 비교

```bash
# 여러 모델 성능 비교
python evaluate.py checkpoints_tiny_*/best_model.pth --output_dir eval_tiny
python evaluate.py checkpoints_base_*/best_model.pth --output_dir eval_base

# 결과 비교
grep "Average Loss" eval_*/evaluation_results.json
```

이 가이드를 참고하여 체크포인트 평가를 수행하세요!
