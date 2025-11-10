# Transformer 번역 모델 프로젝트

이 프로젝트는 PyTorch를 사용하여 Transformer 아키텍처를 구현한 영어-한국어 번역 모델입니다.

## 프로젝트 구조

```
transformer2/
├── src/
│   ├── __init__.py
│   ├── model.py          # Transformer 모델 구현
│   └── data_utils.py     # 데이터 전처리 및 토크나이저 유틸리티
├── train.py              # 모델 학습 스크립트
├── inference.py          # 번역 추론 스크립트
├── demo.ipynb           # 데모 노트북
├── requirements.txt      # 의존성 패키지
└── README.md            # 이 파일
```

## 주요 기능

### 1. Transformer 모델 구현

- Multi-Head Attention
- Positional Encoding
- Encoder-Decoder 아키텍처
- Layer Normalization
- Feed Forward Networks

### 2. 데이터 처리

- BPE (Byte Pair Encoding) 토크나이저
- 커스텀 데이터셋 클래스
- 배치 처리 및 패딩

### 3. 학습 기능

- Label Smoothing Loss
- Adam 옵티마이저
- 학습률 스케줄링
- 체크포인트 저장
- 학습 곡선 시각화

### 4. 추론 기능

- Greedy Decoding
- Beam Search Decoding
- 인터랙티브 번역

## 설치 및 사용법

### 1. 환경 설정

이 프로젝트는 uv를 사용하여 관리됩니다:

```bash
# 프로젝트 디렉토리로 이동
cd transformer2

# 의존성이 이미 설치되어 있습니다
uv run python --version  # Python 3.11 확인
```

### 2. 모델 학습

```bash
# 학습 시작
uv run python train.py
```

학습이 완료되면 다음 파일들이 생성됩니다:

- `checkpoints/best_model.pth`: 최고 성능 모델
- `tokenizers/src_tokenizer.json`: 소스 언어 토크나이저
- `tokenizers/tgt_tokenizer.json`: 타겟 언어 토크나이저

### 3. 번역 추론

```bash
# 번역 실행
uv run python inference.py
```

### 4. Jupyter 노트북 실행

```bash
# Jupyter 노트북 시작
uv run jupyter notebook demo.ipynb
```

## 모델 아키텍처

### Transformer 구조

- **Embedding Layer**: 토큰을 벡터로 변환
- **Positional Encoding**: 위치 정보 추가
- **Encoder**: 6개의 인코더 레이어
  - Multi-Head Self-Attention
  - Feed Forward Network
  - Residual Connection & Layer Normalization
- **Decoder**: 6개의 디코더 레이어
  - Masked Multi-Head Self-Attention
  - Multi-Head Cross-Attention
  - Feed Forward Network
  - Residual Connection & Layer Normalization

### 하이퍼파라미터

- Model Dimension (d_model): 256
- Number of Heads: 8
- Number of Layers: 4
- Feed Forward Dimension: 1024
- Vocabulary Size: 5000 (각 언어)
- Max Sequence Length: 128

## 샘플 데이터

프로젝트에는 영어-한국어 번역을 위한 샘플 데이터가 포함되어 있습니다:

```python
sample_data = [
    {"src": "Hello, how are you?", "tgt": "안녕하세요, 어떻게 지내세요?"},
    {"src": "I love machine learning.", "tgt": "저는 머신러닝을 사랑합니다."},
    # ... 더 많은 예시
]
```

## 확장 가능성

1. **더 큰 데이터셋**: 실제 번역 코퍼스 사용
2. **다른 언어 쌍**: 토크나이저만 변경하면 다른 언어 지원
3. **모델 크기 조정**: 하이퍼파라미터 수정으로 모델 크기 변경
4. **고급 디코딩**: Top-k, Top-p 샘플링 추가

## 성능 최적화

- **혼합 정밀도 학습**: `torch.cuda.amp` 사용
- **그래디언트 누적**: 큰 배치 크기 시뮬레이션
- **데이터 병렬화**: 여러 GPU 사용
- **모델 병렬화**: 큰 모델을 위한 분산 처리

## 문제 해결

### 메모리 부족

- 배치 크기 줄이기
- 시퀀스 길이 줄이기
- 모델 크기 줄이기

### 학습 불안정

- 학습률 조정
- 그래디언트 클리핑 추가
- Warmup 스케줄러 사용

## 참고 자료

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [PyTorch Tutorial](https://pytorch.org/tutorials/beginner/transformer_tutorial.html)
