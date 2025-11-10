"""
체크포인트 평가 사용 예시 및 가이드
"""

# 1. 독립적인 평가 스크립트 사용
# python evaluate.py checkpoint_path
# 예시:
# python evaluate.py checkpoints_tiny_20241110_142345/best_model.pth

# 2. main.py를 통한 평가
# python main.py --evaluate checkpoint_path
# 예시:
# python main.py --evaluate checkpoints_tiny_20241110_142345/best_model.pth

# 3. 추가 옵션들
# python evaluate.py checkpoint_path --data_type validation --max_batches 50 --num_samples 3
# python main.py --evaluate checkpoint_path --eval_data_type train --eval_max_batches 100

"""
사용 가능한 평가 옵션들:

1. 기본 평가 (validation 데이터, 전체 배치):
   python evaluate.py checkpoints/best_model.pth

2. 학습 데이터로 평가 (overfitting 확인):
   python evaluate.py checkpoints/best_model.pth --data_type train

3. 빠른 평가 (제한된 배치 수):
   python evaluate.py checkpoints/best_model.pth --max_batches 20

4. 샘플 분석 생략:
   python evaluate.py checkpoints/best_model.pth --no_samples

5. 커스텀 출력 디렉토리:
   python evaluate.py checkpoints/best_model.pth --output_dir my_evaluation

6. main.py를 통한 통합 평가:
   python main.py --evaluate checkpoints/best_model.pth --eval_data_type validation

평가 결과:
- evaluation_results.json: 상세한 평가 결과
- loss_analysis.png: 배치별 손실 분포 그래프
- 콘솔에 샘플 번역 결과 출력
"""

import os
import json
import torch
from datetime import datetime

def create_demo_checkpoint():
    """데모용 가짜 체크포인트 생성"""
    # 먼저 설정 파일이 있는지 확인
    if not os.path.exists('configs/tiny.json'):
        print("configs/tiny.json이 필요합니다. 먼저 학습을 실행하여 config 파일을 생성하세요.")
        return None
    
    # tiny config 로드
    with open('configs/tiny.json', 'r') as f:
        config = json.load(f)
    
    # 가짜 체크포인트 데이터 생성
    demo_checkpoint = {
        'step': 1000,
        'val_loss': 4.5,
        'config': config,
        # 실제로는 여기에 model_state_dict, optimizer_state_dict 등이 들어감
        # 데모용이므로 빈 딕셔너리
        'model_state_dict': {},
        'optimizer_state_dict': {},
        'scheduler_state_dict': {}
    }
    
    # 데모 디렉토리 생성
    demo_dir = "demo_checkpoints"
    os.makedirs(demo_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(demo_dir, "demo_checkpoint.pth")
    
    # 실제로는 torch.save를 사용하지만, 데모용으로는 JSON으로 저장
    with open(checkpoint_path.replace('.pth', '.json'), 'w') as f:
        json.dump(demo_checkpoint, f, indent=2)
    
    print(f"데모 체크포인트 생성: {checkpoint_path}")
    print("실제 평가를 위해서는 학습을 완료한 후 생성된 실제 체크포인트를 사용하세요.")
    
    return checkpoint_path

if __name__ == "__main__":
    print("체크포인트 평가 가이드")
    print("=" * 60)
    print(__doc__)
    
    # 데모 체크포인트 생성 (실제 사용 시에는 불필요)
    create_demo_checkpoint()
