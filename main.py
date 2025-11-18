"""
Transformer 번역 모델 학습을 위한 메인 스크립트
JSON config 파일을 통한 다양한 아키텍처 변형을 지원합니다.
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
from datetime import datetime
import glob

from src.trainer import TransformerTrainer


def load_config(config_path):
    """JSON config 파일 로드"""
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return config


def get_available_configs():
    """사용 가능한 config 파일 목록 반환"""
    config_files = glob.glob("configs/*.json")
    configs = {}
    for config_file in config_files:
        config_name = os.path.basename(config_file).replace(".json", "")
        try:
            config = load_config(config_file)
            configs[config_name] = {
                "path": config_file,
                "description": config.get("description", "No description"),
                "config": config,
            }
        except Exception as e:
            print(f"Warning: Could not load config {config_file}: {e}")
    return configs


def merge_config_with_args(config, args):
    """커맨드라인 인자로 config 덮어쓰기"""
    if args.train_steps is not None:
        config["training"]["train_steps"] = args.train_steps
    if args.batch_tokens is not None:
        config["training"]["batch_tokens"] = args.batch_tokens
    if args.learning_rate is not None:
        config["training"]["learning_rate"] = args.learning_rate
    if args.max_length is not None:
        config["data"]["max_length"] = args.max_length
        config["model"]["max_seq_length"] = args.max_length
    if args.vocab_size is not None:
        config["data"]["vocab_size"] = args.vocab_size
    if args.data_multiplier is not None:
        config["data"]["data_multiplier"] = args.data_multiplier
    if args.warmup_steps is not None:
        config["training"]["warmup_steps"] = args.warmup_steps
    if args.update_freq is not None:
        config["training"]["update_freq"] = args.update_freq

    return config


def find_latest_checkpoint(checkpoint_dir):
    """체크포인트 디렉토리에서 가장 최근 체크포인트 찾기"""
    if not os.path.exists(checkpoint_dir):
        return None, None

    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_step_*.pth"))

    if not checkpoint_files:
        return None, None

    # 스텝 번호로 정렬하여 가장 최근 체크포인트 찾기
    def extract_step(filepath):
        filename = os.path.basename(filepath)
        try:
            step = int(filename.replace("checkpoint_step_", "").replace(".pth", ""))
            return step
        except ValueError:
            return 0

    checkpoint_files.sort(key=extract_step, reverse=True)
    latest_checkpoint = checkpoint_files[0]
    latest_step = extract_step(latest_checkpoint)

    return latest_checkpoint, latest_step


def load_and_resume_training(trainer, checkpoint_path, target_step=None):
    """체크포인트에서 학습 재개"""
    print(f"Loading checkpoint from: {checkpoint_path}")

    step, val_loss = trainer.load_checkpoint(checkpoint_path)

    print(f"✓ Resumed from step {step}")
    if val_loss != float("inf"):
        print(f"✓ Previous validation loss: {val_loss:.4f}")

    # 특정 스텝부터 시작하려는 경우
    if target_step is not None and target_step != step:
        print(f"⚠️  Requested step {target_step} but checkpoint is at step {step}")
        print(f"   Continuing from checkpoint step {step}")

    return step


# TransformerTrainer와 LabelSmoothingLoss는 src.trainer에서 import됨


def main():
    parser = argparse.ArgumentParser(
        description="Transformer 번역 모델 학습 (JSON Config 지원)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="base",
        help="Config 파일 이름 (configs/ 디렉토리의 .json 파일)",
    )
    parser.add_argument(
        "--config_path", type=str, default=None, help="Config 파일의 전체 경로"
    )
    parser.add_argument(
        "--train_steps",
        type=int,
        default=None,
        help="학습 스텝 수 (config 파일 덮어쓰기)",
    )
    parser.add_argument(
        "--batch_tokens",
        type=int,
        default=None,
        help="배치 토큰 수 (config 파일 덮어쓰기)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="학습률 (config 파일 덮어쓰기)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=None,
        help="최대 시퀀스 길이 (config 파일 덮어쓰기)",
    )
    parser.add_argument(
        "--vocab_size", type=int, default=None, help="어휘 크기 (config 파일 덮어쓰기)"
    )
    parser.add_argument(
        "--data_multiplier",
        type=int,
        default=None,
        help="데이터 확장 배수 (config 파일 덮어쓰기)",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=None,
        help="워밍업 스텝 (config 파일 덮어쓰기)",
    )
    parser.add_argument(
        "--update_freq",
        type=int,
        default=None,
        help="업데이트 빈도 (gradient accumulation)",
    )
    parser.add_argument("--save_dir", type=str, default=None, help="저장 디렉토리")
    parser.add_argument(
        "--list_configs", action="store_true", help="사용 가능한 설정 목록 출력"
    )
    parser.add_argument(
        "--create_config",
        type=str,
        default=None,
        help="새로운 config 파일 생성 (파일명)",
    )
    parser.add_argument(
        "--evaluate", type=str, default=None, help="체크포인트 파일 경로 (평가 모드)"
    )
    parser.add_argument(
        "--eval_data_type",
        type=str,
        default="validation",
        choices=["validation", "train"],
        help="평가할 데이터 타입",
    )
    parser.add_argument(
        "--eval_max_batches", type=int, default=None, help="평가 시 최대 배치 수"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="체크포인트 폴더에서 재시작 (폴더 경로)",
    )
    parser.add_argument(
        "--resume_from_step",
        type=int,
        default=None,
        help="특정 스텝부터 재시작 (선택적)",
    )

    args = parser.parse_args()

    # config 파일들이 있는지 확인하고 없으면 생성
    if not os.path.exists("configs"):
        os.makedirs("configs")
        print("configs/ 디렉토리가 생성되었습니다.")

    # 사용 가능한 config 목록 출력
    if args.list_configs:
        configs = get_available_configs()
        if not configs:
            print("사용 가능한 config 파일이 없습니다.")
            print(
                "configs/ 디렉토리에 .json 파일을 생성하거나 --create_config 옵션을 사용하세요."
            )
            return

        print("사용 가능한 모델 설정:")
        print("=" * 80)
        for name, info in configs.items():
            config = info["config"]
            print(f"{name:15s}: {info['description']}")
            model_cfg = config["model"]
            training_cfg = config["training"]
            data_cfg = config["data"]
            print(
                f"{'':15s}  Model: N={model_cfg['N']}, d_model={model_cfg['d_model']}, "
                f"d_ff={model_cfg['d_ff']}, h={model_cfg['h']}"
            )
            print(
                f"{'':15s}  Training: train_steps={training_cfg['train_steps']}, "
                f"batch_tokens={training_cfg['batch_tokens']}, lr={training_cfg['learning_rate']}"
            )
            print(
                f"{'':15s}  Data: vocab_size={data_cfg['vocab_size']}, "
                f"max_length={data_cfg['max_length']}"
            )
            print()
        return

    # 새로운 config 파일 생성
    if args.create_config:
        create_sample_config(args.create_config)
        return

    # 평가 모드
    if args.evaluate:
        from evaluate import ModelEvaluator

        print("Transformer 모델 평가 모드")
        print("=" * 60)
        print(f"체크포인트: {args.evaluate}")
        print(f"데이터 타입: {args.eval_data_type}")
        print("=" * 60)

        # 평가자 생성
        evaluator = ModelEvaluator(args.evaluate)

        # 체크포인트 로드
        checkpoint = evaluator.load_checkpoint()

        # 토크나이저 로드
        evaluator.load_tokenizers()

        # 모델 구성
        evaluator.build_model(checkpoint)

        # 데이터 준비
        evaluator.prepare_data(args.eval_data_type)

        # 평가 실행
        results = evaluator.evaluate_full(args.eval_max_batches)

        # 몇 개 샘플 분석
        evaluator.evaluate_samples(num_samples=3)

        # 결과 저장
        checkpoint_name = os.path.basename(args.evaluate).replace(".pth", "")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"evaluation_{checkpoint_name}_{timestamp}"
        evaluator.save_results(results, output_dir)

        print(f"\n평가 완료! 결과는 {output_dir}에 저장되었습니다.")
        return

    # Config 파일 로드
    if args.config_path:
        config_path = args.config_path
        config_name = os.path.basename(config_path).replace(".json", "")
    else:
        config_name = args.config
        config_path = f"configs/{config_name}.json"

    if not os.path.exists(config_path):
        print(f"Error: Config 파일을 찾을 수 없습니다: {config_path}")
        print("사용 가능한 config 목록을 보려면 --list_configs 옵션을 사용하세요.")
        return

    try:
        config = load_config(config_path)
        print(f"Config 파일 로드 완료: {config_path}")
    except Exception as e:
        print(f"Error: Config 파일 로드 실패: {e}")
        return

    # 커맨드라인 인자로 config 덮어쓰기
    config = merge_config_with_args(config, args)

    # 저장 디렉토리 설정
    if args.save_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.save_dir = f"checkpoints_{config_name}_{timestamp}"

    print("Transformer 번역 모델 학습 시작")
    print("=" * 80)
    print(f"Config: {config_name} - {config.get('description', 'No description')}")
    print(f"학습 스텝: {config['training']['train_steps']}")
    print(f"배치 토큰 수: {config['training']['batch_tokens']}")
    print(f"학습률: {config['training']['learning_rate']}")
    print(f"모델 차원: {config['model']['d_model']}, 레이어: {config['model']['N']}")
    print(f"저장 디렉토리: {args.save_dir}")
    print("=" * 80)

    # 트레이너 생성
    trainer = TransformerTrainer(config)

    # 데이터 준비
    trainer.prepare_data()

    # 모델 구성
    trainer.build_model()

    # 학습 설정
    trainer.setup_training()

    # 체크포인트에서 재시작 여부 확인
    start_step = 0
    if args.checkpoint:
        print(f"\n🔄 체크포인트에서 재시작 모드")
        print(f"체크포인트 디렉토리: {args.checkpoint}")

        # 최신 체크포인트 찾기
        latest_checkpoint, latest_step = find_latest_checkpoint(args.checkpoint)

        if latest_checkpoint:
            print(
                f"✓ 최신 체크포인트 발견: {os.path.basename(latest_checkpoint)} (step {latest_step})"
            )

            # 체크포인트 로드 및 재시작
            start_step = load_and_resume_training(
                trainer, latest_checkpoint, args.resume_from_step
            )

            # 저장 디렉토리를 기존 체크포인트 디렉토리로 설정
            args.save_dir = args.checkpoint
            print(f"✓ 저장 디렉토리: {args.save_dir} (기존 체크포인트 디렉토리 사용)")

        else:
            print(
                f"⚠️  체크포인트 디렉토리 '{args.checkpoint}'에서 체크포인트를 찾을 수 없습니다."
            )
            print(f"   새로운 학습을 시작합니다.")

            # 디렉토리가 존재하지 않으면 생성
            os.makedirs(args.checkpoint, exist_ok=True)
            args.save_dir = args.checkpoint

    # 남은 학습 스텝 계산
    total_steps = config["training"]["train_steps"]
    remaining_steps = max(0, total_steps - start_step)

    if remaining_steps > 0:
        print(f"\n📈 학습 진행 정보:")
        print(f"   시작 스텝: {start_step}")
        print(f"   목표 스텝: {total_steps}")
        print(f"   남은 스텝: {remaining_steps}")
        print(f"   진행률: {start_step/total_steps*100:.1f}%")

        # 학습 시작
        steps, train_losses, val_losses = trainer.train(
            train_steps=total_steps, save_dir=args.save_dir
        )
    else:
        print(
            f"\n✅ 학습이 이미 완료되었습니다! (현재 스텝: {start_step}, 목표: {total_steps})"
        )
        return

    print(f"\n학습 완료! 결과는 {args.save_dir}에 저장되었습니다.")


def create_sample_config(config_name):
    """샘플 config 파일 생성"""
    sample_config = {
        "model": {
            "N": 6,
            "d_model": 512,
            "d_ff": 2048,
            "h": 8,
            "d_k": 64,
            "d_v": 64,
            "P_drop": 0.1,
            "max_seq_length": 128,
        },
        "training": {
            "train_steps": 10000,
            "batch_tokens": 25000,
            "learning_rate": 1e-4,
            "warmup_steps": 4000,
            "label_smoothing": 0.1,
            "grad_clip": 1.0,
            "eval_every": 500,
            "save_every": 1000,
        },
        "data": {"vocab_size": 5000, "max_length": 128, "data_multiplier": 10},
        "description": f"Custom config: {config_name}",
    }

    config_path = f"configs/{config_name}.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(sample_config, f, indent=2, ensure_ascii=False)

    print(f"샘플 config 파일이 생성되었습니다: {config_path}")
    print("파일을 편집하여 원하는 설정으로 변경하세요.")


if __name__ == "__main__":
    main()
