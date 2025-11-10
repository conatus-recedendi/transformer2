"""
실제 데이터 로딩을 위한 모듈
./data/{problem}/ 구조의 실제 데이터 파일들을 로드
"""

import os
import logging
from typing import List, Tuple, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


def load_language_file(file_path: str, max_lines: Optional[int] = None) -> List[str]:
    """언어 파일 로드 (train.en, train.de 등)"""
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return []
    
    sentences = []
    try:
        with open(file_path, 'r', encoding='utf-8', newline='\n') as f:
            for line_num, line in enumerate(f, 1):
                if max_lines and line_num > max_lines:
                    break
                    
                line = line.strip()
                if line:  # 빈 줄 제외
                    sentences.append(line)
                    
                # 진행 상황 로깅
                if line_num % 100000 == 0:
                    logger.info(f"  Loaded {line_num:,} lines from {file_path}")
        
        logger.info(f"Successfully loaded {len(sentences):,} sentences from {file_path}")
        
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        return []
    
    return sentences


def load_translation_pairs(
    src_file: str, 
    tgt_file: str, 
    max_lines: Optional[int] = None
) -> Tuple[List[str], List[str]]:
    """번역 쌍 로드 (소스-타겟 파일)"""
    logger.info(f"Loading translation pairs:")
    logger.info(f"  Source: {src_file}")
    logger.info(f"  Target: {tgt_file}")
    
    # 파일 존재 확인
    if not os.path.exists(src_file):
        logger.error(f"Source file not found: {src_file}")
        return [], []
    
    if not os.path.exists(tgt_file):
        logger.error(f"Target file not found: {tgt_file}")
        return [], []
    
    # 파일 라인 수 확인
    def count_lines(file_path):
        with open(file_path, 'rb') as f:
            return f.read().count(b'\n')
    
    src_lines = count_lines(src_file)
    tgt_lines = count_lines(tgt_file)
    
    logger.info(f"File line counts:")
    logger.info(f"  {src_file}: {src_lines:,} lines")
    logger.info(f"  {tgt_file}: {tgt_lines:,} lines")
    
    if src_lines != tgt_lines:
        logger.warning(f"Line count mismatch! Source: {src_lines}, Target: {tgt_lines}")
        logger.warning("Will load up to the minimum count")
    
    # 데이터 로드
    src_sentences = load_language_file(src_file, max_lines)
    tgt_sentences = load_language_file(tgt_file, max_lines)
    
    # 길이 맞추기
    min_len = min(len(src_sentences), len(tgt_sentences))
    if len(src_sentences) != len(tgt_sentences):
        logger.warning(f"Trimming to {min_len:,} pairs to match lengths")
        src_sentences = src_sentences[:min_len]
        tgt_sentences = tgt_sentences[:min_len]
    
    logger.info(f"Final loaded pairs: {len(src_sentences):,}")
    
    return src_sentences, tgt_sentences


def load_problem_data(config: dict) -> Tuple[
    Tuple[List[str], List[str]],  # train
    Tuple[List[str], List[str]],  # valid  
    Tuple[List[str], List[str]]   # test
]:
    """문제별 데이터 로드"""
    data_config = config['data']
    problem = data_config['problem']
    src_lang = data_config['src_lang']
    tgt_lang = data_config['tgt_lang']
    data_dir = data_config['data_dir']
    
    problem_dir = Path(data_dir) / problem
    
    logger.info(f"Loading problem data: {problem}")
    logger.info(f"Language pair: {src_lang} -> {tgt_lang}")
    logger.info(f"Data directory: {problem_dir}")
    
    if not problem_dir.exists():
        logger.error(f"Problem directory not found: {problem_dir}")
        return ([], []), ([], []), ([], [])
    
    # 파일 경로 생성
    train_src = problem_dir / f"train.{src_lang}"
    train_tgt = problem_dir / f"train.{tgt_lang}"
    valid_src = problem_dir / f"valid.{src_lang}"
    valid_tgt = problem_dir / f"valid.{tgt_lang}"
    test_src = problem_dir / f"test.{src_lang}"
    test_tgt = problem_dir / f"test.{tgt_lang}"
    
    # 데이터 로드
    logger.info("=" * 50)
    logger.info("Loading TRAIN data...")
    train_data = load_translation_pairs(str(train_src), str(train_tgt))
    
    logger.info("=" * 50)
    logger.info("Loading VALID data...")
    valid_data = load_translation_pairs(str(valid_src), str(valid_tgt))
    
    logger.info("=" * 50)
    logger.info("Loading TEST data...")
    test_data = load_translation_pairs(str(test_src), str(test_tgt))
    
    logger.info("=" * 50)
    logger.info("Data loading summary:")
    logger.info(f"  Train pairs: {len(train_data[0]):,}")
    logger.info(f"  Valid pairs: {len(valid_data[0]):,}")
    logger.info(f"  Test pairs: {len(test_data[0]):,}")
    
    return train_data, valid_data, test_data


def get_bpe_training_files(config: dict) -> List[str]:
    """BPE 학습용 파일 경로 반환"""
    data_config = config['data']
    problem = data_config['problem']
    src_lang = data_config['src_lang']
    tgt_lang = data_config['tgt_lang']
    data_dir = data_config['data_dir']
    
    problem_dir = Path(data_dir) / problem
    
    # BPE 학습에 사용할 파일들 (train 데이터만 사용)
    bpe_files = [
        str(problem_dir / f"train.{src_lang}"),
        str(problem_dir / f"train.{tgt_lang}")
    ]
    
    # 파일 존재 확인
    existing_files = []
    for file_path in bpe_files:
        if os.path.exists(file_path):
            existing_files.append(file_path)
        else:
            logger.warning(f"BPE training file not found: {file_path}")
    
    logger.info(f"BPE training files: {existing_files}")
    return existing_files


def clean_sentence_pairs(src_sentences: List[str], tgt_sentences: List[str]) -> Tuple[List[str], List[str]]:
    """Tensor2Tensor 스타일 데이터 클리닝"""
    logger.info("Applying Tensor2Tensor data cleaning rules...")
    
    cleaned_src = []
    cleaned_tgt = []
    
    original_count = len(src_sentences)
    
    for src, tgt in zip(src_sentences, tgt_sentences):
        # 1. 빈 문장 제거
        if not src.strip() or not tgt.strip():
            continue
            
        # 2. 너무 짧은 문장 제거 (3글자 미만)
        if len(src.strip()) < 3 or len(tgt.strip()) < 3:
            continue
            
        # 3. 너무 긴 문장 제거 (1000자 초과)
        if len(src) > 1000 or len(tgt) > 1000:
            continue
            
        # 4. 길이 비율 체크 (1:9 비율 초과하는 경우 제거)
        src_len = len(src.split())
        tgt_len = len(tgt.split())
        
        if src_len > 0 and tgt_len > 0:
            ratio = max(src_len, tgt_len) / min(src_len, tgt_len)
            if ratio > 9.0:
                continue
        
        # 5. 특수 문자만으로 구성된 문장 제거
        if src.strip().replace(' ', '').replace('.', '').replace(',', '').replace('!', '').replace('?', '') == '':
            continue
        if tgt.strip().replace(' ', '').replace('.', '').replace(',', '').replace('!', '').replace('?', '') == '':
            continue
            
        cleaned_src.append(src)
        cleaned_tgt.append(tgt)
    
    cleaned_count = len(cleaned_src)
    removed_count = original_count - cleaned_count
    
    logger.info(f"Data cleaning results:")
    logger.info(f"  Original pairs: {original_count:,}")
    logger.info(f"  Cleaned pairs: {cleaned_count:,}")
    logger.info(f"  Removed pairs: {removed_count:,} ({removed_count/original_count*100:.1f}%)")
    
    return cleaned_src, cleaned_tgt


def create_data_sample_for_testing(data_dir: str = "./data", problem: str = "sample_en_de"):
    """테스트용 샘플 데이터 생성"""
    problem_dir = Path(data_dir) / problem
    problem_dir.mkdir(parents=True, exist_ok=True)
    
    # 샘플 데이터
    en_sentences = [
        "Hello, how are you?",
        "I love machine learning.",
        "The weather is nice today.",
        "What time is it?",
        "I am studying artificial intelligence.",
        "This is a beautiful day.",
        "Can you help me?",
        "I enjoy reading books.",
        "The food was delicious.",
        "Thank you very much."
    ] * 100  # 1000개로 확장
    
    de_sentences = [
        "Hallo, wie geht es dir?",
        "Ich liebe maschinelles Lernen.",
        "Das Wetter ist heute schön.",
        "Wie spät ist es?",
        "Ich studiere künstliche Intelligenz.",
        "Das ist ein schöner Tag.",
        "Können Sie mir helfen?",
        "Ich lese gerne Bücher.",
        "Das Essen war lecker.",
        "Vielen Dank."
    ] * 100  # 1000개로 확장
    
    # 파일 저장
    splits = ['train', 'valid', 'test']
    split_sizes = [800, 100, 100]  # 8:1:1 비율
    
    start_idx = 0
    for split, size in zip(splits, split_sizes):
        end_idx = start_idx + size
        
        # 영어 파일
        with open(problem_dir / f"{split}.en", 'w', encoding='utf-8') as f:
            for sentence in en_sentences[start_idx:end_idx]:
                f.write(sentence + '\n')
        
        # 독일어 파일  
        with open(problem_dir / f"{split}.de", 'w', encoding='utf-8') as f:
            for sentence in de_sentences[start_idx:end_idx]:
                f.write(sentence + '\n')
        
        start_idx = end_idx
    
    logger.info(f"Sample data created in {problem_dir}")
    logger.info(f"  Train: {split_sizes[0]} pairs")
    logger.info(f"  Valid: {split_sizes[1]} pairs") 
    logger.info(f"  Test: {split_sizes[2]} pairs")
