"""
학습된 Transformer 모델을 사용한 번역 추론
"""
import torch
import torch.nn.functional as F
from src.model import Transformer
from src.data_utils import load_tokenizer

class TranslationInference:
    def __init__(self, model_path, src_tokenizer_path, tgt_tokenizer_path, device='cpu'):
        self.device = device
        
        # 토크나이저 로드
        self.src_tokenizer = load_tokenizer(src_tokenizer_path)
        self.tgt_tokenizer = load_tokenizer(tgt_tokenizer_path)
        
        # 특수 토큰 ID
        self.src_pad_id = self.src_tokenizer.token_to_id("[PAD]")
        self.tgt_pad_id = self.tgt_tokenizer.token_to_id("[PAD]")
        self.tgt_bos_id = self.tgt_tokenizer.token_to_id("[BOS]")
        self.tgt_eos_id = self.tgt_tokenizer.token_to_id("[EOS]")
        
        # 모델 로드
        src_vocab_size = self.src_tokenizer.get_vocab_size()
        tgt_vocab_size = self.tgt_tokenizer.get_vocab_size()
        
        self.model = Transformer(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            d_model=256,
            n_heads=8,
            n_layers=4,
            d_ff=1024,
            max_seq_length=128
        ).to(device)
        
        # 체크포인트 로드
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Model loaded from {model_path}")
        print(f"Device: {device}")
    
    def encode_text(self, text, max_length=128):
        """입력 텍스트를 토큰 ID로 변환"""
        encoding = self.src_tokenizer.encode(text)
        token_ids = encoding.ids[:max_length-1] + [self.src_tokenizer.token_to_id("[EOS]")]
        
        # 패딩
        if len(token_ids) < max_length:
            token_ids += [self.src_pad_id] * (max_length - len(token_ids))
        
        return torch.tensor(token_ids[:max_length], dtype=torch.long).unsqueeze(0)
    
    def decode_tokens(self, token_ids):
        """토큰 ID를 텍스트로 변환"""
        # 특수 토큰 제거
        tokens = []
        for token_id in token_ids:
            if token_id in [self.tgt_pad_id, self.tgt_bos_id, self.tgt_eos_id]:
                if token_id == self.tgt_eos_id:
                    break
                continue
            tokens.append(token_id)
        
        if tokens:
            return self.tgt_tokenizer.decode(tokens)
        return ""
    
    def greedy_decode(self, src_text, max_length=128):
        """Greedy 디코딩을 사용한 번역"""
        with torch.no_grad():
            # 소스 텍스트 인코딩
            src = self.encode_text(src_text, max_length).to(self.device)
            
            # 디코더 입력 초기화 (BOS 토큰으로 시작)
            tgt = torch.tensor([[self.tgt_bos_id]], dtype=torch.long).to(self.device)
            
            for _ in range(max_length - 1):
                # 현재까지의 출력으로 다음 토큰 예측
                output = self.model(src, tgt, src_pad_idx=self.src_pad_id, tgt_pad_idx=self.tgt_pad_id)
                
                # 마지막 위치의 예측 결과
                next_token_logits = output[0, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1)
                
                # 다음 토큰을 디코더 입력에 추가
                tgt = torch.cat([tgt, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
                
                # EOS 토큰이 생성되면 종료
                if next_token.item() == self.tgt_eos_id:
                    break
            
            # 결과 디코딩
            result = self.decode_tokens(tgt[0].cpu().numpy())
            return result
    
    def beam_search_decode(self, src_text, beam_size=3, max_length=128):
        """Beam search 디코딩을 사용한 번역"""
        with torch.no_grad():
            # 소스 텍스트 인코딩
            src = self.encode_text(src_text, max_length).to(self.device)
            
            # 빔 초기화
            beams = [(torch.tensor([[self.tgt_bos_id]], dtype=torch.long).to(self.device), 0.0)]
            
            for _ in range(max_length - 1):
                candidates = []
                
                for tgt, score in beams:
                    # 현재 빔에 대해 다음 토큰 예측
                    output = self.model(src, tgt, src_pad_idx=self.src_pad_id, tgt_pad_idx=self.tgt_pad_id)
                    next_token_logits = output[0, -1, :]
                    log_probs = F.log_softmax(next_token_logits, dim=-1)
                    
                    # 상위 beam_size개 토큰 선택
                    top_log_probs, top_indices = torch.topk(log_probs, beam_size)
                    
                    for i in range(beam_size):
                        next_token = top_indices[i].unsqueeze(0).unsqueeze(0)
                        new_tgt = torch.cat([tgt, next_token], dim=1)
                        new_score = score + top_log_probs[i].item()
                        candidates.append((new_tgt, new_score))
                
                # 상위 beam_size개 후보 선택
                candidates.sort(key=lambda x: x[1], reverse=True)
                beams = candidates[:beam_size]
                
                # 모든 빔이 EOS로 끝났는지 확인
                all_finished = all(beam[0][0, -1].item() == self.tgt_eos_id for beam in beams)
                if all_finished:
                    break
            
            # 최고 점수의 빔 선택
            best_beam = max(beams, key=lambda x: x[1])
            result = self.decode_tokens(best_beam[0][0].cpu().numpy())
            return result
    
    def translate(self, text, method='greedy'):
        """텍스트 번역"""
        if method == 'greedy':
            return self.greedy_decode(text)
        elif method == 'beam_search':
            return self.beam_search_decode(text)
        else:
            raise ValueError("Method must be 'greedy' or 'beam_search'")

def main():
    # 추론 객체 생성 (모델과 토크나이저 경로 지정)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        translator = TranslationInference(
            model_path='checkpoints/best_model.pth',
            src_tokenizer_path='tokenizers/src_tokenizer.json',
            tgt_tokenizer_path='tokenizers/tgt_tokenizer.json',
            device=device
        )
        
        # 테스트 번역
        test_sentences = [
            "Hello, how are you?",
            "I love machine learning.",
            "The weather is nice today.",
            "What time is it?",
            "Thank you very much."
        ]
        
        print("Translation Results:")
        print("=" * 50)
        
        for sentence in test_sentences:
            greedy_result = translator.translate(sentence, method='greedy')
            beam_result = translator.translate(sentence, method='beam_search')
            
            print(f"Source: {sentence}")
            print(f"Greedy: {greedy_result}")
            print(f"Beam Search: {beam_result}")
            print("-" * 30)
        
        # 인터랙티브 번역
        print("\\nInteractive Translation (type 'quit' to exit):")
        while True:
            user_input = input("Enter English text: ")
            if user_input.lower() == 'quit':
                break
            
            translation = translator.translate(user_input, method='beam_search')
            print(f"Translation: {translation}\\n")
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please make sure you have trained the model and saved the checkpoints and tokenizers.")

if __name__ == "__main__":
    main()
