import sys
from pathlib import Path
import json
from tqdm import tqdm

from src.utils import load_pinyin_dict, viterbi_decode, evaluate
from src.train import train

def load_data(data_dir: Path):
    """加载所需的数据文件"""
    print('开始加载数据...', file=sys.stderr)
    
    # 加载拼音汉字映射表
    pinyin_dict = load_pinyin_dict(str(data_dir / '拼音汉字表.txt'))
    
    # 加载一元词频表
    unigram_freq = {}
    with open(data_dir / '1_word.txt', 'r', encoding='utf-8') as f:
        data = json.load(f)
        print('处理一元词频表...', file=sys.stderr)
        for pinyin, info in tqdm(data.items(), desc='处理一元词频表', file=sys.stderr):
            for char, count in zip(info['words'], info['counts']):
                unigram_freq[char] = count
    
    # 加载二元词频表
    bigram_freq = {}
    with open(data_dir / '2_word.txt', 'r', encoding='utf-8') as f:
        data = json.load(f)
        print('处理二元词频表...', file=sys.stderr)
        for pinyin_pair, info in tqdm(data.items(), desc='处理二元词频表', file=sys.stderr):
            char1, char2 = zip(*[word.split() for word in info['words']])
            for chars, count in zip(zip(char1, char2), info['counts']):
                bigram_freq[chars] = count
    
    return pinyin_dict, unigram_freq, bigram_freq

def main():
    # 设置数据目录路径
    data_dir = Path(__file__).parent / 'data'
    
    # 检查模型文件是否存在
    model_files_exist = (data_dir / '1_word.txt').exists() and (data_dir / '2_word.txt').exists()
    
    if not model_files_exist:
        # 如果模型文件不存在，训练模型
        print("模型文件不存在，开始训练模型...", file=sys.stderr)
        train()
        print("模型训练完成！", file=sys.stderr)
    else:
        print("发现已有模型文件，跳过训练步骤...", file=sys.stderr)
    
    # 加载训练好的模型数据
    pinyin_dict, unigram_freq, bigram_freq = load_data(data_dir)
    
    # 存储预测结果
    predictions = []
    
    # 标准输入输出模式
    for line in sys.stdin:
        pinyin_list = line.strip().split()
        result = viterbi_decode(pinyin_list, pinyin_dict, unigram_freq, bigram_freq)
        if not result:
            result = ''.join([pinyin_dict[p][0] if p in pinyin_dict else p for p in pinyin_list])
        predictions.append(result)
        print(result)
    
    # 评估模型性能
    try:
        with open(data_dir / 'answer.txt', 'r', encoding='utf-8') as f:
            answer_lines = f.readlines()
            answer_lines = [line.strip() for line in answer_lines]
            
            # 计算准确率
            char_precision, sent_precision = evaluate(predictions, answer_lines)
            
            # 将结果写入result.txt
            with open(data_dir / 'result.txt', 'w', encoding='utf-8') as f_result:
                f_result.write(f"字准确率: {char_precision:.2%}\n")
                f_result.write(f"句准确率: {sent_precision:.2%}\n")
                # f_result.write(f"得分: {5 * (sent_precision // 0.05):.1f}\n")
            
            # 同时输出到stderr
            print(f"\n评估结果：", file=sys.stderr)
            print(f"字准确率: {char_precision:.2%}", file=sys.stderr)
            print(f"句准确率: {sent_precision:.2%}", file=sys.stderr)
            # print(f"得分: {5 * (sent_precision // 0.05):.1f}", file=sys.stderr)
    except FileNotFoundError:
        print("\n未找到answer.txt文件，跳过评估步骤", file=sys.stderr)

if __name__ == '__main__':
    main()