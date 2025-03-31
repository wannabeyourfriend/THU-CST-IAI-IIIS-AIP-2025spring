import os
import json
from pathlib import Path
from tqdm import tqdm
from .utils import train_bigram_model

def load_sina_news_corpus(corpus_dir: str, char_set: set):
    """加载新浪新闻语料"""
    unigram_freq = {}
    bigram_freq = {}
    
    files = [f for f in os.listdir(corpus_dir) if f.endswith('.txt')]
    for file_name in tqdm(files, desc='处理新浪新闻语料'):
        file_path = os.path.join(corpus_dir, file_name)
        uni_freq, bi_freq = train_bigram_model(file_path, char_set)
        
        # 合并频率统计
        for char, freq in uni_freq.items():
            unigram_freq[char] = unigram_freq.get(char, 0) + freq
        for bigram, freq in bi_freq.items():
            bigram_freq[bigram] = bigram_freq.get(bigram, 0) + freq
    
    return unigram_freq, bigram_freq

def save_model(unigram_freq: dict, bigram_freq: dict, output_dir: str, pinyin_to_chars: dict, char_to_pinyin: dict):
    """保存训练好的模型
    
    Args:
        unigram_freq: 一元字频统计
        bigram_freq: 二元字频统计
        output_dir: 输出目录
        pinyin_to_chars: 拼音到汉字的映射
        char_to_pinyin: 汉字到拼音的映射
    """
    # 保存一元字频
    unigram_data = {}
    
    # 按拼音组织数据
    for pinyin, chars in pinyin_to_chars.items():
        if pinyin not in unigram_data:
            unigram_data[pinyin] = {"words": [], "counts": []}
        
        # 对于每个拼音下的汉字，记录其频率
        for char in chars:
            if char in unigram_freq:
                unigram_data[pinyin]["words"].append(char)
                unigram_data[pinyin]["counts"].append(unigram_freq[char])
    
    # 保存一元模型
    with open(os.path.join(output_dir, '1_word.txt'), 'w', encoding='utf-8') as f:
        json.dump(unigram_data, f, ensure_ascii=False, indent=2)
    
    # 保存二元字频
    bigram_data = {}
    
    # 处理二元模型
    for (char1, char2), freq in bigram_freq.items():
        # 获取汉字对应的拼音，如果没有则跳过
        if char1 not in char_to_pinyin or char2 not in char_to_pinyin:
            continue
            
        pinyin1 = char_to_pinyin[char1]
        pinyin2 = char_to_pinyin[char2]
        
        # 拼音对作为键
        key = f"{pinyin1} {pinyin2}"
        
        if key not in bigram_data:
            bigram_data[key] = {"words": [], "counts": []}
        
        # 汉字对作为值
        word_pair = f"{char1} {char2}"
        bigram_data[key]["words"].append(word_pair)
        bigram_data[key]["counts"].append(freq)
    
    # 保存二元模型
    with open(os.path.join(output_dir, '2_word.txt'), 'w', encoding='utf-8') as f:
        json.dump(bigram_data, f, ensure_ascii=False, indent=2)

def load_pinyin_dict(pinyin_file_path: str):
    """加载拼音汉字映射表
    
    Args:
        pinyin_file_path: 拼音汉字映射表文件路径
    
    Returns:
        pinyin_to_chars: 拼音到汉字的映射字典
        char_to_pinyin: 汉字到拼音的映射字典
    """
    pinyin_to_chars = {}
    char_to_pinyin = {}
    
    with open(pinyin_file_path, 'r', encoding='gbk') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                pinyin = parts[0]
                chars = parts[1:]
                
                # 构建拼音到汉字的映射
                if pinyin not in pinyin_to_chars:
                    pinyin_to_chars[pinyin] = []
                pinyin_to_chars[pinyin].extend(chars)
                
                # 构建汉字到拼音的映射
                for char in chars:
                    char_to_pinyin[char] = pinyin
    
    return pinyin_to_chars, char_to_pinyin

def train():
    """训练模型主函数"""
    print('开始训练模型...')
    
    # 设置路径
    project_root = Path(__file__).parent.parent
    data_dir = project_root / 'data'
    corpus_dir = project_root / 'corpus' / 'sina_news_gbk'
    
    # 加载拼音字典和构建字符集
    pinyin_to_chars, char_to_pinyin = load_pinyin_dict(str(data_dir / '拼音汉字表.txt'))
    
    # 构建合法汉字集合
    char_set = set()
    for chars in pinyin_to_chars.values():
        char_set.update(chars)
    
    # 训练模型
    unigram_freq, bigram_freq = load_sina_news_corpus(str(corpus_dir), char_set)
    
    # 保存模型
    save_model(unigram_freq, bigram_freq, str(data_dir), pinyin_to_chars, char_to_pinyin)
    print('模型训练完成！')

if __name__ == '__main__':
    train()