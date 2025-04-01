import os
import json
from pathlib import Path
from tqdm import tqdm
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
# 修复导入问题
try:
    from .utils import train_language_model, load_pinyin_dict
except ImportError:
    from utils import train_language_model, load_pinyin_dict

def load_sina_news_corpus(corpus_dir: str, char_set: set, models_to_train: set):
    """加载新浪新闻语料
    
    Args:
        corpus_dir: 语料目录路径
        char_set: 合法汉字集合
        models_to_train: 需要训练的模型集合
    """
    unigram_freq = {}
    bigram_freq = {}
    trigram_freq = {}
    
    # 确保目录存在
    if not os.path.exists(corpus_dir):
        print(f"警告: 语料目录 {corpus_dir} 不存在!")
        return unigram_freq, bigram_freq, trigram_freq
    
    files = [f for f in os.listdir(corpus_dir) if f.endswith('.txt')]
    if not files:
        print(f"警告: 在 {corpus_dir} 中没有找到任何txt文件!")
        return unigram_freq, bigram_freq, trigram_freq
        
    file_paths = [os.path.join(corpus_dir, f) for f in files]
    
    # 使用多进程并行处理文件
    num_cores = max(1, multiprocessing.cpu_count() - 1)  # 留一个核心给系统
    print(f"使用 {num_cores} 个进程并行处理语料...")
    
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        # 创建处理单个文件的任务
        future_to_file = {
            executor.submit(
                train_language_model, file_path, char_set, models_to_train
            ): file_path for file_path in file_paths
        }
        
        # 使用tqdm显示进度
        for future in tqdm(future_to_file, desc='并行处理新浪新闻语料'):
            file_path = future_to_file[future]
            try:
                uni_freq, bi_freq, tri_freq = future.result()
                
                # 合并频率统计
                if 1 in models_to_train:
                    for char, freq in uni_freq.items():
                        unigram_freq[char] = unigram_freq.get(char, 0) + freq
                if 2 in models_to_train:
                    for bigram, freq in bi_freq.items():
                        bigram_freq[bigram] = bigram_freq.get(bigram, 0) + freq
                if 3 in models_to_train:
                    for trigram, freq in tri_freq.items():
                        trigram_freq[trigram] = trigram_freq.get(trigram, 0) + freq
            except Exception as e:
                print(f"处理文件 {file_path} 时出错: {e}")
    
    # 检查是否成功统计了频率
    print(f"一元模型统计了 {len(unigram_freq)} 个字符")
    print(f"二元模型统计了 {len(bigram_freq)} 个二元组")
    print(f"三元模型统计了 {len(trigram_freq)} 个三元组")
    
    return unigram_freq, bigram_freq, trigram_freq

def save_model(unigram_freq: dict, bigram_freq: dict, trigram_freq: dict, output_dir: str, pinyin_to_chars: dict, char_to_pinyin: dict, models_to_train: set):
    """保存训练好的模型
    
    Args:
        unigram_freq: 一元字频统计
        bigram_freq: 二元字频统计
        trigram_freq: 三元字频统计
        output_dir: 输出目录
        pinyin_to_chars: 拼音到汉字的映射
        char_to_pinyin: 汉字到拼音的映射
        models_to_train: 需要训练的模型集合
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存一元字频
    if 1 in models_to_train:
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
        unigram_path = os.path.join(output_dir, '1_word.txt')
        with open(unigram_path, 'w', encoding='utf-8') as f:
            json.dump(unigram_data, f, ensure_ascii=False, indent=2)
        print(f"一元模型已保存到: {unigram_path}")
    
    # 保存二元字频
    if 2 in models_to_train:
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
        bigram_path = os.path.join(output_dir, '2_word.txt')
        with open(bigram_path, 'w', encoding='utf-8') as f:
            json.dump(bigram_data, f, ensure_ascii=False, indent=2)
        print(f"二元模型已保存到: {bigram_path}")
    
    # 保存三元字频
    if 3 in models_to_train:
        trigram_data = {}
        
        # 处理三元模型
        for (char1, char2, char3), freq in trigram_freq.items():
            # 获取汉字对应的拼音，如果没有则跳过
            if char1 not in char_to_pinyin or char2 not in char_to_pinyin or char3 not in char_to_pinyin:
                continue
                
            pinyin1 = char_to_pinyin[char1]
            pinyin2 = char_to_pinyin[char2]
            pinyin3 = char_to_pinyin[char3]
            
            # 拼音三元组作为键
            key = f"{pinyin1} {pinyin2} {pinyin3}"
            
            if key not in trigram_data:
                trigram_data[key] = {"words": [], "counts": []}
            
            # 汉字三元组作为值
            word_triple = f"{char1} {char2} {char3}"
            trigram_data[key]["words"].append(word_triple)
            trigram_data[key]["counts"].append(freq)
        
        # 保存三元模型
        trigram_path = os.path.join(output_dir, '3_word.txt')
        with open(trigram_path, 'w', encoding='utf-8') as f:
            json.dump(trigram_data, f, ensure_ascii=False, indent=2)
        print(f"三元模型已保存到: {trigram_path}")

# 删除重复的load_pinyin_dict函数，使用utils.py中的函数
def build_char_to_pinyin(pinyin_to_chars):
    """从拼音到汉字的映射构建汉字到拼音的映射"""
    char_to_pinyin = {}
    for pinyin, chars in pinyin_to_chars.items():
        for char in chars:
            char_to_pinyin[char] = pinyin
    return char_to_pinyin

def train(models_to_train: set):
    """训练模型主函数
    
    Args:
        models_to_train: 需要训练的模型集合
    """
    print('开始训练模型...')
    
    # 设置路径
    project_root = Path(__file__).parent.parent
    data_dir = project_root / 'data'
    corpus_dir = project_root / 'corpus' / 'sina_news_gbk'
    
    print(f"项目根目录: {project_root}")
    print(f"数据目录: {data_dir}")
    print(f"语料库目录: {corpus_dir}")
    
    # 确保目录存在
    if not os.path.exists(data_dir):
        print(f"警告: 数据目录 {data_dir} 不存在!")
        os.makedirs(data_dir, exist_ok=True)
    
    pinyin_table_path = str(data_dir / '拼音汉字表.txt')
    hanzi_table_path = str(data_dir / '一二级汉字表.txt')
    
    # 检查文件是否存在
    if not os.path.exists(pinyin_table_path):
        print(f"错误: 拼音汉字表文件 {pinyin_table_path} 不存在!")
        return
    
    if not os.path.exists(hanzi_table_path):
        print(f"错误: 一二级汉字表文件 {hanzi_table_path} 不存在!")
        return
    
    # 加载拼音字典
    pinyin_to_chars = load_pinyin_dict(pinyin_table_path)
    char_to_pinyin = build_char_to_pinyin(pinyin_to_chars)
    
    print(f"加载了 {len(pinyin_to_chars)} 个拼音和 {len(char_to_pinyin)} 个汉字")
    
    # 构建合法汉字集合
    char_set = set()
    try:
        with open(hanzi_table_path, 'r', encoding='gbk') as f:
            for line in f:
                chars = line.strip()
                char_set.update(chars)
        print(f"从一二级汉字表加载了 {len(char_set)} 个合法汉字")
    except Exception as e:
        print(f"加载一二级汉字表时出错: {e}")
        return
    
    # 训练模型
    unigram_freq, bigram_freq, trigram_freq = load_sina_news_corpus(str(corpus_dir), char_set, models_to_train)
    
    # 检查是否成功统计了频率
    if not unigram_freq and 1 in models_to_train:
        print("警告: 一元模型没有统计到任何字符!")
    if not bigram_freq and 2 in models_to_train:
        print("警告: 二元模型没有统计到任何二元组!")
    if not trigram_freq and 3 in models_to_train:
        print("警告: 三元模型没有统计到任何三元组!")
    
    # 保存模型
    save_model(unigram_freq, bigram_freq, trigram_freq, str(data_dir), pinyin_to_chars, char_to_pinyin, models_to_train)
    print('模型训练完成！')

if __name__ == '__main__':
    # 创建一个包含所有模型的集合，表示需要训练所有模型
    train({1, 2, 3})