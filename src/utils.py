import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import math
import os
from tqdm import tqdm
import json

def load_pinyin_dict(pinyin_table_path: str) -> Dict[str, List[str]]:
    """加载拼音汉字映射表"""
    pinyin_dict = {}
    try:
        with open(pinyin_table_path, 'r', encoding='gbk', errors='ignore') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    pinyin = parts[0]
                    chars = parts[1:]
                    if pinyin not in pinyin_dict:
                        pinyin_dict[pinyin] = []
                    pinyin_dict[pinyin].extend(chars)
    except Exception as e:
        print(f"加载拼音汉字映射表时出错: {str(e)}")
    return pinyin_dict

def train_language_model(corpus_path: str, char_set: set, models_to_train: set) -> Tuple[Dict[str, int], Dict[Tuple[str, str], int], Dict[Tuple[str, str, str], int]]:
    """训练语言模型，收集一元、二元和三元字频统计
    
    Args:
        corpus_path: 语料文件路径
        char_set: 合法汉字集合
        models_to_train: 需要训练的模型集合
    
    Returns:
        unigram_freq: 一元字频统计
        bigram_freq: 二元字频统计
        trigram_freq: 三元字频统计
    """
    unigram_freq = {}
    bigram_freq = {}
    trigram_freq = {}
    
    # 预先确定哪些模型需要训练，避免在循环中重复判断
    train_unigram = 1 in models_to_train
    train_bigram = 2 in models_to_train
    train_trigram = 3 in models_to_train
    
    try:
        with open(corpus_path, 'r', encoding='gbk', errors='ignore') as f:
            for line in f:
                # 修改：统计所有字符，但只保留合法汉字用于n-gram模型
                line_chars = line.strip()
                
                # 对所有字符进行一元统计（如果需要）
                if train_unigram:
                    for char in line_chars:
                        if char in char_set:  # 只统计合法汉字
                            unigram_freq[char] = unigram_freq.get(char, 0) + 1
                
                # 过滤后的字符列表，用于二元和三元模型
                chars = [c for c in line_chars if c in char_set]
                if not chars:
                    continue
                
                # 字符串长度，避免重复计算
                n = len(chars)
                
                if train_bigram and n > 1:
                    # 一次性处理所有二元组合
                    for i in range(n-1):
                        bigram = (chars[i], chars[i+1])
                        bigram_freq[bigram] = bigram_freq.get(bigram, 0) + 1
                
                if train_trigram and n > 2:
                    # 一次性处理所有三元组合
                    for i in range(n-2):
                        trigram = (chars[i], chars[i+1], chars[i+2])
                        trigram_freq[trigram] = trigram_freq.get(trigram, 0) + 1
    except Exception as e:
        print(f"处理文件 {corpus_path} 时出错: {str(e)}")
    
    return unigram_freq, bigram_freq, trigram_freq

def viterbi_decode(pinyin_list: List[str], pinyin_dict: Dict[str, List[str]], 
                  unigram_freq: Dict[str, int], bigram_freq: Dict[Tuple[str, str], int],
                  trigram_freq: Dict[Tuple[str, str, str], int] = None, use_trigram: bool = False,
                  lambda1: float = 0.0, lambda2: float = 0.5, lambda3: float = 0.5) -> str:
    """使用维特比算法进行解码
    
    Args:
        pinyin_list: 拼音列表
        pinyin_dict: 拼音到汉字的映射字典
        unigram_freq: 一元字频统计
        bigram_freq: 二元字频统计
        trigram_freq: 三元字频统计（可选）
        use_trigram: 是否使用三元模型
        lambda1: 一元模型权重
        lambda2: 二元模型权重
        lambda3: 三元模型权重
    
    Returns:
        解码后的汉字序列
    """
    if not pinyin_list:
        return ""
    
    # 获取总字符数用于计算概率
    total_chars = sum(unigram_freq.values())
    
    # 初始化
    V = [{}]  # Viterbi矩阵
    path = {}
    
    # 初始状态
    first_pinyin = pinyin_list[0]
    if first_pinyin not in pinyin_dict:
        return ""
        
    candidates = pinyin_dict[first_pinyin]
    
    for candidate in candidates:
        # 使用一元模型的概率作为初始概率
        count = unigram_freq.get(candidate, 0)
        # 使用对数概率避免下溢
        prob = math.log(count + 1) - math.log(total_chars + len(unigram_freq))  # 加1平滑
        
        V[0][candidate] = prob
        path[candidate] = [candidate]
    
    # 动态规划过程
    for t in range(1, len(pinyin_list)):
        V.append({})
        new_path = {}
        
        current_pinyin = pinyin_list[t]
        if current_pinyin not in pinyin_dict:
            continue
            
        candidates = pinyin_dict[current_pinyin]
        
        for candidate in candidates:
            max_prob = float('-inf')
            best_prev = None
            
            # 查找最佳前一个字符
            for prev_char in V[t-1]:
                # 计算转移概率
                if use_trigram and t > 1:
                    # 使用三元模型
                    prev_prev_char = path[prev_char][-2]  # 获取前前一个字符
                    
                    # 三元组: (prev_prev_char, prev_char, candidate)
                    trigram_key = (prev_prev_char, prev_char, candidate)
                    trigram_count = trigram_freq.get(trigram_key, 0)
                    
                    # 二元组1: (prev_prev_char, prev_char)
                    bigram_key1 = (prev_prev_char, prev_char)
                    bigram_count1 = bigram_freq.get(bigram_key1, 0)
                    
                    # 二元组2: (prev_char, candidate)
                    bigram_key2 = (prev_char, candidate)
                    bigram_count2 = bigram_freq.get(bigram_key2, 0)
                    
                    # 计算三元概率 P(candidate|prev_prev_char,prev_char)
                    if bigram_count1 > 0:
                        # 条件概率: P(candidate|prev_prev_char,prev_char) = P(prev_prev_char,prev_char,candidate) / P(prev_prev_char,prev_char)
                        trigram_prob = math.log(trigram_count + 1) - math.log(bigram_count1 + len(unigram_freq))
                    else:
                        # 平滑处理
                        trigram_prob = math.log(1) - math.log(len(unigram_freq))
                    
                    # 计算二元概率 P(candidate|prev_char)
                    prev_count = unigram_freq.get(prev_char, 0)
                    if prev_count > 0:
                        # 条件概率: P(candidate|prev_char) = P(prev_char,candidate) / P(prev_char)
                        bigram_prob = math.log(bigram_count2 + 1) - math.log(prev_count + len(unigram_freq))
                    else:
                        # 平滑处理
                        bigram_prob = math.log(1) - math.log(len(unigram_freq))
                    
                    # 计算一元概率 P(candidate)
                    unigram_prob = math.log(unigram_freq.get(candidate, 0) + 1) - math.log(total_chars + len(unigram_freq))
                    
                    # 插值计算最终概率：λ1*P(w3) + λ2*P(w3|w2) + λ3*P(w3|w1,w2)
                    trans_prob = (lambda1 * unigram_prob + 
                                lambda2 * bigram_prob + 
                                lambda3 * trigram_prob)
                else:
                    # 使用二元模型
                    bigram_key = (prev_char, candidate)
                    trans_count = bigram_freq.get(bigram_key, 0)
                    prev_count = unigram_freq.get(prev_char, 0)
                    
                    # 条件概率 P(candidate|prev_char) = P(prev_char,candidate) / P(prev_char)
                    if prev_count > 0:
                        trans_prob = math.log(trans_count + 1) - math.log(prev_count + len(unigram_freq))
                    else:
                        trans_prob = math.log(1) - math.log(len(unigram_freq))  # 平滑处理
                
                # 计算当前概率
                prob = V[t-1][prev_char] + trans_prob
                
                if prob > max_prob:
                    max_prob = prob
                    best_prev = prev_char
            
            if best_prev is not None:
                V[t][candidate] = max_prob
                new_path[candidate] = path[best_prev] + [candidate]
        
        path = new_path
    
    # 找出最优路径
    if not V[-1]:
        return ""
        
    max_prob = float('-inf')
    best_last_char = None
    
    for last_char, prob in V[-1].items():
        if prob > max_prob:
            max_prob = prob
            best_last_char = last_char
    
    if best_last_char is None:
        return ""
    
    return ''.join(path[best_last_char])

def evaluate(pred_lines: List[str], true_lines: List[str]) -> Tuple[float, float]:
    """评估模型性能"""
    correct_chars = 0
    total_chars = 0
    correct_sents = 0
    total_sents = 0
    
    for pred, true in zip(pred_lines, true_lines):
        pred = pred.strip()
        true = true.strip()
        
        # 计算字准确率
        for p_char, t_char in zip(pred, true):
            if p_char == t_char:
                correct_chars += 1
            total_chars += 1
        
        # 计算句准确率
        if pred == true:
            correct_sents += 1
        total_sents += 1
    
    char_precision = correct_chars / total_chars if total_chars > 0 else 0
    sent_precision = correct_sents / total_sents if total_sents > 0 else 0
    
    return char_precision, sent_precision
