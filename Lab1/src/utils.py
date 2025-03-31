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
    with open(pinyin_table_path, 'r', encoding='gbk') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                pinyin = parts[0]
                chars = parts[1:]
                if pinyin not in pinyin_dict:
                    pinyin_dict[pinyin] = []
                pinyin_dict[pinyin].extend(chars)
    return pinyin_dict

def train_bigram_model(corpus_path: str, char_set: set) -> Tuple[Dict[str, int], Dict[Tuple[str, str], int]]:
    """训练二元字模型"""
    unigram_freq = {}
    bigram_freq = {}
    
    with open(corpus_path, 'r', encoding='gbk', errors='ignore') as f:
        for line in f:
            chars = [c for c in line.strip() if c in char_set]
            if not chars:
                continue
            
            # 统计一元字频
            for char in chars:
                unigram_freq[char] = unigram_freq.get(char, 0) + 1
            
            # 统计二元字频
            for i in range(len(chars)-1):
                bigram = (chars[i], chars[i+1])
                bigram_freq[bigram] = bigram_freq.get(bigram, 0) + 1
    
    return unigram_freq, bigram_freq

def viterbi_decode(pinyin_list: List[str], pinyin_dict: Dict[str, List[str]], 
                  unigram_freq: Dict[str, int], bigram_freq: Dict[Tuple[str, str], int]) -> str:
    """使用维特比算法进行解码"""
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
                # 计算转移概率 (二元模型)
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
