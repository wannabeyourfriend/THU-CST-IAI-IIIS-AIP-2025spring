import sys
import json
import math
import gc  

def log_memory(message):
    pass

def initialize_language_model():
    log_memory("初始化开始")
    
    pronunciation_mapping = {}
    with open('./word2pinyin.txt', 'r', encoding='utf-8') as input_file:
        for text_line in input_file:
            tokens = text_line.strip().split()
            if len(tokens) >= 2:
                hanzi, pronunciation = tokens[0], tokens[1]
                if pronunciation not in pronunciation_mapping:
                    pronunciation_mapping[pronunciation] = []
                pronunciation_mapping[pronunciation].append(hanzi)
    
    log_memory("读取拼音映射后")
    single_char_statistics = {}
    try:
        with open('./1_word.txt', 'r', encoding='utf-8') as input_file:
            raw_data = json.load(input_file)
            for pinyin_key, frequency_info in raw_data.items():
                for i in range(len(frequency_info['words'])):
                    single_char_statistics[frequency_info['words'][i]] = frequency_info['counts'][i]
    except json.JSONDecodeError:
        with open('./1_word.txt', 'r', encoding='utf-8') as input_file:
            content = input_file.read()
            try:
                raw_data = json.loads(content)
                for pinyin_key, frequency_info in raw_data.items():
                    for i in range(len(frequency_info['words'])):
                        single_char_statistics[frequency_info['words'][i]] = frequency_info['counts'][i]
            except json.JSONDecodeError:
                pass
    for char in list(single_char_statistics.keys()):
        if single_char_statistics[char] <= 0:
            single_char_statistics[char] = 1
    
    log_memory("读取单字统计后")
    char_pair_statistics = {}
    try:
        with open('./2_word.txt', 'r', encoding='utf-8') as input_file:
            raw_data = json.load(input_file)
            for pinyin_pair, frequency_info in raw_data.items():
                words = frequency_info['words']
                counts = frequency_info['counts']
                for i in range(len(words)):
                    chars = words[i].split()
                    if len(chars) == 2:
                        char_pair_statistics[chars[0] + "|" + chars[1]] = max(1, counts[i])  # 确保频率为正
    except json.JSONDecodeError:
        with open('./2_word.txt', 'r', encoding='utf-8') as input_file:
            content = input_file.read()
            try:
                raw_data = json.loads(content)
                for pinyin_pair, frequency_info in raw_data.items():
                    words = frequency_info['words']
                    counts = frequency_info['counts']
                    for i in range(len(words)):
                        chars = words[i].split()
                        if len(chars) == 2:
                            char_pair_statistics[chars[0] + "|" + chars[1]] = max(1, counts[i])
            except json.JSONDecodeError:
                pass
    gc.collect()
    
    log_memory("读取二元组统计后")
    return pronunciation_mapping, single_char_statistics, char_pair_statistics

def decode_pinyin_sequence(pinyin_sequence, pronunciation_map, char_freq, pair_freq):
    if not pinyin_sequence:
        return ""
    total_frequency = max(1, sum(char_freq.values()))
    vocab_size = len(char_freq)
    prev_probs = {}
    prev_paths = {}
    
    initial_pinyin = pinyin_sequence[0]
    if initial_pinyin not in pronunciation_map:
        return ""
        
    possible_chars = pronunciation_map[initial_pinyin]
    
    for candidate_char in possible_chars:
        char_count = max(1, char_freq.get(candidate_char, 0))
        log_probability = math.log(char_count) - math.log(total_frequency + vocab_size)
        prev_probs[candidate_char] = log_probability
        prev_paths[candidate_char] = candidate_char
    
    for position in range(1, len(pinyin_sequence)):
        current_pinyin = pinyin_sequence[position]
        if current_pinyin not in pronunciation_map:
            continue
        curr_probs = {}
        curr_paths = {}
        
        possible_chars = pronunciation_map[current_pinyin]
        
        for candidate_char in possible_chars:
            highest_prob = float('-inf')
            optimal_previous = None
            
            for previous_char in prev_probs:
                pair_key = previous_char + "|" + candidate_char
                transition_count = max(1, pair_freq.get(pair_key, 0))
                previous_count = max(1, char_freq.get(previous_char, 0))
                transition_prob = math.log(transition_count) - math.log(previous_count + vocab_size)
                current_prob = prev_probs[previous_char] + transition_prob
                
                if current_prob > highest_prob:
                    highest_prob = current_prob
                    optimal_previous = previous_char
            
            if optimal_previous is not None:
                curr_probs[candidate_char] = highest_prob
                curr_paths[candidate_char] = prev_paths[optimal_previous] + candidate_char
        if not curr_probs:
            return ""
        prev_probs = curr_probs
        prev_paths = curr_paths
    
    if not prev_probs:
        return ""
    highest_prob = float('-inf')
    optimal_final_char = None
    
    for final_char, probability in prev_probs.items():
        if probability > highest_prob:
            highest_prob = probability
            optimal_final_char = final_char
    
    if optimal_final_char is None:
        return ""
    return prev_paths[optimal_final_char]

def process_input_stream():
    log_memory("处理输入前")
    pronunciation_map, char_freq, pair_freq = initialize_language_model()
    log_memory("加载模型后")
    for input_line in sys.stdin:
        pinyin_sequence = input_line.strip().split()
        conversion_result = decode_pinyin_sequence(pinyin_sequence, pronunciation_map, char_freq, pair_freq)
        if conversion_result:
            print(conversion_result)
        else:
            fallback_result = ""
            for pinyin in pinyin_sequence:
                if pinyin in pronunciation_map and pronunciation_map[pinyin]:
                    fallback_result += pronunciation_map[pinyin][0]
                else:
                    fallback_result += pinyin
            print(fallback_result)
        gc.collect()

if __name__ == '__main__':
    process_input_stream()