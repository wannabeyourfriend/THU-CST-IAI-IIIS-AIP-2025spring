import sys
from pathlib import Path
import json
from tqdm import tqdm
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import datetime

from src.utils import load_pinyin_dict, viterbi_decode, evaluate
from src.train import train

# 设置日志文件
log_file = Path(__file__).parent / 'performance_log.txt'

def log_message(message):
    """写入日志消息"""
    with open(log_file, 'a', encoding='utf-8') as f:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{timestamp}] {message}\n")
    print(message, file=sys.stderr)

def load_data(data_dir: Path, use_trigram: bool = False):
    """加载所需的数据文件
    
    Args:
        data_dir: 数据目录路径
        use_trigram: 是否使用三元模型
    """
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
    
    # 加载三元词频表（如果启用）
    trigram_freq = {}
    if use_trigram:
        try:
            with open(data_dir / '3_word.txt', 'r', encoding='utf-8') as f:
                data = json.load(f)
                print('处理三元词频表...', file=sys.stderr)
                for pinyin_triple, info in tqdm(data.items(), desc='处理三元词频表', file=sys.stderr):
                    char1, char2, char3 = zip(*[word.split() for word in info['words']])
                    for chars, count in zip(zip(char1, char2, char3), info['counts']):
                        trigram_freq[chars] = count
        except FileNotFoundError:
            print('警告：未找到三元模型文件，将使用二元模型', file=sys.stderr)
            use_trigram = False
    
    return pinyin_dict, unigram_freq, bigram_freq, trigram_freq, use_trigram

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='拼音转汉字程序')
    parser.add_argument('--trigram', action='store_true', help='使用三元模型（如果可用）')
    parser.add_argument('--train', choices=['1', '2', '3', 'all'], help='指定要训练的模型: 1=一元模型, 2=二元模型, 3=三元模型, all=所有模型')
    parser.add_argument('--force-train', action='store_true', help='强制重新训练指定的模型，即使模型文件已存在')
    parser.add_argument('--lambda1', type=float, default=0.0, help='一元模型权重')
    parser.add_argument('--lambda2', type=float, default=0.5, help='二元模型权重')
    parser.add_argument('--lambda3', type=float, default=0.5, help='三元模型权重')
    parser.add_argument('--grid-search', action='store_true', help='执行权重网格搜索并绘制图表')
    parser.add_argument('--benchmark', action='store_true', help='运行性能基准测试')
    parser.add_argument('--train-only', action='store_true', help='仅训练模型，不进行推断')
    args = parser.parse_args()
    
    # 记录启动参数
    params_str = ' '.join(sys.argv[1:])
    log_message(f"程序启动参数: {params_str}")
    
    # 设置数据目录路径
    data_dir = Path(__file__).parent / 'data'
    
    # 检查已训练的模型
    trained = set()
    if (data_dir / '1_word.txt').exists():
        trained.add(1)
    if (data_dir / '2_word.txt').exists():
        trained.add(2)
    if (data_dir / '3_word.txt').exists():
        trained.add(3)
    
    # 确定需要训练哪些模型
    models_to_train = set()
    
    # 如果指定了--train参数
    if args.train:
        if args.train == 'all':
            models_to_train = {1, 2, 3}
        else:
            models_to_train.add(int(args.train))
        
        # 如果指定了--force-train，则无论文件是否存在都重新训练
        if args.force_train:
            # 强制训练指定的模型
            pass
        else:
            # 移除已训练的模型
            models_to_train = models_to_train - trained
        
        # 设置train-only标志，如果只使用了--train参数
        if not (args.grid_search or args.benchmark):
            args.train_only = True
    else:
        # 默认行为：训练缺失的必要模型
        if not (data_dir / '1_word.txt').exists():
            models_to_train.add(1)
        if not (data_dir / '2_word.txt').exists():
            models_to_train.add(2)
        if args.trigram and not (data_dir / '3_word.txt').exists():
            models_to_train.add(3)
    
    # 模型训练部分
    training_start_time = time.time()
    
    if models_to_train:
        # 如果有需要训练的模型，训练模型
        log_message(f"需要训练模型 {models_to_train}，开始训练...")
        train(models_to_train)
        training_time = time.time() - training_start_time
        log_message(f"模型训练完成！训练时间: {training_time:.2f}秒")
    else:
        log_message("所有需要的模型文件已存在，跳过训练步骤...")
        training_time = 0
    
    # 如果只是训练模型，则在此退出
    if args.train_only:
        log_message("仅训练模式，程序退出")
        return
    
    # 加载模型数据开始时间
    loading_start_time = time.time()
    
    # 加载训练好的模型数据
    pinyin_dict, unigram_freq, bigram_freq, trigram_freq, use_trigram = load_data(data_dir, args.trigram)
    
    # 加载模型数据结束时间
    loading_time = time.time() - loading_start_time
    log_message(f"模型加载完成！加载时间: {loading_time:.2f}秒")
    
    # 检查是否执行网格搜索并绘制图表
    if args.grid_search and use_trigram:
        grid_search_weights(pinyin_dict, unigram_freq, bigram_freq, trigram_freq, data_dir)
        return
    
    # 基准测试模式
    if args.benchmark:
        run_benchmark(pinyin_dict, unigram_freq, bigram_freq, trigram_freq, use_trigram, 
                     args.lambda1, args.lambda2, args.lambda3, data_dir)
        return
    
    # 存储预测结果
    predictions = []
    
    # 推断开始时间
    inference_start_time = time.time()
    
    # 标准输入输出模式
    input_lines = []
    for line in sys.stdin:
        input_lines.append(line.strip())
    
    log_message(f"处理 {len(input_lines)} 行输入...")
    
    for line in input_lines:
        pinyin_list = line.strip().split()
        result = viterbi_decode(
            pinyin_list, pinyin_dict, unigram_freq, bigram_freq, trigram_freq, use_trigram,
            args.lambda1, args.lambda2, args.lambda3
        )
        if not result:
            result = ''.join([pinyin_dict[p][0] if p in pinyin_dict else p for p in pinyin_list])
        predictions.append(result)
        print(result)
    
    # 推断结束时间
    inference_time = time.time() - inference_start_time
    log_message(f"推断完成！处理 {len(input_lines)} 行输入耗时: {inference_time:.2f}秒")
    
    # 计算每行平均时间
    if len(input_lines) > 0:
        avg_time_per_line = inference_time / len(input_lines)
        log_message(f"平均每行处理时间: {avg_time_per_line:.4f}秒")
    
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
                if use_trigram:
                    f_result.write(f"使用三元模型 (λ1={args.lambda1}, λ2={args.lambda2}, λ3={args.lambda3})\n")
                else:
                    f_result.write("使用二元模型\n")
                f_result.write(f"训练时间: {training_time:.2f}秒\n")
                f_result.write(f"处理 {len(input_lines)} 行输入耗时: {inference_time:.2f}秒\n")
                f_result.write(f"平均每行处理时间: {avg_time_per_line:.4f}秒\n")
            
            # 同时输出到stderr和日志
            log_message(f"\n评估结果：")
            log_message(f"字准确率: {char_precision:.2%}")
            log_message(f"句准确率: {sent_precision:.2%}")
            if use_trigram:
                log_message(f"使用三元模型 (λ1={args.lambda1}, λ2={args.lambda2}, λ3={args.lambda3})")
            else:
                log_message("使用二元模型")
            
            # 记录总时间
            total_time = training_time + loading_time + inference_time
            log_message(f"总运行时间: {total_time:.2f}秒")
            
    except FileNotFoundError:
        log_message("\n未找到answer.txt文件，跳过评估步骤")

def run_benchmark(pinyin_dict, unigram_freq, bigram_freq, trigram_freq, use_trigram, 
                lambda1, lambda2, lambda3, data_dir):
    """运行性能基准测试
    
    Args:
        pinyin_dict: 拼音到汉字的映射字典
        unigram_freq: 一元字频统计
        bigram_freq: 二元字频统计
        trigram_freq: 三元字频统计
        use_trigram: 是否使用三元模型
        lambda1, lambda2, lambda3: 模型权重
        data_dir: 数据目录路径
    """
    log_message("开始运行性能基准测试...")
    
    try:
        # 加载测试数据
        with open(data_dir / 'input.txt', 'r', encoding='utf-8') as f:
            input_lines = f.readlines()
            input_lines = [line.strip().split() for line in input_lines]
        
        # 测试不同模型配置的性能
        model_configs = []
        
        # 二元模型配置
        model_configs.append({
            'name': '二元模型',
            'use_trigram': False,
            'lambda1': 0.0,
            'lambda2': 1.0,
            'lambda3': 0.0
        })
        
        # 三元模型配置
        if use_trigram:
            model_configs.append({
                'name': '三元模型 (λ1=0.0, λ2=0.5, λ3=0.5)',
                'use_trigram': True,
                'lambda1': 0.0,
                'lambda2': 0.5,
                'lambda3': 0.5
            })
            
            model_configs.append({
                'name': '三元模型 (λ1=0.1, λ2=0.3, λ3=0.6)',
                'use_trigram': True,
                'lambda1': 0.1,
                'lambda2': 0.3,
                'lambda3': 0.6
            })
        
        # 测试结果存储
        benchmark_results = []
        
        # 对每个模型配置进行测试
        for config in model_configs:
            log_message(f"测试 {config['name']}...")
            
            # 运行多次获取平均性能
            num_runs = 3
            total_time = 0
            
            for run in range(num_runs):
                start_time = time.time()
                
                predictions = []
                for pinyin_list in tqdm(input_lines, desc=f"运行 {run+1}/{num_runs}", file=sys.stderr):
                    result = viterbi_decode(
                        pinyin_list, pinyin_dict, unigram_freq, bigram_freq, trigram_freq,
                        config['use_trigram'], config['lambda1'], config['lambda2'], config['lambda3']
                    )
                    if not result:
                        result = ''.join([pinyin_dict[p][0] if p in pinyin_dict else p for p in pinyin_list])
                    predictions.append(result)
                
                run_time = time.time() - start_time
                total_time += run_time
                log_message(f"  运行 {run+1}: {run_time:.2f}秒")
            
            # 计算平均时间
            avg_time = total_time / num_runs
            avg_time_per_line = avg_time / len(input_lines)
            
            # 评估准确率
            with open(data_dir / 'answer.txt', 'r', encoding='utf-8') as f:
                answer_lines = f.readlines()
                answer_lines = [line.strip() for line in answer_lines]
                
                char_precision, sent_precision = evaluate(predictions, answer_lines)
            
            # 记录结果
            result = {
                'config': config['name'],
                'avg_time': avg_time,
                'avg_time_per_line': avg_time_per_line,
                'char_precision': char_precision,
                'sent_precision': sent_precision
            }
            benchmark_results.append(result)
            
            log_message(f"  平均时间: {avg_time:.2f}秒")
            log_message(f"  平均每行时间: {avg_time_per_line:.4f}秒")
            log_message(f"  字准确率: {char_precision:.2%}")
            log_message(f"  句准确率: {sent_precision:.2%}")
        
        # 保存基准测试结果
        with open(data_dir / 'benchmark_results.txt', 'w', encoding='utf-8') as f:
            f.write("性能基准测试结果\n")
            f.write("=================\n\n")
            
            for result in benchmark_results:
                f.write(f"模型配置: {result['config']}\n")
                f.write(f"平均处理时间: {result['avg_time']:.2f}秒\n")
                f.write(f"平均每行处理时间: {result['avg_time_per_line']:.4f}秒\n")
                f.write(f"字准确率: {result['char_precision']:.2%}\n")
                f.write(f"句准确率: {result['sent_precision']:.2%}\n\n")
        
        log_message(f"基准测试完成！结果已保存到 {data_dir / 'benchmark_results.txt'}")
        
    except FileNotFoundError:
        log_message("未找到测试数据文件，无法执行基准测试")

def grid_search_weights(pinyin_dict, unigram_freq, bigram_freq, trigram_freq, data_dir):
    """执行权重网格搜索并绘制图表
    
    Args:
        pinyin_dict: 拼音到汉字的映射字典
        unigram_freq: 一元字频统计
        bigram_freq: 二元字频统计
        trigram_freq: 三元字频统计
        data_dir: 数据目录路径
    """
    log_message("执行权重网格搜索...")
    
    # 加载测试数据
    try:
        with open(data_dir / 'input.txt', 'r', encoding='utf-8') as f:
            input_lines = f.readlines()
            input_lines = [line.strip().split() for line in input_lines]
        
        with open(data_dir / 'answer.txt', 'r', encoding='utf-8') as f:
            answer_lines = f.readlines()
            answer_lines = [line.strip() for line in answer_lines]
        
        # 创建权重网格
        step = 0.05
        lambda1_values = np.arange(0, 0.3, step)
        
        # 结果存储
        results = []
        
        # 记录搜索开始时间
        search_start_time = time.time()
        
        # 图表目录
        charts_dir = Path(__file__).parent / 'charts'
        os.makedirs(charts_dir, exist_ok=True)
        
        log_message("测试不同权重组合...")
        # 测试不同的权重组合
        for lambda1 in tqdm(lambda1_values, desc="λ1"):
            for lambda2 in np.arange(0, 1.0-lambda1, step):
                lambda3 = 1.0 - lambda1 - lambda2
                
                # 使用当前权重组合进行预测
                predictions = []
                for pinyin_list in input_lines:
                    result = viterbi_decode(
                        pinyin_list, pinyin_dict, unigram_freq, bigram_freq, trigram_freq, True,
                        lambda1, lambda2, lambda3
                    )
                    if not result:
                        result = ''.join([pinyin_dict[p][0] if p in pinyin_dict else p for p in pinyin_list])
                    predictions.append(result)
                
                # 评估性能
                char_precision, sent_precision = evaluate(predictions, answer_lines)
                
                # 存储结果
                results.append((lambda1, lambda2, lambda3, char_precision, sent_precision))
        
        # 记录搜索结束时间
        search_time = time.time() - search_start_time
        log_message(f"网格搜索耗时: {search_time:.2f}秒")
        
        # 转换为NumPy数组便于操作
        results_array = np.array(results)
        
        # 查找最佳权重组合
        best_char_idx = np.argmax(results_array[:, 3])
        best_sent_idx = np.argmax(results_array[:, 4])
        
        best_char_weights = results_array[best_char_idx, :3]
        best_sent_weights = results_array[best_sent_idx, :3]
        best_char_prec = results_array[best_char_idx, 3]
        best_sent_prec = results_array[best_sent_idx, 4]
        
        log_message(f"\n最佳字准确率权重: λ1={best_char_weights[0]:.1f}, λ2={best_char_weights[1]:.1f}, λ3={best_char_weights[2]:.1f}, 准确率: {best_char_prec:.2%}")
        log_message(f"最佳句准确率权重: λ1={best_sent_weights[0]:.1f}, λ2={best_sent_weights[1]:.1f}, λ3={best_sent_weights[2]:.1f}, 准确率: {best_sent_prec:.2%}")
        
        # 将结果保存到文件
        with open(charts_dir / 'weight_search_results.txt', 'w', encoding='utf-8') as f:
            f.write(f"最佳字准确率权重: λ1={best_char_weights[0]:.1f}, λ2={best_char_weights[1]:.1f}, λ3={best_char_weights[2]:.1f}, 准确率: {best_char_prec:.2%}\n")
            f.write(f"最佳句准确率权重: λ1={best_sent_weights[0]:.1f}, λ2={best_sent_weights[1]:.1f}, λ3={best_sent_weights[2]:.1f}, 准确率: {best_sent_prec:.2%}\n\n")
            f.write("详细结果:\n")
            f.write("λ1,λ2,λ3,字准确率,句准确率\n")
            for result in results:
                f.write(f"{result[0]:.1f},{result[1]:.1f},{result[2]:.1f},{result[3]:.4f},{result[4]:.4f}\n")
            f.write(f"\n网格搜索耗时: {search_time:.2f}秒\n")
        
        # 绘制字准确率热力图
        # plot_heatmap(lambda1_values, results_array, 3, 'word', charts_dir / 'char_precision_heatmap.png')
        
        # 绘制句准确率热力图
        # plot_heatmap(lambda1_values, results_array, 4, 'sentence', charts_dir / 'sent_precision_heatmap.png')
        
        # 绘制λ1为最佳值时的字准确率和句准确率随λ2变化的曲线
        plot_lambda2_curve(results_array, best_char_weights[0], charts_dir / 'lambda2_curve.png')
        
        log_message(f"图表已保存到 {charts_dir} 目录")
        
    except FileNotFoundError:
        log_message("未找到测试数据文件，无法执行网格搜索")

def plot_heatmap(lambda1_values, results_array, metric_idx, title, save_path):
    """绘制热力图
    
    Args:
        lambda1_values: λ1的值数组
        results_array: 结果数组
        metric_idx: 指标索引 (3=字准确率, 4=句准确率)
        title: 图表标题
        save_path: 保存路径
    """
    plt.figure(figsize=(10, 8))
    
    # 创建网格
    grid_size = len(lambda1_values)
    heatmap_data = np.zeros((grid_size, grid_size))
    
    # 填充数据
    for i, lambda1 in enumerate(lambda1_values):
        lambda1_results = results_array[np.isclose(results_array[:, 0], lambda1, atol=0.01)]
        for j, lambda2 in enumerate(np.arange(0, 1.01-lambda1, 0.1)):
            if j >= grid_size:  # 防止索引超出边界
                continue
            lambda2_results = lambda1_results[np.isclose(lambda1_results[:, 1], lambda2, atol=0.01)]
            if len(lambda2_results) > 0:
                # 直接索引所需的列，而不使用二维索引
                heatmap_data[i, j] = lambda2_results[0][metric_idx]
    
    # 绘制热力图
    plt.imshow(heatmap_data, cmap='viridis', aspect='auto')
    plt.colorbar(label=title)
    
    # 设置坐标轴
    x_ticks = np.arange(0, grid_size)
    y_ticks = np.arange(0, grid_size)
    plt.xticks(x_ticks, [f'{x:.1f}' for x in np.arange(0, 1.01, 0.1)[:grid_size]])
    plt.yticks(y_ticks, [f'{y:.1f}' for y in lambda1_values])
    
    plt.xlabel('λ2')
    plt.ylabel('λ1')
    plt.title(f'Model weight influence on {title} accuracy (λ3 = 1-λ1-λ2)')
    
    # 保存图表
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_lambda2_curve(results_array, best_lambda1, save_path):
    """绘制λ2曲线图
    
    Args:
        results_array: 结果数组
        best_lambda1: 最佳λ1值
        save_path: 保存路径
    """
    plt.figure(figsize=(10, 6))
    
    # 筛选出λ1为最佳值的结果
    best_lambda1_results = results_array[np.isclose(results_array[:, 0], best_lambda1, atol=0.01)]
    
    # 按λ2值排序
    sorted_indices = np.argsort(best_lambda1_results[:, 1])
    sorted_results = best_lambda1_results[sorted_indices]
    
    # 绘制曲线
    plt.plot(sorted_results[:, 1], sorted_results[:, 3], 'b-', label='word precesion')
    plt.plot(sorted_results[:, 1], sorted_results[:, 4], 'r-', label='sentence precesion')
    
    plt.xlabel('λ2 value')
    plt.ylabel('accuracy')
    plt.title(f'λ1={best_lambda1:.1f}, λ2 influence on accuracy')
    plt.legend()
    plt.grid(True)
    
    # 保存图表
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

if __name__ == '__main__':
    main()