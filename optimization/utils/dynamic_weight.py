import numpy as np

def calculate_dynamic_weighted_score(scenarios_scores, alpha=1.0):
    """
    计算动态加权总分，根据每个任务的评分表现动态调整权重。

    参数:
    scenarios_scores: 一个列表或数组，包含每个场景的评分。
    alpha: 调整因子，控制权重调整的灵敏度。

    return:
    总分（float类型）
    """
    min_score = np.min(scenarios_scores)
    max_score = np.max(scenarios_scores)

    normalized_scores = [(score - min_score) / (max_score - min_score) for score in scenarios_scores]

    mean_score = np.mean(normalized_scores)

    differences = [abs(score - mean_score) for score in normalized_scores]

    max_diff = max(differences)
    weights = [(1 / len(scenarios_scores)) * (1 + alpha * diff / max_diff) for diff in differences]

    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]

    total_score = sum(w * s for w, s in zip(normalized_weights, normalized_scores))

    return total_score

# 示例数据：假设有5个场景的评分
scenarios_scores = [80, 90, 85, 95, 88]

# 计算总分
total_score = calculate_dynamic_weighted_score(scenarios_scores, alpha=1.5)

# 输出结果
print("动态加权后的总分：", total_score)
