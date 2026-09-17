# 配置文件示例：启用logits收集
# 在你的配置文件中添加以下设置来启用logits收集

config = {
    "data": {
        # 启用logits收集
        "collect_logits": True,
        
        # 其他数据相关配置
        "max_prompt_length": 2048,
        "truncation": "left",
        "return_raw_chat": False,
        "train_batch_size": 32,
    },
    
    "env": {
        "rollout": {
            "n": 4,  # 环境分组数量
        },
        "max_steps": 100,  # 最大步数
    },
    
    "algorithm": {
        "filter_groups": {
            "enable": False,  # 是否启用动态采样
            "max_num_gen_batches": 10,
        },
    },
}

# 使用说明：
# 1. 设置 "collect_logits": True 来启用logits收集
# 2. 设置 "collect_logits": False 来禁用logits收集（默认行为）
# 3. logits将被添加到每个轨迹的每个动作中
# 4. 可以通过检查轨迹数据中的'logits'字段来访问logits
