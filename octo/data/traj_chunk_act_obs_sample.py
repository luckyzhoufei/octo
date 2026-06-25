import numpy as np

# 假设总步数 (traj_len) = 100
T = 100
W = 2
A = 4
D = 7

trajectory = {
    # ==========================================
    # 1. Observation (观测：机器人在每一帧看到的、感受到的)
    # 所有的第一维都是 100！
    # ==========================================
    "observation": {
        # 记录了从第 1 帧到第 100 帧的主相机完整录像
        "image_primary": np.zeros((T, 256, 256, 3), dtype=np.uint8),

        "history_image_primary": np.zeros((T, 2, 256, 256, 3), dtype=np.uint8),

        # 记录了 100 帧的腕部相机录像
        "image_wrist": np.zeros((T, 128, 128, 3), dtype=np.uint8),

        "history_image_wrist": np.zeros((T, 2 ,128, 128, 3), dtype=np.uint8),   # 训练的时候，T会被展开，打混，然后组成batch

        # 记录了这 100 步里，每一时刻机械臂 6个关节的角度 + 1个夹爪的状态 (7维)
        "proprio": np.zeros((T, 7), dtype=np.float32),
        "history_proprio": np.zeros((T, 2, 7), dtype=np.float32),

        # add_pad_mask_dict
        "pad_mask_dict": {
            "image_primary": [True, ..., True],
            "image_wrist": [True, ..., True],
            "proprio": [True, ..., True],
        },
        "history_pad_mask_dict": {
            "image_primary": [[True, True], ..., [True, True]],
            "image_wrist": [[True, True], ..., [True, True]],
            "proprio": [[True, True], ..., [True, True]],
        },
        "timestep_pad_mask": [[False, True], [True, True], [True, True], [True, True]],   # 标记traj中，每步history pad—mask
    },

    # ==========================================
    # 2. Action (动作：机器人在每一帧实际执行的运动指令)
    # ==========================================
    # 记录了从第 1 步到第 100 步发给电机的指令（比如每一帧关节要转多少度）
    "action": np.zeros((T, 7), dtype=np.float32),
    "chunk_action": np.zeros((T, 7, 4), dtype=np.float32),   # 最后一个时间步的chunk中，每个action都是一样的，都是用最后一步action替换
    "history_chunk_action": np.zeros((T, 2, 7, 4), dtype=np.float32),

    # ==========================================
    # 3. Task (任务：人类给它的指令是什么)
    # 为了满足“第一维度必须是 T”的铁律，任务指令通常会被复制 T 次！
    # ==========================================
    "task": {
        # 同一句话被复制了 100 遍，对应这 100 个时间步
        "language_instruction": np.array(["把红色方块放进碗里"] * T),

        # 如果是看图做事，目标图片也会被复制 100 遍
        "image_primary": np.zeros((T, 256, 256, 3), dtype=np.uint8),

        # 记录当前是第几步（可选），用于计算什么时候到达目标
        "timestep": np.arange(T, dtype=np.int32),  # [0, 1, 2, ..., 99]

        # add_pad_mask_dict
        "pad_mask_dict": {
            "language_instruction": [True, ..., True],
            "image_primary": [True, ..., True],
            "timestep": [True, ..., True],
        },
    },

    "action_pad_mask": [T, W, A, D]
}