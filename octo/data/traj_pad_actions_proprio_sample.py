import numpy as np

# 假设总步数 (traj_len) = 100
T = 100

trajectory = {
    # ==========================================
    # 1. Observation (观测：机器人在每一帧看到的、感受到的)
    # 所有的第一维都是 100！
    # ==========================================
    "observation": {
        # 记录了从第 1 帧到第 100 帧的主相机完整录像
        "image_primary": np.zeros((T, 256, 256, 3), dtype=np.uint8),

        # 记录了 100 帧的腕部相机录像
        "image_wrist": np.zeros((T, 128, 128, 3), dtype=np.uint8),

        # 记录了这 100 步里，每一时刻机械臂 6个关节的角度 + 1个夹爪的状态 (7维)
        "proprio": np.zeros((T, 4), dtype=np.float32),
        "pad_proprio": [[0.1, 0.2, 0.5, 0.3, 0], [], ..., [], [0.1, 0.2, 0.5, 0.3, 0]],   # pad_actions_and_proprio 替换 proprio

        "pad_mask_dict": {
            "image_primary": [True, ..., True],
            "image_wrist": [True, ..., True],
            "proprio": [True, ..., True],
        },
    },

    # ==========================================
    # 2. Action (动作：机器人在每一帧实际执行的运动指令)
    # ==========================================
    # 记录了从第 1 步到第 100 步发给电机的指令（比如每一帧关节要转多少度）
    # max_action_dim = 5
    "action": np.random((T, 4), dtype=np.float32),
    "pad_action": [[0.1, 0.2, 0.5, 0.3, 0], [], ..., [], [0.1, 0.2, 0.5, 0.3, 0]],   # pad_actions_and_proprio   这个替换action
    "action_pad_mask": [[True, True, True, True, False], [], ..., [], [True, True, True, True, False]],  # pad_actions_and_proprio


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

        "pad_mask_dict": {
            "language_instruction": [True, ..., True],
            "image_primary": [True, ..., True],
            "timestep": [True, ..., True],
        },
    }
}