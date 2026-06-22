import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

# 随便选一个子数据集，比如 bridge 数据集
DATASET_NAME = 'bridge'
# 指定你的云端路径
GCS_PATH = f'gs://rail-orca-central2/resize_256_256/{DATASET_NAME}'

print(f"正在从 {GCS_PATH} 流式加载一条数据...")

# 构建 builder（不需要下载全量数据）
builder = tfds.builder_from_directory(GCS_PATH)
dataset = builder.as_dataset(split='train')

# 取出一条轨迹（Episode）
for episode in dataset.take(1):
    print("成功获取一条轨迹！\n")

    # 取出轨迹中的第一步（Step）
    for step in episode['steps'].take(1):
        # 1. 打印语言指令
        instruction = step['observation']['natural_language_instruction'].numpy().decode('utf-8')
        print(f"指令 (Instruction): {instruction}")

        # 2. 打印动作向量
        action = step['action'].numpy()
        print(f"动作 (Action): {action}")

        # 3. 显示主摄像头图片
        image = step['observation']['image_0'].numpy()
        plt.imshow(image)
        plt.title(f"Instruction: {instruction}")
        plt.axis('off')
        plt.show()