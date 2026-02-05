import tensorflow as tf

# 检查是否识别到GPU
gpus = tf.config.list_physical_devices('GPU')
print(f"可用的GPU数量: {len(gpus)}")

# 测试GPU是否正常工作
if gpus:
    # 设置内存增长，避免占用所有GPU内存
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    
    # 执行一个简单的计算
    with tf.device('/GPU:0'):
        a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3])
        b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2])
        c = tf.matmul(a, b)
        print(c)