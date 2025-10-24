"""
常用数据集格式完全指南

这个模块详细介绍深度学习中常用的数据集存储格式：
1. Arrow (Apache Arrow) - Hugging Face 主流格式
2. HDF5 (Hierarchical Data Format) - 科学计算标准
3. TFRecord - TensorFlow 专用格式
4. LMDB (Lightning Memory-Mapped Database) - 高性能键值数据库
5. Parquet - 列式存储格式
6. NPZ/NPY - NumPy 原生格式
7. 各格式对比和使用场景

作者: Seeback
日期: 2025-10-23
"""

import numpy as np
import os
import struct
import pickle
import matplotlib.pyplot as plt
from pathlib import Path


def explain_arrow_format():
    """解释 Apache Arrow 格式"""
    print("=" * 70)
    print("Apache Arrow 格式详解")
    print("=" * 70)

    print("\n📦 1. 什么是 Apache Arrow?")
    print("-" * 70)
    print("""
    Apache Arrow 是一个跨语言的内存列式数据格式标准

    核心特点:
    ✅ 零拷贝读取 (Zero-copy) - 极快的数据访问
    ✅ 列式存储 - 高效的列访问和压缩
    ✅ 跨语言支持 - Python, R, Java, C++ 等
    ✅ 内存映射 - 支持超大数据集
    ✅ 流式处理 - 支持增量读取

    主要应用:
    - Hugging Face Datasets (🤗 核心格式)
    - Pandas 2.0+ (PyArrow backend)
    - Spark, Dask 等大数据框架
    """)

    print("\n📁 2. 文件结构")
    print("-" * 70)
    print("""
    典型的 Arrow 数据集结构:
    dataset/
    ├── dataset_info.json       - 数据集元信息
    ├── state.json              - 处理状态
    └── data/
        ├── train-00000-of-00001.arrow  - 训练数据
        ├── validation-00000.arrow      - 验证数据
        └── test-00000.arrow            - 测试数据

    Arrow 文件内部结构:
    - Schema: 定义列名和数据类型
    - RecordBatches: 实际数据批次
    - Metadata: 附加元信息
    """)

    print("\n🔍 3. Arrow vs 传统格式")
    print("-" * 70)
    print("""
    ┌─────────────────┬───────────────┬───────────────────┐
    │     特性        │    Arrow      │   Pickle/CSV      │
    ├─────────────────┼───────────────┼───────────────────┤
    │ 读取速度        │   ⚡ 极快      │   🐢 慢           │
    │ 内存效率        │   ✅ 高        │   ❌ 低           │
    │ 跨语言          │   ✅ 支持      │   ❌ 有限         │
    │ 随机访问        │   ✅ O(1)      │   ❌ O(n)         │
    │ 压缩支持        │   ✅ 优秀      │   ⚠️ 有限        │
    │ 大数据集        │   ✅ 完美      │   ❌ 内存爆炸     │
    └─────────────────┴───────────────┴───────────────────┘
    """)

    print("\n💡 4. 使用 Arrow 的库")
    print("-" * 70)
    print("""
    # 1. Hugging Face Datasets (最常用)
    from datasets import load_dataset
    dataset = load_dataset('imdb')  # 自动使用 Arrow

    # 2. PyArrow (底层库)
    import pyarrow as pa
    import pyarrow.parquet as pq
    table = pa.Table.from_pandas(df)

    # 3. Pandas with PyArrow backend
    import pandas as pd
    df = pd.read_parquet('data.parquet', engine='pyarrow')
    """)

    print("\n📊 5. Arrow 的性能优势")
    print("-" * 70)
    print("""
    示例: 读取 100 万行数据

    格式          读取时间     内存占用      随机访问
    ────────────────────────────────────────────────
    CSV           12.5 秒      2.5 GB       不支持
    Pickle        8.2 秒       2.8 GB       不支持
    Arrow         0.8 秒       0.5 GB       ✅ 支持
    HDF5          1.5 秒       1.2 GB       ✅ 支持

    结论: Arrow 在速度和内存上都有显著优势!
    """)


def explain_hdf5_format():
    """解释 HDF5 格式"""
    print("\n" + "=" * 70)
    print("HDF5 格式详解")
    print("=" * 70)

    print("\n📦 1. 什么是 HDF5?")
    print("-" * 70)
    print("""
    HDF5 (Hierarchical Data Format version 5) 是科学计算标准格式

    核心特点:
    ✅ 层次化结构 - 类似文件系统
    ✅ 支持大数据 - TB 级数据集
    ✅ 部分读取 - 不需要加载全部数据
    ✅ 压缩支持 - gzip, lzf 等
    ✅ 跨平台 - C, Python, MATLAB, R 等

    主要应用:
    - 科学数据存储 (天文、生物信息学)
    - Keras 模型保存 (model.h5)
    - 大规模图像数据集
    """)

    print("\n📁 2. 文件结构")
    print("-" * 70)
    print("""
    HDF5 文件内部是层次化的:

    mydata.h5
    ├── /images                    (Group)
    │   ├── /train                 (Group)
    │   │   ├── data (Dataset)     [50000, 32, 32, 3]
    │   │   └── labels (Dataset)   [50000]
    │   └── /test                  (Group)
    │       ├── data (Dataset)     [10000, 32, 32, 3]
    │       └── labels (Dataset)   [10000]
    └── /metadata                  (Group)
        └── class_names (Dataset)  ['cat', 'dog', ...]

    概念:
    - Group: 类似文件夹,可包含其他 Group 或 Dataset
    - Dataset: 实际数据,多维数组
    - Attributes: 附加元信息
    """)

    print("\n💻 3. 使用代码")
    print("-" * 70)
    print("""
    # 安装: pip install h5py

    import h5py

    # 写入 HDF5
    with h5py.File('data.h5', 'w') as f:
        f.create_dataset('images', data=images)
        f.create_dataset('labels', data=labels)
        f.attrs['description'] = 'CIFAR-10 dataset'

    # 读取 HDF5
    with h5py.File('data.h5', 'r') as f:
        images = f['images'][:]      # 全部读取
        subset = f['images'][0:100]  # 部分读取 (高效!)

        # 查看结构
        print(list(f.keys()))
        print(f['images'].shape)
    """)

    print("\n⚡ 4. HDF5 vs NumPy")
    print("-" * 70)
    print("""
    ┌─────────────────┬───────────────┬───────────────────┐
    │     特性        │    HDF5       │    NPY/NPZ        │
    ├─────────────────┼───────────────┼───────────────────┤
    │ 部分读取        │   ✅ 支持      │   ❌ 全部加载     │
    │ 层次化结构      │   ✅ 支持      │   ❌ 平面         │
    │ 压缩            │   ✅ 多种      │   ✅ 单一         │
    │ 超大文件        │   ✅ 完美      │   ❌ 受限         │
    │ 简单性          │   ⚠️ 中等     │   ✅ 简单         │
    └─────────────────┴───────────────┴───────────────────┘
    """)


def explain_tfrecord_format():
    """解释 TFRecord 格式"""
    print("\n" + "=" * 70)
    print("TFRecord 格式详解")
    print("=" * 70)

    print("\n📦 1. 什么是 TFRecord?")
    print("-" * 70)
    print("""
    TFRecord 是 TensorFlow 的官方数据格式

    核心特点:
    ✅ 流式读取 - 支持超大数据集
    ✅ 高效序列化 - Protocol Buffers
    ✅ TF 优化 - 与 TensorFlow 深度集成
    ✅ 数据管道 - tf.data.Dataset 支持
    ✅ 并行读取 - 多线程加载

    主要应用:
    - TensorFlow 训练数据
    - Google Cloud ML
    - 大规模分布式训练
    """)

    print("\n📁 2. 文件结构")
    print("-" * 70)
    print("""
    TFRecord 文件是二进制序列化格式:

    record.tfrecord
    ├── Record 1 (Example)
    │   ├── Feature: image (bytes)
    │   ├── Feature: label (int64)
    │   └── Feature: height (int64)
    ├── Record 2 (Example)
    │   ├── Feature: image (bytes)
    │   ├── Feature: label (int64)
    │   └── Feature: height (int64)
    └── ...

    每个 Example 包含多个 Feature:
    - BytesList: 字节串 (图像、文本)
    - Int64List: 整数 (标签、尺寸)
    - FloatList: 浮点数 (特征向量)
    """)

    print("\n💻 3. 使用代码")
    print("-" * 70)
    print("""
    import tensorflow as tf

    # 写入 TFRecord
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    with tf.io.TFRecordWriter('data.tfrecord') as writer:
        for img, label in zip(images, labels):
            feature = {
                'image': _bytes_feature(img.tobytes()),
                'label': _int64_feature(label),
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

    # 读取 TFRecord
    def parse_fn(serialized):
        features = {
            'image': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64),
        }
        parsed = tf.io.parse_single_example(serialized, features)
        image = tf.io.decode_raw(parsed['image'], tf.uint8)
        return image, parsed['label']

    dataset = tf.data.TFRecordDataset('data.tfrecord')
    dataset = dataset.map(parse_fn)
    """)


def explain_lmdb_format():
    """解释 LMDB 格式"""
    print("\n" + "=" * 70)
    print("LMDB 格式详解")
    print("=" * 70)

    print("\n📦 1. 什么是 LMDB?")
    print("-" * 70)
    print("""
    LMDB (Lightning Memory-Mapped Database) 是高性能键值数据库

    核心特点:
    ✅ 极速读取 - 内存映射
    ✅ 零拷贝 - 直接访问磁盘数据
    ✅ 并发安全 - 多进程读取
    ✅ 事务支持 - ACID 保证
    ✅ 紧凑存储 - 无碎片

    主要应用:
    - Caffe 数据集
    - 大规模图像检索
    - 缓存系统
    """)

    print("\n📁 2. 文件结构")
    print("-" * 70)
    print("""
    LMDB 是目录形式:

    dataset.lmdb/
    ├── data.mdb    - 实际数据
    └── lock.mdb    - 锁文件

    内部是键值对:
    Key: '0000000001'  ->  Value: <image_bytes>
    Key: '0000000002'  ->  Value: <image_bytes>
    ...

    特点:
    - 键通常是字符串 ID
    - 值是序列化的数据 (pickle, msgpack)
    """)

    print("\n💻 3. 使用代码")
    print("-" * 70)
    print("""
    # 安装: pip install lmdb

    import lmdb
    import pickle

    # 写入 LMDB
    env = lmdb.open('dataset.lmdb', map_size=10 * 1024**3)  # 10GB
    with env.begin(write=True) as txn:
        for i, (img, label) in enumerate(zip(images, labels)):
            key = f'{i:08d}'.encode()
            value = pickle.dumps((img, label))
            txn.put(key, value)
    env.close()

    # 读取 LMDB
    env = lmdb.open('dataset.lmdb', readonly=True)
    with env.begin() as txn:
        # 随机访问
        value = txn.get(b'00000001')
        img, label = pickle.loads(value)

        # 顺序遍历
        cursor = txn.cursor()
        for key, value in cursor:
            img, label = pickle.loads(value)
    env.close()
    """)


def explain_other_formats():
    """解释其他常用格式"""
    print("\n" + "=" * 70)
    print("其他常用格式")
    print("=" * 70)

    print("\n1️⃣ Parquet (列式存储)")
    print("-" * 70)
    print("""
    特点: 类似 Arrow,列式压缩格式
    应用: Spark, Pandas, Dask
    优势: 高压缩率,列访问快

    使用:
    import pandas as pd
    df.to_parquet('data.parquet')
    df = pd.read_parquet('data.parquet')
    """)

    print("\n2️⃣ NPY/NPZ (NumPy 原生)")
    print("-" * 70)
    print("""
    特点: NumPy 数组的二进制格式
    应用: Python 科学计算
    优势: 简单,快速

    使用:
    import numpy as np
    # NPY - 单数组
    np.save('data.npy', array)
    array = np.load('data.npy')

    # NPZ - 多数组 (压缩)
    np.savez('data.npz', images=imgs, labels=lbls)
    data = np.load('data.npz')
    imgs = data['images']
    """)

    print("\n3️⃣ MessagePack")
    print("-" * 70)
    print("""
    特点: 比 JSON 快且更紧凑
    应用: 高性能 API, 序列化
    优势: 跨语言,比 pickle 安全

    使用:
    import msgpack
    data = {'images': imgs, 'labels': lbls}
    packed = msgpack.packb(data)
    unpacked = msgpack.unpackb(packed)
    """)

    print("\n4️⃣ Feather")
    print("-" * 70)
    print("""
    特点: 基于 Arrow 的文件格式
    应用: Pandas/R 数据交换
    优势: 极快读写,保留元信息

    使用:
    import pandas as pd
    df.to_feather('data.feather')
    df = pd.read_feather('data.feather')
    """)


def comprehensive_comparison():
    """所有格式的综合对比"""
    print("\n" + "=" * 70)
    print("数据集格式综合对比")
    print("=" * 70)

    print("\n📊 性能对比 (100万样本, 32x32 RGB图像)")
    print("-" * 70)
    print("""
    ┌──────────┬────────┬─────────┬──────────┬──────────┬────────┐
    │  格式    │ 读取时间│ 写入时间 │ 文件大小  │ 随机访问  │ 压缩率 │
    ├──────────┼────────┼─────────┼──────────┼──────────┼────────┤
    │ Arrow    │  0.8s  │  2.1s   │  800 MB  │   ✅      │  优秀  │
    │ HDF5     │  1.5s  │  3.2s   │  850 MB  │   ✅      │  良好  │
    │ TFRecord │  1.8s  │  4.5s   │  900 MB  │   ❌      │  良好  │
    │ LMDB     │  0.9s  │  2.8s   │  950 MB  │   ✅      │  中等  │
    │ Parquet  │  1.2s  │  3.5s   │  780 MB  │   ✅      │  优秀  │
    │ NPZ      │  2.5s  │  1.8s   │  920 MB  │   ❌      │  中等  │
    │ Pickle   │  8.2s  │  5.5s   │ 1200 MB  │   ❌      │  差    │
    └──────────┴────────┴─────────┴──────────┴──────────┴────────┘

    注: 时间基于 SSD, 实际性能取决于硬件和数据特点
    """)

    print("\n🎯 使用场景推荐")
    print("-" * 70)
    print("""
    场景                          推荐格式            理由
    ─────────────────────────────────────────────────────────
    🤗 Hugging Face 数据集        Arrow              官方标准
    🔬 科学计算/研究              HDF5               成熟稳定
    🎮 TensorFlow 训练           TFRecord            深度集成
    🖼️ 大规模图像数据集           LMDB               快速随机访问
    📊 数据分析                   Parquet/Arrow      列式高效
    🐍 简单 Python 项目          NPZ                简单易用
    🚀 高性能 API                MessagePack         跨语言
    💾 临时缓存                   Pickle             方便快速

    通用建议:
    - 新项目 → Arrow/Parquet (现代标准)
    - PyTorch → 自定义 Dataset + Arrow/HDF5
    - TensorFlow → TFRecord
    - 小数据集 → NPZ/Pickle (够用就好)
    """)


def demo_arrow_usage():
    """演示 Arrow 的实际使用"""
    print("\n" + "=" * 70)
    print("Arrow 实战演示")
    print("=" * 70)

    try:
        print("\n1️⃣ 安装检查:")
        print("-" * 70)
        try:
            import pyarrow as pa
            print(f"✅ PyArrow 已安装 (版本: {pa.__version__})")
        except ImportError:
            print("❌ PyArrow 未安装")
            print("   安装命令: pip install pyarrow")
            return

        print("\n2️⃣ 创建 Arrow 数据:")
        print("-" * 70)
        # 创建示例数据
        data = {
            'id': list(range(100)),
            'image': [np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8).tobytes()
                     for _ in range(100)],
            'label': np.random.randint(0, 10, 100).tolist(),
        }

        # 转换为 Arrow Table
        table = pa.table(data)
        print(f"   表结构: {table.schema}")
        print(f"   行数: {table.num_rows}")
        print(f"   列数: {table.num_columns}")

        print("\n3️⃣ 保存和加载:")
        print("-" * 70)
        output_path = 'artifacts/demo.arrow'
        os.makedirs('artifacts', exist_ok=True)

        # 保存
        import pyarrow.feather as feather
        feather.write_feather(table, output_path)
        file_size = os.path.getsize(output_path) / 1024
        print(f"   ✅ 已保存到: {output_path} ({file_size:.1f} KB)")

        # 加载
        loaded_table = feather.read_table(output_path)
        print(f"   ✅ 已加载: {loaded_table.num_rows} 行")

        print("\n4️⃣ 性能优势:")
        print("-" * 70)
        print("   - 零拷贝读取: 直接访问磁盘数据")
        print("   - 列式访问: 只读取需要的列")
        print("   - 内存映射: 支持超大文件")

    except Exception as e:
        print(f"❌ 演示失败: {e}")


def demo_hdf5_usage():
    """演示 HDF5 的实际使用"""
    print("\n" + "=" * 70)
    print("HDF5 实战演示")
    print("=" * 70)

    try:
        print("\n1️⃣ 安装检查:")
        print("-" * 70)
        try:
            import h5py
            print(f"✅ h5py 已安装 (版本: {h5py.__version__})")
        except ImportError:
            print("❌ h5py 未安装")
            print("   安装命令: pip install h5py")
            return

        print("\n2️⃣ 创建 HDF5 文件:")
        print("-" * 70)
        output_path = 'artifacts/demo.h5'
        os.makedirs('artifacts', exist_ok=True)

        # 创建示例数据
        images = np.random.randint(0, 255, (100, 32, 32, 3), dtype=np.uint8)
        labels = np.random.randint(0, 10, 100)

        with h5py.File(output_path, 'w') as f:
            # 创建 Group
            train_group = f.create_group('train')

            # 创建 Dataset
            train_group.create_dataset('images', data=images, compression='gzip')
            train_group.create_dataset('labels', data=labels)

            # 添加属性
            f.attrs['description'] = 'Demo dataset'
            f.attrs['num_classes'] = 10

        file_size = os.path.getsize(output_path) / 1024
        print(f"   ✅ 已保存到: {output_path} ({file_size:.1f} KB)")

        print("\n3️⃣ 读取 HDF5 文件:")
        print("-" * 70)
        with h5py.File(output_path, 'r') as f:
            print(f"   文件结构: {list(f.keys())}")
            print(f"   图像形状: {f['train/images'].shape}")
            print(f"   描述: {f.attrs['description']}")

            # 部分读取 (关键特性!)
            subset = f['train/images'][0:10]
            print(f"   部分读取: {subset.shape} (无需加载全部数据!)")

    except Exception as e:
        print(f"❌ 演示失败: {e}")


def demo_npz_usage():
    """演示 NPZ 的实际使用"""
    print("\n" + "=" * 70)
    print("NPZ 实战演示")
    print("=" * 70)

    print("\n1️⃣ 创建 NPZ 文件:")
    print("-" * 70)
    output_path = 'artifacts/demo.npz'
    os.makedirs('artifacts', exist_ok=True)

    # 创建示例数据
    images = np.random.randint(0, 255, (100, 32, 32, 3), dtype=np.uint8)
    labels = np.random.randint(0, 10, 100)

    # 保存 (压缩)
    np.savez_compressed(output_path, images=images, labels=labels)
    file_size = os.path.getsize(output_path) / 1024
    print(f"   ✅ 已保存到: {output_path} ({file_size:.1f} KB)")

    print("\n2️⃣ 读取 NPZ 文件:")
    print("-" * 70)
    data = np.load(output_path)
    print(f"   文件中的数组: {list(data.keys())}")
    print(f"   图像形状: {data['images'].shape}")
    print(f"   标签形状: {data['labels'].shape}")

    print("\n3️⃣ NPZ 的优缺点:")
    print("-" * 70)
    print("   ✅ 简单易用 - 一行代码搞定")
    print("   ✅ Python 原生 - 无需额外依赖")
    print("   ❌ 全部加载 - 不支持部分读取")
    print("   ❌ 大文件慢 - 不适合 GB 级数据")


def main():
    """主函数"""
    print("\n" + "=" * 70)
    print("🎓 常用数据集格式完全指南")
    print("=" * 70)

    # 1. Arrow 格式
    explain_arrow_format()

    # 2. HDF5 格式
    explain_hdf5_format()

    # 3. TFRecord 格式
    explain_tfrecord_format()

    # 4. LMDB 格式
    explain_lmdb_format()

    # 5. 其他格式
    explain_other_formats()

    # 6. 综合对比
    comprehensive_comparison()

    # 7. 实战演示
    demo_arrow_usage()
    demo_hdf5_usage()
    demo_npz_usage()

    print("\n" + "=" * 70)
    print("✅ 所有格式介绍完成!")
    print("=" * 70)
    print("\n💡 快速选择指南:")
    print("   🤗 使用 Hugging Face → Arrow")
    print("   🔬 科学计算项目 → HDF5")
    print("   🎮 TensorFlow 训练 → TFRecord")
    print("   🖼️ 大规模图像集 → LMDB")
    print("   📊 数据分析 → Parquet/Arrow")
    print("   🐍 简单 Python → NPZ")
    print("=" * 70)


if __name__ == "__main__":
    main()
