# FID

在**已有训练环境**中，对生成好的图片 NPZ 一键计算 ImageNet FID。
首次运行自动下载参考 NPZ 和 ADM Inception 权重，后续校验并复用缓存，结果同时打印并保存为 JSON。

本仓库只负责 **NPZ → FID**，不包含模型采样、checkpoint、VAE 或训练数据。使用与 SIT 原评估器一致的 OpenAI ADM TensorFlow Inception `pool_3` 和 FID 距离实现。

## 一键使用

先激活你已经配置好的环境：

```bash
git clone https://github.com/Jeneveuxpas/FID.git
cd FID

bash fid.sh /path/to/samples.npz
```

默认 ImageNet 256，使用 GPU 0，特征提取 batch=64。结果保存在输入文件旁边：`samples.npz.fid.json`。
脚本使用当前环境的 `python`，不会安装依赖、创建环境或修改已有 PyTorch。

```bash
# 指定 GPU，检查确实是 50k 样本，并指定结果位置
bash fid.sh /path/to/samples.npz \
  --gpu 0 --expected-count 50000 --output results/step1M.fid.json

# 512 分辨率
bash fid.sh /path/to/samples512.npz --resolution 512

# 显存不足时减小 batch
bash fid.sh /path/to/samples.npz --batch-size 16

# CPU 评估（速度较慢）
bash fid.sh /path/to/samples.npz --gpu cpu
```

也可以直接使用 `python run.py /path/to/samples.npz`。从其他工作目录调用 `fid.sh` 时，相对输入路径仍以调用者的工作目录为准。

**4 卡、1M checkpoint、CFG=1.7、guidance interval=[0,0.7] 属于生成样本时的参数。** 请先用训练仓库的采样器按这些设置生成 NPZ，再将 NPZ 传给本工具。FID 本身是单进程特征提取，不需要 `torchrun`，也不需要四张 GPU；本工具不会重新采样或修改上述设置。

## 提前下载 / 离线使用

无需加载 TensorFlow，就能先下载资源：

```bash
bash fid.sh --prepare
# 或选择 512 分辨率及缓存目录
bash fid.sh --prepare --resolution 512 --cache-dir /data/fid-assets
```

默认缓存位置是仓库内的 `assets/`：

| 文件 | 用途 |
| --- | --- |
| `VIRTUAL_imagenet256_labeled.npz`（约 2.04 GB） | ImageNet 256 参考数据和全数据集统计 |
| `classify_image_graph_def.pb`（约 95.7 MB） | ADM Inception 图和权重 |

512 分辨率自动使用 `VIRTUAL_imagenet512.npz`。以上资源从 OpenAI 官方公开存储下载，不上传到此 Git 仓库。
下载失败会重试；未完成文件写到 `.partial`，下次重新下载。完成的资源用本地 SHA256 收据检查后复用；该收据用于检测缓存变化，不是官方签名。请勿同时向同一缓存目录启动多个下载。

联网机器准备好后，复制 `assets/`（包括 `.download.json` 收据）到目标机器：

```bash
bash fid.sh /path/to/samples.npz --offline
```

如果已经有参考 NPZ 和 Inception 权重，可直接指定，不需要缓存收据：

```bash
bash fid.sh /path/to/samples.npz --offline \
  --ref-batch /data/VIRTUAL_imagenet256_labeled.npz \
  --inception /data/classify_image_graph_def.pb
```

自备参考文件必须对应正确的数据集和分辨率。

## 输入格式与指标口径

NPZ 内必须包含 `arr_0`：

- shape：`[N, 256, 256, 3]`，或指定 `--resolution 512` 后的 `[N, 512, 512, 3]`。
- dtype：`uint8`，RGB、NHWC、C-order，像素值为 0–255。
- N 至少为 2；正式 FID-50k 请用 50,000 样本，并加 `--expected-count 50000`。

不接受 checkpoint、图片目录、NCHW 或归一化浮点数组。不会自动调整大小或转换颜色。
参考统计直接读取 OpenAI NPZ 的 `mu/sigma`，不把包中的 10k 参考图片重新当作完整训练集。
只计算 FID，不运行 IS、sFID、Precision/Recall。其他实现或预处理方式得到的 FID 不一定可直接比较。

按 batch 读取原始图像，保留 2048 维特征。50k 样本的特征矩阵约 410 MB，拼接和协方差计算还需要额外 RAM。
JSON 包含 FID、样本数、分辨率、输入/参考/权重路径，以及 NumPy、TensorFlow 版本；数值不是有限值时会报错，不保存成功结果。

## 已有环境需要哪些依赖？

需要 `numpy`、`scipy`、`tensorflow`、`requests`、`tqdm`。**已配置训练环境不一定包含 TensorFlow。** 默认不会安装或升级任何包。

若缺包，脚本会提示名称。先检查当前环境，再按需安装。原实现使用 TensorFlow v1 compatibility API；一个保守的 FID 版本组合是 Python 3.10/3.11、NumPy 1.26.4、SciPy 1.12.0、TensorFlow 2.16.1。
不要为了 FID 直接覆盖正常训练环境中的 CUDA/NumPy 依赖；需要隔离时可用已有的 FID 环境调用：

```bash
PYTHON=/path/to/fid-env/bin/python bash fid.sh /path/to/samples.npz
```

TensorFlow GPU 安装请参照 [官方说明](https://www.tensorflow.org/install/pip)。有 GPU 不等于 TensorFlow 能使用它；必要时检查：

```bash
python -c 'import tensorflow as tf; print(tf.config.list_physical_devices("GPU"))'
```

## 来源和验证

- 评估实现：[OpenAI guided-diffusion evaluations](https://github.com/openai/guided-diffusion/tree/main/evaluations)，经 SIT 工作树沿用；上游 MIT 许可证附在 `LICENSE`。
- 新入口仅执行 FID 所需的部分，保留原 Inception 特征图和 Fréchet distance 实现。
- 已做 Python/shell 语法及本地输入、缓存、命令入口检查。尚未在目标 Linux GPU 上完成真实样本的端到端 FID 验证。
