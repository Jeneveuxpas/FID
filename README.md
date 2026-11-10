# FID

**SIT checkpoint → 多 GPU 生成图片 → NPZ → OpenAI ADM FID**，一条命令完成。
沿用已经配置好的训练环境，不自动安装依赖或修改 PyTorch。首次运行自动下载 ImageNet 参考 NPZ、Inception 权重和 iREPA MSE VAE，后续校验并复用缓存。

## 一键评估 1M checkpoint

先激活已有训练环境，然后：

```bash
git clone https://github.com/Jeneveuxpas/FID.git
cd FID
bash eval.sh /path/to/checkpoints/1000000.pt
```

已克隆仓库的用户先 `git pull`。把路径换成你的 **1,000,000 步 checkpoint**；模型权重不随仓库提供。

`eval.sh` 固定了以下默认设置，可通过末尾参数覆盖：

| 设置 | 默认值 |
| --- | --- |
| 模型 | SiT-XL/2，读取 checkpoint 的 EMA 权重 |
| GPU | `0,1,2,3`，4 张 GPU |
| CFG / guidance interval | **1.7 / [0.0, 0.7]** |
| 样本数 | 50,000 |
| 采样 | SDE，250 步 |
| VAE | iREPA 官方转换的 MSE VAE，配套 latent 统计 |
| 每 GPU batch | 32，全局 batch 128 |
| seed / 精度 | 0 / FP32，TF32 开启 |
| 分辨率 | 从 checkpoint args 读取，缺失时默认 256 |
| 输出目录 | `results/step1M-cfg1.7` |

4 张卡用于生成图片；生成结束后自动运行单进程 FID。无需手动 `torchrun`。

```bash
# 指定不同结果目录
bash eval.sh /path/to/1000000.pt --output-dir results/my-model-1M

# 从同一任务中断处继续，其他参数保持不变
bash eval.sh /path/to/1000000.pt --resume

# 显存不足：在首次运行时减小 batch，并选新的输出目录
bash eval.sh /path/to/1000000.pt --batch-size 16 --output-dir results/step1M-bs16
```

## 提前下载与离线使用

```bash
python run.py prepare --resolution 256
# 若模型使用 512 分辨率：
python run.py prepare --resolution 512
```

`assets/` 自动缓存：ImageNet 256 参考 NPZ（约 2.04 GB）、ADM Inception（约 95.7 MB）、iREPA MSE VAE（约 334.7 MB）。512 参考包会使用相应文件。
配套 `sdvae-ft-mse-f8d4-latents-stats.pt`（1,704 字节）已随仓库提供，并按 iREPA 官方 LFS SHA256 校验；VAE 下载也核对官方 SHA256。
如训练使用了自定义 latent 统计，请传入 `--latent-stats /path/to/your-stats.pt`。

下载失败会重试，未完成文件保存在 `.partial`，重新启动会从头下载未完成资源；完整缓存检查哈希后复用。请勿让多个进程同时下载到同一缓存目录。

准备好后，复制仓库、完整 `assets/`（包括 `.download.json`）和 checkpoint 到离线 GPU 机器，在已装好依赖的环境运行：

```bash
bash eval.sh /path/to/1000000.pt --offline
```

用 `--cache-dir /data/fid-assets` 更换缓存位置；已有参考资源可传入 `--ref-batch /path/reference.npz --inception /path/classify_image_graph_def.pb`。

## 输出和断点续跑

```text
results/step1M-cfg1.7/
  samples/         # 生成的 PNG
  samples.npz      # 前 50,000 个样本，uint8 NHWC RGB
  sampling.json    # 采样参数、checkpoint/统计/VAE/代码哈希
  sampling.log
  fid.json         # FID、样本数、分辨率、资源路径及数值库版本
  fid.log
```

续跑需使用原命令追加 `--resume`。程序保留有效 PNG，补齐缺失或损坏图片；GPU 顺序、GPU 数量、batch、seed、路径及其他采样设置必须保持一致。参数、权重或代码变化会拒绝混用原样本，需选择新输出目录。
PNG 逐张打包为 NPZ，不把所有原图堆入内存。50k 个 256×256 RGB 的 NPZ 约 9.83 GB，另有 PNG、checkpoint、资源，建议预留至少 30 GB；512 分辨率需要更多。
FID 保存全部 2048 维特征，50k 特征矩阵约 410 MB，拼接和协方差计算还需要额外 RAM。

如果图片已经生成成功，只重新计算 FID：

```bash
bash fid.sh results/step1M-cfg1.7/samples.npz \
  --expected-count 50000 --output results/step1M-cfg1.7/fid.json
```

更多仅 NPZ 评估选项见 [NPZ_FID.md](NPZ_FID.md)。

## 自定义完整命令

```bash
python run.py run \
  --ckpt /path/to/checkpoints/1000000.pt \
  --model SiT-XL/2 --resolution 256 \
  --gpus 0,1,2,3 --batch-size 32 --num-samples 50000 \
  --mode sde --num-steps 250 --vae mse \
  --cfg-scale 1.7 --guidance-low 0.0 --guidance-high 0.7 \
  --seed 0 --output-dir results/step1M-cfg1.7
```

`eval.sh` 提供上述实验预设；`run.py` 是通用入口，其默认单卡、CFG=1.0、interval=[0,1]。使用通用入口时请显式指定实验设置。
支持本 SIT 训练保存的含 `ema` 的 checkpoint，模型 SiT-S/B/L/XL/2。省略 `--model` 可从 checkpoint 的 `args.model` 推断，`-EncoderKV` 名称映射到推理 SiT。qk_norm 和 path_type 默认读 checkpoint，显式命令行值优先。
推理不注入 DINO 特征，训练专用投影忽略；缺少实际推理参数会报错。仅加载可信 `.pt` 文件（训练 checkpoint 使用 `weights_only=False` 读取）。

## 环境要求

目标：Linux + NVIDIA GPU，已配置的 Python 训练环境。

- 采样：`torch`、`torchvision`、`numpy`、`timm`、`diffusers`、`huggingface_hub`、`safetensors`、`dictdot`、`Pillow`、`tqdm`。
- FID：`tensorflow`、`numpy`、`scipy`、`requests`、`tqdm`。

**训练环境不一定已经有 TensorFlow**。工具不自动安装/升级任何依赖，避免破坏现有环境。可先做导入检查：

```bash
python -c 'import torch, timm, diffusers, dictdot, PIL; print("PyTorch CUDA:", torch.cuda.is_available())'
python -c 'import tensorflow as tf, scipy, requests, tqdm; print("TensorFlow GPUs:", tf.config.list_physical_devices("GPU"))'
```

有独立 FID 环境时可指定 `--fid-python /path/to/fid-env/bin/python`；默认采样和 FID 均使用当前 Python。TensorFlow 安装参照 [官方说明](https://www.tensorflow.org/install/pip)。
不需要安装 DINO、下载训练数据或提供训练代码。

## 来源与验证范围

- 采样器、模型和 VAE 来自 SIT 工作树，分享版补充了独立路径、缺失权重检查和 NPZ 流式打包。
- VAE/latent 统计来自 [iREPA 官方资源库](https://huggingface.co/REPA-E/iREPA-collections/tree/main/pretrained_models)，项目说明见 [iREPA](https://github.com/end2end-diffusion/irepa)。
- FID 保留 [OpenAI ADM](https://github.com/openai/guided-diffusion/tree/main/evaluations) 的 TensorFlow `pool_3` 与距离计算，直接使用参考包的全数据集 `mu/sigma`。只计算 FID。
- 原始许可证见 `LICENSE` 和 `LICENSE.*`；预训练资源遵循各自来源条款。
- 已运行本地 CPU 回归、Python/shell 语法和入口检查；**尚未在目标 Linux GPU 上完成真实 checkpoint 的 CUDA/DDP 采样及端到端 FID 验证**。

```bash
python -m unittest discover -s . -p 'test_*.py' -v
```
