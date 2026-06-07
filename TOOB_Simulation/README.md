# TOOB Simulation

这个目录是 TOOB 思路的仿真版本。它目前不实现真实的 Pluggable
Transport，也不直接做在线实时流量发送，而是生成一份可以交给其他网站指纹
检测器评测的 defended/adversarial 数据集。

## 核心思路

当前流程是：

1. 把原始 direction trace 转成 burst 序列。
2. 根据每个网站类别的 burst profile，把原始网站类别聚成更少的伪标签类。
3. 每个伪标签 `c` 训练一个生成器 `G_c`。
4. `G_c(z)` 输出非负 burst 扰动。
5. 把扰动加到原始 burst 上：

```text
b_adv = b + G_c(z) * sign(b)
```

6. 使用检测器在防御后的数据上评测 Accuracy、Precision、Recall、F1-score。

最终导出的数据集仍然保留原始网站标签 `labels`。伪标签
`pseudo_labels` 只用于选择哪个生成器 `G_c` 来产生扰动，不作为最终评测标签。

## 输入文件

脚本默认读取 `.npz` 数据集。原始数据通常需要包含：

```text
data   - numpy array，一般是 direction sequence 或 burst sequence
labels - 整数标签或 one-hot 标签
```

如果使用当前项目里的数据，runner 默认路径是：

```text
TOOB_Simulation/data/raw/test.npz
```

## 训练集和验证集

现在 runner 支持显式指定训练集和验证集：

```bash
TRAIN_DATASET=TOOB_Simulation/data/raw/train.npz \
VALID_DATASET=TOOB_Simulation/data/raw/valid.npz \
bash TOOB_Simulation/run_exp_ubuntu.sh full
```

训练集负责三件事：

```text
1. 生成 train burst 数据
2. 根据 train burst profile 聚类得到 website_to_pseudo_label
3. 训练每个伪标签对应的生成器 G_c
```

验证集负责两件事：

```text
1. 使用训练集得到的 website_to_pseudo_label 映射到伪标签
2. 使用训练好的 G_c 生成 defended valid 样本，并在 valid 样本上评测检测器
```

也就是说，指定 `VALID_DATASET` 后，最终用于评测的防御样本来自 valid 集，
不是训练集。输出文件会变成：

```text
TOOB_Simulation/outputs/toob_valid_adv_direction.npz
```

如果没有指定 `VALID_DATASET`，runner 会保持旧行为：直接在训练输入对应的
defended dataset 上做评测。

你的 DF 模型通过 detector adapter 加载。默认模型结构和 checkpoint 是：

```text
TOOB_Simulation/toob/wflib_df.py:DF
TOOB_Simulation/checkpoints/df_cw/max_f1.pth
```

## 一键运行

Ubuntu 上推荐使用：

```bash
bash TOOB_Simulation/run_exp_ubuntu.sh smoke
```

正式完整运行：

```bash
bash TOOB_Simulation/run_exp_ubuntu.sh full
```

如果 Ubuntu 里的 Python 不是 `python3`，可以指定：

```bash
PYTHON_BIN=/path/to/python bash TOOB_Simulation/run_exp_ubuntu.sh smoke
```

Windows PowerShell 上可以用：

```powershell
powershell -ExecutionPolicy Bypass -File TOOB_Simulation/run_exp.ps1 smoke
```

```powershell
powershell -ExecutionPolicy Bypass -File TOOB_Simulation/run_exp.ps1 full
```

Windows Git Bash 也可以用：

```powershell
D:\Git\bin\sh.exe TOOB_Simulation/run_exp.sh smoke
```

只训练某一个伪标签对应的生成器：

```powershell
$env:PSEUDO_LABEL="0"
D:\Git\bin\sh.exe TOOB_Simulation/run_exp.sh one
```

常用环境变量：

```powershell
$env:TRAIN_DATASET="TOOB_Simulation\data\raw\train.npz"
$env:VALID_DATASET="TOOB_Simulation\data\raw\valid.npz"
$env:FULL_EPOCHS="50"
$env:FULL_BATCH_SIZE="16"
$env:OUT_DIR="TOOB_Simulation/outputs_v1"
```

评测相关环境变量：

```powershell
$env:RUN_EVAL="1"
$env:EVAL_METRICS="accuracy precision recall f1"
$env:EVAL_AVERAGE="macro"
```

如果只想生成防御数据集，暂时不做检测器评测，可以设置：

```powershell
$env:RUN_EVAL="0"
```

## Step 1：Direction 转 Burst

对应脚本：

```text
TOOB_Simulation/EXP/01_make_burst_dataset.py
```

单独运行示例：

```powershell
python TOOB_Simulation/EXP/01_make_burst_dataset.py `
  --input TOOB_Simulation/data/raw/test.npz `
  --output TOOB_Simulation/outputs/burst_dataset.npz `
  --data-key X `
  --labels-key y `
  --max-bursts 2000
```

输出文件：

```text
TOOB_Simulation/outputs/burst_dataset.npz
```

主要字段：

```text
data   - 从 direction trace 转换得到的 burst 序列
labels - 原始网站标签，不改变
```

这一步只做表示转换，不产生伪标签。`labels` 仍然是真实网站类别，例如
`0..94`。

## Step 2：聚类并生成伪标签

对应脚本：

```text
TOOB_Simulation/EXP/02_make_pseudo_labels.py
```

单独运行示例：

```powershell
python TOOB_Simulation/EXP/02_make_pseudo_labels.py `
  --labels-npz TOOB_Simulation/outputs/burst_dataset.npz `
  --output TOOB_Simulation/outputs/pseudo_labels.npz `
  --json-output TOOB_Simulation/outputs/pseudo_labels.json `
  --set-size 30 `
  --rounds 1 `
  --profile-method super `
  --exclude-labels 95 `
  --drop-unmapped
```

输出文件：

```text
TOOB_Simulation/outputs/pseudo_labels.npz
TOOB_Simulation/outputs/pseudo_labels.json
```

`pseudo_labels.npz` 是后续训练真正使用的文件，主要字段是：

```text
labels        - 过滤后的原始网站标签，仍然是真实标签
pseudo_labels - 每条样本对应的伪标签
keep_indices  - 从 burst_dataset.npz 中保留下来的样本下标
```

`pseudo_labels.json` 是方便人检查的可读文件，主要字段是：

```text
website_to_pseudo_label  - 原始网站标签 -> 伪标签
pseudo_label_to_websites - 伪标签 -> 该伪标签包含哪些原始网站标签
pseudo_label_counts      - 每个伪标签下有多少条样本
cluster                  - 聚类参数和元信息
samples                  - 每条样本的原始标签和伪标签
```

这一阶段的语义是：把原本的 95 个网站类别，根据每个类别的 burst profile
聚成更少的伪标签大类。比如网站标签 `0`、`5`、`8` 可能都被分到伪标签
`0`。后面训练时，所有伪标签为 `0` 的样本都会用来训练同一个生成器
`G_0`。

`--exclude-labels 95` 用来排除 open-world 类别。最终评测时仍然使用真实
`labels`，不是 `pseudo_labels`。

## Step 3：使用 DF 训练生成器

对应脚本：

```text
TOOB_Simulation/EXP/03_train_generators.py
```

单独运行示例：

```powershell
python TOOB_Simulation/EXP/03_train_generators.py `
  --burst-npz TOOB_Simulation/outputs/burst_dataset.npz `
  --pseudo-npz TOOB_Simulation/outputs/pseudo_labels.npz `
  --detector-builder TOOB_Simulation/toob/wflib_df.py:DF `
  --detector-checkpoint TOOB_Simulation/checkpoints/df_cw/max_f1.pth `
  --num-classes 95 `
  --detector-input-kind direction `
  --detector-input-layout ncl `
  --detector-input-length 5000 `
  --output-dir TOOB_Simulation/outputs/generators
```

输出文件：

```text
TOOB_Simulation/outputs/generators/generator_pseudo_0.pt
TOOB_Simulation/outputs/generators/generator_pseudo_1.pt
...
TOOB_Simulation/outputs/generators/manifest.json
```

每个 `generator_pseudo_*.pt` 对应一个伪标签的生成器 `G_c`。脚本会读取：

```text
burst_dataset.npz  - burst 输入
pseudo_labels.npz  - 每条样本属于哪个伪标签
```

然后对每个伪标签分别训练一个生成器。

`manifest.json` 记录训练时使用的 detector checkpoint、训练参数，以及生成了
哪些 `G_c` 文件。

当前 DF 模型输入是 direction sequence，形状为：

```text
[N, 1, 5000]
```

但是 TOOB 的扰动是在 burst 空间里生成的。因此训练时会使用一个可微的
soft burst-to-direction projection，把 burst 近似展开成 direction sequence，
让 DF 的梯度可以反传回 `G_c`。

当前 CW checkpoint 是 95 类输出，所以训练时使用：

```text
--num-classes 95
```

如果只是快速测试某一个伪标签：

```powershell
python TOOB_Simulation/EXP/03_train_generators.py `
  --burst-npz TOOB_Simulation/outputs/burst_dataset.npz `
  --pseudo-npz TOOB_Simulation/outputs/pseudo_labels.npz `
  --detector-builder TOOB_Simulation/toob/wflib_df.py:DF `
  --detector-checkpoint TOOB_Simulation/checkpoints/df_cw/max_f1.pth `
  --num-classes 95 `
  --pseudo-labels 0 `
  --epochs 1 `
  --batch-size 8 `
  --detector-input-kind direction `
  --detector-input-layout ncl `
  --detector-input-length 5000 `
  --output-dir TOOB_Simulation/outputs/generators_smoke
```

如果 GPU 显存不够，可以优先调小：

```text
--batch-size 4 --projection-chunk-size 64
```

`--soft-projection-tau` 控制 soft burst 展开时的锐利程度。值越小越接近硬的
direction sequence，但梯度可能更不稳定。建议先用默认 `1.5`，再尝试 `1.0`
或 `0.5`。

## Step 4：生成最终防御数据集

对应脚本：

```text
TOOB_Simulation/EXP/04_generate_dataset.py
```

导出 burst 版本：

```powershell
python TOOB_Simulation/EXP/04_generate_dataset.py `
  --burst-npz TOOB_Simulation/outputs/burst_dataset.npz `
  --pseudo-npz TOOB_Simulation/outputs/pseudo_labels.npz `
  --generator-dir TOOB_Simulation/outputs/generators `
  --output TOOB_Simulation/outputs/toob_adv_burst.npz `
  --output-kind burst `
  --round
```

导出 direction 版本，方便交给其他检测器评测：

```powershell
python TOOB_Simulation/EXP/04_generate_dataset.py `
  --burst-npz TOOB_Simulation/outputs/burst_dataset.npz `
  --pseudo-npz TOOB_Simulation/outputs/pseudo_labels.npz `
  --generator-dir TOOB_Simulation/outputs/generators `
  --output TOOB_Simulation/outputs/toob_adv_direction.npz `
  --output-kind direction `
  --max-trace-len 5000 `
  --round
```

输出文件：

```text
TOOB_Simulation/outputs/toob_adv_direction.npz
```

如果 runner 指定了 `VALID_DATASET`，Step 4 会先额外生成：

```text
TOOB_Simulation/outputs/valid_burst_dataset.npz
TOOB_Simulation/outputs/valid_pseudo_labels.npz
TOOB_Simulation/outputs/valid_pseudo_labels.json
```

然后使用训练好的 `generators/` 给 valid 样本加防御，最终输出：

```text
TOOB_Simulation/outputs/toob_valid_adv_direction.npz
```

当 `--output-kind direction` 时，主要字段是：

```text
data          - 防御后的 direction sequence，可直接交给检测器评测
labels        - 原始网站标签，不改变
pseudo_labels - 该样本使用了哪个伪标签生成器
overhead      - 每条样本的扰动开销
```

当 `--output-kind burst` 时，`data` 和 `burst_data` 都是防御后的 burst 数据。

当 `--output-kind both` 时，文件里会同时保存：

```text
burst_data
direction_data
```

最终评测检测器时应该使用 `labels`，不是 `pseudo_labels`。`pseudo_labels`
只是说明该样本由哪个聚类生成器产生扰动。

## Step 5：评测防御效果

对应脚本：

```text
TOOB_Simulation/EXP/05_evaluate_defense.py
```

runner 默认会在 Step 4 后自动评测生成好的 defended direction 数据集：

```text
TOOB_Simulation/outputs/toob_adv_direction.npz
```

如果指定了 `VALID_DATASET`，runner 会自动改为评测：

```text
TOOB_Simulation/outputs/toob_valid_adv_direction.npz
```

单独评测已经生成好的防御数据集：

```powershell
python TOOB_Simulation/EXP/05_evaluate_defense.py `
  --input-npz TOOB_Simulation/outputs/toob_adv_direction.npz `
  --input-kind direction `
  --data-key data `
  --labels-key labels `
  --detector-builder TOOB_Simulation/toob/wflib_df.py:DF `
  --detector-checkpoint TOOB_Simulation/checkpoints/df_cw/max_f1.pth `
  --num-classes 95 `
  --detector-input-kind direction `
  --detector-input-layout ncl `
  --max-trace-len 5000 `
  --metrics accuracy precision recall f1 `
  --average macro `
  --output-json TOOB_Simulation/outputs/defense_eval_metrics.json
```

输出文件：

```text
TOOB_Simulation/outputs/defense_eval_metrics.json
```

其中会记录：

```text
metrics.accuracy
metrics.precision_macro
metrics.recall_macro
metrics.f1_score_macro
overhead.mean
overhead.median
detector
defense
num_samples
```

如果你有单独的 valid 集，也可以让脚本在 valid 集上现场部署 TOOB 防御，
然后直接评测：

```powershell
python TOOB_Simulation/EXP/05_evaluate_defense.py `
  --input-npz TOOB_Simulation/data/raw/valid.npz `
  --input-kind direction `
  --data-key X `
  --labels-key y `
  --defense toob `
  --generator-dir TOOB_Simulation/outputs/generators `
  --pseudo-json TOOB_Simulation/outputs/pseudo_labels.json `
  --drop-unmapped `
  --max-bursts 2000 `
  --max-trace-len 5000 `
  --round `
  --detector-builder TOOB_Simulation/toob/wflib_df.py:DF `
  --detector-checkpoint TOOB_Simulation/checkpoints/df_cw/max_f1.pth `
  --num-classes 95 `
  --detector-input-kind direction `
  --detector-input-layout ncl `
  --metrics accuracy precision recall f1 `
  --average macro `
  --output-json TOOB_Simulation/outputs/valid_defense_eval_metrics.json
```

这里的 `--detector-builder` 和 `--detector-checkpoint` 用来选择评测模型。
目前默认使用 DF，但之后如果要换成别的检测器，只要提供对应的模型构造函数和
checkpoint 即可。

`--metrics` 可以选择要记录哪些指标，例如：

```text
--metrics accuracy f1
```

`--average` 可以选择：

```text
macro
micro
weighted
```

当前默认是 `macro`，比较适合多类别网站指纹分类结果的整体报告。

## Runner 输出目录

一键脚本会自动跑完五步，并根据模式写到不同目录：

```text
smoke -> TOOB_Simulation/outputs_smoke/
one   -> TOOB_Simulation/outputs_one/
full  -> TOOB_Simulation/outputs/
```

每个输出目录里的主要文件是：

```text
burst_dataset.npz          - Step 1 得到的 burst 表示
pseudo_labels.npz          - Step 2 得到的样本级伪标签
pseudo_labels.json         - Step 2 得到的可读聚类映射
generators/                - Step 3 训练得到的 G_c checkpoints
toob_adv_direction.npz     - Step 4 得到的最终 direction 防御数据集
toob_valid_adv_direction.npz - 指定 VALID_DATASET 时的 valid 防御数据集
defense_eval_metrics.json  - Step 5 得到的检测器评测结果
```

## 运行模式说明

`smoke` 用于快速检查流程是否能跑通，默认只取少量样本，训练轮数也很少。这个
模式下的聚类结果不代表正式实验结果。

`one` 用于只训练一个伪标签对应的生成器，适合调试 loss、显存和训练速度。

`full` 用于正式生成完整实验数据集。

## 评测时要注意

最终给其他检测器评测时，通常使用：

```text
TOOB_Simulation/outputs/toob_adv_direction.npz
```

其中：

```text
data   - 检测器输入
labels - 检测器评测标签
```

`pseudo_labels` 不参与检测器分类评测，它只用于 TOOB 内部选择生成器。
