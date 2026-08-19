# TOOB 实验记录与记忆

> 本文档用于恢复上下文。记录 WFP-Defence-lib 项目中 TOOB(Think Out Of The Box)防御方法的理解、实验过程、结论、代码改动与待办事项。
> 最后更新:2026-08-19

---

## 1. 项目与任务概述

TOOB = **"Think Out Of The Box"**,是用户设计的网站指纹(WF)防御方法,对应论文 *Think Out Of The Box: Deep Privacy-Preserving Defense Method for Website Fingerprinting Protection*(论文中方法名为 PETO)。

核心思路:在 **burst(突发)空间** 对流量注入 dummy padding 包,使 WF 检测器(DF 等)分类失效,同时用 cost loss 控制带宽开销。每个网站类先按 burst profile 聚成伪标签(匿名集),每个伪标签训练一个扰动生成器 `G_c`。

任务目标是:跑通方法、验证有效性、做消融实验、回复审稿人意见。

---

## 2. 关键资源路径

### 数据
| 资源 | 路径 | 说明 |
|---|---|---|
| CW 数据集 | `/root/autodl-fs/datasets/CW/{train,valid,test}.npz` | 95 类 monitored,key=`X`/`y`,X 为 signed-timestamp (10000 长) |
| OW 数据集 | `/root/autodl-fs/datasets/OW/{train,valid,test}.npz` | 96 类 = 95 monitored + 1 open-world(label 95),unmonitored train 32979 条 |
| 小数据集(train) | `TOOB_Simulation/data/raw/cw_small_train.npz` | 分层抽样 每类 30,共 2850 |
| 小数据集(valid) | `TOOB_Simulation/data/raw/cw_small_valid.npz` | 每类 10,共 950 |

小数据集由 `/tmp/opencode/subsample_cw.py` 生成(分层抽样,seed=2024)。

### 检测器/攻击模型 checkpoint
| 模型 | 路径 | 类别 |
|---|---|---|
| DF (闭世界) | `/root/Website-Fingerprinting-Library/checkpoints/CW/DF/max_f1.pth` | 95 |
| DF (开世界) | `/root/Website-Fingerprinting-Library/checkpoints/OW/DF/max_f1.pth` | 96(第 96 类 = unmonitored) |
| 其他攻击模型 | `/root/Website-Fingerprinting-Library/checkpoints/CW/{AWF,TF,BAPM,TMWF,VarCNN,ARES,RF,...}/max_f1.pth` | 95 |

各攻击模型特征/长度:DF=DIR/5000、AWF=DIR/3000、TF=DIR/5000(kNN)、BAPM=DIR/8500、TMWF=DIR/30720、VarCNN=DT2/5000、ARES=MTAF/8000、RF=TAM/1800。

### 论文 PDF
`/autodl-fs/data/WFP_CyberSecurity (2).pdf`(32 页,提取文本在 `/tmp/opencode/wfp_pdf.txt`)

---

## 3. TOOB 方法理解

五步流水线(`TOOB_Simulation/`):
1. Direction → Burst(`toob/burst.py:direction_to_burst`,取 sign 压缩成带符号 burst 计数)
2. 聚类生成伪标签(`toob/cluster.py`,默认 greedy 最近邻 + set_size=30)
3. 训练生成器(`toob/train.py`,每个伪标签一个生成器)
4. 生成防御数据集(加扰动,展开回 direction)
5. 评测(DF 检测 accuracy/overhead)

扰动公式:`b_adv = b + G(x) * sign(b)`(非负 padding,方向不变)。
可微投影:`soft/hard/STE` 三种 burst→direction 展开(`toob/burst.py`)。

三个 loss(对应论文 Ld/Ll/Lc):
- Ld(distance/attack):`untargeted_attack_loss`,降真实类置信度
- Ll(logit):`unknown_logit_loss`,开世界,拽向虚拟未知类
- Lc(cost):`overhead_budget_loss`,控制带宽开销

---

## 4. 关键发现(实验过程中的洞察)

1. **原始无条件 generator `G(z)` 学不动**。训练/导出都用随机 z,generator 要学"任意 z → 有效扰动"(本质学 z 不变映射),极难收敛。→ 改为 **conditional generator `G(x)`**(输入 burst,输出 delta)。
2. **attack loss 从 `true_prob` 改 `true_logit`**。PGD 实验证明 true_logit 更有效(80 步 accuracy 0.975→0.54 @ overhead 9%)。
3. **burst padding 方向本身可行**(PGD 直接优化 delta 能骗过 DF)。
4. **encoder 降维 + greedy**:聚类质量指标提升(0.079→0.133)但聚类结果没变(greedy set_size 硬性决定),防御持平。
5. **encoder + kmeans**:聚类指标大幅提升(0.397)但簇严重不均衡(51/12/8/24 网站),防御反而变差(0.36)。
6. **核心结论:均衡匿名集 > 几何紧凑簇**。对 TOOB 防御目标,equally-sized 匿名集比高 silhouette 更重要。
7. **unknown loss 落地方式**:直接用 OW 数据集的 96 类 DF,第 96 类(unmonitored)就是"虚拟未知类",无需额外构造。

---

## 5. 实验结果

### 5.1 闭世界(conditional generator + true_logit)
小数据集,DF accuracy:**0.975 → 0.085~0.127**。

λ_overhead 扫描(20 epoch):
| λ_overhead | accuracy | overhead |
|---|---|---|
| 2.0 | 0.085 | 0.44 |
| 5.0 | 0.108 | 0.32 |
| 10.0 | 0.356 | 0.25 |

### 5.2 开世界(unknown loss)
OW DF + `to_unknown`,λ_unknown=1,λ_overhead=5:
- monitored 被判为 unknown 的比例:**1.05% → 48.95%**(overhead 15.9%)

### 5.3 多攻击模型迁移性(5 个方向类模型)
| 模型 | clean | 闭世界(v2) | 开世界(ow) |
|---|---|---|---|
| DF | 0.9747 | 0.1274 | 0.3516 |
| AWF | 0.9474 | 0.0768 | 0.2389 |
| BAPM | 0.9474 | 0.0958 | 0.3968 |
| TMWF | 0.9653 | 0.0811 | 0.2179 |
| TF | 0.9726 | 0.1105 | 0.3242 |

结论:针对 DF 训练,但 AWF/BAPM/TMWF/TF 等不同架构全部大幅下降(闭世界 8~13%),迁移性极好。

### 5.4 消融实验(回答审稿人)
| 配置 | cw_acc↓ | ow_unknown↑ | overhead |
|---|---|---|---|
| full(α=1,β=1,γ=5) | 0.320 | 0.574 | 0.137 |
| no_Ll(α=1,β=0,γ=5) | 0.457 | 0.337 | 0.127 |
| no_Ld(α=0,β=1,γ=5) | 0.354 | 0.730 | 0.121 |
| g0(γ=0) | 0.310 | 0.593 | 0.146 |
| g1(γ=1) | 0.241 | 0.606 | 0.187 |
| g2(γ=2) | 0.321 | 0.594 | 0.248 |
| g10(γ=10) | 0.321 | 0.583 | 0.152 |

消融结论:
- **Ll 显著改善 open-world**:no_Ll→full,unknown 33.7%→57.4%(+23.7pp)。
- **只有 Ld 不够**:no_Ll 时闭世界 45.7%、开世界 33.7%。
- **Ll 也间接改善闭世界**(45.7%→32.0%),因为拽向 unknown 比 untargeted 降真实类更有效。
- **γ≥5 能压住 overhead(~0.14)且不损防御**;γ 太小(1~2)时 overhead 反而失控。
- α=0(只 Ll)开世界最好(73%)、闭世界 35%;β=0(只 Ld)两者都不好。

---

## 6. 代码改动与版本管理

### 分支
- `main` = 原版(可交付,提交 2a81c65)
- `toob-v2` = 实验版(所有改动)

### toob-v2 提交历史
- `55ff52f` conditional generator (x→delta) + true_logit + 可选 autoencoder 聚类
- `c72c9cf` unknown/open-world logit loss
- `ee00285` lambda_attack 开关(消融用)

### 核心代码改动(均在 toob-v2 分支)
- `toob/generator.py`:BurstGenerator 从 `G(z)` 改 conditional `G(x)`(输入 burst)
- `toob/train.py`:delta=model(x);attack_loss 默认 true_logit;加 lambda_attack/lambda_unknown
- `toob/losses.py`:新增 `unknown_logit_loss`(to_unknown/peto/combined 三种)
- `toob/cluster.py`:提取 greedy 核心,新增 kmeans 聚类 + build_website_profiles + evaluate 支持预计算 profile
- `toob/encoder.py`:新增 MLP autoencoder(可选聚类降维)
- `EXP/02`:加 --use-encoder/--cluster-method/--num-clusters
- `EXP/03`:加 --lambda-attack/--lambda-unknown/--unknown-loss
- `EXP/04/05`:delta=model(x)
- `run_exp_ubuntu.sh`:加 USE_ENCODER/CLUSTER_METHOD/LAMBDA_ATTACK/LAMBDA_UNKNOWN/UNKNOWN_LOSS 等环境变量

### 环境变量速查(run_exp_ubuntu.sh full)
- 数据集:`TRAIN_DATASET`/`VALID_DATASET`(key 默认 X/y)
- 检测器:`DF_CHECKPOINT`/`DF_BUILDER`/`NUM_CLASSES`
- 训练:`FULL_EPOCHS`/`FULL_BATCH_SIZE`/`LAMBDA_OVERHEAD`/`OVERHEAD_THRESHOLD`/`ATTACK_LOSS`
- 开世界:`LAMBDA_UNKNOWN`/`UNKNOWN_LOSS`
- 聚类:`USE_ENCODER`/`CLUSTER_METHOD`/`NUM_CLUSTERS`

---

## 7. 审稿人意见与应对

### 意见 1:缺消融实验(Ld/Ll/Lc)
已做,结果见 5.4。核心:三个 loss 各司其职,Ll 最关键。

### 意见 2:虚拟标签构造细节不足
审稿人问 4 点(多少 unmonitored、与 random padded 的比例、random padding 分布/幅度、输出层表示)。
- Q1 unmonitored 数量:train 32979 条。
- Q4 表示:surrogate 输出层第 M+1(实现里第 96)个单元。
- Q2/Q3 random padded:论文缺口,需用户确认实际做法(方案 A 澄清"纯 unmonitored" 或 方案 B 补充 random padding 细节)。

### 意见 3(待办):时间类攻击模型未评测
VarCNN(DT2)、ARES(MTAF)、RF(TAM)需要时间戳特征,我们的方向序列防御不直接适用,需重构时间戳后评测。

---

## 8. 下一步待办

- [ ] 回复审稿人意见 2(虚拟标签细节,需用户确认 Q2/Q3)
- [ ] 时间类模型(VarCNN/ARES/RF)评测(均匀时间戳重构)
- [ ] γ sweep 用 target_l2 做更干净的 trade-off 曲线(当前 hinge 有噪声)
- [ ] 完整 CW/OW 大数据集上验证(当前是小数据集结论)
- [ ] 正式 open-world 评测(TPR/FPR/PR/ROC)集成进 05 脚本

---

## 9. 临时脚本位置

诊断/评测脚本在 `/tmp/opencode/`:
- `subsample_cw.py` 小数据集生成
- `diag_ste.py`/`diag_cond.py`/`diag_ow.py`/`diag_ow_gen.py` 诊断实验
- `eval_attacks.py` 多攻击模型评测
- `eval_ablation.py` 消融评测
- `run_ablation.sh` 消融运行脚本
- `wfp_pdf.txt` 论文提取文本
