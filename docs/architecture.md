# Secure INT8 AI Accelerator Tile 架构说明

## 1. 项目定位

本项目不是完整商用 NPU，而是一个可综合、可验证的小型 AI 芯片计算 Tile。主线是展示量化推理芯片的数据通路、控制通路、存储层次和后处理单元；算法适配与安全监控均为扩展功能。

## 2. 总体架构

```text
                 配置/命令接口
                       │
       ┌───────────────▼────────────────┐
       │ Controller + Status + Counter  │
       │ IDLE/PREP/COMPUTE/ACTIVATE/DONE│
       └───────┬───────────────┬────────┘
               │               │
      ┌────────▼───────┐ ┌─────▼──────────┐
      │ Activation SRAM│ │ Weight SRAM     │
      │ 4 × signed INT8│ │ 4 × signed INT8│
      └────────┬───────┘ └─────┬──────────┘
               └───────┬───────┘
                       ▼
        ┌───────────────────────────────┐
        │ 2×2 Output-Stationary Array   │
        │ PE00 ───────────────► PE01    │
        │  │                    │       │
        │  ▼                    ▼       │
        │ PE10 ───────────────► PE11    │
        │ INT8 × INT8 → INT32 MAC       │
        └──────────────┬────────────────┘
                       ▼
             ┌───────────────────┐
             │ Per-channel Bias  │
             │ ReLU / Bypass     │
             │ Shift + Saturate  │
             └──────┬───────┬────┘
                    │       │
                 INT32    INT8
                 Debug    Inference
```

安全监控器位于配置接口与缓冲区旁路，对写入时机和存储奇偶校验进行检查；它不是主计算数据通路。

## 3. 推理公式

```text
acc[i,j] = sum(A[i,k] × W[k,j])
biased[i,j] = acc[i,j] + bias[j]
C32[i,j] = ReLU_or_bypass(biased[i,j])
C8[i,j] = saturate_INT8(ReLU_or_bypass(biased[i,j] >>> shift))
```

Bias 按输出列共享，等价于神经网络中的输出通道 Bias。保留 C32 便于调试，同时输出 C8 供下一层量化计算使用。

## 4. 数据流

阵列采用 output-stationary：部分和固定保存在 PE 内，激活值向右移动，权重向下移动。第二行激活值与第二列权重延迟一个周期注入，使正确数据在正确时钟相遇。

| COMPUTE 周期 | row0 | row1 | col0 | col1 |
|---:|---:|---:|---:|---:|
| 0 | A00 | 0 | W00 | 0 |
| 1 | A01 | A10 | W10 | W01 |
| 2 | 0 | A11 | 0 | W11 |
| 3 | 0 | 0 | 0 | 0（排空） |

## 5. 控制状态机

```text
IDLE → PREP → COMPUTE → ACTIVATE → DONE → IDLE
```

- `IDLE`：加载激活、权重、Bias 和量化参数。
- `PREP`：清空阵列流水寄存器和累加器。
- `COMPUTE`：4 个周期完成注入、MAC 和排空。
- `ACTIVATE`：Bias、激活、重定标、饱和及结果写回。
- `DONE`：产生一个周期完成信号。

一次任务需要 6 个内部周期。后处理与结果写回合并在 ACTIVATE 状态，因此新增 Bias 和 INT8 重量化没有增加周期数。

## 6. 性能分析

| 指标 | 数值 |
|---|---:|
| PE 数量 | 4 |
| 峰值吞吐 | 4 MAC/周期 |
| 单个 2×2 乘法有效 MAC | 8 |
| COMPUTE 周期 | 4 |
| 单 Tile 阵列利用率 | 8/(4×4)=50% |
| 端到端周期 | 6 |

若时钟为 100 MHz，理论峰值为 400 MMAC/s；若将一次乘法和一次加法分别计为一个 operation，则为 0.8 GOPS。2×2 阵列的填充/排空占比较高，因此本项目重点是架构原理而不是绝对性能。

## 7. 文件与模块映射

| 模块 | 作用 |
|---|---|
| `mac_pe.v` | 单个 INT8×INT8→INT32 PE |
| `systolic_array_2x2.v` | 2×2 output-stationary 数据通路 |
| `relu.v` | INT32 ReLU/旁路 |
| `requantize_int8.v` | 右移重定标与 INT8 饱和 |
| `ai_accelerator_top.v` | 缓冲区、控制器、后处理、安全监控 |
| `ai_accelerator_tb.v` | 功能、随机、饱和与安全验证 |
| `adapter_tb.v` | 算法适配文件驱动验证 |

## 8. 可继续发展的方向

合理的下一步是将阵列参数化为 4×4、加入双缓冲隐藏加载延迟，或增加简单 valid/ready 流接口。AXI、DMA 和片外内存控制器应作为更后期的 SoC 集成工作。
