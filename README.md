# SystolicGuard — Secure INT8 AI Accelerator Tile

## 项目介绍

**SystolicGuard** 是一个面向神经网络量化推理的可综合、可验证小型 AI 加速器 Tile。核心计算单元为 2×2 INT8 脉动阵列，支持有符号矩阵乘法、按输出通道 Bias、ReLU/旁路、定点重缩放与 INT8 饱和，并同时提供 INT32 调试结果和 INT8 推理结果。

项目覆盖 RTL 设计、测试平台、Python/PyTorch 黄金模型适配及 Yosys 综合，形成从算法参考、位精度验证到硬件实现与资源评估的完整闭环。配置侧还加入了非法写入检测、存储奇偶校验和安全清除，用于演示轻量级硬件安全机制。

![SystolicGuard Architecture](docs/architecture.svg)

## 技术路线

1. **低精度数据表示**：激活和权重采用 signed INT8，乘法结果扩展后使用 INT32 累加，兼顾推理效率与累加精度。
2. **并行计算阵列**：构建 4 个 MAC PE 组成的 2×2 output-stationary 脉动阵列；激活值向右传播、权重向下传播，部分和固定保存在 PE 内。
3. **量化后处理**：在矩阵乘法后依次执行按输出通道 Bias、ReLU/旁路、算术右移和 INT8 饱和，并保留双精度输出：

   ```text
   C32 = ReLU_or_bypass(A × W + bias)
   C8  = saturate_INT8(ReLU_or_bypass((A × W + bias) >>> shift))
   ```

4. **控制与性能计数**：使用 `IDLE → PREP → COMPUTE → ACTIVATE → DONE` 五状态控制器协调加载、计算和结果写回；单次任务需要 6 个内部周期，峰值吞吐为 4 MAC/周期。
5. **轻量安全防护**：限制计算期间的配置写入，对激活、权重和 Bias 缓冲区执行奇偶校验，并支持故障锁存与安全清除。
6. **软硬件协同验证**：通过 Verilog 测试平台覆盖基础计算、有符号数据、随机回归、量化边界和安全攻击场景；使用 Python/PyTorch 生成参考结果并进行 RTL 位精度对比，最后通过 Yosys 完成综合与资源统计。

## 实验结果

### RTL 功能回归

使用 Icarus Verilog 运行测试平台，10 组测试全部通过，其中包含 20 组随机架构回归。测试范围如下：

| 测试类别 | 验证内容 | 结果 |
|---|---|---|
| 基础计算 | 2×2 矩阵乘法、signed INT8、ReLU 与旁路 | 通过 |
| 量化后处理 | 按通道 Bias、算术右移、正负 INT8 饱和 | 通过 |
| 随机回归 | 随机激活、权重、Bias、ReLU 和 shift，共 20 组 | 通过 |
| 安全机制 | 非法写入拦截、权重奇偶校验、`secure_clear` 清除 | 通过 |

仿真最终输出：

```text
TEST 7: 20 randomized architecture regressions
  PASS 20 randomized cases
TEST 8: illegal write attack is blocked and logged
  PASS illegal write blocked, fault_code=1
TEST 9: parity detects weight fault injection
  PASS weight tamper detected, execution refused
TEST 10: secure_clear zeroizes architectural state
  PASS secure zeroization complete
ALL TESTS PASSED
```

### 算法适配波形

下图展示了 Python/PyTorch 参考输入驱动 RTL 后的关键波形。示例配置为：

```text
A    = [1, 2; 3, 4]
W    = [5, 6; 7, 8]
bias = [5, -10]
shift = 1, ReLU = true

C32 = [24, 12; 48, 40]
C8  = [12,  6; 24, 20]
```

波形中 `cycle_count=6` 时 `done` 拉高，四个地址依次读出上述 INT32 与 INT8 结果，且 `security_fault=0`。

![RTL adapter waveform](docs/adapter_waveform.png)

### Yosys 综合结果

以 `ai_accelerator_top` 为顶层进行 Yosys 通用逻辑综合，统计结果如下。该结果是综合后的通用 cell 数量，不等同于特定 FPGA 的 LUT/DSP 映射结果。

| 指标 | 数量 |
|---|---:|
| Cells | 351 |
| Multipliers (`$mul`) | 4 |
| Adders (`$add`) | 10 |
| Multiplexers (`$mux`) | 162 |
| 带使能触发器 (`$adffe`) | 45 |
| 算术右移单元 (`$sshr`) | 4 |
| Wires / Wire bits | 446 / 3891 |
