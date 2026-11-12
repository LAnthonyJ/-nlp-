# NLP 课程代码实现

本仓库展示 Transformer、LoRA、解码策略和数学答案评测的实现、修复与测试。

[English README](README.md) · [接口说明](docs/api.md) · [验证记录](docs/validation.md)

## 当前成果

- **29 项测试全部通过**，覆盖掩码信息泄漏、梯度、LoRA 合并、采样边界、评测批次和错误处理。
- Transformer 修复层参数共享、掩码维度、归一化和残差连接，并支持奇数维位置编码。
- LoRA 补上直接构造时的初始化、参数冻结、设备与精度匹配，以及合并状态保存。
- 解码支持 Greedy、Top-K、Top-P、相对最大概率定义的 Min-P 和重复惩罚。
- 评测支持离线 JSONL 输入、精确数值比较、无效答案统计及逐条结果输出。
- 提供无需模型下载的 CPU 示例，包含真实执行的微型 LoRA 优化。

原始代码位于 `coursework/`，保持不变。请使用 `src/nlp_course/` 中的修复版本。

## 在当前电脑运行

在本仓库目录打开 PowerShell，已有 `.venv` 可直接使用：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe examples/cpu_demo.py
.\.venv\Scripts\python.exe -m nlp_course.evaluation examples/toy_predictions.jsonl --output outputs/toy-report.json
```

新电脑的安装步骤见英文 README。当前电脑的虚拟环境复用了已有 CPU PyTorch，并已验证本项目的 editable 安装；未声称在全新环境重新下载验证所有依赖。

## 实际运行结果

测试环境：Windows / Python 3.13.7 / PyTorch 2.9.0+cpu。

CPU 示例中，Transformer 输出形状为 `[2,5,16]`；LoRA 在合成数据上优化 80 步，MSE 从约 `0.045949` 降至 `0.000006364`，合并前后最大输出差约 `1.79e-7`。这些数字只验证小型示例，不代表大模型性能提升。

## 评测边界

`math-strict` 只进行受支持的数值等价或去除首尾格式后的文本比较，不是通用符号数学判等器，也不是官方 MATH 评测。原始脚本缺失的 `util` 没有被冒充恢复。尚未运行真实大模型训练和完整基准评测。

## 贡献说明

五份原始课程实现由你完成。本次仓库整理、修复、测试与文档由 AI 辅助完成，详情见 [贡献说明](docs/provenance.md)。GitHub 仓库：https://github.com/LAnthonyJ/-nlp- 。提交使用 GitHub 隐私邮箱。
