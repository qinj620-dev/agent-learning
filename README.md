# agent-learning

我的 Agent 工程师学习仓库 —— 从 2026 年 10 月开始，用"动手做项目"的方式学 Agent 开发。

## 仓库里有什么

| 文件 | 说明 |
|---|---|
| `weather_agent.py` | ReAct 天气助手：工具调用 + 异常处理（城市不存在、工具报错后重试） |
| `Plan_agent.py` | Plan-and-Execute：规划 → 执行 → 汇总，带人工确认环节 |
| `eval_weather.py` | 12 道测试题的 eval 脚本：规则评分 + LLM-as-judge |

## 学习里程碑

- **任务三（Plan-and-Execute）**：规划→执行→汇总三段式跑通；ReAct vs Plan-and-Execute 取舍复盘
- **任务四（Evals）**：规则评分 12/12，LLM-as-judge 平均 5.0/5；亲手抓出"兰兰→兰州静默替换"、"星期四幻觉"等真实 bug，完整走通"跑分 → 定位 → 改进 → 再跑分"

## 本地运行

```bash
# 先设置 DeepSeek API key（不要写进代码）
setx DEEPSEEK_API_KEY "你的key"

python eval_weather.py
- 2026-10-13：第一次从本地 push 成功
```
