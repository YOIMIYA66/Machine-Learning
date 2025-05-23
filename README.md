# 🎯 启航者 AI - 智能学习导航助手

一个基于AI大模型与机器学习集成的教育平台，支持个性化学习路径、智能问答、数据分析与实验对比。

---

## 🚀 项目背景与创新点

- **RAG+ML深度融合**：结合检索增强生成（RAG）与多种机器学习模型，既能知识问答又能数据预测。
- **对话式AI体验**：用自然语言即可训练模型、分析数据、生成学习路径。
- **个性化学习路径**：基于知识图谱和用户背景，自动规划循序渐进的学习计划。
- **可解释AI**：所有模型结果均有可视化和解释，降低AI"黑箱"感。
- **模块化架构**：前后端分离，易于扩展和维护。

---

## 🏗️ 系统架构简图

```
┌─────────────┐   浏览器/移动端   ┌─────────────┐
│   前端UI    │◀──────────────▶│   Flask API │
└─────────────┘                  └─────────────┘
         │                              │
         ▼                              ▼
   ┌────────────┐   ┌────────────┐   ┌────────────┐
   │  RAG系统   │   │ ML代理系统 │   │ 学习路径   │
   └────────────┘   └────────────┘   └────────────┘
         │              │                  │
         ▼              ▼                  ▼
   ┌────────────┐   ┌────────────┐   ┌────────────┐
   │ 向量数据库 │   │ ML模型库   │   │ 知识图谱   │
   └────────────┘   └────────────┘   └────────────┘
```

---

## 🔬 主要技术原理

- **RAG（检索增强生成）**：先用向量相似度检索知识，再让大模型生成答案，兼顾准确性与创新性。
- **向量数据库（ChromaDB）**：将文档/知识转为向量，支持高效语义检索。
- **ML代理系统**：大模型自动选择"训练/预测/可视化"等工具，用户一句话即可跑实验。
- **个性化学习路径**：拓扑排序+先验知识，自动生成适合你的学习计划。
- **集成学习**：支持投票法、平均法、堆叠法等多模型集成，提升预测准确率。

---

## 🧩 关键代码片段（核心模块）

**RAG 问答入口**
```python
# rag_core.py
class RAGSystem:
    def query_rag(self, question: str) -> Dict[str, Any]:
        result = self.qa_chain.invoke({"query": question})
        return self.format_response(result)
```

**机器学习代理**
```python
# ml_agents.py
class MLAgent:
    def query_ml_agent(self, question: str) -> Dict[str, Any]:
        result = self.agent.run(question)
        return self.enhance_with_visualization(result)
```

**学习路径生成**
```python
# learning_planner.py
def generate_learning_path(user_id, goal, prior_knowledge, weekly_hours):
    modules = build_learning_path(goal, prior_knowledge)
    total_hours = sum(m['estimated_hours'] for m in modules)
    return {"modules": modules, "total_hours": total_hours}
```

---

## 🏁 新手快速上手（10分钟体验）

1. **克隆并安装依赖**
   ```bash
   git clone <repo>
   cd "Machine Learning"
   python run_app.py   # 自动安装+启动
   ```
2. **访问** `http://localhost:5000`，体验"学习导航"与"数据与模型"功能。
3. **示例提问**：
   - "我想零基础学深度学习，每周10小时，帮我制定路径"
   - "上传air_data.csv，选择随机森林，预测PM2.5"
4. **常见问题FAQ**
   - "Cannot set properties of null"：刷新页面，检查HTML完整
   - "Chart.js未加载"：检查网络或CDN
   - "API调用失败"：检查.env中的API密钥

---

## 📚 术语表（精选）

| 术语 | 通俗解释 |
|------|-----------|
| RAG | 检索增强生成，先查知识再让大模型写答案 |
| 向量嵌入 | 把文本变成数字向量，便于比"语义距离" |
| LLM | 大语言模型，如GPT/ERNIE |
| 集成学习 | 多个模型组合预测，提升准确率 |
| 拓扑排序 | 一种保证先后顺序的排序算法 |
| 可解释AI | 让AI的决策过程变得透明、可理解 |

---

## 🌟 项目亮点速览

| 特性 | 启航者AI | 传统ML平台 |
|------|:--------:|:----------:|
| RAG+ML集成 | ✅ | ❌ |
| 对话式AI | ✅ | ❌ |
| 个性化学习路径 | ✅ | ❌ |
| 可解释AI | ✅ | 部分 |
| 多模型集成 | ✅ | 部分 |
| 响应式UI | ✅ | ❌ |
| 新手友好 | ✅ | ❌ |

---

**启航者 AI** —— 让机器学习教育更智能、更个性化！如需详细原理和代码解析，请查阅《项目介绍文档.md》。