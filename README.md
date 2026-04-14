# 🏥 WardLung Compass（慧脉守护）

> 一个面向病房场景的多角色协作原型系统：患者端、护士端、医生端、家属端统一协同，连接评估、交接、医嘱、消息与知情同意流程。

![患者端-仪表盘](images/患者/仪表盘.png)

## ✨ 项目简介

WardLung Compass 是一个可运行的病房协作原型，围绕肺炎相关护理与临床协作流程设计。

系统目标：
- 让患者表达更结构化。
- 让护士工作流更连续（巡视、录入、评估、交接、转发）。
- 让医生决策更聚焦（Patient 360、证据、风险、计划）。
- 让家属参与更顺畅（查看概览、给患者留言、签署知情同意书）。

## 🧩 核心能力

- 👤 患者端：每日打卡、护理卡片、聊天、收件箱、设置。
- 👩‍⚕️ 护士端：病区看板、生命体征与 MAR、评估生成、交班总结、护士收件箱。
- 👨‍⚕️ 医生端：医生看板、Patient 360、医嘱与计划、医生收件箱。
- 👨‍👩‍👧 家属端：家属总览、给患者留言、知情同意签署、设置。
- 🤖 AI 能力：MedGemma 推理、MedSigLIP 视觉分支、RAG 证据召回（ASR链路保留，可按需启用）。
- 💾 持久化：SQLite（账号、评估、交接、收件箱、风险快照、签字留痕等）。

## 🏗️ 技术栈

- 后端：`Python`、`FastAPI`、`Uvicorn`
- 页面渲染：服务端渲染 HTML + 原生 JS 交互
- 数据层：`SQLite`
- 模型与工具：
- `google/medgemma-1.5-4b-it`
- `google/medsiglip-448`
- `LlamaIndex + FAISS + HuggingFace Embedding`

## 📁 目录结构

```text
.
├─ app.py
├─ requirements.txt
├─ requirements.no_torch.txt
├─ src/
│  ├─ agents/
│  ├─ auth/
│  ├─ store/
│  ├─ tools/
│  ├─ ui/
│  └─ utils/
├─ data/
│  ├─ ward_demo.db
│  ├─ rag/
│  ├─ rag_index/
│  ├─ uploads/
│  └─ agree/
└─ images/
   ├─ 患者/
   ├─ 护士/
   ├─ 医生/
   └─ 家属/
```

## 🚀 快速开始

### 1) 环境准备

- Python `3.10+`（建议 `3.10`）
- 可选 GPU 环境（若启用较完整模型链路）
- 系统依赖：`ffmpeg`

### 2) 安装依赖

```bash
pip install -r requirements.txt
```

如果你已经有可用的 torch/cuda 环境，建议：

```bash
pip install -r requirements.no_torch.txt
```

### 3) 启动服务

```bash
python app.py
```

默认访问：

```text
http://127.0.0.1:8000
```

## 🔐 账号与登录

- 支持登录页自助注册（患者/护士/医生/家属）。
- 默认兜底密码环境变量：`DEMO_DEFAULT_PASSWORD`。
- 默认值：`Demo@123`。

说明：如果你已手动修改数据库账号密码，以数据库实际值为准。

## 🖼️ 界面预览（按端展示）

### 👤 患者端

- 仪表盘（`images/患者/仪表盘.png`）

![患者端-仪表盘](images/患者/仪表盘.png)

- 每日打卡（`images/患者/每日打卡.png`）

![患者端-每日打卡](images/患者/每日打卡.png)

- 护理卡片总览（`images/患者/护理卡片总览.png`）

![患者端-护理卡片总览](images/患者/护理卡片总览.png)

- 护理卡片详细查看（`images/患者/护理卡片详细查看.png`）

![患者端-护理卡片详细查看](images/患者/护理卡片详细查看.png)

- 聊天（`images/患者/聊天.png`）

![患者端-聊天](images/患者/聊天.png)

- 呼叫护士（`images/患者/呼叫护士.png`）

![患者端-呼叫护士](images/患者/呼叫护士.png)

- 收件箱（`images/患者/收件箱.png`）

![患者端-收件箱](images/患者/收件箱.png)

- 设置（`images/患者/设置.png`）

![患者端-设置](images/患者/设置.png)

### 👩‍⚕️ 护士端

- 护士看板（`images/护士/护士看板.png`）

![护士端-护士看板](images/护士/护士看板.png)

- 生命体征与用药 1（`images/护士/生命体征与用药1.png`）

![护士端-生命体征与用药1](images/护士/生命体征与用药1.png)

- 生命体征与用药 2（`images/护士/生命体征与用药2.png`）

![护士端-生命体征与用药2](images/护士/生命体征与用药2.png)

- 生成评估 1（`images/护士/生成评估1.png`）

![护士端-生成评估1](images/护士/生成评估1.png)

- 生成评估 2（`images/护士/生成评估2.png`）

![护士端-生成评估2](images/护士/生成评估2.png)

- 生成评估详情 1（`images/护士/生成评估详情1.png`）

![护士端-生成评估详情1](images/护士/生成评估详情1.png)

- 生成评估详情 2（`images/护士/生成评估详情2.png`）

![护士端-生成评估详情2](images/护士/生成评估详情2.png)

- 交班总结（`images/护士/交班总结.png`）

![护士端-交班总结](images/护士/交班总结.png)

- 交班总结详情（`images/护士/交班总结详情.png`）

![护士端-交班总结详情](images/护士/交班总结详情.png)

- 护士收件箱（`images/护士/护士收件箱.png`）

![护士端-收件箱](images/护士/护士收件箱.png)

### 👨‍⚕️ 医生端

- 医生看板（`images/医生/医生看板.png`）

![医生端-医生看板](images/医生/医生看板.png)

- 患者360 1（`images/医生/患者3601.png`）

![医生端-患者3601](images/医生/患者3601.png)

- 患者360 2（`images/医生/患者3602.png`）

![医生端-患者3602](images/医生/患者3602.png)

- 患者360 3（`images/医生/患者3603.png`）

![医生端-患者3603](images/医生/患者3603.png)

- 患者360 4（`images/医生/患者3604.png`）

![医生端-患者3604](images/医生/患者3604.png)

- 医嘱与计划（`images/医生/医嘱与计划.png`）

![医生端-医嘱与计划](images/医生/医嘱与计划.png)

- 医生收件箱（`images/医生/医生收件箱.png`）

![医生端-医生收件箱](images/医生/医生收件箱.png)

- 设置（`images/医生/设置.png`）

![医生端-设置](images/医生/设置.png)

### 👨‍👩‍👧 家属端

- 家属总览（`images/家属/家属总览.png`）

![家属端-家属总览](images/家属/家属总览.png)

- 给患者留言（`images/家属/给患者留言.png`）

![家属端-给患者留言](images/家属/给患者留言.png)

- 知情同意书（`images/家属/知情同意书.png`）

![家属端-知情同意书](images/家属/知情同意书.png)

- 设置（`images/家属/设置.png`）

![家属端-设置](images/家属/设置.png)

## 🧠 评估链路（简要）

1. 收集患者上下文（日志、体征、用药、历史评估）。
2. 可选处理图像/语音输入。
3. 检索 RAG 证据片段。
4. 生成诊断摘要、审校结果、反向鉴别建议。
5. 输出风险提示、信息缺口与建议动作。

## ⚙️ 常用环境变量

- 服务：`HOST`、`PORT`
- HuggingFace：`HF_ENDPOINT`、`HF_HUB_OFFLINE`、`HF_HUB_DISABLE_XET`
- 模型行为：`FORCE_CUDA`、`MEDSIGLIP_DEVICE`、`FUN_ASR_DEVICE`
- 认证：`DEMO_DEFAULT_PASSWORD`

## ❓ 常见问题

### 启动后页面没变化

这是服务端渲染项目。修改 Python 后请重启 `python app.py`，再刷新浏览器。

### 模型下载慢或失败

建议提前准备 `models/` 目录缓存后再启动，离线运行更稳定。

### 语音链路不可用

当前语音链路是可选能力，不影响文本与图像主流程。

## ⚠️ 免责声明

本项目为临床协作原型系统，用于产品验证、流程演示与技术集成，不可替代正式医疗诊断与处方决策。

---

如果你准备把它部署到云服务器或做二次开发，建议先从 `app.py`、`src/ui/`、`src/agents/` 三个目录开始阅读。
