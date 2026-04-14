# 🏥 WardLung Compass（慧脉守护）

> 一个面向病房场景的多角色协作原型系统：患者端、护士端、医生端、家属端统一协同，连接评估、交接、医嘱、消息与知情同意流程。

![登录页](images/登录页.png)

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

| 界面 | 界面 |
|---|---|
| **仪表盘**<br/>患者当前状态总览与关键入口。<br/><img src="images/患者/仪表盘.png" alt="患者端-仪表盘" width="100%" /> | **每日打卡**<br/>结构化提交当日症状与主观感受。<br/><img src="images/患者/每日打卡.png" alt="患者端-每日打卡" width="100%" /> |
| **护理卡片总览**<br/>查看护理建议与优先级列表。<br/><img src="images/患者/护理卡片总览.png" alt="患者端-护理卡片总览" width="100%" /> | **护理卡片详情**<br/>展开查看单条建议的细节与说明。<br/><img src="images/患者/护理卡片详细查看.png" alt="患者端-护理卡片详细查看" width="100%" /> |
| **聊天**<br/>支持文本/图像等输入进行健康问答。<br/><img src="images/患者/聊天.png" alt="患者端-聊天" width="100%" /> | **呼叫护士**<br/>一键提交护理请求并附加补充信息。<br/><img src="images/患者/呼叫护士.png" alt="患者端-呼叫护士" width="100%" /> |
| **收件箱**<br/>接收医生/护士消息并查看历史通知。<br/><img src="images/患者/收件箱.png" alt="患者端-收件箱" width="100%" /> | **设置**<br/>管理语言、账号信息与密码等偏好。<br/><img src="images/患者/设置.png" alt="患者端-设置" width="100%" /> |

### 👩‍⚕️ 护士端

| 界面 | 界面 |
|---|---|
| **护士看板**<br/>病区患者状态、风险与任务集中查看。<br/><img src="images/护士/护士看板.png" alt="护士端-护士看板" width="100%" /> | **生命体征与用药 1**<br/>录入体征与给药信息，更新临床上下文。<br/><img src="images/护士/生命体征与用药1.png" alt="护士端-生命体征与用药1" width="100%" /> |
| **生命体征与用药 2**<br/>连续记录与校验关键生理指标。<br/><img src="images/护士/生命体征与用药2.png" alt="护士端-生命体征与用药2" width="100%" /> | **生成评估 1**<br/>发起评估生成并查看摘要结果。<br/><img src="images/护士/生成评估1.png" alt="护士端-生成评估1" width="100%" /> |
| **生成评估 2**<br/>补充输入后刷新评估结论。<br/><img src="images/护士/生成评估2.png" alt="护士端-生成评估2" width="100%" /> | **评估详情 1**<br/>查看评估细节、证据与建议动作。<br/><img src="images/护士/生成评估详情1.png" alt="护士端-生成评估详情1" width="100%" /> |
| **评估详情 2**<br/>对评估文本进行人工核对与编辑。<br/><img src="images/护士/生成评估详情2.png" alt="护士端-生成评估详情2" width="100%" /> | **交班总结**<br/>自动生成交接摘要用于班次衔接。<br/><img src="images/护士/交班总结.png" alt="护士端-交班总结" width="100%" /> |
| **交班总结详情**<br/>查看 SBAR 结构化交班内容。<br/><img src="images/护士/交班总结详情.png" alt="护士端-交班总结详情" width="100%" /> | **护士收件箱**<br/>处理患者请求、分发与状态回写。<br/><img src="images/护士/护士收件箱.png" alt="护士端-护士收件箱" width="100%" /> |

### 👨‍⚕️ 医生端

| 界面 | 界面 |
|---|---|
| **医生看板**<br/>医生视角下的病区优先级与待处理项。<br/><img src="images/医生/医生看板.png" alt="医生端-医生看板" width="100%" /> | **患者360 1**<br/>患者全景信息与时间线入口。<br/><img src="images/医生/患者3601.png" alt="医生端-患者3601" width="100%" /> |
| **患者360 2**<br/>查看风险标签、证据与评估摘要。<br/><img src="images/医生/患者3602.png" alt="医生端-患者3602" width="100%" /> | **患者360 3**<br/>复核临床建议与模型输出细节。<br/><img src="images/医生/患者3603.png" alt="医生端-患者3603" width="100%" /> |
| **患者360 4**<br/>继续查看补充信息与历史上下文。<br/><img src="images/医生/患者3604.png" alt="医生端-患者3604" width="100%" /> | **医嘱与计划**<br/>编辑医嘱并生成患者友好版计划。<br/><img src="images/医生/医嘱与计划.png" alt="医生端-医嘱与计划" width="100%" /> |
| **医生收件箱**<br/>处理上行请求并进行回复与流转。<br/><img src="images/医生/医生收件箱.png" alt="医生端-医生收件箱" width="100%" /> | **设置**<br/>维护医生侧账号与系统偏好。<br/><img src="images/医生/设置.png" alt="医生端-设置" width="100%" /> |

### 👨‍👩‍👧 家属端

| 界面 | 界面 |
|---|---|
| **家属总览**<br/>非专业视角查看患者近况与待办。<br/><img src="images/家属/家属总览.png" alt="家属端-家属总览" width="100%" /> | **给患者留言**<br/>给患者发送关怀或补充信息。<br/><img src="images/家属/给患者留言.png" alt="家属端-给患者留言" width="100%" /> |
| **知情同意书**<br/>接收医生同意书并完成电子签字。<br/><img src="images/家属/知情同意书.png" alt="家属端-知情同意书" width="100%" /> | **设置**<br/>管理家属信息与账户密码。<br/><img src="images/家属/设置.png" alt="家属端-设置" width="100%" /> |

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
