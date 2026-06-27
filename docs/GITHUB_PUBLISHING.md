# GitHub 公开发布指南

这份指南用于把当前项目发布成 public GitHub repository，方便面试或简历展示。

## 1. 发布前检查

先确认当前仓库状态：

```bash
git status --short
```

重点检查不要提交这些内容：

- 模型权重：`*.safetensors`、`*.bin`、`*.pt`、`*.pth`；
- Triton 编译缓存：`triton_cache*/`；
- Python 缓存：`__pycache__/`；
- 虚拟环境：`.venv/`、`venv/`；
- 私有 token、服务器地址、账号密码等敏感信息。

如果权重文件已经被 `git add` 到暂存区，但还没有 commit，可以从 Git 索引中移除，保留本地文件：

```bash
git rm --cached DS-R1-Distill-Qwen-1.5B/*.safetensors
git rm --cached DS-R1-Distill-Qwen-1.5B/*.bin
```

如果这些文件本来就没有被加入 Git，上面的命令可以不用执行。

## 2. 建议提交内容

建议至少提交：

- `nanonona/`
- `test/`
- `benchmarks/`，包括 `benchmarks/results/`
- `web/`
- `web_server.py`
- `config.yaml`
- `setup.py`
- `requirements.txt`
- `.gitignore`
- `README.md`
- `docs/GITHUB_PUBLISHING.md`
- `DS-R1-Distill-Qwen-1.5B/config.json`
- `DS-R1-Distill-Qwen-1.5B/tokenizer_config.json`

如果 README 里的 Web UI 截图链接需要在 GitHub 上正常显示，也把 `web/页面效果.png` 加入提交。

## 3. 创建 commit

如果这是第一次提交整个项目：

```bash
git add .
git status --short
git commit -m "Initial release of nanonona inference engine"
```

如果你只想先提交文档变更：

```bash
git add README.md requirements.txt .gitignore docs/GITHUB_PUBLISHING.md
git commit -m "docs: add bilingual README and publishing guide"
```

## 4. 创建 public GitHub 仓库

### 方案 A：使用 GitHub CLI

先登录：

```bash
gh auth login
```

在当前目录直接创建 public repo 并 push：

```bash
gh repo create nanonona --public --source=. --remote=origin --push
```

如果你想换仓库名，把 `nanonona` 替换成你的目标 repo 名。

### 方案 B：使用 GitHub 网页

1. 打开 GitHub，点击 New repository。
2. Repository name 填 `nanonona` 或你想展示的名字。
3. Visibility 选择 Public。
4. 不要勾选自动生成 README、`.gitignore` 或 License，避免和本地仓库冲突。
5. 创建完成后，GitHub 会给出远程地址。

然后在本地执行：

```bash
git remote add origin https://github.com/<your-name>/<repo-name>.git
git branch -M main
git push -u origin main
```

如果已经存在 `origin`：

```bash
git remote -v
git remote set-url origin https://github.com/<your-name>/<repo-name>.git
git push -u origin main
```

## 5. 发布后检查

push 成功后，在 GitHub 页面检查：

- README 是否正常渲染；
- Mermaid 架构图是否显示；
- `web/页面效果.png` 是否能打开；
- `benchmarks/results/` 是否能看到结果文件；
- 没有模型权重或敏感文件被上传；
- About 区域可以加 topics，例如 `llm-inference`、`triton`、`paged-attention`、`continuous-batching`、`prefix-cache`。

## 6. 面试展示建议

可以按这条线讲项目：

1. 先说明目标：学习并复现 LLM serving 的关键机制，而不是做生产级 vLLM 替代。
2. 讲控制层：`Scheduler` 如何在 waiting/running 队列之间切换 prefill 和 decode。
3. 讲内存层：`BlockManager` 如何用 paged KV cache 和 block table 支撑非连续 KV 访问。
4. 讲 prefix cache：完整 block 的链式 hash 如何复用共享前缀，benchmark 中 prefill token 从 33792 降到 2048。
5. 讲算子层：Triton 实现了 Linear、RMSNorm、RoPE、SwiGLU 和 prefill/decode attention。
6. 讲验证：机制测试、算子 reference 对齐、Transformers greedy 100% match、vLLM 对比中诚实展示差距。
