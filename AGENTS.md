# Repository Guidelines

## 项目结构与模块组织
仓库根目录仅包含 `main.py`，该文件提供最小化的示例入口。未来扩展时，请在 `src/` 下按功能域拆分模块，并将实验脚本放入 `experiments/`。IDE 元数据保存在 `.idea/`，请勿提交个人化配置。若新增数据或模型权重，请使用 `data/` 与 `models/` 并通过 `.gitignore` 排除大文件。

## 构建、测试与开发命令
- `python3 -m venv .venv && source .venv/bin/activate`：创建并启用隔离环境，避免污染全局依赖。
- `pip install -r requirements.txt`：安装依赖。若文件不存在，请在添加新依赖前生成最小列表。
- `python3 main.py`：运行示例脚本，验证仓库可执行状态。

## 编码风格与命名约定
遵循 PEP 8：4 空格缩进、每行不超过 100 列。模块和函数使用 `snake_case`，类使用 `PascalCase`，常量全大写。保持单一职责，避免在同一文件混合训练、评估和数据预处理逻辑。提交前请使用 `ruff` 或 `flake8` 进行静态检查，使用 `black` 做可选格式化。

## 测试指南
建议使用 `pytest`。测试文件放在 `tests/`，命名为 `test_<module>.py`，用 `Test` 前缀的类组织场景。保持快速、可重复，必要时使用固定随机种子。面向深度学习模块时，为模型推断添加最小化的烟雾测试，并通过 `pytest --maxfail=1 --disable-warnings` 作为 PR 前置校验。

## 提交与 Pull Request 指南
采用 Conventional Commits，例如 `feat: 添加卷积网络骨架` 或 `fix: 修正数据加载路径`。每次提交聚焦单一变更，附带必要的文档与脚本更新。提交 PR 时，请在描述中列出变更摘要、验证步骤及潜在回归风险；若包含可视化结果，请附图或记录路径。至少等待一名同伴确认后合并。

## 安全与配置提示
避免将 API 密钥、模型权重等敏感文件提交到仓库，可通过 `.env` 加载。对外部数据源进行校验，并在 README 或 `docs/` 目录说明复现步骤及资源位置。