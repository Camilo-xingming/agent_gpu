# Contributing to RalphGPU

> 开发工作流和合并后验证标准操作流程 (SOP)。

## Pre-merge: PR 提交前检查

每个 PR 提交前，coder 必须在本地通过以下检查：

```bash
make lint          # Verilator lint，0 error
make test          # 全部 testbench pass
```

如果修改了 RTL 文件，还需运行：

```bash
python3 tools/rtl_frm_compare.py --all   # RTL-FRM 对比
```

PR 描述必须使用 `.github/PULL_REQUEST_TEMPLATE.md` 模板，勾选所有适用的 checklist 项。

## Post-merge SOP: 合并后验证

**每个 PR 合并到 master 后，提交 PR 的 coder 必须执行以下验证：**

### 步骤

1. **确认 CI 通过**：检查 merge commit 的 CI 状态
   ```bash
   gh run list --repo ssql2014/RalphGPU --branch master --limit 1
   ```

2. **本地回归测试**：
   ```bash
   make regression
   ```

3. **本地 lint**：
   ```bash
   make lint
   ```

4. **报告结果**：在对应 issue 中 comment 结果
   ```
   Post-merge verification:
   - CI: green (run #XXXXX)
   - make regression: PASS
   - make lint: 0 errors
   ```

5. **如果失败**：立即开 fix PR 或 revert merge commit，并在 #ralphgpu-dev 通知团队

### 常见失败原因及预防

| 失败类型 | 占比 | 预防措施 |
|----------|------|----------|
| Verilog 语法错误 | 18% | 提交前运行 `make lint` |
| Makefile 格式错误 | 9% | 使用 tab 缩进，不要用空格 |
