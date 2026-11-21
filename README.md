# 剧场扬声器共享 EQ 试算 API

剧场扬声器只能加载**一组**均衡参数。本服务用 FastAPI + SciPy 试算一枚 **32 抽头因果 FIR**，让它同时服务于 2～4 个座位：

- 输入各座位**等长**脉冲响应（采样率相同、已按直达声起点对齐、每条 ≤ 256 点）、目标延时（0～31 样点）、增益上限（≥ 1）；
- 约束 FIR 系数的 L1 范数（绝对值之和）不超过增益上限；
- 以**全长卷积** `h_k * g` 与延时单位脉冲的**均方误差（MSE）**评估每个座位；
- 与同延时直通（单位脉冲 FIR）相比，**每个座位的误差都不得增加**，并在此前提下**最小化最差座位误差**；
- **仅当最差误差确实降低时**才返回系数和各座位前后误差；`no_improvement`（无改善）与 `failed`（求解失败）分别报告。

## 数学模型

设座位 `k` 的脉冲响应为 `h_k`（长 `L`），FIR 为 `g`（32 抽头），全长卷积长度 `N = L + 31`，目标为 `d[n] = δ[n - n0]`（`n0` 为目标延时）：

```
minimize   max_k  (1/N) · ‖ h_k * g − d ‖²
subject to Σ|g_i| ≤ gain_limit
           MSE_k(g) ≤ MSE_k([1,0,…,0])   ∀k      （各座位不得变差）
```

这是凸优化问题（凸二次函数的最大值 + 凸约束），以 epigraph 形式用
`scipy.optimize.minimize(method="SLSQP")` 求解（解析梯度）。

**L1 预算的精确建模。** 早期实现把 `Σ|g| ≤ gain_limit` 写成
`gain_limit − |g|₁ ≥ 0`，并向 SLSQP 提供 `−sign(g)` 作为次梯度。这在
`gain_limit = 1` 时必然失效：最优解正好落在 L1 球面的非光滑边界上，
SLSQP 线搜索停滞，最终以 "Iteration limit reached" 报 `failed`，尽管
附近已有可用解。现在改用拆分变量 `g = g₊ − g₋`（`g₊, g₋ ≥ 0`），预算
约束变为**线性**的 `Σ(g₊ + g₋) ≤ gain_limit`，处处可微，13 次迭代即可
收敛。

**返回前的独立复核。** 求解器迭代出的候选系数先按预算缩放（若 SLSQP
有微小越界），再用**全长卷积**逐座位重新计算误差；若某个活跃约束的
误差比直通高出几个 ulp，会沿该座位的梯度做阻尼修正并重新投影到预算内。
只有复核后同时满足"各座位误差 ≤ 直通"且"最差误差严格下降（相对阈值
`1e-9·max(1, worst_before)`）"时才返回 `improved`，报告中所有前后误差
都来自这同一次全长卷积复核。直通解 `[1,0,…,0]` 的 L1 范数为 1，因此
`gain_limit ≥ 1` 时约束集恒非空。

## API

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/v1/eq/optimize` | 试算共享 FIR（三种结果状态见下） |
| GET  | `/v1/eq/example`  | 返回一个可直接 POST 的示例请求体 |
| GET  | `/health`         | 健康检查 |
| GET  | `/docs`           | Swagger UI 交互式文档 |

### 请求体

```json
{
  "impulse_responses": [[1.0, 0.5], [1.0, -0.5]],
  "target_delay": 1,
  "gain_limit": 2.0,
  "sample_rate": 48000
}
```

- `impulse_responses`：2～4 条等长脉冲响应，每条 1～256 点，数值须有限；
- `target_delay`：整数，0～31；
- `gain_limit`：≥ 1；
- `sample_rate`：可选，仅作元数据（各条脉冲响应采样率本就相同）。

### 响应（`status` 判别字段，HTTP 均为 200；参数非法为 422）

- `improved`：返回 `coefficients`（32 个系数）、`l1_norm`、各座位
  `error_before`/`error_after`、`worst_error_before`/`worst_error_after` 等；
- `no_improvement`：不返回系数，只给出各座位直通基线误差与说明；
- `failed`：求解器异常/未收敛，返回 `detail` 说明。

## 运行

### Docker Compose（推荐）

```bash
docker compose up --build
# 服务监听 http://127.0.0.1:8000 ，文档在 http://127.0.0.1:8000/docs
```

镜像和 Compose 都固定了 `OMP_NUM_THREADS=1` / `OPENBLAS_NUM_THREADS=1` /
`MKL_NUM_THREADS=1`。SLSQP 每步只解很小的稠密系统，多线程 BLAS 在此
规模下反而慢一个数量级（24 线程时约 0.7 s，单线程约 28 ms）。

### 复现报告中的请求

服务启动后直接执行（请求体已放在仓库中）：

```bash
curl -X POST http://127.0.0.1:8000/v1/eq/optimize \
  -H 'Content-Type: application/json' \
  -d @examples/request_reported.json
```

修复前该请求返回 `{"status":"failed", ... "optimizer did not converge:
Iteration limit reached"}`；现在返回 `improved`，32 个系数满足
`sum|g| ≤ 1`，两座位误差分别为 0.028237 → 0.019263 与 0.006671 → 0.006671
（不高于直通），最差误差 0.028237 → 0.019263。

### 本地开发

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt
uvicorn app.main:app --reload --port 8000
```

### 演示

```bash
./examples/demo.sh                 # 依次演示报告案例/改进/无改善/422
curl -X POST http://127.0.0.1:8000/v1/eq/optimize \
  -H 'Content-Type: application/json' \
  -d @examples/request_improved.json

# 报告中的紧预算案例（gain_limit = 1），修复后应返回 improved
curl -X POST http://127.0.0.1:8000/v1/eq/optimize \
  -H 'Content-Type: application/json' \
  -d @examples/request_reported.json
```

### 测试

```bash
python -m pytest tests/ -v
```

## 项目结构

```
app/
  main.py      # FastAPI 应用、路由、422 异常处理
  schemas.py   # 请求/响应模型（Pydantic 校验）
  solver.py    # 凸优化求解（SciPy SLSQP，split-L1 epigraph + 全长卷积复核）
tests/test_api.py
examples/      # 可直接 POST 的示例请求（含报告案例）与 demo 脚本
Dockerfile
docker-compose.yml
requirements.txt / requirements-dev.txt
```

## 设计说明与限制

- MSE 在全长卷积（`L + 31` 点）上取平均；目标为单位幅度延时脉冲。
- "改善"判定阈值：`worst_after < worst_before − 1e-9·max(1, worst_before)`，
  避免把数值噪声当作改善。
- "各座位不变差"按全长卷积复核结果判定；对直通误差本身已为 0 的座位，
  允许 `1e-9·max(1, error_before)` 的浮点容差（任何 FIR 都无法使其更低）。
- 若求解器未收敛但最终迭代点经核验可行且严格更优，仍会返回 `improved`
  （`solver_success=false` 并在 `detail` 中注明）；无法得到可信结果时返回 `failed`。
- 单线程同步计算：问题规模很小（65 个变量），单次请求通常在几十毫秒内。
- `sample_rate` 不参与计算，仅作元数据；如需按频率加权或限制相位，需要扩展模型。
