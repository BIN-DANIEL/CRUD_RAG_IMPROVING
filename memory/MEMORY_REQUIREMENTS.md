# Agent Memory 需求记录

> 状态：已实现（领域类 / DAO / task / evaluator / deepseek 客户端均已落地，`keep_alive` 已接自对话逻辑）

## 设计要点（近期变更）

- **对象自我创建**：`Session()` / `Conversation(session_id=None)` / `Memory(session_id=None)` 在 `__init__` 时**自动落库并取得自己的 ID**，外部不传 `id` / `dao`。
- **默认 DAO 单例**：所有落库操作走 `dao.get_dao()` 单例（`models._dao()` 内部延迟导入），不再外部传入。
- **`Conversation(persist=False)`**：仅内存运行，不落库，用于 `keep_alive` 里的 `_long_conversation`（无限增长、永不摘要）。
- **`Memory` / `Conversation` 通过 `Session` 共享同一 `session_id`**，保证 summary 的 `conversation_id` / `memory_id` / `session_id` 三者一致。

## 文件清单

| 文件 | 说明 |
|---|---|
| `models.py` | 领域类：`Session` / `Conversation` / `Memory` / `Summary` / `Query` |
| `dao.py` | 数据访问层 `DAO`（落库 / 生成 UUID / 查询） |
| `schema.sql` | SQLite 建表脚本（表名与类同名，单数） |
| `create_db.py` | 建库脚本（幂等） |
| `memory.db` | 已生成的 SQLite 数据库 |
| `demo.py` | 端到端演示（可运行验证） |
| `memory_evaluator.py` | 记忆增强评估器（继承根目录 `evaluator.BaseEvaluator`） |
| `deepseek_client.py` | DeepSeek 客户端（`quest` / `request`，驱动自对话） |
| `../src/tasks/mem_hallucinated_modified.py` | 记忆版任务（继承 `HalluModified`，拆出 `build_query` / `generate_from_query`） |

---

## 正常使用流程（用户口述，记录）

### 流程总览

两阶段：
1. **阶段一（打分）**：BM25 检索 + 模型回答，得到一个基线分数。
2. **阶段二（模拟对话）**：进入记忆增强的对话循环，验证记忆机制。

### 阶段一：基线打分

> 采用 bm25 检索，让模型回答问题，得出一个分数。（作为后续记忆方案的对比基线）

### 阶段二：模拟对话（伪代码）

> 结构：`run_memory_loop` 先解决第一次 task，前置生成初始问题，之后 `keep_alive` 内部自循环直到 memory 爆。

```python
def run_memory_loop(data_point):
    # ① 第一次解决 task：走 RAG，query 与 real_res 均进 memory
    generated_text = task_generation(data_point)

    # ② 前置：让 deepseek 抛出初始问题
    simulate_query = deepseek.quest("请你结合时下热点，向我抛出一个问题。")
    query_size = 500
    initial_prompt = (f"用不少于{query_size}字回答我下面的问题：{simulate_query}，"
                      f"且以后对我的回复的字数也要在{query_size}左右")

    # ③ keep_alive 运行直到 memory 爆满（循环内置）
    keep_alive(initial_prompt, query_size)

    # ④ 终局：清空 conversation，重问最初问题，不走 RAG
    conversation.clear()
    final_prompt = system_prompt + memory.generate() \
                 + "\n-------------以下是目前的历史\n" + initial_question
    final_res = generate_from_query(final_prompt)   # 不用 retriever，纯凭记忆
    return final_res


def keep_alive(query, query_size):
    _long_conversation = 无限增长、永不摘要的对话（仅内存）
    while memory.size() < MEM_THRESHOLD:      # MEM_THRESHOLD = 2000（token）
        _long_conversation.add(query, 'user')
        conversation.add(query, 'user')       # 每次 summarize 后清空窗口，避免冗余

        response = deepseek.request(query)
        _long_conversation.add(response, 'model')
        conversation.add(response, 'model')

        summary = conversation.summarize()    # 满了才返回 Summary（并清空窗口）
        if summary:
            memory.add_summary(summary)

        query = deepseek.request(
            f"基于下面的对话，生成一个新的query，不少于{query_size}字\n"
            + _long_conversation.history()
        )
```

### 流程要点

- **`run_memory_loop` 职责**：解决第一次 task → 前置初始问题 → `keep_alive` 直到 memory 爆 → 终局测试。
- **`keep_alive` 职责**：自对话循环（循环内置，无外层 while），直到 memory 爆满才返回。
- **`_long_conversation`**：无限增长、永不摘要、仅内存的对话，用于给 deepseek 生成下一个 query。
- **`conversation`**：每次 `summarize()` 清空窗口（`pending_queries` 与 `count` 归零），避免已摘要内容重复。
- **注入顺序（终局）**：`system_prompt + memory.generate() + 分隔线 + initial_question`。
- **触发压缩**：`conversation.summarize()` 显式调用，未满 2000 token 返回 `None`。
- **memory 上限**：`memory.size() >= 2000`（**token 数**）判定记忆已满，进入终局测试（**不走 RAG**）。

### 与当前实现的方法名对照（已对齐）

| 用户口述方法 | 当前实现 | 状态 |
|---|---|---|
| `conversation.add(q)` | `conversation.add(text, role)`（`cache_query` 保留为别名） | ✅ |
| `conversation.history()` | `conversation.history()` | ✅ |
| `conversation.summarize()` | `conversation.summarize()`（显式，未满返回 None） | ✅ |
| `conversation.clear()` | `conversation.clear()` | ✅ |
| `mem.get_memories()` | `memory.get_memories()`（返回 Summary 列表） | ✅ |
| `mem.add_summary(s)` | `memory.add_summary(summary)` | ✅ |
| `mem.size()` | `memory.size()`（token 数，用注入 tokenizer 统计） | ✅ |
| `mem = Memory()`（无参） | `Memory(memory_id, session_id, dao)` | ⚠️ 构造仍需 DAO 生成 ID，见「先落库再取 ID」约定 |

---

## 0. 通用约定：ID 一律由数据库分配（UUID）

- 所有对象（`conversation_id` / `session_id` / `memory_id` / `summary_id` / `query_id`）的 ID **均由数据库生成，类型统一为 UUID（TEXT）**。
- 创建流程统一为：**先在数据库中创建实例（落库），再从对应表中取回 ID**。
- 即：`id = dao.create_xxx(...)` → 后续使用 `id` 作为外键引用，调用方不自造 ID。

---

## 1. Conversation

### 1.1 类定义

```python
class Conversation:
    def __init__(self, conversation_id, session_id):
        ...
```

- 一个 `Conversation` 对应一个 `session_id`（会话）。
- `conversation_id` 由**数据库分配**（先落库再取 ID）。

### 1.2 方法：`cache_query`

```python
def cache_query(self, model=True, text):
    ...
```

- `model=True`  → 该条 `text` 属于**模型（助手）**输出。
- `model=False` → 该条 `text` 属于**用户**输入。

### 1.3 query 对象

`text` 被封装为 **query 对象**，字段如下：

| 字段 | 说明 |
|---|---|
| `query_id` | 唯一 ID（UUID） |
| `timestamp` | 时间戳 |
| `session_id` | 属于哪个会话 |
| `summary_id` | 属于哪个摘要 |
| `role` | `'model'` 或 `'user'` |

### 1.4 token 计数与 count

- 每次 `cache_query` 调用 **qwen tokenizer** 计算该条 `text` 的 token 数。
- 维护一个 `count`，累计所有已缓存 query 的 token 数。
- **`count` 是每个 `Conversation` 实例内的状态**（非全局、非跨会话共享）。

### 1.5 摘要压缩触发

- 若 `count > 2000`，触发**摘要压缩**。
- 触发摘要生成时，写入 **conversation ID** 和 **session ID**（即将被压缩的这批 query 通过 summary 与 session/conversation 关联）。

---

## 2. Session

### 2.1 类定义

```python
class Session:
    def __init__(self, session_id, conversation, memory):
        ...
```

- 一个 `Session` **持有一个 `Memory` 对象**和**一个 `Conversation` 对象**（1:1）。
- 本质上是用于**索引**的容器：把会话（Conversation）与记忆（Memory）关联起来。
- `session_id` 由**数据库分配**（先落库再取 ID）。

---

## 3. Summary

### 3.1 类定义

```python
class Summary:
    def __init__(self, queries: list[Query], conversation_id, session_id, memory_id, level=1):
        ...
```

- `queries`：本次摘要关联的一组 `query` 对象。
- `conversation_id`：属于哪个会话（对话）。
- `session_id`：属于哪个会话（session）。
- `memory_id`：属于哪个记忆。
- `level`：摘要层级，默认 `1`（一级摘要；可扩展为二级/递归摘要）。

### 3.2 与 query 的关联

- 每个 `query` 需加入 `summary_id`，从而把 **summary 与一系列 query 关联**起来。
- 即：`query.summary_id` → 指向生成它的 `Summary`，可据此从摘要反向定位到原始 query。

### 3.3 `generate_content`

- 初始化时会调用 `generate_content()`，**生成摘要内容**。
- 即摘要文本是构造时即时生成的。

---

## 4. Memory

### 4.1 类定义

```python
class Memory:
    def __init__(self, memory_id, session_id):
        ...
```

- 一个 `Memory` **包含一系列 `Summary`**。
- 持有其所属的 `session_id`（`Memory` 属于某个 `Session`）。
- `memory_id` 由**数据库分配**（先落库再取 ID）。
- 当 `Summary` 存入 `Memory` 时，写入 `memory_id`（即 `summary.memory_id`）。

### 4.2 `generate`

- 调用 `generate()` 时，**抽取对应的 summary 文本**（即各 `Summary.content`）。
- 按**从早到晚（时间顺序）**排序后拼接输出。
- 即：Memory 的最终产物是一段按时间先后排列的摘要串，用于注入上下文。

---

## 5. DAO（数据访问层）

> 按常见 Repository / DAO 模式设计，待你审核。

### 5.1 定位

- DAO 是**唯一与数据库打交道的层**，所有「落库 / 取 ID / 查询」都经由此层。
- 上层（Conversation / Session / Memory / Summary）不直接操作 SQL / ES / 文件，只调用 DAO 接口。

### 5.2 接口设计

```python
class DAO:
    # ---- Session ----
    def create_session(self) -> session_id
    def get_session(self, session_id) -> Session

    # ---- Conversation ----
    def create_conversation(self, session_id) -> conversation_id
    def get_conversation(self, conversation_id) -> Conversation

    # ---- Query ----
    def create_query(self, text, timestamp, session_id, role, summary_id=None) -> query_id
    def get_queries_by_summary(self, summary_id) -> list[Query]

    # ---- Summary ----
    def create_summary(self, queries, conversation_id, session_id, memory_id, level, content) -> summary_id
    def get_summaries_by_memory(self, memory_id) -> list[Summary]   # Memory.generate 使用
    def get_summaries_by_session(self, session_id) -> list[Summary]

    # ---- Memory ----
    def create_memory(self, session_id) -> memory_id
    def get_memory(self, memory_id) -> Memory
```

### 5.3 设计说明

- 每个实体的 `create_*` 负责落库并**返回数据库生成的 ID**（对应第 0 节约定）。
- `get_*` 负责按 ID / 外键查询，返回领域对象。
- `Summary.content` 由 `Summary.generate_content()` 生成后，再通过 `create_summary(...)` 落库（内容与元信息一起持久化）。

### 5.4 存储后端（已定）

- 使用 **SQLite**（文件 `memory.db`），主键统一 **UUID（TEXT）**。

---

## 6. 数据库表结构（SQLite）

> 关系：Session 持有 Conversation 与 Memory（各 1:1）；Conversation 包含 Query；Memory 包含 Summary；Summary 汇总 Query；Summary 通过 `parent_summary_id` 自引用支持二级摘要。

### 6.1 `sessions`

| 列 | 类型 | 约束 |
|---|---|---|
| `session_id` | TEXT (UUID) | PK |
| `created_at` | TEXT | |

### 6.2 `conversations`

| 列 | 类型 | 约束 |
|---|---|---|
| `conversation_id` | TEXT (UUID) | PK |
| `session_id` | TEXT (UUID) | FK → sessions，UNIQUE（1:1） |
| `created_at` | TEXT | |

### 6.3 `memories`

| 列 | 类型 | 约束 |
|---|---|---|
| `memory_id` | TEXT (UUID) | PK |
| `session_id` | TEXT (UUID) | FK → sessions，UNIQUE（1:1） |
| `created_at` | TEXT | |

### 6.4 `summaries`

| 列 | 类型 | 约束 |
|---|---|---|
| `summary_id` | TEXT (UUID) | PK |
| `memory_id` | TEXT (UUID) | FK → memories，NOT NULL |
| `conversation_id` | TEXT (UUID) | FK → conversations，NOT NULL |
| `session_id` | TEXT (UUID) | FK → sessions（冗余，反规范化） |
| `parent_summary_id` | TEXT (UUID) | FK → summaries，NULLABLE（自引用） |
| `level` | INTEGER | 默认 1 |
| `content` | TEXT | 摘要文本 |
| `created_at` | TEXT | |

### 6.5 `queries`

| 列 | 类型 | 约束 |
|---|---|---|
| `query_id` | TEXT (UUID) | PK |
| `conversation_id` | TEXT (UUID) | FK → conversations，NOT NULL |
| `summary_id` | TEXT (UUID) | FK → summaries，NULLABLE |
| `role` | TEXT | `'model'` 或 `'user'`（CHECK 约束） |
| `text` | TEXT | 原文 |
| `token_count` | INTEGER | tokenizer 计数 |
| `timestamp` | TEXT | |

### 6.6 关键设计

- `query.summary_id` 始终指向**直接生成它的 level-1 摘要**；二级摘要通过 `parent_summary_id` 间接关联。
- `summary.session_id` / `conversation_id` 为冗余列（反规范化），用于加速按会话查询（`Memory.generate` 高频操作）。
- 排序键：`(timestamp, summary_id)` 保证同时间戳下排序稳定。

---

## 7. 待定项（留空）

- [ ] 摘要压缩的具体算法 / 触发后 count 如何重置
- [ ] `generate_content` 的生成方式（调用哪个 LLM / prompt 模板）
- [ ] `Summary` 的完整字段（是否需要 `token_count`）

---

## 8. 建议补充项（供讨论，非定稿）

1. **`count` 重置策略**：明确「当前未压缩窗口」的 token 累计，压缩后应重置或减去已压缩部分的 token。
2. **原文保留**：`query` 对象除简介外应保留**未压缩原文引用**，召回时按 `query_id` 拉回原文，避免摘要丢细节（呼应 Lost in the Middle 实验）。
3. **区分 user / model 的权重**：摘要时 user 的意图通常比 model 的冗余输出更重要，可考虑分开计数或加权。
4. **时间戳与顺序**：摘要压缩时保留 `timestamp` 与身份（user/model）标记，不能丢失「谁在什么时候说了什么」的顺序感。
5. **遗忘 / 淘汰策略**：`count` 只增不减会有隐患，建议补充重要性打分 + 时间衰减，或固定上限 + LRU。
6. **阈值可配置**：`2000` 建议做成可配置参数（不同模型 context 不同）。
