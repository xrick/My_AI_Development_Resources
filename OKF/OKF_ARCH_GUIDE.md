這是一份為你的 **AI Coding Partner（例如 Cursor、GitHub Copilot Workspace、Claude Code 或自訂 Agent）** 精心整理的架構設計與實作指南文件。

你可以直接複製下方內容並儲存為 **`OKF_ARCH_GUIDE.md`** 放在專案根目錄中，作為 AI Code 夥伴進行開發時的系統架構指引（System Prompt / Knowledge Context）。

---

```markdown
# Open Knowledge Format (OKF) 企業級 AI Agent 架構設計與實作指南

## 1. Executive Summary & Design Philosophy
本專案採用 **Open Knowledge Format (OKF)** 作為 AI Agent 與企業知識庫的單一事實來源（Source of Truth）。
OKF 採用 **「極簡、檔碼化（As Code）、無 Vendor 綁定」** 的設計哲學：

* **Format, not Platform**：完全基於本地檔案系統與 Git 版控，不需專屬 SDK 或封閉平台。
* **Single Concept Principle**：每個 Markdown 檔案代表一個獨立的「概念 (Concept)」，嚴禁將多個獨立主題混在同一檔案中。
* **Minimal Opinionation**：YAML Frontmatter 中**唯一強制要求的欄位只有 `type`**，其餘可自由擴充。
* **Path-as-Identity**：概念的唯一識別碼 (URI) 即為其相對於 Bundle 根目錄的相對檔案路徑（如 `datasets/orders.md`）。

---

## 2. Directory Structure & Naming Conventions

### 2.1 Standard Knowledge Bundle Layout
```text
acme_knowledge_bundle/
├── okf.yaml                     # [選填] Bundle 層級元數據 (定義 spec_version, author)
├── log.md                       # [保留檔名] 知識庫變更日誌 (Changelog)
├── datasets/                    # [概念分類目錄] 小寫 kebab-case
│   ├── customers.md
│   └── orders.md
├── metrics/
│   ├── arr.md
│   └── customer-churn.md
└── runbooks/
    └── handle-failed-payout.md

```

### 2.2 Filename & Linking Standard

* **檔名規範**：全小寫英文字母，單字間以連字號 `-` 分隔（例如：`customer-churn.md`）。
* **跨文件連結 (Cross-linking)**：使用標準 Markdown **相對路徑連結**：
```markdown
Refer to the [Orders Dataset](../datasets/orders.md) and [Failed Payout SOP](../runbooks/handle-failed-payout.md).

```



---

## 3. System Architecture & Context Gateway

本架構將 OKF 與 **Vector DB**、**Graph Traversal** 及 **Prompt Caching** 結合，打造高精準、低延遲、低 Token 成本的 Production-ready RAG 系統。

### 3.1 Data Flow Diagram

```text
[ OKF Bundle (Git) ] ──► Ingestion Pipeline
                              │
                              ├──► 1. Metadata -> Vector DB (Payload Filter)
                              ├──► 2. Document Body -> Vector DB (Dense Embedding)
                              └──► 3. Cross-links -> In-Memory Graph Index
                              
[ User Query ] ──► Context Gateway
                        │
                        ├── Step 1: Pre-filtering (Metadata Query)
                        ├── Step 2: Vector Search -> Primary Concept (Top-1)
                        ├── Step 3: Graph Traversal -> Expand 1-2 Hops
                        │
                        ▼
                [ Prompt Caching Layer ]
                ┌──────────────────────────────────────┐
                │ Block 1: Static System Guardrails     │ (90%+ Cache Hit)
                │ Block 2: Retrieved OKF Context        │ (Ephemeral Cache)
                │ Block 3: Dynamic User Query           │ (Full Cost Token)
                └──────────────────────────────────────┘
                        │
                        ▼
                  [ LLM Inference ]

```

### 3.2 Memory Architecture Mapping (The Hardware Metaphor)

| 計算機記憶體階層 (Hardware) | AI Agent Context 階層 | 實作策略 |
| --- | --- | --- |
| **SRAM (L1/L2/L3 Cache)** | **Prompt Caching (Prefix Cache)** | 固定靜態規範與熱點 OKF，節省 ~90% Token 成本與首字延遲。 |
| **DRAM (Main Memory)** | **Active Context Window** | 經 Context Gateway 檢索並透過 Graph 展開載入的 OKF 上下文。 |
| **Disk (SSD/NVMe)** | **Vector DB & OKF Git Repository** | 長期持久化存儲，提供高維度搜尋索引。 |

---

## 4. Core Implementation Specifications

### 4.1 Python OKF Parser & Graph Ingestion Engine

AI Code 夥伴在編寫後端檢索模組時，請參考以下核心實作邏輯：

```python
import os
import re
import frontmatter
from typing import Dict, List, Set, Tuple
from pydantic import BaseModel, Field

# 1. OKF Frontmatter Schema 驗證
class OKFFrontmatter(BaseModel):
    type: str                                  # 唯一強制欄位
    title: str | None = None
    description: str | None = None
    tags: List[str] = Field(default_factory=list)
    status: str | None = "active"

# 2. OKF 知識庫解析與圖譜建構引擎
class OKFKnowledgeEngine:
    def __init__(self, bundle_root: str):
        self.bundle_root = bundle_root
        self.concepts: Dict[str, dict] = {}   # key: relative_path
        self.graph: Dict[str, Set[str]] = {}  # Adjacency list for links
        self._load_bundle()

    def _load_bundle(self):
        for root, _, files in os.walk(self.bundle_root):
            for file in files:
                if file.endswith(".md") and file not in ["log.md", "okf.yaml"]:
                    abs_path = os.path.join(root, file)
                    rel_path = os.path.relpath(abs_path, self.bundle_root).replace("\\", "/")
                    
                    post = frontmatter.load(abs_path)
                    meta = OKFFrontmatter(**post.metadata)
                    
                    self.concepts[rel_path] = {
                        "metadata": meta.model_dump(),
                        "content": post.content,
                        "type": meta.type
                    }
                    self.graph[rel_path] = self._extract_relative_links(abs_path, post.content)

    def _extract_relative_links(self, current_file_abs: str, content: str) -> Set[str]:
        links = set()
        matches = re.findall(r'\[.*?\]\((?!http)(.*?\.md)\)', content)
        current_dir = os.path.dirname(current_file_abs)
        
        for link in matches:
            target_abs = os.path.normpath(os.path.join(current_dir, link))
            if os.path.exists(target_abs):
                target_rel = os.path.relpath(target_abs, self.bundle_root).replace("\\", "/")
                links.add(target_rel)
        return links

    def retrieve_with_graph_expansion(self, primary_rel_path: str, max_hops: int = 1) -> List[dict]:
        """雙軌檢索：從 Vector DB 查出的起點，沿 OKF 圖譜展開 1~2 Hops"""
        visited = set()
        queue = [(primary_rel_path, 0)]
        results = []

        while queue:
            current_path, depth = queue.pop(0)
            if current_path in visited or current_path not in self.concepts:
                continue
            
            visited.add(current_path)
            concept_data = self.concepts[current_path]
            results.append({
                "path": current_path,
                "metadata": concept_data["metadata"],
                "content": concept_data["content"]
            })

            if depth < max_hops:
                for neighbor in self.graph.get(current_path, []):
                    if neighbor not in visited:
                        queue.append((neighbor, depth + 1))
        return results

```

---

## 5. CI/CD Quality Gate (GitHub Actions)

為確保 AI Agent 產出或人類工程師編寫的 OKF 檔案合規，必須在 CI/CD 中加入以下校驗腳本：

```yaml
# .github/workflows/okf-lint.yml
name: OKF Knowledge Bundle Validation
on: [push, pull_request]

jobs:
  okf-validation:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install Dependencies
        run: pip install python-frontmatter pydantic
      - name: Validate OKF Frontmatter & Dead Links
        run: |
          python -c "
          import os, sys, frontmatter, re

          bundle_root = '.'
          errors = []

          for root, _, files in os.walk(bundle_root):
              for f in files:
                  if f.endswith('.md') and f not in ['log.md', 'README.md']:
                      path = os.path.join(root, f)
                      post = frontmatter.load(path)
                      if 'type' not in post.metadata:
                          errors.append(f' Missing type in Frontmatter: {path}')
                      
                      # Dead link checking
                      links = re.findall(r'\[.*?\]\((?!http)(.*?\.md)\)', post.content)
                      for l in links:
                          target = os.path.normpath(os.path.join(os.path.dirname(path), l))
                          if not os.path.exists(target):
                              errors.append(f' Dead link in {path} -> {l}')

          if errors:
              print('\n'.join(errors))
              sys.exit(1)
          print('OKF Validation Passed successfully!')
          "

```

---

## 6. Developer & AI Assistant Instructions

當你在實作本專案的程式碼時，請遵循以下方針：

1. **絕不安裝專屬 OKF SDK**：解析 Frontmatter 統一使用 `python-frontmatter` + `pydantic`。
2. **Vector DB Ingestion**：強制將 YAML 中的所有欄位存入 Payload/Metadata，嚴禁「盲目切分 Chunking」，預設以「一個 `.md` 檔案 = 一個 Vector Point」為單位。
3. **Prompt Construction**：將檢索出的 OKF 上下文放入 Prompt 的前綴快取區塊（Prefix Cache Block），確保符合 API 的 Caching 規範。

```

```