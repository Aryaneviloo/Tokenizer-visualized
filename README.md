An interactive, first-principles visualization tool built to deconstruct Byte Pair Encoding (BPE) tokenization and high-dimensional vector spaces. This application exposes how text fragments are mapped into continuous coordinates, contrasting traditional additive position frameworks with modern Rotary Position Embeddings (RoPE).

<img width="1824" height="913" alt="Image" src="https://github.com/user-attachments/assets/97ab9ad2-3147-4b47-8796-7fd66ed120e1" />
<img width="1781" height="907" alt="Image" src="https://github.com/user-attachments/assets/e0e056ab-7fda-4657-8701-a3dd852ad502" />
---

## 🏛️ Project Architecture

The codebase is split into three transparent, decoupled modules:

* **`tokenizer.py`:** A raw Python implementation of a BPE tokenizer.
* **`geometry.py`:** Synthesizes multi-dimensional continuous tensor arrays ($d_{model} = 16$) and executes the underlying coordinate mathematics.
* **`app.py`:** The frontend orchestration layer that maps the mathematical engines to interactive data tables and dynamic vector plots.

---

## 🚀 Quickstart

### 1. Prerequisites
You only need a standard Python installation and three external libraries for rendering and math. Install them via your terminal:

```bash
pip install streamlit plotly numpy
