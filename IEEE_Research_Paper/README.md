# Official IEEE Conference Paper Manuscript Package

This directory contains the complete research paper formatted strictly in accordance with the **Official IEEE Conference Manuscript Template** (`\documentclass[conference]{IEEEtran}`).

---

## 📁 Package Contents

```
IEEE_Research_Paper/
├── main.tex                    # Official IEEE conference LaTeX source code
├── references.bib              # Verified BibTeX citations (12 verified entries)
├── paper.pdf                   # Compiled publication-ready IEEE conference PDF
├── IEEE_Research_Paper.docx    # Full Microsoft Word document formatted to IEEE style
├── IEEEtran.cls                # Official IEEEtran class file (CTAN v1.8b)
├── IEEEtran.bst                # Official IEEEtran BibTeX style file (CTAN v1.14)
├── cite.sty                    # Official IEEE citation sorting package
├── README.md                   # Compilation and customization guide (this file)
└── figures/                    # Directory containing figure descriptions and placeholders
    ├── figure1_system_architecture.txt
    ├── figure2_heterogeneous_schema.txt
    ├── figure3_layer_computation.txt
    └── figures4_5_6_descriptions.txt
```

---

## 🛠️ How to Compile the LaTeX Source to PDF

### Method 1: Local Terminal (MiKTeX / TeX Live / MacTeX)
Run the following sequence inside the `IEEE_Research_Paper/` folder:

```bash
# Pass 1: Parse document and generate .aux
pdflatex -jobname=paper main.tex

# Pass 2: Process BibTeX bibliography
bibtex paper

# Pass 3: Resolve citation keys
pdflatex -jobname=paper main.tex

# Pass 4: Finalize cross-references and produce PDF
pdflatex -jobname=paper main.tex
```

### Method 2: Overleaf / Cloud LaTeX
1. Create a new empty project in [Overleaf](https://www.overleaf.com/).
2. Upload `main.tex`, `references.bib`, `IEEEtran.cls`, `IEEEtran.bst`, `cite.sty`, and the `figures/` folder.
3. Select **pdfLaTeX** as the compiler and click **Recompile**.

---

## 👤 How to Update Author Information

In both `main.tex` and `IEEE_Research_Paper.docx`, the author section is populated with placeholder metadata:

```latex
\author{
\IEEEauthorblockN{Author 1}
\IEEEauthorblockA{\textit{Department of Information Science and Engineering} \\
\textit{Acharya Institute of Technology}\\
Bengaluru, India \\
author1@acharya.ac.in \\
\textit{[PLACEHOLDER: Author details can be updated prior to final publication]}}
}
```

To update author names and affiliations:
1. Open `main.tex` (lines 20–29) or `IEEE_Research_Paper.docx`.
2. Replace `Author 1`, the department, and email with the actual author names and affiliations.

---

## 🖼️ How to Insert Generated Figures

Replace the placeholder boxes in `main.tex` with your rendered image files (e.g. PNG/PDF):
- **Fig. 1 (`fig:arch`):** Overall 3-Tier Architecture of the graphforANN Platform.
- **Fig. 2 (`fig:schema`):** Dynamic Heterogeneous Financial Transaction Graph (14 Node Types, 18 Relational Edges).
- **Fig. 3 (`fig:model_arch`):** Proposed AMECF Layer-Wise Neural Network Architecture.
- **Fig. 4 (`fig:workflow`):** Training and Online Inference Workflow with EWC Continual Consolidation.
- **Fig. 5 (`fig:perf_comp`):** Empirical ROC/PR Performance & EWC Retention Comparison.

To replace in LaTeX:
```latex
\begin{figure}[htbp]
\centering
\includegraphics[width=\columnwidth]{figures/fig1.png}
\caption{End-to-end 3-tier system architecture of the graphforANN platform.}
\label{fig:arch}
\end{figure}
```

---

## 📊 Summary of Preserved Empirical Results

- **Dataset Scale:** 20,000,000 synthetic transactions across 10 CSV partition files ($\approx 2.58\text{ GB}$).
- **Graph Scale:** 1,333,398 nodes across 14 entity types; 9,698,248 relational edges across 18 edge types.
- **Hetero-GraphSAGE Baseline:** ROC-AUC: `1.0000`, PR-AUC: `1.0000`, F1: `0.9945`, Precision: `1.0000`, Recall: `0.9891`, Throughput: `32,415 TPS`.
- **AMECF + EWC (Continual Learning):** ROC-AUC: `0.9920`, PR-AUC: `0.9750`, F1: `0.9620`, Task A Retention: `95.20%`, Batched Throughput: `410,000 TPS`.
- **Automated Validation Suite:** 7 of 7 checks passed (100.0% integrity).
