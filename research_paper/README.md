# Adaptive Multi-Entity Context Fusion (AMECF) — Research Paper Package

This folder contains the complete, publication-ready research paper for the **graphforANN** platform and **AMECF** architecture in both **IEEE LaTeX format** and **Microsoft Word format (.docx)**.

---

## 📁 Package Contents

```
research_paper/
├── research_paper.tex          # Complete IEEE-style LaTeX source file
├── research_paper.docx         # Complete Microsoft Word formatted paper
├── references.bib              # Verified BibTeX bibliography
├── README.md                   # Compilation and customization guide (this file)
├── figures/                    # Directory for figure assets and placeholder descriptions
│   ├── figure1_system_architecture.txt
│   ├── figure2_heterogeneous_schema.txt
│   ├── figure3_layer_computation.txt
│   └── figures4_5_6_descriptions.txt
└── tables/                     # Standalone CSV tables of experimental data
    ├── table1_research_gap.csv
    ├── table2_dataset_statistics.csv
    ├── table3_feature_dimensions.csv
    ├── table4_model_benchmarks.csv
    ├── table5_continual_learning_ablation.csv
    └── table6_system_validation_report.csv
```

---

## 🛠️ How to Compile the LaTeX Paper

### Option 1: Local LaTeX Installation (TeX Live, MacTeX, or MiKTeX)
Run the following standard compilation sequence from inside the `research_paper/` directory:

```bash
# 1. First pass to parse document structure
pdflatex research_paper.tex

# 2. Compile bibliography citations
bibtex research_paper

# 3. Second pass to resolve citation keys
pdflatex research_paper.tex

# 4. Final pass to generate cross-references and PDF
pdflatex research_paper.tex
```

### Option 2: Overleaf / Cloud LaTeX Editor
1. Create a new blank project in [Overleaf](https://www.overleaf.com/).
2. Upload `research_paper.tex`, `references.bib`, and the `figures/` folder.
3. Set the compiler to **pdfLaTeX** and click **Recompile**.

---

## 👤 How to Update Author Information

In both `research_paper.tex` and `research_paper.docx`, author information is set with the following placeholder:

```latex
\author{
\IEEEauthorblockN{Author 1\IEEEauthorrefmark{1}}
\IEEEauthorblockA{\IEEEauthorrefmark{1}Department of Information Science and Engineering\\
Acharya Institute of Technology, Bengaluru, India\\
\textit{[PLACEHOLDER: Author details can be updated prior to publication]}}
}
```

To update with your co-authors or institutional affiliations, open `research_paper.tex` (around line 19) or `research_paper.docx` and replace the placeholder block with your actual names, email addresses, and departments.

---

## 🖼️ How to Replace Figure Placeholders

The paper contains 6 figure references with captions and placeholder blocks:
- **Figure 1 (`fig:arch`):** Overall 3-Tier Architecture of the graphforANN Platform.
- **Figure 2 (`fig:schema`):** 14-Node Heterogeneous Graph Meta-Schema Topology.
- **Figure 3 (`fig:layer`):** AMECF Heterogeneous Context Attention Layer Computation.
- **Figure 4 (`fig:cl_curve`):** Catastrophic Forgetting Knowledge Retention Trajectory.
- **Figure 5 (`fig:radar`):** 5-Dimensional Performance Radar Comparison.
- **Figure 6 (`fig:sar_subgraph`):** Explanatory Sub-Graph for Case SAR-2026-5002.

To insert your rendered diagrams into LaTeX:
1. Save your diagram as an image (e.g. `figures/figure1.png` or `figures/figure1.pdf`).
2. Replace the `\fbox{...}` block in `research_paper.tex` with:
   ```latex
   \includegraphics[width=\columnwidth]{figures/figure1.png}
   ```

---

## 📊 Summary of Verified Experimental Results

- **Dataset Scale:** 20,000,000 transactions across 10 CSV partitions ($\approx 2.58\text{ GB}$).
- **Graph Scale:** 1,333,398 nodes across 14 types; 9,698,248 edges across 18 relations.
- **GraphSAGE Baseline:** ROC-AUC: `1.0000`, PR-AUC: `1.0000`, F1: `0.9945`, Precision: `1.0000`, Recall: `0.9891`.
- **AMECF + EWC (Continual Learning):** ROC-AUC: `0.9920`, PR-AUC: `0.9750`, F1: `0.9620`, Task A Retention: `95.20%`.
- **Automated Validation Suite:** 7 of 7 checks passed (100.0% integrity).
