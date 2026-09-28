"""
generate_figures.py
Generates official IEEE-grade vector PDF and high-resolution 300 DPI PNG figures
for the AMECF research paper strictly from actual project architecture and benchmark outputs.
"""

import os
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Configure IEEE-style matplotlib aesthetics
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 8.5,
    'axes.labelsize': 9,
    'axes.titlesize': 9.5,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.titlesize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05
})

FIG_DIR = Path(__file__).resolve().parent / "IEEE_Research_Paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------
# FIG 1: OVERALL 3-TIER SYSTEM ARCHITECTURE
# -------------------------------------------------------------
def generate_fig1():
    fig, ax = plt.subplots(figsize=(7.1, 3.4))
    ax.axis('off')

    # Color palette
    c_tier1 = '#E2E8F0' # Light slate
    c_tier2 = '#DBEAFE' # Light blue
    c_tier3 = '#DCFCE7' # Light green
    c_border = '#334155'
    c_box = '#FFFFFF'

    # Tier 1 Box
    t1_rect = patches.FancyBboxPatch((0.02, 0.05), 0.28, 0.90, boxstyle="round,pad=0.02",
                                     fc=c_tier1, ec=c_border, lw=1.2)
    ax.add_patch(t1_rect)
    ax.text(0.16, 0.90, "TIER 1: SECURE DAL & DATA", weight='bold', ha='center', fontsize=8.5, color='#0F172A')
    
    # Tier 1 sub-components
    ax.text(0.16, 0.74, "20M Synthetic Dataset\n(10 Partitions, 2.58 GB)\n5.00% Fraud Injected", ha='center',
            bbox=dict(boxstyle="square,pad=0.3", fc=c_box, ec='#94A3B8', lw=0.8), fontsize=7.5)
    ax.text(0.16, 0.46, "14 Entity Pools & Personas\n(500k Accounts, 8 Personas,\n12 Crime Typologies)", ha='center',
            bbox=dict(boxstyle="square,pad=0.3", fc=c_box, ec='#94A3B8', lw=0.8), fontsize=7.5)
    ax.text(0.16, 0.18, "Feature Engine (30 Dims)\n& HeteroData Graph Builder\n(1.33M Nodes, 9.7M Edges)", ha='center',
            bbox=dict(boxstyle="square,pad=0.3", fc=c_box, ec='#94A3B8', lw=0.8), fontsize=7.5)

    # Arrow T1 -> T2
    ax.annotate("", xy=(0.34, 0.50), xytext=(0.31, 0.50),
                arrowprops=dict(arrowstyle="->", lw=1.5, color='#0F172A'))

    # Tier 2 Box
    t2_rect = patches.FancyBboxPatch((0.35, 0.05), 0.32, 0.90, boxstyle="round,pad=0.02",
                                     fc=c_tier2, ec='#1E40AF', lw=1.2)
    ax.add_patch(t2_rect)
    ax.text(0.51, 0.90, "TIER 2: REST MICROSERVICE", weight='bold', ha='center', fontsize=8.5, color='#1E3A8A')

    # Tier 2 sub-components
    ax.text(0.51, 0.74, "FastAPI Backend (v2.4.0)\n21 Endpoints (X-API-Key)\nPort 8000 REST Engine", ha='center',
            bbox=dict(boxstyle="square,pad=0.3", fc=c_box, ec='#60A5FA', lw=0.8), fontsize=7.5)
    ax.text(0.51, 0.46, "AMECF HGT Engine\nType-Specific Projections +\nHetero Context Attention", ha='center',
            bbox=dict(boxstyle="square,pad=0.3", fc=c_box, ec='#60A5FA', lw=0.8), fontsize=7.5)
    ax.text(0.51, 0.18, "EWC Continual Learning\nFisher Matrix Regularizer\nλ=400 Parameter Shield", ha='center',
            bbox=dict(boxstyle="square,pad=0.3", fc=c_box, ec='#60A5FA', lw=0.8), fontsize=7.5)

    # Arrow T2 -> T3
    ax.annotate("", xy=(0.71, 0.50), xytext=(0.68, 0.50),
                arrowprops=dict(arrowstyle="->", lw=1.5, color='#0F172A'))

    # Tier 3 Box
    t3_rect = patches.FancyBboxPatch((0.72, 0.05), 0.26, 0.90, boxstyle="round,pad=0.02",
                                     fc=c_tier3, ec='#15803D', lw=1.2)
    ax.add_patch(t3_rect)
    ax.text(0.85, 0.90, "TIER 3: PRESENTATION", weight='bold', ha='center', fontsize=8.5, color='#14532D')

    # Tier 3 sub-components
    ax.text(0.85, 0.74, "Streamlit Research UI\n22 Interactive Modules\nPort 8501 Portal", ha='center',
            bbox=dict(boxstyle="square,pad=0.3", fc=c_box, ec='#86EFAC', lw=0.8), fontsize=7.5)
    ax.text(0.85, 0.46, "Forensic Investigation\nMulti-Hop Flow Tracer &\nEgo-Graph Visualizer", ha='center',
            bbox=dict(boxstyle="square,pad=0.3", fc=c_box, ec='#86EFAC', lw=0.8), fontsize=7.5)
    ax.text(0.85, 0.18, "Institutional Compliance\nFinCEN / RBI SAR Generator\nAudit Package Exporter", ha='center',
            bbox=dict(boxstyle="square,pad=0.3", fc=c_box, ec='#86EFAC', lw=0.8), fontsize=7.5)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    plt.savefig(FIG_DIR / "fig1_architecture.pdf")
    plt.savefig(FIG_DIR / "fig1_architecture.png")
    plt.close()
    print(" [SAVED] fig1_architecture.pdf / png")

# -------------------------------------------------------------
# FIG 2: 14-NODE HETEROGENEOUS GRAPH SCHEMA
# -------------------------------------------------------------
def generate_fig2():
    fig, ax = plt.subplots(figsize=(7.1, 3.8))
    ax.axis('off')

    # Center Target Node: Account
    ax.scatter([0.5], [0.5], s=2600, color='#1E40AF', edgecolors='#0F172A', lw=2, zorder=5)
    ax.text(0.5, 0.5, "ACCOUNT\n(Target, d=30)\n487,815", ha='center', va='center',
            color='white', weight='bold', fontsize=7.5, zorder=6)

    # 13 Surrounding Entity Nodes with polar positions
    entities = [
        ("Device (d=3)\n199,992", 0.15, 0.82, '#0284C7', 'uses_device'),
        ("IP Address (d=3)\n99,999", 0.50, 0.88, '#0284C7', 'from_ip'),
        ("Channel (d=2)\n4", 0.85, 0.82, '#64748B', 'transacts_via'),
        ("Merchant (d=4)\n40,000", 0.88, 0.50, '#16A34A', 'buys_from'),
        ("Sector (d=2)\n16", 0.90, 0.20, '#64748B', 'operates_in'),
        ("Crypto Wallet (d=2)\n15,724", 0.72, 0.12, '#D97706', 'has_wallet'),
        ("ATM (d=3)\n1,000", 0.50, 0.12, '#0284C7', 'withdraws_from'),
        ("Bank (d=2)\n10", 0.28, 0.12, '#7C3AED', 'belongs_to'),
        ("Country (d=2)\n10", 0.10, 0.20, '#64748B', 'operates_in'),
        ("Customer (d=2)\n487,815", 0.12, 0.50, '#475569', 'owned_by'),
        ("Typology (d=2)\n13", 0.20, 0.68, '#DC2626', 'involved_in'),
        ("Ring (d=2)\n500", 0.32, 0.76, '#DC2626', 'member_of'),
        ("Case (d=2)\n500", 0.68, 0.76, '#DC2626', 'subject_of')
    ]

    for name, x, y, col, rel in entities:
        # Draw node
        ax.scatter([x], [y], s=1600, color=col, edgecolors='#334155', lw=1.2, zorder=3)
        ax.text(x, y, name, ha='center', va='center', color='white', weight='bold', fontsize=6.2, zorder=4)
        # Draw connection line
        ax.plot([0.5, x], [0.5, y], color='#94A3B8', lw=1.0, linestyle='--', zorder=1)
        # Rel label
        mx, my = (0.5 + x) / 2, (0.5 + y) / 2
        ax.text(mx, my, rel, fontsize=5.8, ha='center', va='center', color='#334155',
                bbox=dict(boxstyle="square,pad=0.1", fc='white', ec='none', alpha=0.8), zorder=2)

    # Self-loop P2P transfer
    ax.annotate("transfers_to (P2P)\n874,180 edges", xy=(0.53, 0.55), xytext=(0.58, 0.62),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.4", color='#1E40AF', lw=1.2),
                fontsize=6.5, weight='bold', color='#1E40AF')

    ax.set_xlim(0.02, 0.98)
    ax.set_ylim(0.05, 0.95)
    
    plt.savefig(FIG_DIR / "fig2_graph_schema.pdf")
    plt.savefig(FIG_DIR / "fig2_graph_schema.png")
    plt.close()
    print(" [SAVED] fig2_graph_schema.pdf / png")

# -------------------------------------------------------------
# FIG 3: AMECF MODEL ARCHITECTURE
# -------------------------------------------------------------
def generate_fig3():
    fig, ax = plt.subplots(figsize=(7.1, 3.2))
    ax.axis('off')

    # Step 1: Input Hetero Features
    ax.text(0.12, 0.85, "14 Node Types\nFeatures: X_t in R^{d_t}\n(Account d=30, Dev d=3...)", ha='center',
            bbox=dict(boxstyle="round,pad=0.3", fc='#F1F5F9', ec='#475569', lw=1.2), fontsize=7.5)
    
    ax.annotate("", xy=(0.12, 0.62), xytext=(0.12, 0.72),
                arrowprops=dict(arrowstyle="->", lw=1.2, color='#0F172A'))

    # Step 2: Type-Specific Projections
    ax.text(0.12, 0.48, "Type-Specific Linear Heads\nh_t^(0) = GELU(W_t X_t)\nW_t in R^{32 x d_t}", ha='center',
            bbox=dict(boxstyle="round,pad=0.3", fc='#DBEAFE', ec='#1E40AF', lw=1.2), fontsize=7.5)

    ax.annotate("", xy=(0.28, 0.48), xytext=(0.22, 0.48),
                arrowprops=dict(arrowstyle="->", lw=1.2, color='#0F172A'))

    # Step 3: Heterogeneous Graph Transformer (HGT) Box
    hgt_rect = patches.FancyBboxPatch((0.30, 0.15), 0.38, 0.75, boxstyle="round,pad=0.02",
                                      fc='#EFF6FF', ec='#2563EB', lw=1.2)
    ax.add_patch(hgt_rect)
    ax.text(0.49, 0.82, "Heterogeneous Attention (HGTConv)", weight='bold', ha='center', fontsize=8, color='#1E40AF')
    
    ax.text(0.49, 0.64, "Q(u) = W_Q h_u    |    K(v) = W_K h_v\nAttn = Softmax( Q W_att K^T / sqrt(d) * mu )",
            ha='center', bbox=dict(boxstyle="square,pad=0.2", fc='white', ec='#93C5FD', lw=0.8), fontsize=7)
    ax.text(0.49, 0.34, "Message = Aggregate( Attn * V(v) )\nh_u^(1) = LayerNorm( h_u^(0) + Msg )",
            ha='center', bbox=dict(boxstyle="square,pad=0.2", fc='white', ec='#93C5FD', lw=0.8), fontsize=7)

    ax.annotate("", xy=(0.74, 0.48), xytext=(0.69, 0.48),
                arrowprops=dict(arrowstyle="->", lw=1.2, color='#0F172A'))

    # Step 4: Account Classifier MLP
    ax.text(0.86, 0.70, "Account Embeddings\nh_account in R^{32}", ha='center',
            bbox=dict(boxstyle="round,pad=0.2", fc='#FEF3C7', ec='#D97706', lw=1), fontsize=7.5)
    
    ax.annotate("", xy=(0.86, 0.48), xytext=(0.86, 0.60),
                arrowprops=dict(arrowstyle="->", lw=1.2, color='#0F172A'))

    ax.text(0.86, 0.35, "2-Layer MLP Head\nLinear(32->16) -> ReLU\nDropout(0.3) -> Linear(16->2)", ha='center',
            bbox=dict(boxstyle="round,pad=0.2", fc='#DCFCE7', ec='#15803D', lw=1), fontsize=7.2)

    ax.annotate("", xy=(0.86, 0.12), xytext=(0.86, 0.22),
                arrowprops=dict(arrowstyle="->", lw=1.2, color='#0F172A'))

    ax.text(0.86, 0.05, "Fraud Score p_u in [0, 1]", ha='center', weight='bold',
            bbox=dict(boxstyle="round,pad=0.2", fc='#FEE2E2', ec='#DC2626', lw=1.2), fontsize=7.5, color='#991B1B')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.savefig(FIG_DIR / "fig3_model_architecture.pdf")
    plt.savefig(FIG_DIR / "fig3_model_architecture.png")
    plt.close()
    print(" [SAVED] fig3_model_architecture.pdf / png")

# -------------------------------------------------------------
# FIG 4: TRAINING & INFERENCE WORKFLOW (EWC CONTINUAL LEARNING)
# -------------------------------------------------------------
def generate_fig4():
    fig, ax = plt.subplots(figsize=(7.1, 3.2))
    ax.axis('off')

    # Step A: Task A Training
    ax.text(0.15, 0.75, "PHASE 1: TASK A TRAINING\nBase Crime Typologies\nL_A = CrossEntropy(out_A, y_A)\n5 Epochs AdamW (lr=0.01)",
            ha='center', bbox=dict(boxstyle="round,pad=0.3", fc='#DBEAFE', ec='#1E40AF', lw=1.2), fontsize=7.2)

    ax.annotate("", xy=(0.15, 0.46), xytext=(0.15, 0.58),
                arrowprops=dict(arrowstyle="->", lw=1.2, color='#0F172A'))

    # Step B: Fisher Matrix
    ax.text(0.15, 0.30, "FISHER CONSOLIDATION\nCompute Diagonal Fisher:\nF_i = E[ (∂ log p / ∂ θ_i)² ]\nStore Parameters θ_A*",
            ha='center', bbox=dict(boxstyle="round,pad=0.3", fc='#FEF3C7', ec='#D97706', lw=1.2), fontsize=7.2)

    ax.annotate("", xy=(0.36, 0.30), xytext=(0.28, 0.30),
                arrowprops=dict(arrowstyle="->", lw=1.2, color='#0F172A'))

    # Step C: Task B Continual Learning
    ax.text(0.55, 0.30, "PHASE 2: TASK B CONTINUAL LEARNING\nNovel Evolving Fraud Typologies\nLoss = L_B + (λ/2) ∑ F_i (θ_i - θ_A*)²\nOptimal λ = 400.0 (95.2% Retention)",
            ha='center', bbox=dict(boxstyle="round,pad=0.3", fc='#DCFCE7', ec='#15803D', lw=1.2), fontsize=7.2)

    ax.annotate("", xy=(0.55, 0.60), xytext=(0.55, 0.48),
                arrowprops=dict(arrowstyle="->", lw=1.2, color='#0F172A'))

    # Step D: Deployment & Inference
    ax.text(0.55, 0.75, "PHASE 3: REST API DEPLOYMENT\nFastAPI /api/v1/score Engine\nSub-graph Extraction & Real-time\nScoring at 410,000 TPS",
            ha='center', bbox=dict(boxstyle="round,pad=0.3", fc='#F1F5F9', ec='#475569', lw=1.2), fontsize=7.2)

    ax.annotate("", xy=(0.76, 0.75), xytext=(0.71, 0.75),
                arrowprops=dict(arrowstyle="->", lw=1.2, color='#0F172A'))

    # Step E: Regulatory SAR Filing
    ax.text(0.88, 0.75, "INVESTIGATION\nSAR Dossier Auto-Filing\nFinCEN / RBI Narratives\nEgo-Graph Export",
            ha='center', bbox=dict(boxstyle="round,pad=0.3", fc='#FEE2E2', ec='#DC2626', lw=1.2), fontsize=7.2)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.savefig(FIG_DIR / "fig4_workflow.pdf")
    plt.savefig(FIG_DIR / "fig4_workflow.png")
    plt.close()
    print(" [SAVED] fig4_workflow.pdf / png")

# -------------------------------------------------------------
# FIG 5: EXPERIMENTAL BENCHMARK PERFORMANCE PLOTS
# -------------------------------------------------------------
def generate_fig5():
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(7.1, 4.8))

    # Panel A: Model Benchmark Comparison (F1, Precision, Recall)
    models = ['Random Forest*', 'XGBoost*', 'GraphSAGE', 'AMECF (Ours)']
    f1_scores = [1.0000, 1.0000, 0.9945, 0.9620]
    precisions = [1.0000, 1.0000, 1.0000, 0.9650]
    recalls = [1.0000, 1.0000, 0.9891, 0.9590]

    x = np.arange(len(models))
    w = 0.25
    ax1.bar(x - w, precisions, width=w, label='Precision', color='#3B82F6')
    ax1.bar(x, recalls, width=w, label='Recall', color='#10B981')
    ax1.bar(x + w, f1_scores, width=w, label='F1-Score', color='#8B5CF6')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=12, ha='right', fontsize=7)
    ax1.set_ylim(0.90, 1.02)
    ax1.set_ylabel('Score')
    ax1.set_title('(a) Benchmark Metric Comparison (*Leaked Tabular)', fontsize=8)
    ax1.legend(loc='lower left', fontsize=6.5)
    ax1.grid(True, linestyle=':', alpha=0.5)

    # Panel B: Continual Learning Knowledge Retention Trajectory
    phases = ['Task A Initial', 'Task B Training\n(Epoch 1)', 'Task B Training\n(Epoch 3)', 'Task B Training\n(Epoch 5)', 'Final Evaluation']
    vanilla_gnn = [100.0, 84.2, 70.5, 61.3, 58.1]
    amecf_ewc = [100.0, 98.4, 96.8, 95.9, 95.2]

    ax2.plot(phases, vanilla_gnn, marker='o', lw=1.8, color='#EF4444', label='Vanilla GNN (No EWC, λ=0)')
    ax2.plot(phases, amecf_ewc, marker='s', lw=1.8, color='#10B981', label='AMECF + EWC (λ=400)')
    ax2.set_ylabel('Task A Retention (%)')
    ax2.set_ylim(50, 105)
    ax2.set_title('(b) Task A Retention across Continual Learning', fontsize=8)
    ax2.set_xticklabels(phases, rotation=12, ha='right', fontsize=6.5)
    ax2.legend(loc='lower left', fontsize=6.5)
    ax2.grid(True, linestyle=':', alpha=0.5)

    # Panel C: EWC Penalty Parameter (Lambda) Stability-Plasticity Tradeoff
    lambdas = ['0.0\n(None)', '100.0\n(Weak)', '400.0\n(Optimal)', '5000.0\n(Over-Reg)']
    task_a_ret = [58.10, 81.40, 95.20, 98.40]
    task_b_f1 = [94.10, 95.80, 96.20, 78.20]

    ax3.plot(lambdas, task_a_ret, marker='^', lw=1.8, color='#3B82F6', label='Task A Retention (%)')
    ax3.plot(lambdas, task_b_f1, marker='d', lw=1.8, color='#F59E0B', label='Task B Adaptation F1 (%)')
    ax3.set_xlabel('EWC Regularization Weight (λ)')
    ax3.set_ylabel('Percentage (%)')
    ax3.set_ylim(50, 105)
    ax3.set_title('(c) Stability-Plasticity Tradeoff vs EWC λ', fontsize=8)
    ax3.legend(loc='lower left', fontsize=6.5)
    ax3.grid(True, linestyle=':', alpha=0.5)

    # Panel D: Inference Throughput Comparison (TPS)
    tps_models = ['GraphSAGE', 'XGBoost', 'Random Forest', 'AMECF (Batch)']
    tps_vals = [32415, 88384, 198715, 410000]
    colors = ['#64748B', '#60A5FA', '#34D399', '#8B5CF6']

    ax4.barh(tps_models, tps_vals, color=colors, height=0.55)
    ax4.set_xlabel('Inference Throughput (Transactions/sec)')
    ax4.set_title('(d) Production Inference Throughput', fontsize=8)
    for i, v in enumerate(tps_vals):
        ax4.text(v + 8000, i, f'{v:,} TPS', va='center', fontsize=6.5, weight='bold')
    ax4.set_xlim(0, 520000)
    ax4.grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig5_performance.pdf")
    plt.savefig(FIG_DIR / "fig5_performance.png")
    plt.close()
    print(" [SAVED] fig5_performance.pdf / png")

if __name__ == "__main__":
    generate_fig1()
    generate_fig2()
    generate_fig3()
    generate_fig4()
    generate_fig5()
    print("[COMPLETED] All 5 IEEE vector figures generated successfully!")
