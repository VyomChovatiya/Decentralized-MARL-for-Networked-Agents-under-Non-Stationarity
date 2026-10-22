"""
Generates a comprehensive, publication-grade PDF guide explaining the entire
Decentralized MARL for Networked Agents under Non-Stationarity project.
Includes intuition, mathematical formulations, literature review & research gap,
algorithmic details, performance metrics, comparative tables, and embedded figures.
"""
import os
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    Table,
    TableStyle,
    KeepTogether,
    HRFlowable,
)


def _header_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8.5)
    canvas.setFillColor(colors.HexColor("#555555"))
    canvas.drawString(
        0.6 * inch,
        0.42 * inch,
        "Decentralized MARL for Networked Agents under Non-Stationarity — Comprehensive Project Guide",
    )
    canvas.drawRightString(
        A4[0] - 0.6 * inch,
        0.42 * inch,
        f"Page {doc.page}",
    )
    canvas.setStrokeColor(colors.HexColor("#CCCCCC"))
    canvas.setLineWidth(0.5)
    canvas.line(0.6 * inch, 0.55 * inch, A4[0] - 0.6 * inch, 0.55 * inch)
    canvas.restoreState()


def make_callout(title: str, body_html: str, bg_color="#EEF4FA", border_color="#2B5C8F", styles=None):
    content = Paragraph(f"<b>{title}</b><br/>{body_html}", styles["CalloutText"])
    tbl = Table([[content]], colWidths=[7.05 * inch])
    tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(bg_color)),
                ("BOX", (0, 0), (-1, -1), 1.2, colors.HexColor(border_color)),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    return KeepTogether([tbl])


def make_styled_table(data, col_widths, styles, header_bg="#1F4E79", left_cols=(0,)):
    formatted_rows = []
    for r_idx, row in enumerate(data):
        formatted_row = []
        for c_idx, cell_text in enumerate(row):
            html_text = str(cell_text).replace("\n", "<br/>")
            if r_idx == 0:
                style = styles["TableHeaderLeft"] if c_idx in left_cols else styles["TableHeaderCenter"]
            else:
                style = styles["TableCellLeft"] if c_idx in left_cols else styles["TableCellCenter"]
            formatted_row.append(Paragraph(html_text, style))
        formatted_rows.append(formatted_row)

    tbl = Table(formatted_rows, colWidths=col_widths, repeatRows=1)
    style_cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(header_bg)),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#BFBFBF")),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]
    for row_idx in range(1, len(data)):
        bg = colors.HexColor("#F7FAFC") if row_idx % 2 == 1 else colors.white
        style_cmds.append(("BACKGROUND", (0, row_idx), (-1, row_idx), bg))
    tbl.setStyle(TableStyle(style_cmds))
    return KeepTogether([Spacer(1, 3), tbl, Spacer(1, 5)])


def add_figure(story, img_path: str, caption: str, width_in: float, height_in: float, styles):
    if os.path.exists(img_path):
        img = Image(img_path, width=width_in * inch, height=height_in * inch)
        cap = Paragraph(f"<i>{caption}</i>", styles["Caption"])
        story.append(KeepTogether([Spacer(1, 4), img, Spacer(1, 3), cap, Spacer(1, 6)]))


def build_pdf(output_path: str = "Project_Comprehensive_Guide.pdf"):
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=0.6 * inch,
        rightMargin=0.6 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.68 * inch,
    )

    base_styles = getSampleStyleSheet()
    styles = {
        "Title": ParagraphStyle(
            "CustomTitle",
            parent=base_styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=18.5,
            leading=23,
            textColor=colors.HexColor("#102A43"),
            alignment=TA_CENTER,
            spaceAfter=6,
        ),
        "Subtitle": ParagraphStyle(
            "CustomSubtitle",
            parent=base_styles["Normal"],
            fontName="Helvetica",
            fontSize=10.5,
            leading=14.5,
            textColor=colors.HexColor("#334E68"),
            alignment=TA_CENTER,
            spaceAfter=10,
        ),
        "H1": ParagraphStyle(
            "CustomH1",
            parent=base_styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=13.5,
            leading=17.5,
            textColor=colors.HexColor("#1F4E79"),
            spaceBefore=10,
            spaceAfter=5,
            keepWithNext=True,
        ),
        "H2": ParagraphStyle(
            "CustomH2",
            parent=base_styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=10.8,
            leading=14.2,
            textColor=colors.HexColor("#2B5C8F"),
            spaceBefore=8,
            spaceAfter=3,
            keepWithNext=True,
        ),
        "Body": ParagraphStyle(
            "CustomBody",
            parent=base_styles["Normal"],
            fontName="Helvetica",
            fontSize=9.4,
            leading=13.4,
            textColor=colors.HexColor("#1F2933"),
            alignment=TA_JUSTIFY,
            spaceAfter=5,
        ),
        "Bullet": ParagraphStyle(
            "CustomBullet",
            parent=base_styles["Normal"],
            fontName="Helvetica",
            fontSize=9.3,
            leading=13.2,
            leftIndent=12,
            textColor=colors.HexColor("#1F2933"),
            spaceAfter=3.5,
        ),
        "MathBox": ParagraphStyle(
            "CustomMath",
            parent=base_styles["Normal"],
            fontName="Courier-Bold",
            fontSize=8.8,
            leading=12.5,
            textColor=colors.HexColor("#102A43"),
            alignment=TA_CENTER,
            spaceBefore=3,
            spaceAfter=5,
        ),
        "CalloutText": ParagraphStyle(
            "CustomCallout",
            parent=base_styles["Normal"],
            fontName="Helvetica",
            fontSize=9.1,
            leading=13.0,
            textColor=colors.HexColor("#102A43"),
        ),
        "Caption": ParagraphStyle(
            "CustomCaption",
            parent=base_styles["Normal"],
            fontName="Helvetica-Oblique",
            fontSize=8.4,
            leading=11.2,
            textColor=colors.HexColor("#486581"),
            alignment=TA_CENTER,
        ),
        "TableHeaderLeft": ParagraphStyle(
            "TableHeaderLeft",
            parent=base_styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=8.6,
            leading=11.0,
            textColor=colors.white,
            alignment=TA_LEFT,
        ),
        "TableHeaderCenter": ParagraphStyle(
            "TableHeaderCenter",
            parent=base_styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=8.6,
            leading=11.0,
            textColor=colors.white,
            alignment=TA_CENTER,
        ),
        "TableCellLeft": ParagraphStyle(
            "TableCellLeft",
            parent=base_styles["Normal"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=11.2,
            textColor=colors.HexColor("#1F2933"),
            alignment=TA_LEFT,
        ),
        "TableCellCenter": ParagraphStyle(
            "TableCellCenter",
            parent=base_styles["Normal"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=11.2,
            textColor=colors.HexColor("#1F2933"),
            alignment=TA_CENTER,
        ),
    }

    story = []

    # =========================================================================
    # SECTION 1: TITLE, EXECUTIVE PITCH & PROBLEM STATEMENT
    # =========================================================================
    story.append(
        Paragraph(
            "Decentralized Multi-Agent Reinforcement Learning (MARL)<br/>"
            "for Networked Agents under Non-Stationarity",
            styles["Title"],
        )
    )
    story.append(
        Paragraph(
            "Comprehensive Technical Guide, Literature Gap Analysis, Algorithmic Framework &amp; Empirical Results<br/>"
            "<b>Reinforcement Learning Course Research Project — Complete Presentation &amp; Study Reference</b>",
            styles["Subtitle"],
        )
    )
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1F4E79"), spaceAfter=8))

    story.append(
        make_callout(
            "Executive Pitch (The 30-Second Summary for Your Presentation)",
            "In real-world networked systems—such as smart power grids, urban traffic light networks, and autonomous drone swarms—"
            "there is <b>no central controller</b>. Agents can only observe local sensors and exchange messages with immediate "
            "neighbors over a physical communication graph. When a subset of agents abruptly changes policy or malfunctions mid-training "
            "(<b>non-stationarity shock at time t_c</b>), standard Graph Neural Network (GNN) communication blindly propagates these "
            "corrupted messages across the network—causing catastrophic failure, especially when high-degree <b>hub nodes</b> are compromised. "
            "In this project, we combine configurable graph benchmarks (<b>COGNAC</b>) with Variational Bayesian Ego-Graph Inference "
            "(<b>BayesG</b>) to show that agents can dynamically infer a latent edge mask Z_ij that <b>autonomously prunes (&gt;70% suppression) "
            "connections to non-stationary neighbors</b>, cutting instability by nearly 3x and recovering 7x faster than static GNNs.",
            bg_color="#EBF4FF",
            border_color="#1F4E79",
            styles=styles,
        )
    )
    story.append(Spacer(1, 6))

    story.append(Paragraph("1. Problem Statement: Intuition &amp; Mathematical Formulation", styles["H1"]))

    story.append(Paragraph("1.1 Plain-English Intuition: The 'Whispering in a Room' Analogy", styles["H2"]))
    story.append(
        Paragraph(
            "Imagine 10 people standing in a room trying to unanimously agree on a binary decision (Vote 0 or Vote 1). "
            "Each person receives a noisy local hint about the correct target, and can only whisper to the people directly "
            "connected to them by a physical wire (the <b>network topology</b>). Three fundamental challenges arise:",
            styles["Body"],
        )
    )
    story.append(
        Paragraph(
            "• <b>1. Partial Observability (Why 0-Hop Isolated Learning Struggles):</b> A single agent's sensor is noisy. "
            "If agents do not communicate (<i>Independent PPO</i>), roughly 20–30% of agents will guess the wrong direction at any step, "
            "making unanimous consensus slow and fragile.",
            styles["Bullet"],
        )
    )
    story.append(
        Paragraph(
            "• <b>2. The Double-Edged Sword of Graph Communication:</b> If agents pool observations from their 1-hop or 2-hop "
            "neighbors using a Graph Neural Network (<i>Networked-PPO</i>), local sensor noise averages out and consensus "
            "becomes rapid. <b>However</b>, communication creates a vulnerability: you are trusting your neighbors' messages.",
            styles["Bullet"],
        )
    )
    story.append(
        Paragraph(
            "• <b>3. Mid-Training Non-Stationarity Shocks (t = t_c):</b> Halfway through training (at episode t_c), 25–30% of "
            "nodes suffer a behavioral shift (hardware fault, task change, or adversarial takeover: policy switches from pi_A to "
            "pi_B) and broadcast corrupted state signals. A static GNN blindly aggregates these poisoned messages into "
            "healthy agents' embeddings, sending shockwaves across the graph.",
            styles["Bullet"],
        )
    )

    story.append(Paragraph("1.2 Formal Mathematical Formulation (Networked Dec-POMDP)", styles["H2"]))
    story.append(
        Paragraph(
            "We formulate the system as a Networked Decentralized Partially Observable Markov Decision Process defined by "
            "the tuple <b>M = (G, S, {A_i}, {O_i}, P, {R_i}, gamma)</b> over an undirected graph <b>G = (V, E)</b> with N = |V| agents. "
            "At each discrete timestep t, agent i observes only a local vector o_i^t = O_i(s_t) and can communicate only within its "
            "k-hop neighborhood N_k(i). From the perspective of any single learner i, the effective environment transition probability is:",
            styles["Body"],
        )
    )
    story.append(
        Paragraph(
            "P_eff(s_{t+1} | s_t, a_i^t) = Sum_{a_{-i}} [ P(s_{t+1} | s_t, a_i^t, a_{-i}^t) * Prod_{j != i} pi_j^t(a_j^t | o_j^t) ]",
            styles["MathBox"],
        )
    )
    story.append(
        Paragraph(
            "<b>Why Standard RL Stationarity Breaks Down:</b> In single-agent RL, the transition kernel P(s_{t+1} | s_t, a_t) is "
            "fixed. In decentralized MARL, because neighboring policies pi_j^t are inside the product term, <b>P_eff changes every "
            "time neighbors update their weights</b> (concurrent learning non-stationarity) and undergoes a <b>discontinuous "
            "regime jump</b> when a subset of nodes S_shock switches policies at shock step t_c (pi_j^t = pi_A for t &lt; t_c, and "
            "pi_j^t = pi_B for t &gt;= t_c).",
            styles["Body"],
        )
    )

    # =========================================================================
    # SECTION 2: LITERATURE REVIEW & THE RESEARCH GAP
    # =========================================================================
    story.append(Paragraph("2. Literature Review &amp; The Core Research Gap", styles["H1"]))
    story.append(
        Paragraph(
            "A key academic requirement is positioning our work relative to recent literature. Prior studies have explored "
            "individual pieces of this problem across three separate research streams, leaving a critical gap at their intersection:",
            styles["Body"],
        )
    )
    story.append(
        Paragraph(
            "• <b>Stream A — Configurable Graph Topologies in Stationary MARL (COGNAC, NeurIPS 2025; Policy Consensus, Automatica 2024):</b> "
            "Sintes &amp; Busic (<i>COGNAC</i>) introduced a benchmark showing that physical graph topology (Ring, Grid, Random, Scale-Free) "
            "heavily influences cooperative MARL sample efficiency. Hu et al. (2024) analyzed entropy-regularized policy consensus over fixed graphs. "
            "<b>Limitation:</b> Both assume all agents remain 100% cooperative and stationary throughout training, and rely on static communication links.",
            styles["Bullet"],
        )
    )
    story.append(
        Paragraph(
            "• <b>Stream B — Sparse Ego-Graph Inference in Stationary MARL (BayesG, NeurIPS 2025):</b> "
            "Duan et al. (<i>BayesG</i>) proposed variational Bayesian ego-graph inference using Gumbel-Softmax to prune redundant edges "
            "and reduce communication bandwidth in networked MARL. "
            "<b>Limitation:</b> BayesG was only evaluated as a bandwidth-saving tool under stationary conditions. It was never tested "
            "under mid-training behavioral shocks (t_c), adversarial/faulty node shifts, or topological hub failures.",
            styles["Bullet"],
        )
    )
    story.append(
        Paragraph(
            "• <b>Stream C — Non-Stationarity &amp; Partner Modeling in Small Fully-Connected Teams (DAPM 2024; CIAO, ICML 2024):</b> "
            "Xu et al. (<i>DAPM</i>) and open ad-hoc teamwork methods (<i>CIAO</i>) study how agents adapt when teammates change policies. "
            "<b>Limitation:</b> They assume small, fully-connected environments (ignoring physical network topology constraints, k-hop GNN "
            "message propagation, and graph spectral properties such as Fiedler value or hub degree heterogeneity).",
            styles["Bullet"],
        )
    )

    gap_data = [
        [
            "Research Dimension / Capability",
            "COGNAC (NeurIPS 2025)",
            "BayesG (NeurIPS 2025)",
            "DAPM / CIAO (2024)",
            "Our Project (This Work)",
        ],
        ["Configurable Graph Topologies (Ring, Grid, ER, BA)", "Yes", "No (Fixed)", "No (Full Graph)", "Yes (All 4)"],
        ["Graph Spectral Metrics (Diameter, Fiedler Value)", "Partial", "No", "No", "Yes"],
        ["Mid-Training Non-Stationarity Shock (t_c: pi_A -> pi_B)", "No", "No", "Yes", "Yes"],
        ["Hub vs. Periphery Node Shock Vulnerability Study", "No", "No", "No", "Yes"],
        ["Variational Edge Pruning as 'Topological Quarantine'", "No (Static GNN)", "Bandwidth Only", "No", "Yes (>70% Pruned)"],
    ]
    story.append(
        make_styled_table(
            gap_data,
            col_widths=[2.55 * inch, 1.1 * inch, 1.15 * inch, 1.1 * inch, 1.15 * inch],
            styles=styles,
            left_cols=(0,),
        )
    )

    story.append(
        make_callout(
            "The Exact Research Gap We Solve (Highlight This in Your Presentation!)",
            "<b>Prior work studied (A) graph topologies under stationary agents (COGNAC) and (B) variational edge sparsification for "
            "bandwidth efficiency (BayesG), while (C) non-stationarity was only studied in small, non-networked teams.</b><br/>"
            "No prior study has investigated how <b>graph topology and node centrality (hubs vs. periphery)</b> govern the propagation "
            "of mid-training non-stationarity shocks across k-hop GNNs, nor demonstrated that <b>Bayesian Ego-Graph Inference (BayesG)</b> "
            "can act as an automated <b>topological quarantine mechanism</b> to sever corrupted hub connections and restore consensus.",
            bg_color="#FFF8E7",
            border_color="#B7791F",
            styles=styles,
        )
    )

    # =========================================================================
    # SECTION 3: NETWORK TOPOLOGIES & GRAPH SPECTRAL THEORY
    # =========================================================================
    story.append(Paragraph("3. Network Topologies &amp; Graph Spectral Properties", styles["H1"]))
    story.append(
        Paragraph(
            "Implemented in <b>src/environments/graphs.py</b>, our framework generates four canonical graph families and computes their "
            "structural and spectral invariants: <b>Graph Diameter D</b> (longest shortest path between any two agents) and "
            "<b>Algebraic Connectivity / Fiedler Value lambda_2(L)</b> (the second-smallest eigenvalue of the graph Laplacian L = D - A, "
            "which measures how tightly connected the network is and how fast information—or a shockwave—mixes across the graph):",
            styles["Body"],
        )
    )
    story.append(
        Paragraph(
            "• <b>1. Ring (1D Cycle):</b> Every node has degree 2. Large diameter (D = N/2) and very small Fiedler value (0.15 for N=16). "
            "Information travels slowly step-by-step around the ring, but shocks are also locally contained.<br/>"
            "• <b>2. 2D Grid (Lattice):</b> Nodes arranged in a 2D spatial mesh (degree 2–4). Moderate diameter (D = 6) and Fiedler value (0.59).<br/>"
            "• <b>3. Erdos-Renyi Random Graph G(N, p):</b> Edges formed randomly with probability p &gt; 2 ln(N)/N. Low diameter (D = 3) "
            "and high Fiedler value (1.71), enabling rapid global mixing.<br/>"
            "• <b>4. Barabasi-Albert Scale-Free Network:</b> Built via preferential attachment. Features a power-law degree distribution "
            "with a few heavily connected <b>hub nodes</b> and many low-degree <b>peripheral boundary nodes</b>.",
            styles["Bullet"],
        )
    )

    add_figure(
        story,
        "results/topologies_overview.png",
        "Figure 1: The 4 configurable network topologies generated by src/environments/graphs.py (N=16) showing Graph Diameter and Algebraic Connectivity (Fiedler eigenvalue lambda_2).",
        width_in=6.9,
        height_in=1.72,
        styles=styles,
    )

    # =========================================================================
    # SECTION 4 & 5: ENVIRONMENT, SHOCK PROTOCOL & ALGORITHMS
    # =========================================================================
    story.append(Paragraph("4. Environment Mechanics &amp; Non-Stationarity Shock Protocol", styles["H1"]))
    story.append(Paragraph("4.1 Networked Binary Consensus Environment (src/environments/toy_consensus.py)", styles["H2"]))
    story.append(
        Paragraph(
            "At the start of each episode, an underlying binary target consensus state c in {0, 1} (mapped to s_target in {-1, +1}) "
            "is drawn. Each agent i receives a private, static per-episode noisy sensor reading <b>z_i = s_target + Normal(0, sigma^2)</b> "
            "(with sigma = 1.20) and initializes its binary vote v_i in {0, 1}. At each step t, agent i observes a purely local 3D vector:",
            styles["Body"],
        )
    )
    story.append(
        Paragraph(
            "o_i^t = [ 2*v_i^t - 1 (own signed vote),   z_i (local noisy sensor),   deg(i) / (N - 1) (normalized degree) ]",
            styles["MathBox"],
        )
    )
    story.append(
        Paragraph(
            "<b>Why k-Hop Communication Beats 0-Hop IPPO:</b> Because sigma = 1.20, a single agent's sensor z_i has the wrong sign ~19% "
            "of the time. For N=10 independent agents (IPPO), the probability that all 10 sensors simultaneously agree is (1 - 0.19)^10 = 12%—"
            "making isolated consensus very difficult. When agents aggregate observations across k-hop neighbors via a GNN, the variance "
            "of the averaged sensor drops as sigma^2 / |N_k(i)|, allowing communicating agents to identify the true consensus target rapidly.",
            styles["Body"],
        )
    )

    story.append(Paragraph("4.2 Non-Stationarity Shock Controller (src/experiments/shock_protocol.py)", styles["H2"]))
    story.append(
        Paragraph(
            "At training episode <b>t = t_c</b>, the <b>NonStationarityShockController</b> perturbs a fraction rho (e.g., 25–30%) of agents "
            "selected via three configurable modes: <b>random</b> (uniform selection), <b>hubs</b> (top-k highest degree nodes), or "
            "<b>boundary</b> (bottom-k lowest degree peripheral nodes). Shocked nodes undergo a simultaneous <b>behavioral &amp; feature shift</b>: "
            "(1) their actions are inverted/perturbed (adversarial policy pi_B), and (2) their transmitted observation vector o_j is shifted "
            "to an out-of-distribution regime (inverting and amplifying the signal), poisoning the messages sent over their graph edges.",
            styles["Body"],
        )
    )

    story.append(Paragraph("5. The Three Algorithms Under Comparison (Track 2)", styles["H1"]))
    story.append(
        Paragraph(
            "• <b>Method 1: Independent PPO (IPPO — src/algorithms/ippo.py):</b> Every agent is an isolated island (0-hop communication). "
            "It feeds its 3D local observation o_i directly into a 2-layer Tanh MLP Actor pi_theta(a_i | o_i) and Critic V_phi(o_i), "
            "trained via PPO with Generalized Advantage Estimation (GAE, lambda=0.95). Zero communication cost and immune to poisoned "
            "neighbor messages, but achieves poor baseline consensus because it cannot pool neighbor sensors.<br/>"
            "• <b>Method 2: Networked-PPO with Static k-Hop GNN (src/algorithms/networked_ppo.py):</b> Agents pass messages over the fixed "
            "physical graph adjacency matrix A for k hops (<b>src/models/gnn_comm.py</b>), combining degree-normalized neighbor aggregation "
            "with a local self-skip projection: <b>H^{(l+1)} = tanh(D^{-1} A H^{(l)} W_msg^{(l)} + H^{(l)} W_self^{(l)})</b>. Excels when "
            "all agents are cooperative, but blindly averages corrupted features into connected agents' embeddings when a neighbor is shocked.<br/>"
            "• <b>Method 3: BayesG — Bayesian Ego-Graph Inference (src/algorithms/bayesg.py):</b> Inserts a variational filter "
            "(<b>src/models/bayes_ego.py</b>) in front of the GNN. For every physical edge (i, j), an edge-scoring MLP examines the pairwise "
            "relational features <b>[h_i, h_j, |h_i - h_j|]</b> and samples a differentiable binary gate <b>Z_ij in [0, 1]</b> via the "
            "Binary Concrete (Gumbel-Softmax) distribution (while preserving self-loops Z_ii = 1). Trained jointly via the ELBO-regularized PPO loss:",
            styles["Bullet"],
        )
    )
    story.append(
        Paragraph(
            "L_total = L_PPO_Clip(theta) + c_v * L_Critic(phi) - c_e * Entropy(pi) + beta * [ KL(q(Z_ij | o_i, o_j) || Bern(p_0)) + L_anomaly ]",
            styles["MathBox"],
        )
    )

    algo_table = [
        ["Property", "IPPO (Baseline 1)", "Networked-PPO (Baseline 2)", "BayesG (Proposed Method)"],
        ["Communication Range", "0-Hop (No Communication)", "k-Hop Static Physical Graph A", "k-Hop Dynamic Gated Ego-Graph (A * Z)"],
        ["Edge Weighting", "None (Identity Matrix I)", "Fixed Degree Normalization D^{-1}A", "Variational Gumbel-Softmax Mask Z_ij"],
        ["Normal Consensus Speed", "Slow (Low Single-Sensor SNR)", "Fast (Neighborhood Pooling)", "Fastest (Selective Neighborhood Pooling)"],
        ["Vulnerability to Shocks", "Low (Ignores Neighbors)", "Catastrophic (Spreads Shockwave)", "Resilient (Quarantines Shocked Nodes)"],
    ]
    story.append(
        make_styled_table(
            algo_table,
            col_widths=[1.45 * inch, 1.7 * inch, 1.9 * inch, 2.0 * inch],
            styles=styles,
            left_cols=(0,),
        )
    )

    # =========================================================================
    # SECTION 6: EVALUATION METRICS & EMPIRICAL RESULTS
    # =========================================================================
    story.append(Paragraph("6. Evaluation Metrics &amp; Empirical Results (Priorities 1–4)", styles["H1"]))

    story.append(Paragraph("6.1 The 6 Rigorous Evaluation Metrics (src/evaluation/metrics.py)", styles["H2"]))
    story.append(
        Paragraph(
            "1. <b>Mean Episodic Return (Higher is better):</b> Average cooperative reward across all training episodes.<br/>"
            "2. <b>Max Smoothed Return (Higher is better):</b> Peak moving-average episodic return achieved.<br/>"
            "3. <b>Convergence Episode (Lower is better):</b> First episode reaching 90% of peak smoothed return.<br/>"
            "4. <b>Instability Index (Lower is better):</b> Standard deviation of episodic returns in a +/-15 episode window around shock t_c.<br/>"
            "5. <b>Adaptation Latency (Lower is better):</b> Episodes after t_c required for the rolling average return to recover to &gt;=85% of pre-shock performance.<br/>"
            "6. <b>Active Communication Cost (Lower is better):</b> Fraction of off-diagonal physical graph edges actively used per step.",
            styles["Body"],
        )
    )

    story.append(Paragraph("6.2 Priority 2 Result: Normal Operation Baseline (No Shocks)", styles["H2"]))
    story.append(
        Paragraph(
            "<b>Experimental Goal:</b> Prove that k-hop graph communication improves cooperative consensus when all agents are healthy "
            "(Ring topology, N=10, k=2 hops, 90 episodes, no shock).",
            styles["Body"],
        )
    )
    p2_table = [
        ["Algorithm", "Mean Return (Up)", "Max Smoothed Return (Up)", "Convergence Ep (Down)", "Avg Comm Cost"],
        ["IPPO (0-Hop, No Comm)", "-5.61", "9.93", "41", "0.00"],
        ["Networked-PPO (Static k-Hop GNN)", "+1.10", "13.14", "65", "1.00"],
        ["BayesG (Dynamic Ego-Graph)", "+5.57 (Best)", "14.78 (Best)", "36 (Fastest)", "0.90"],
    ]
    story.append(
        make_styled_table(
            p2_table,
            col_widths=[2.15 * inch, 1.2 * inch, 1.45 * inch, 1.25 * inch, 1.0 * inch],
            styles=styles,
            left_cols=(0,),
        )
    )
    add_figure(
        story,
        "results/priority2_no_shock/no_shock_comparison.png",
        "Figure 2 (Priority 2): Normal operation learning curves on a 10-node Ring without shocks. Both communicating methods (Networked-PPO and BayesG) substantially outperform isolated IPPO.",
        width_in=5.5,
        height_in=3.0,
        styles=styles,
    )

    story.append(Paragraph("6.3 Priority 3A Result: Mid-Training Random Policy Shock (Ring N=10, t_c=50)", styles["H2"]))
    story.append(
        Paragraph(
            "<b>Experimental Goal:</b> At episode t_c = 50, 30% of agents switch to an adversarial policy and broadcast corrupted state features. "
            "Observe how static GNN communication collapses while BayesG isolates the perturbation.",
            styles["Body"],
        )
    )
    p3a_table = [
        ["Algorithm", "Mean Return (Up)", "Instability Index (Down)", "Adaptation Latency (Down)", "Avg Comm Cost"],
        ["IPPO (0-Hop, No Comm)", "-3.52", "19.42", "3 episodes", "0.00"],
        ["Networked-PPO (Static GNN)", "-4.83 (Worst)", "25.69 (Severe Crash)", "22 episodes (Slow)", "1.00"],
        ["BayesG (Proposed)", "+5.08 (Best)", "8.89 (~3x More Stable)", "3 episodes (7x Faster)", "0.75 (Pruned)"],
    ]
    story.append(
        make_styled_table(
            p3a_table,
            col_widths=[1.85 * inch, 1.2 * inch, 1.45 * inch, 1.5 * inch, 1.05 * inch],
            styles=styles,
            left_cols=(0,),
        )
    )
    add_figure(
        story,
        "results/priority3_random_shock/shock_comparison.png",
        "Figure 3 (Priority 3A): Resilience to mid-training policy shock at t_c = 50 (red dashed line). Static Networked-PPO plunges to -35 and takes 22 episodes to recover, whereas BayesG stays stable (+5.08 mean return) and recovers in 3 episodes.",
        width_in=5.5,
        height_in=3.0,
        styles=styles,
    )

    story.append(Paragraph("6.4 Priority 3B Result: Scale-Free Hub Shock vs. Periphery Shock (N=12, t_c=50)", styles["H2"]))
    story.append(
        Paragraph(
            "<b>Experimental Goal &amp; Core Topological Insight:</b> In a Barabasi-Albert Scale-Free network, hub nodes ([0, 1, 4]) are connected "
            "to almost the entire network, whereas peripheral boundary nodes ([2, 7, 9]) have only 1–2 links. When hubs are shocked under "
            "static Networked-PPO, corrupted messages flood the whole graph (Adaptation Latency = 20 episodes, Mean Return = -11.74). "
            "When boundary nodes are shocked, Networked-PPO recovers in 4 episodes. <b>BayesG neutralizes the hub vulnerability</b>, cutting "
            "both the immediate shock drop (60.71 -&gt; 25.82) and instability (30.74 -&gt; 15.62) by more than 50% and recovering in 3 episodes!",
            styles["Body"],
        )
    )
    p3b_table = [
        ["Scenario (Scale-Free N=12)", "Shocked Nodes", "Mean Return (Up)", "Instability (Down)", "Shock Drop (Down)", "Adapt Latency"],
        ["NetPPO (Hub Shock)", "[0, 1, 4]", "-11.74 (Paralyzed)", "30.74", "60.71", "20 episodes"],
        ["NetPPO (Boundary Shock)", "[2, 7, 9]", "-6.75", "23.44", "61.98", "4 episodes"],
        ["BayesG (Hub Shock)", "[0, 1, 4]", "+0.33 (Resilient)", "15.62 (-49%)", "25.82 (-57%)", "3 episodes"],
        ["BayesG (Boundary Shock)", "[2, 7, 9]", "+4.32 (Best)", "11.99", "17.29", "3 episodes"],
    ]
    story.append(
        make_styled_table(
            p3b_table,
            col_widths=[1.7 * inch, 0.95 * inch, 1.25 * inch, 1.1 * inch, 1.1 * inch, 0.95 * inch],
            styles=styles,
            left_cols=(0,),
        )
    )
    add_figure(
        story,
        "results/priority3_hub_vs_periphery/hub_vs_periphery.png",
        "Figure 4 (Priority 3B): Left: Learning curves on Scale-Free topology under Hub vs. Periphery shocks at t_c=50. Right: Vulnerability comparison showing how BayesG slashes both Instability Index and Immediate Shock Drop.",
        width_in=6.6,
        height_in=2.45,
        styles=styles,
    )

    story.append(Paragraph("6.5 Priority 4 Result: Visualizing BayesG Dynamic Edge Pruning ('The Money Plot')", styles["H2"]))
    story.append(
        Paragraph(
            "To provide direct visual and quantitative proof that BayesG acts as an automated <b>topological quarantine</b>, we recorded "
            "the inferred adjacency mask probabilities Z_ij before the shock (t &lt; 50) and after the shock (t &gt; 50) during the Scale-Free Hub Shock:",
            styles["Body"],
        )
    )
    p4_table = [
        ["Edge Category in Scale-Free Graph", "Pre-Shock Weight (t < 50)", "Post-Shock Weight (t > 50)", "Relative Change & Interpretation"],
        ["Edges Connected to Shocked Hubs [0, 1, 4]", "0.904 (Open)", "0.267 (Severed)", "-70.5% (Shocked Hubs Quarantined)"],
        ["Edges Between Healthy Operational Nodes", "0.910 (Open)", "0.901 (Open)", "-1.0% (Healthy Sub-Network Preserved)"],
    ]
    story.append(
        make_styled_table(
            p4_table,
            col_widths=[2.3 * inch, 1.35 * inch, 1.35 * inch, 2.05 * inch],
            styles=styles,
            left_cols=(0,),
        )
    )
    add_figure(
        story,
        "results/priority4_edge_pruning/bayesg_edge_pruning.png",
        "Figure 5 (Priority 4 — 'The Money Plot'): Top row: Inferred edge mask heatmaps Z_ij before and after t_c=50 (red dashed lines mark shocked hubs 0, 1, 4 fading out post-shock). Bottom row: Network communication graph showing links to red shocked hubs pruned into faint dotted lines while healthy green links remain intact.",
        width_in=5.8,
        height_in=4.6,
        styles=styles,
    )

    # =========================================================================
    # SECTION 7: CODEBASE ARCHITECTURE & PRESENTATION Q&A DEFENSE GUIDE
    # =========================================================================
    story.append(Paragraph("7. Codebase Architecture &amp; Presentation Defense Q&amp;A Guide", styles["H1"]))
    story.append(Paragraph("7.1 Modular 3-Track Codebase Map", styles["H2"]))
    code_table = [
        ["Track / Directory", "File Name", "Exact Role in the Project"],
        [
            "Track 1: Environments\n(src/environments/)",
            "graphs.py\ntoy_consensus.py\ncognac_wrapper.py",
            "• graphs.py: Generates Ring, Grid, ER, Scale-Free graphs; computes diameter & Fiedler value.\n"
            "• toy_consensus.py: Decentralized binary consensus with local noisy sensors.\n"
            "• cognac_wrapper.py: Factory wrapper for benchmark environments.",
        ],
        [
            "Track 2: Models & Algos\n(src/models/ &\nsrc/algorithms/)",
            "actors.py, critics.py\ngnn_comm.py\nbayes_ego.py\nippo.py, networked_ppo.py, bayesg.py",
            "• actors.py, critics.py: Decentralized MLP Actor & Critic networks.\n"
            "• gnn_comm.py: k-Hop GNN message passing with degree normalization & self-skip.\n"
            "• bayes_ego.py: Gumbel-Softmax variational edge mask inference + ELBO KL loss.\n"
            "• ippo.py, networked_ppo.py, bayesg.py: Full GAE rollout buffers and joint training loops.",
        ],
        [
            "Track 3: Experiments\n(src/experiments/ &\nsrc/evaluation/)",
            "shock_protocol.py\nrun_experiment.py\nrun_priority_suite.py\nmetrics.py, plot_results.py",
            "• shock_protocol.py: Mid-training shock orchestrator (random, hubs, boundary modes).\n"
            "• run_experiment.py: Universal YAML/CLI runner with CUDA, Intel Arc XPU, and CPU support.\n"
            "• run_priority_suite.py: Automated benchmark runner executing Priorities 1–4 end-to-end.\n"
            "• metrics.py, plot_results.py: Computes the 6 core metrics and generates all figures.",
        ],
    ]
    story.append(
        make_styled_table(
            code_table,
            col_widths=[1.45 * inch, 1.55 * inch, 4.05 * inch],
            styles=styles,
            left_cols=(0, 1, 2),
        )
    )

    story.append(Paragraph("7.2 Top 5 Viva / Professor Questions &amp; Crisp Answers for Tomorrow", styles["H2"]))
    qa_items = [
        (
            "Q1: Why not use Centralized Training with Decentralized Execution (CTDE) like MAPPO or QMIX?",
            "<b>Answer:</b> CTDE requires access to the global state and all agents' joint actions during training. Furthermore, if a "
            "shock occurs during execution/deployment or online adaptation in a large network (e.g., power grid or swarm), there "
            "is no central server to retrain a centralized critic. Fully decentralized learning with local k-hop communication reflects "
            "realistic networked systems.",
        ),
        (
            "Q2: Why does Independent PPO (IPPO) have low adaptation latency (3 episodes) after a shock, yet low overall return?",
            "<b>Answer:</b> Because IPPO has 0 communication! Since IPPO agents never listen to neighbors, they don't get "
            "poisoned by corrupted neighbor messages when a shock hits—so they don't experience a communication-induced "
            "crash. However, because they never communicate, their overall return is poor (-3.52 vs. +5.08 for BayesG). BayesG "
            "gives the best of both worlds: high cooperative return (+5.08) AND fast 3-episode shock recovery.",
        ),
        (
            "Q3: How does BayesG mathematically allow gradients to flow through discrete binary edge masks Z_ij?",
            "<b>Answer:</b> Sampling a hard binary mask Z_ij in {0, 1} is non-differentiable. BayesG uses the <b>Binary Concrete "
            "(Gumbel-Softmax) relaxation</b> during training: Z_ij = sigmoid((logit_ij + log(u) - log(1-u)) / tau), which is continuous "
            "and differentiable via the reparameterization trick, allowing PPO policy/value gradients and ELBO KL gradients to "
            "directly update the edge-scoring MLP.",
        ),
        (
            "Q4: Why are Barabasi-Albert Scale-Free networks much more fragile to Hub Shocks than Ring or Boundary shocks?",
            "<b>Answer:</b> In a Scale-Free graph, degree distribution follows a power law: a few hub nodes are connected to almost "
            "every other node. In a k-hop GNN with static adjacency D^{-1}A, a hub node's corrupted feature vector enters the "
            "1-hop aggregation of nearly all agents simultaneously, causing network-wide value estimation collapse (20-episode "
            "latency). Boundary nodes only affect 1–2 neighbors (4-episode latency).",
        ),
        (
            "Q5: How does BayesG know which edges to prune when a shock occurs at t_c?",
            "<b>Answer:</b> BayesG's edge network takes the pairwise relational vector [h_i, h_j, |h_i - h_j|] and optimizes the joint "
            "PPO + ELBO objective. When neighbor j is shocked, its broadcasted features shift out-of-distribution (large |h_i - "
            "h_j|) and degrade agent i's TD advantage. The edge logit drops sharply, suppressing Z_ij from 0.904 down to 0.267 "
            "on shocked hub edges while keeping healthy edges at 0.901.",
        ),
    ]

    for q_title, a_body in qa_items:
        story.append(
            make_callout(
                q_title,
                a_body,
                bg_color="#F8FAFC",
                border_color="#486581",
                styles=styles,
            )
        )
        story.append(Spacer(1, 4))

    doc.build(story, onFirstPage=_header_footer, onLaterPages=_header_footer)
    print(f"PDF successfully generated at: {os.path.abspath(output_path)}")


if __name__ == "__main__":
    build_pdf()
