# Review of `implementation_plan.md`

## Bottom line

The plan gets the shape of the project right (topology sweep, comm radius sweep, non-stationarity shock, COGNAC + BayesG). The two things wrong with it are: it treats BayesG integration as a one-week task when the released BayesG code is built for a completely different stack than the rest of the plan, and it compresses a genuinely large experiment matrix into an 8-week roadmap with no seed count, no compute budget, and no operational definitions for the metrics it promises to report. Neither is fatal, but both will blow the timeline if not fixed before week 1.

I verified the claims about COGNAC and BayesG against their papers and repos (sources at the bottom). Where I could not verify something, I say so directly rather than guessing.

---

## 1. The plan ships with unresolved decisions baked into it, and one is self-contradictory

The "Open Questions" section asks whether Binary Consensus should be the starting environment, but Milestone 3 already assumes it ("Run IPPO on COGNAC's Binary Consensus environment on a 5-node ring"). Pick one and delete the other. Given COGNAC's own comparison table, Binary Consensus and SysAdmin are actually the *only two* of COGNAC's five environments marked "graph agnostic" (the other three: Firefighting Graph 1D/2D and Multi-commodity Flow, only work on fixed graph structures). Since the whole point of this project is varying topology, Binary Consensus and SysAdmin aren't just a reasonable starting pair, they're the only pair that supports the experiment at all. The plan should say this explicitly instead of presenting it as an open preference question. Start with Binary Consensus (smaller joint state space, 2^N vs SysAdmin's 9^N) and treat SysAdmin as the harder follow-up, which is what the plan already implies.

GitHub visibility and commit email are not technical decisions worth blocking on. Public repo, GitHub noreply email, done. Spend zero more roadmap time on this.

## 2. BayesG integration is scoped as one week. It's closer to a reimplementation, and that changes the whole roadmap.

This is the most important thing I found. I pulled the actual BayesG repo (`Wei9711/BayesG`, official NeurIPS 2025 code). It requires:

- **Python 3.8** and **SUMO >= 1.1.0** as hard dependencies, not Python 3.11 with PyTorch/PyG.
- An **A2C-family** training loop (`IA2C`, `MA2C` with CommNet/NeurComm baselines), not PPO.
- Environments that are entirely SUMO traffic-signal simulations: ATSC Grid, ATSC Monaco, and NewYork33/51/167 (Manhattan road networks built from OSM via `osmWebWizard`). There is no COGNAC-style abstract graph environment anywhere in the codebase; the "graph" is a traffic network, and the env wrapper (`envs/Large_city.py`) is written specifically against SUMO's traci API.

So "integrate BayesG" cannot mean "import their module and point it at COGNAC." It means: read `agents/` to understand how they parameterize the variational posterior q(Z_i; phi_i) over the ego-graph mask, how they sample it (their paper describes learning a latent communication mask via variational inference with an ELBO objective; I could not confirm from what I read whether they use a literal Gumbel-Softmax/Concrete relaxation as opposed to some other reparameterization, so treat that detail in the plan as unverified until you read the actual paper's method section), and then **rewrite that module from scratch** against your own PPO-based, PyTorch actor-critic running on COGNAC's discrete state/action spaces. That is a legitimate research contribution in its own right (nobody has published a COGNAC-native reimplementation of this), but it is not a one-week task, especially not for a week 6 slot that comes right after a week 5 that's already introducing a new experimental protocol.

What I'd do: move a *feasibility spike* into week 1 or 2, not week 6. Clone the BayesG repo, read `agents/` and the paper's method section closely enough to know exactly what you're reimplementing, and confirm before you commit the rest of the roadmap around it. Finding out in week 6 that the "integration" is actually a from-scratch VI module is a much worse place to discover this than week 1.

## 3. The timeline is overcommitted, in a specific and fixable way

Two spots where "learn the concept" and "ship the full experiment on it" are scheduled in the same 7-day window:

- **Week 4**: learn GCN/GAT for the first time, implement k-hop message passing into a PPO actor-critic, *and* benchmark IPPO vs Networked-GNN across all 4 topologies at N=10 and N=25, in one week. That's 2 algorithms x 4 topologies x 2 scales = 16 configurations, before multiplying by seeds, on top of learning the architecture for the first time.
- **Week 7**: full sweep across N in {5,10,25,50,100} x 4 topologies x up to 3 comm settings x 3 algorithms x non-stationarity fractions, with no wall-clock estimate anywhere in the document and no VRAM budget check against the 8GB RTX 5070. A 100-agent GNN rollout is a meaningfully different memory footprint than a 5-agent one, and that's before BayesG's per-agent ego-graph inference adds its own overhead.

Neither of these needs to be cut entirely, but the plan should say explicitly what the "core" result is (the thing you'd present even if everything after it slips) versus what's "if time permits." My suggestion: make N in {5, 10, 25} the core sweep across all 4 topologies and all 3 algorithms, and treat N in {50, 100} as a stretch goal run only on the 1-2 conditions that look most interesting from the smaller runs. That's a defensible scoping choice for a research report; running a threadbare version of everything at all 5 scales is not.

## 4. No seed count, anywhere

RL results are noisy across random seeds, often more so than across the experimental conditions you're trying to compare. The verification plan has unit tests for code correctness (graph generators, env stepping, forward/backward passes) but nothing for the *scientific* claims: how many seeds per configuration, and what statistic you'll report (mean +/- std, confidence interval, etc.). Without this, a claim like "BayesG recovers faster after a shock than static k-hop" isn't a claim yet, it's an anecdote from one run. Put a number on this now (3-5 seeds minimum per configuration is the usual floor for RL work at this scale, more if you can afford it) and multiply it into the week 7 time budget before you commit to the full sweep.

## 5. The non-stationarity shock protocol is underspecified

The plan says agents switch "from cooperative policy pi_A to perturbed/adversarial/stochastic policy pi_B," hedging across three different things in one phrase. These are not interchangeable and they measure different things:
- A **random/uniform** pi_B tests robustness to noise.
- An **adversarial** pi_B (trained to actively hurt cooperation) tests robustness to hostile disruption.
- A **frozen earlier checkpoint** tests robustness to distributional shift from a stale-but-not-hostile policy.

Pick one as the primary condition and justify it, or run two variants (e.g., random vs adversarial) as an explicit ablation. Right now the roadmap can't actually be executed as written because this design choice hasn't been made.

Related: the Project Overview promises varying "topological roles (e.g., hub vs. peripheral nodes)" for the shock, but this disappears completely by the time you get to Milestone 6 and the week 7 sweep variables, which only vary the *fraction* rho of switched agents. Either add "which nodes get switched" back in as a real experimental factor (this is actually one of the more interesting things you could show, since it connects directly to BayesG's ability to prune a corrupted hub vs a corrupted leaf), or drop the promise from the overview so the document doesn't overclaim.

## 6. Metrics named but not defined

"Instability Index" and "Adaptation Speed" are listed as things you'll measure in week 7, but nowhere does the plan pin down the actual formula: what window size for the rolling variance, what threshold counts as "recovered" for adaptation speed (within X% of pre-shock steady-state return, for how many consecutive steps). These need to be nailed down in `src/evaluation/metrics.py` during week 5 or 6, before you're computing them across a full sweep in week 7, or you'll end up re-running things because the definition changed after the fact.

One more small thing worth naming explicitly in the plan: "Communication Overhead" is trivially zero for IPPO and a fixed, known-in-advance number for static k-hop. The only place this metric is actually interesting is BayesG, where the active-edge fraction should be plotted over training time, especially around t_c, since watching it drop when a neighbor gets corrupted is the headline result the whole BayesG comparison is trying to produce. Say that outright in the plan so it doesn't get treated as "one more metric to compute" alongside the others.

## 7. Environment and tooling claims worth double-checking (verified where I could)

- **COGNAC**: real, NeurIPS 2025 Datasets & Benchmarks paper by Jules Sintes and Ana Bušić (Inria/ENS), repo at `github.com/yojul/cognac`. Confirmed it ships 5 environments (Firefighting 1D/2D, Binary Consensus, SysAdmin, Multi-commodity Flow) and confirmed the "graph agnostic" distinction described in section 1 above.
- **BayesG**: real, NeurIPS 2025 poster by Wei Duan, Jie Lu, Junyu Xuan (UTS), repo at `github.com/Wei9711/BayesG`. Confirmed Python 3.8 + SUMO requirement, A2C baselines, traffic-control-only environments (up to 167 agents on a NewYork road network), as described in section 2.
- **CUDA 13.3**: this is a real, current NVIDIA release (dated May 2026), and CUDA 13.x does still support Blackwell (your RTX 5070). The plan's claim here checks out, I was initially skeptical of the version number and it turned out to be accurate.
- **"Python 3.14 is too new for standard PyTorch/PyG wheels"**: I'd flag this as likely outdated rather than confirmed. Recent PyTorch releases (2.9 and later) do ship official CUDA-enabled wheels for Python 3.14 on Linux. What I could *not* confirm is whether PyG's optional compiled extension wheels (`torch-scatter`, `torch-sparse`, `pyg-lib`, etc., which you'll want for efficient k-hop message passing) currently build for the Python 3.14 / CUDA 13.3 combination specifically, since those extensions have historically lagged mainline PyTorch's Python support by months. Practically this doesn't change the recommendation: pin to Python 3.11 anyway, both because it's the safer bet across the whole dependency chain and because BayesG's own code needs Python 3.8 regardless, so you're going to need at least two isolated environments no matter what you pick for the main stack. Just don't repeat "3.14 is unsupported" as settled fact in the README; say "we pinned to 3.11 for dependency stability across PyG and BayesG's SUMO stack" instead, which is true regardless of the current 3.14 wheel situation.
- **Not verified, check yourself in week 1**: exact PyG extension wheel availability for your torch/CUDA combo, and actual VRAM headroom for a 100-agent networked rollout on the 8GB 5070. Run `python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"` and a quick PyG import check before you build anything on top of the environment, rather than assuming the "uv installs PyTorch + CUDA in seconds" line covers PyG too.

## 8. Smaller things

- No fully-connected (global communication) baseline appears anywhere. IPPO (zero comm) is your floor and BayesG/k-hop are the methods under test, but without a fully-connected upper bound you can't say how much of the achievable ceiling your sparse methods are recovering. Worth adding as a fifth condition, at least at small N where it's cheap to run.
- Milestone 2's toy 2-agent Dec-POMDP notebook is good pedagogy, but make sure its logging harness gets reused for the main experiments rather than thrown away after week 1. Otherwise you're writing the same reward-curve/TensorBoard plumbing twice.

---

## What I'd suggest doing next, concretely

1. Resolve the two contradictions (env choice, GitHub trivia) before writing another line of the roadmap.
2. Move the BayesG feasibility read into week 1-2, and replan weeks 5-8 once you know what "integration" actually costs.
3. Decide the shock protocol (pi_B definition, and whether hub-vs-peripheral is in or out) and write it into `shock_protocol.py`'s spec before week 5.
4. Pin down seed count and the exact formulas for instability index and adaptation speed before week 6.
5. Split the week 7 sweep into a "core" matrix you're committed to and a "stretch" matrix you'll attempt only if the core finishes early.

I can draft any of these next: a tightened week-by-week roadmap with seed counts and a core/stretch split, the operational metric definitions for `metrics.py`, or a week-1 environment feasibility script (`torch`/PyG/CUDA check plus a BayesG repo readthrough checklist). Say which.

---

## Sources checked
- COGNAC paper (OpenReview, NeurIPS 2025 D&B track) and NeurIPS poster slides (`neurips.cc/media/neurips-2025/Slides/121486.pdf`)
- BayesG paper (arXiv 2509.16606, OpenReview, NeurIPS 2025 poster page)
- BayesG official repo, `github.com/Wei9711/BayesG`
- NVIDIA CUDA Toolkit 13.3 release notes (`docs.nvidia.com/cuda/archive/13.3.0`)
- PyTorch GitHub issues on Python 3.14 wheel support (`pytorch/pytorch#156856`, `#169929`) and the Astral `uv` PyTorch integration docs
