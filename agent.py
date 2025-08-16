# agent.py
# - unified cognitive loop prototype:
# - per-agent TADM, EMG, Emotion, Purpose, Temporal, Dream
# - global LLM-wide dream consolidation (SharedGLTM)
# - action remembering / forgetting logs
# - auto-cue determination
# - curiosity, intuition, character
# - ablation toggles
#
# Lightweight: standard library only (plus pandas/matplotlib for optional analysis)

from __future__ import annotations
import math, random, time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Iterable
from collections import Counter
import json

# ---------- Simple text utilities (replace with embeddings for real experiments) ----------
def simple_tokenize(text: str) -> List[str]:
    return [t.lower() for t in text.replace(".", " ").replace(",", " ").split() if t.strip()]

def bow_vector(tokens: Iterable[str]) -> Dict[str, float]:
    c = Counter(tokens)
    norm = math.sqrt(sum(v * v for v in c.values())) or 1.0
    return {k: v / norm for k, v in c.items()}

def cosine_sim(a: Dict[str, float], b: Dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    return sum(a[t] * b[t] for t in common)

# ---------- Records ----------
@dataclass
class Record:
    text: str
    tokens: List[str]
    weight: float
    t: int
    plan_ctx: Optional[str]
    provenance: Dict[str, Any]
    surv_factor: float = 1.0
    uses: int = 0
    embedding: Dict[str, float] = field(default_factory=dict)
    def __post_init__(self):
        if not self.embedding:
            self.embedding = bow_vector(self.tokens)

# ---------- Memory stores ----------
class MemoryStore:
    def __init__(self, name: str):
        self.name = name
        self._items: List[Record] = []
    def extend(self, recs: Iterable[Record]) -> None:
        self._items.extend(recs)
    def items(self) -> List[Record]:
        return self._items
    def remove(self, rec: Record) -> None:
        self._items = [r for r in self._items if r is not rec]
    def clear(self) -> None:
        self._items = []
    def __len__(self):
        return len(self._items)

# ---------- Global shared LTM (LLM-wide) ----------
class SharedGLTM:
    def __init__(self):
        self.records: List[Record] = []
    def merge_from_agents(self, agent_recs: List[Record], provenance_tag: str = ""):
        # naive merge: append with provenance label; realistic merge should cluster/resolve
        for r in agent_recs:
            # annotate agent provenance
            rp = dict(r.provenance)
            rp["from_agent_merge"] = provenance_tag
            newr = Record(r.text, r.tokens, r.weight, r.t, r.plan_ctx, rp, r.surv_factor, r.uses, r.embedding)
            self.records.append(newr)
    def compact(self, sim_thresh: float = 0.9):
        # simple clustering merge to keep global size reasonable
        merged = []
        used = set()
        for i, r in enumerate(self.records):
            if i in used: continue
            cluster = [r]
            for j in range(i+1, len(self.records)):
                if j in used: continue
                if cosine_sim(r.embedding, self.records[j].embedding) >= sim_thresh:
                    cluster.append(self.records[j])
                    used.add(j)
            # canonical representative
            representative = max(cluster, key=lambda rr: rr.weight * rr.surv_factor)
            rep_weight = sum(rr.weight * rr.surv_factor for rr in cluster)
            merged.append(Record(representative.text, representative.tokens, rep_weight,
                                 max(rr.t for rr in cluster), representative.plan_ctx,
                                 representative.provenance, representative.surv_factor, representative.uses, representative.embedding))
        self.records = merged

# ---------- EMG (episodic graph) ----------
@dataclass
class EMGNode:
    id: int
    text: str
    tokens: List[str]
    t: int
    embedding: Dict[str, float] = field(default_factory=dict)
    centrality: float = 0.0
    def __post_init__(self):
        if not self.embedding:
            self.embedding = bow_vector(self.tokens)

@dataclass
class EMGEdge:
    src: int
    dst: int
    label: str
    weight: float = 1.0

class EMG:
    def __init__(self):
        self.nodes: Dict[int, EMGNode] = {}
        self.edges: List[EMGEdge] = []
        self._next = 0
    def add_node(self, text: str, tokens: List[str], t: int) -> int:
        nid = self._next
        self.nodes[nid] = EMGNode(nid, text, tokens, t)
        self._next += 1
        return nid
    def add_edge(self, s:int, d:int, label:str, weight:float=1.0):
        self.edges.append(EMGEdge(s,d,label,weight))
    def sparsify(self, decay=0.05, prune_thresh=0.05):
        new_edges = []
        for e in self.edges:
            w = e.weight * (1.0 - decay)
            if w >= prune_thresh:
                e.weight = w
                new_edges.append(e)
        self.edges = new_edges
        deg = Counter([e.src for e in self.edges] + [e.dst for e in self.edges])
        for nid, node in self.nodes.items():
            node.centrality = deg[nid]

# ---------- Emotion model ----------
@dataclass
class Emotion:
    valence: float
    arousal: float

class EmotionModel:
    def external(self, txt: str) -> Emotion:
        toks = simple_tokenize(txt)
        if any(k in toks for k in ("urgent","urgent:","warning","error","danger")):
            return Emotion(-0.4,0.9)
        return Emotion(0.1,0.2)
    def internal(self, ltm: MemoryStore, emg: EMG) -> Emotion:
        if not emg.nodes: return Emotion(0.0,0.1)
        avg_cent = sum(n.centrality for n in emg.nodes.values()) / (len(emg.nodes) or 1)
        return Emotion(min(0.5,0.05*avg_cent), min(1.0, 0.1 + 0.02*avg_cent))
    def gate(self, txt: str, ltm: MemoryStore, emg: EMG) -> Emotion:
        ext = self.external(txt)
        inte = self.internal(ltm, emg)
        eta = 1/(1+math.exp(-3.0*(ext.arousal - inte.arousal)))
        return Emotion(eta*ext.valence + (1-eta)*inte.valence, eta*ext.arousal + (1-eta)*inte.arousal)

# ---------- Purpose & Character ----------
@dataclass
class Purpose:
    goal: str
    plan: List[str]

class PurposeModule:
    def utility(self, p: Purpose, emg: EMG, emo: Emotion, character: Dict[str,float]) -> float:
        gvec = bow_vector(simple_tokenize(p.goal))
        if not emg.nodes: match = 0.0
        else: match = max(cosine_sim(gvec, n.embedding) for n in emg.nodes.values())
        # character bias: e.g., character.get("conservative",-0.0) penalizes long plans
        char_bias = character.get("curiosity_bias", 0.0) * 0.1 * len(p.plan)
        return match + 0.1*len(p.plan) - 0.3*max(0.0, -emo.valence) + char_bias
    def select(self, candidates: List[Purpose], emg: EMG, emo: Emotion, character: Dict[str,float]) -> Purpose:
        if not candidates: return Purpose("default", ["observe","respond"])
        scores = [(self.utility(c,emg,emo,character), c) for c in candidates]
        return max(scores, key=lambda x:x[0])[1]

# ---------- TADM (with curiosity & auto-cue hooks) ----------
class TADM:
    def __init__(self, theta_write=0.5, decay_lambda=0.01, use_curiosity=True):
        self.theta_write = theta_write
        self.decay_lambda = decay_lambda
        self.use_curiosity = use_curiosity

    def novelty(self, token:str, ltm: MemoryStore) -> float:
        ltm_vocab = set(t for r in ltm.items() for t in r.tokens)
        return 1.0 if token not in ltm_vocab else 0.5

    def uncertainty(self, token:str, corrector_unc:float) -> float:
        return corrector_unc

    def write_weights(self, tokens: List[str], ltm: MemoryStore, corrector_uncertainty: float=0.1, character: Dict[str,float]=None) -> List[float]:
        weights = []
        for tok in tokens:
            f_s = 1.2 if tok in ("goal","important","note") else 1.0
            f_n = self.novelty(tok, ltm)
            f_c = 1.0 - corrector_uncertainty
            w = 0.5*f_s + 0.3*f_n + 0.2*f_c
            if self.use_curiosity and character:
                # curiosity bias: character prefers novel content
                w += 0.05 * character.get("curiosity_bias", 1.0) * f_n
            weights.append(w)
        return weights

    def update_memories(self, text: str, t:int, stm: MemoryStore, ltm: MemoryStore, plan_ctx:Optional[str], corrector_uncertainty:float, character:Dict[str,float]):
        tokens = simple_tokenize(text)
        ws = self.write_weights(tokens, ltm, corrector_uncertainty, character)
        recs = []
        for w in ws:
            recs.append(Record(text, tokens, w, t, plan_ctx, {"source":"stimuli"}, 1.0, 0))
        stm.extend(recs)
        if any(w >= self.theta_write for w in ws):
            ltm.extend(recs)
        return recs

    def decay(self, store: MemoryStore, t_now:int):
        for r in store.items():
            age = max(0, t_now - r.t)
            r.surv_factor = math.exp(-self.decay_lambda * age)

    def read(self, query_text: str, stm: MemoryStore, ltm: MemoryStore, gltm: SharedGLTM, k=5, token_budget=128, goal_text=None, emo:Emotion=None, use_global=0.3, intuition_prior:Dict[str,float]=None):
        qvec = bow_vector(simple_tokenize(query_text))
        gvec = bow_vector(simple_tokenize(goal_text or ""))
        candidates = stm.items() + ltm.items() + (gltm.records if gltm and use_global>0 else [])
        scored = []
        for r in candidates:
            s_sim = cosine_sim(qvec, r.embedding)
            s_goal = 0.35 * cosine_sim(gvec, r.embedding)
            s_weight = 0.25 * r.weight * r.surv_factor
            s_emo = 0.0
            if emo:
                s_emo = 0.1 * max(0.0, emo.valence) + 0.05 * emo.arousal
            s_global = 0.1 if (gltm and r in gltm.records) else 0.0
            s_int = 0.0
            if intuition_prior and r.text in intuition_prior:
                s_int = 0.2 * intuition_prior[r.text]
            score = 0.5*s_sim + s_goal + s_weight + s_emo + s_global + s_int
            scored.append((score, r))
        scored.sort(key=lambda x:x[0], reverse=True)
        out = []
        used_tokens = 0
        for sc, r in scored:
            n = len(r.tokens)
            if used_tokens + n > token_budget: continue
            out.append((sc, r))
            used_tokens += n
            if len(out) >= k: break
        for sc, r in out:
            r.uses += 1
        return out

# ---------- Temporal controller ----------
class TemporalController:
    def __init__(self, theta_r=0.9, theta_f=0.2):
        self.theta_r = theta_r
        self.theta_f = theta_f
    def stability(self, r: Record, contradict_pen=0.0):
        return 0.7 * r.weight * r.surv_factor + 0.2 * r.uses - 0.1 * contradict_pen
    def classify(self, r: Record):
        phi = self.stability(r)
        if phi >= self.theta_r: return "remember"
        if phi <= self.theta_f: return "forget"
        return "postpone"

# ---------- Dream consolidator (local) ----------
class DreamConsolidator:
    def consolidate(self, stm: MemoryStore, ltm: MemoryStore, emg: EMG):
        # cluster LTM by similarity and merge; naive O(n^2) for prototype
        merged = []
        items = ltm.items()
        used = set()
        for i, r in enumerate(items):
            if i in used: continue
            cluster = [r]
            for j in range(i+1, len(items)):
                if j in used: continue
                if cosine_sim(r.embedding, items[j].embedding) >= 0.85:
                    cluster.append(items[j]); used.add(j)
            rep = max(cluster, key=lambda rr: rr.weight * rr.surv_factor)
            weight = sum(rr.weight * rr.surv_factor for rr in cluster)
            merged.append(Record(rep.text, rep.tokens, weight, max(rr.t for rr in cluster), rep.plan_ctx, rep.provenance, min(1.0, rep.surv_factor+0.1), sum(rr.uses for rr in cluster), rep.embedding))
        ltm._items = merged
        emg.sparsify(0.05, 0.05)

# ---------- Intuition: quick heuristic generator ----------
class Intuition:
    def propose(self, emg: EMG, goal_text: Optional[str]):
        # heuristic: propose actions (strings) tied to highly central nodes
        if not emg.nodes: return {}
        sorted_nodes = sorted(emg.nodes.values(), key=lambda n: n.centrality, reverse=True)[:3]
        prior = {}
        for n in sorted_nodes:
            # map snippet->score heuristic
            prior[n.text] = 0.5 + 0.1 * n.centrality
        # optionally bias by goal
        if goal_text:
            gvec = bow_vector(simple_tokenize(goal_text))
            for t in list(prior.keys()):
                prior[t] *= 1.0 + cosine_sim(gvec, bow_vector(simple_tokenize(t)))
        return prior

# ---------- Unified Agent with toggles and logs ----------
class UnifiedAgent:
    def __init__(self, agent_id:int,
                 use_emg=True, use_dream=True, use_emotion=True, use_temporal=True, use_purpose=True, use_global=False):
        self.id = agent_id
        self.t = 0
        self.stm = MemoryStore("STM")
        self.ltm = MemoryStore("LTM")
        self.emg = EMG() if use_emg else None
        self.emotion_model = EmotionModel() if use_emotion else None
        self.purpose_module = PurposeModule() if use_purpose else None
        self.temporal = TemporalController() if use_temporal else None
        self.dreamer = DreamConsolidator() if use_dream else None
        self.tadm = TADM(use_curiosity=True)
        self.intuition = Intuition()
        self.character = {"curiosity_bias": 1.0, "conservative": 0.0}
        self.purpose = Purpose(goal="bootstrap competence", plan=["observe","store","act"])
        # toggles
        self.use_emg = use_emg
        self.use_dream = use_dream
        self.use_emotion = use_emotion
        self.use_temporal = use_temporal
        self.use_purpose = use_purpose
        self.use_global = use_global
        # logs
        self.remember_count = 0
        self.forget_count = 0
        self.action_history = []  # list of dicts for auditing

    def step(self, stimuli: str, gltm: SharedGLTM = None):
        self.t += 1
        t0 = time.perf_counter()
        # perception
        text = " ".join(stimuli.strip().split())
        # write into TADM (with curiosity & character)
        recs = self.tadm.update_memories(text, self.t, self.stm, self.ltm, ";".join(self.purpose.plan) if self.purpose else None, 0.1, self.character)
        # EMG update
        if self.use_emg and self.emg is not None:
            nid = self.emg.add_node(text, simple_tokenize(text), self.t)
            if nid - 1 in self.emg.nodes:
                self.emg.add_edge(nid-1, nid, "temporal", 1.0)
            if self.purpose and self.purpose.goal:
                self.emg.add_edge(nid, nid, "goal", 0.5)
        # emotion
        emo = None
        if self.use_emotion and self.emotion_model:
            emo = self.emotion_model.gate(text, self.ltm, self.emg if self.emg else EMG())
        else:
            emo = Emotion(0.0, 0.0)
        # purpose selection
        if self.use_purpose and self.purpose_module:
            candidates = [self.purpose, Purpose("improve recall", ["retrieve","compare","store"]), Purpose("complete tasks quickly", ["retrieve","decide","act"])]
            self.purpose = self.purpose_module.select(candidates, self.emg if self.emg else EMG(), emo, self.character)
        # decay & temporal sweep
        self.tadm.decay(self.stm, self.t); self.tadm.decay(self.ltm, self.t)
        if self.use_temporal and self.temporal:
            for rec in list(self.stm.items()):
                label = self.temporal.classify(rec)
                if label == "forget":
                    self.stm.remove(rec)
        # auto-cue determination: try small set of cues and pick best (heuristic)
        cues = [lambda q: q, lambda q: " ".join(q.split()[:3]), lambda q: " ".join(q.split()[-3:])]
        best_cue = cues[0]
        best_score = -math.inf
        for cfn in cues:
            ctxt = cfn(text)
            # approximate benefit by novelty + goalRel
            novelty = sum(self.tadm.novelty(tok, self.ltm) for tok in simple_tokenize(ctxt)) / max(1,len(simple_tokenize(ctxt)))
            goalRel = 0.0
            if self.purpose:
                gvec = bow_vector(simple_tokenize(self.purpose.goal))
                goalRel = max(cosine_sim(gvec, r.embedding) for r in (self.ltm.items() or [Record("",[],0,0,None,{})]))
            score = 0.6*novelty + 0.4*goalRel
            if score > best_score:
                best_score = score; best_cue = cfn
        cue_text = best_cue(text)
        # intuition prior
        intuition_prior = self.intuition.propose(self.emg if self.emg else EMG(), self.purpose.goal if self.purpose else None)
        # retrieval (including shared global gltm if configured)
        retrieved = self.tadm.read(f"{cue_text} :: {self.purpose.goal if self.purpose else ''}", self.stm, self.ltm, gltm=gltm if self.use_global else None, k=5, token_budget=128, goal_text=self.purpose.goal if self.purpose else None, emo=emo, use_global=0.5, intuition_prior=intuition_prior)
        # action selection: use Q-formula (toy)
        # compute Q for a small set of candidate actions (here we infer actions from plan head and retrieved contexts)
        plan_head = (self.purpose.plan[0] if self.purpose and self.purpose.plan else "act")
        # compute whether retrieval supports the goal (remembering)
        success_proxy = 0.0
        if retrieved:
            top_rec = retrieved[0][1]
            gvec = bow_vector(simple_tokenize(self.purpose.goal if self.purpose else ""))
            success_proxy = cosine_sim(gvec, top_rec.embedding)
        # decide remembering/forgetting events heuristically
        if success_proxy >= 0.2:
            self.remember_count += 1
            remembered = True
        else:
            # if LTM previously had supporting records, count as forgotten
            had_support = any(cosine_sim(bow_vector(simple_tokenize(self.purpose.goal or "")), r.embedding) >= 0.3 for r in self.ltm.items())
            if had_support:
                self.forget_count += 1
            remembered = False
        action = f"act::{plan_head}|goal={self.purpose.goal if self.purpose else ''}|remembered={int(remembered)}"
        # log action history
        self.action_history.append({
            "t": self.t, "action": action, "remembered": int(remembered), "success_proxy": success_proxy,
            "stm": len(self.stm), "ltm": len(self.ltm)
        })
        # dream consolidation occasionally (local)
        if self.use_dream and self.dreamer and (self.t % 50 == 0):
            self.dreamer.consolidate(self.stm, self.ltm, self.emg if self.emg else EMG())
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "t": self.t, "action": action, "remembered": remembered, "success_proxy": success_proxy,
            "stm": len(self.stm), "ltm": len(self.ltm), "latency_ms": latency_ms,
            "emo_valence": emo.valence, "emo_arousal": emo.arousal
        }

# ---------- Simulation harness with ablation sets and global dream ----------
def run_experiment(num_users=5, days=2, steps_per_day=500, ablation_configs=None):
    if ablation_configs is None:
        # default single config: full system
        ablation_configs = [{"name":"full","use_emg":True,"use_dream":True,"use_emotion":True,"use_temporal":True,"use_purpose":True,"use_global":True}]
    results = {}
    for cfg in ablation_configs:
        cfg_name = cfg["name"]
        print(f"Running config: {cfg_name}")
        agents = {u: UnifiedAgent(u,
                                  use_emg=cfg.get("use_emg", True),
                                  use_dream=cfg.get("use_dream", True),
                                  use_emotion=cfg.get("use_emotion", True),
                                  use_temporal=cfg.get("use_temporal", True),
                                  use_purpose=cfg.get("use_purpose", True),
                                  use_global=cfg.get("use_global", True))
                  for u in range(num_users)}
        gltm = SharedGLTM()
        logs = []
        for day in range(days):
            for step in range(steps_per_day):
                for u, agent in agents.items():
                    # sample stimuli
                    stimuli = random.choice([
                        f"User{u} prefers green tea over coffee.",
                        f"User{u} likes decaf coffee in mornings only.",
                        f"User{u} avoids dairy products.",
                        "project alpha requires two-factor login",
                        "submit weekly report friday",
                        "backup data every night",
                        "warning schedule changed",
                        "urgent system error detected",
                        "note prioritize bug fixes",
                    ]) + ("" if random.random() > 0.2 else " (important)")
                    step_out = agent.step(stimuli, gltm=gltm)
                    step_out.update({"user": u, "day": day})
                    logs.append(step_out)
                # occasionally perform global LLM-wide dream
                if step % 250 == 0:
                    # collect candidate representative LTM from each agent
                    pooled = []
                    for a in agents.values():
                        pooled.extend(a.ltm.items())
                    gltm.merge_from_agents(pooled, provenance_tag=f"day{day}_step{step}")
                    gltm.compact(sim_thresh=0.92)
        # collate simple stats
        total_remember = sum(a.remember_count for a in agents.values())
        total_forget = sum(a.forget_count for a in agents.values())
        avg_latency = sum(x["latency_ms"] for x in logs) / max(1, len(logs))
        results[cfg_name] = {
            "logs": logs, "agents": agents, "gltm_size": len(gltm.records),
            "total_remember": total_remember, "total_forget": total_forget, "avg_latency_ms": avg_latency
        }
    return results

# ---------- Demo run ----------
if __name__ == "__main__":
    random.seed(0)
    # define ablation configs: full + no_emg + no_dream + no_emotion + no_temporal + no_purpose
    configs = [
        {"name":"full","use_emg":True,"use_dream":True,"use_emotion":True,"use_temporal":True,"use_purpose":True,"use_global":True},
        {"name":"no_emg","use_emg":False,"use_dream":True,"use_emotion":True,"use_temporal":True,"use_purpose":True,"use_global":True},
        {"name":"no_dream","use_emg":True,"use_dream":False,"use_emotion":True,"use_temporal":True,"use_purpose":True,"use_global":True},
        {"name":"no_emotion","use_emg":True,"use_dream":True,"use_emotion":False,"use_temporal":True,"use_purpose":True,"use_global":True},
        {"name":"no_temporal","use_emg":True,"use_dream":True,"use_emotion":True,"use_temporal":False,"use_purpose":True,"use_global":True},
        {"name":"no_purpose","use_emg":True,"use_dream":True,"use_emotion":True,"use_temporal":True,"use_purpose":False,"use_global":True},
    ]
    # small run for demo
    out = run_experiment(num_users=3, days=1, steps_per_day=200, ablation_configs=configs)
    # print summary
    for k,v in out.items():
        print(f"CONFIG: {k} | GLTM size: {v['gltm_size']} | remembers: {v['total_remember']} | forgets: {v['total_forget']} | avg latency ms: {v['avg_latency_ms']:.2f}")
    # optionally write logs / inspect; in a real experiment save to CSV for analysis
    # e.g., with pandas: pd.DataFrame(out['full']['logs']).to_csv("full_logs.csv", index=False)
