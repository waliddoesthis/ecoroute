"""The decision graph: prompt -> skills -> difficulty -> models.

    prompt --P(skill | prompt)--> skill --P(level | prompt)--> (skill, level) --acc--> model

Every edge is a number someone can read:
- prompt -> skill: how much the prompt looks like each kind of task (math, code, ...),
  from a classifier trained on the task labels of the outcome data.
- prompt -> difficulty level (easy / medium / hard): from a classifier trained on how many
  models answered each training prompt.
- (skill, level) -> model: the share of training prompts of that skill and level the
  model answered correctly, shrunk toward the model's overall accuracy when the cell has
  few examples.

A second family of edges links the prompt to its most similar past prompts, and each of
those to the models that answered it correctly (Graph B in the design doc). They catch
what three difficulty levels per skill are too coarse to see.

A model's score is the weight of all paths reaching it:
    P(model correct | prompt) = (1 - beta) * sum over skill, level of
                                    P(skill | prompt) * P(level | prompt) * acc[skill, level, model]
                              + beta * (similarity-weighted share of similar past prompts
                                        the model answered correctly)
beta is chosen on validation data (tune_beta). The heaviest paths are the explanation. The graph is a Predictor, so the catalog
anchors, calibration and the router use it like any other.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import LogisticRegression

from ecoroute.predictors.base import Predictor
from ecoroute.predictors.knn import KNNPredictor

LEVELS = ("easy", "medium", "hard")


def level_of(share_correct: np.ndarray) -> np.ndarray:
    """Same thresholds as the evaluation: hard if at most 25% of models were right,
    easy if at least 75%."""
    return np.where(share_correct <= 0.25, 2, np.where(share_correct >= 0.75, 0, 1))


@dataclass
class GraphExplanation:
    skills: list[tuple[str, float]]  # heaviest prompt -> skill edges
    levels: dict[str, float]  # prompt -> difficulty edges
    # model -> [(skill, level, acc, path weight)], weights already scaled by (1 - beta)
    edges: dict[str, list[tuple[str, str, float, float]]]
    beta: float = 0.0
    # model -> (similar past prompts it solved, similar past prompts it answered, weight)
    neighbours: dict[str, tuple[int, int, float]] = field(default_factory=dict)

    def lines(self, model: str | None = None, top: int = 3) -> list[str]:
        skills = ", ".join(f"{s} {w:.2f}" for s, w in self.skills[:top])
        levels = ", ".join(f"{k} {v:.2f}" for k, v in self.levels.items())
        out = [f"Graph: the prompt looks like {skills}; difficulty {levels}."]
        if model is not None and model in self.edges:
            paths = sorted(self.edges[model], key=lambda e: -e[3])[:2]
            via = "; ".join(f"{s}/{lv}: solves {a:.0%}" for s, lv, a, _ in paths)
            line = f"Graph: strongest paths to {model}: {via}"
            if self.beta > 0 and model in self.neighbours:
                solved, answered, _ = self.neighbours[model]
                line += (
                    f"; it solved {solved} of the {answered} most similar past prompts"
                    f" (weight {self.beta:.2f})"
                )
            out.append(line + ".")
        return out


class SkillGraph(Predictor):
    name = "skill_graph"

    def __init__(
        self,
        shrink: float = 20.0,
        C: float = 1.0,
        min_skill_prompts: int = 50,
        k: int = 32,
        beta: float = 0.0,
    ) -> None:
        super().__init__()
        self.shrink = shrink  # pseudo-count pulling small cells toward the model's mean
        self.C = C
        self.min_skill_prompts = min_skill_prompts
        self.k = k  # similar past prompts per prompt
        self.beta = beta  # weight of the similar-prompt edges; set by tune_beta()

    def fit(self, X, Y, models, skills=None, level_extra=None):
        """skills: task label per training prompt. level_extra: optional (X2, levels) of
        prompts with a labelled difficulty (0 easy, 1 medium, 2 hard) but no outcomes,
        added to the prompt -> difficulty classifier only. Outcome data alone is mostly
        exam questions; labelled prompts from elsewhere teach it what "hard" looks like."""
        if skills is None:
            raise ValueError("SkillGraph needs a skill (task) label per training prompt")
        self.models = list(models)
        skills = np.asarray(skills, dtype=object)
        names, counts = np.unique(skills, return_counts=True)
        keep = set(names[counts >= self.min_skill_prompts])
        skills = np.array([s if s in keep else "other" for s in skills], dtype=object)
        self.skills_ = sorted(set(skills))

        share = np.nanmean(Y >= 0.5, axis=1)
        known = ~np.isnan(share)
        level = level_of(np.where(known, share, 0.5))
        self.skill_clf_ = LogisticRegression(C=self.C, max_iter=1000).fit(X[known], skills[known])
        X_lv, y_lv = X[known], level[known]
        if level_extra is not None:
            X_lv = np.concatenate([X_lv, np.asarray(level_extra[0], dtype=X.dtype)])
            y_lv = np.concatenate([y_lv, np.asarray(level_extra[1])])
        self.level_clf_ = LogisticRegression(C=self.C, max_iter=1000).fit(X_lv, y_lv)

        m = len(self.models)
        overall = np.nanmean(Y, axis=0)
        self.acc_ = np.zeros((len(self.skill_clf_.classes_), len(LEVELS), m))
        self.count_ = np.zeros_like(self.acc_)
        for i, s in enumerate(self.skill_clf_.classes_):
            for b in range(len(LEVELS)):
                sel = known & (skills == s) & (level == b)
                obs = ~np.isnan(Y[sel])
                n = obs.sum(axis=0)
                total = np.nansum(Y[sel], axis=0)
                self.acc_[i, b] = (total + self.shrink * overall) / (n + self.shrink)
                self.count_[i, b] = n
        self.knn_ = KNNPredictor(k=self.k).fit(X, Y, self.models)
        return self

    def _similarity(self, X) -> np.ndarray:
        """Mean similarity of each prompt to its nearest past prompts."""
        sim, _ = self.knn_.neighbours(X)
        return np.clip(sim, 0, None).mean(axis=1)

    def beta_for(self, X) -> np.ndarray:
        """Per-prompt weight of the similar-prompt edges.

        Past prompts only say something about a new prompt that resembles them. The weight
        is beta for a prompt as close to its neighbours as a typical validation prompt, and
        shrinks toward 0 as the neighbours get less similar, leaving the skill and
        difficulty paths (which also learned from labelled prompts) to decide.
        """
        if self.beta == 0:
            return np.zeros(len(X))
        if getattr(self, "ref_similarity_", None) is None:
            return np.full(len(X), self.beta)
        return self.beta * np.clip(self._similarity(X) / self.ref_similarity_, 0, 1)

    def tune_beta(self, X_val, Y_val, grid=np.linspace(0, 1, 11)) -> float:
        """Pick beta (the similar-prompt edge weight for a typical prompt) that minimises
        Brier score on validation data."""
        self.ref_similarity_ = float(np.median(self._similarity(X_val)))
        paths, near = self._paths(X_val), self.knn_.predict_proba(X_val)
        seen = ~np.isnan(Y_val)
        scale = np.clip(self._similarity(X_val) / self.ref_similarity_, 0, 1)[:, None]
        scores = [
            np.mean((((1 - b * scale) * paths + b * scale * near) - Y_val)[seen] ** 2) for b in grid
        ]
        self.beta = float(grid[int(np.argmin(scores))])
        return self.beta

    def _edges(self, X):
        ps = self.skill_clf_.predict_proba(X)  # (n, skills)
        pl = np.zeros((len(X), len(LEVELS)))
        pl[:, self.level_clf_.classes_] = self.level_clf_.predict_proba(X)
        return ps, pl

    def _paths(self, X):
        ps, pl = self._edges(X)
        return np.einsum("ns,nb,sbm->nm", ps, pl, self.acc_)

    def predict_proba(self, X):
        paths = self._paths(X)
        if self.beta == 0:
            return paths
        b = self.beta_for(X)[:, None]
        return (1 - b) * paths + b * self.knn_.predict_proba(X)

    def explain(self, x: np.ndarray) -> GraphExplanation:
        ps, pl = self._edges(np.asarray(x, dtype=np.float32).reshape(1, -1))
        ps, pl = ps[0], pl[0]
        order = np.argsort(-ps)
        skills = [(str(self.skill_clf_.classes_[i]), float(ps[i])) for i in order]
        x2 = np.asarray(x, dtype=np.float32).reshape(1, -1)
        beta = float(self.beta_for(x2)[0])
        scale = 1 - beta
        edges = {}
        for j, model in enumerate(self.models):
            edges[model] = [
                (str(self.skill_clf_.classes_[i]), LEVELS[b], float(self.acc_[i, b, j]),
                 float(scale * ps[i] * pl[b] * self.acc_[i, b, j]))
                for i in range(len(ps)) for b in range(len(LEVELS))
            ]  # fmt: skip
        neighbours = {}
        if beta > 0:
            near = self.knn_.predict_proba(x2)[0]
            _, idx = self.knn_.neighbours(x2)
            y = self.knn_.Y_[idx[0]]
            for j, model in enumerate(self.models):
                col = y[:, j][~np.isnan(y[:, j])]
                neighbours[model] = (int((col >= 0.5).sum()), len(col), float(beta * near[j]))
        return GraphExplanation(skills, dict(zip(LEVELS, map(float, pl))), edges, beta, neighbours)
