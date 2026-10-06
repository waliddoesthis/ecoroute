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

A model's score is the weight of all paths reaching it:
    P(model correct | prompt) = sum over skill, level of
                                P(skill | prompt) * P(level | prompt) * acc[skill, level, model]
and the heaviest paths are the explanation. The graph is a Predictor, so the catalog
anchors, calibration and the router use it like any other.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression

from ecoroute.predictors.base import Predictor

LEVELS = ("easy", "medium", "hard")


def level_of(share_correct: np.ndarray) -> np.ndarray:
    """Same thresholds as the evaluation: hard if at most 25% of models were right,
    easy if at least 75%."""
    return np.where(share_correct <= 0.25, 2, np.where(share_correct >= 0.75, 0, 1))


@dataclass
class GraphExplanation:
    skills: list[tuple[str, float]]  # heaviest prompt -> skill edges
    levels: dict[str, float]  # prompt -> difficulty edges
    edges: dict[
        str, list[tuple[str, str, float, float]]
    ]  # model -> [(skill, level, acc, path weight)]

    def lines(self, model: str | None = None, top: int = 3) -> list[str]:
        skills = ", ".join(f"{s} {w:.2f}" for s, w in self.skills[:top])
        levels = ", ".join(f"{k} {v:.2f}" for k, v in self.levels.items())
        out = [f"Graph: the prompt looks like {skills}; difficulty {levels}."]
        if model is not None and model in self.edges:
            paths = sorted(self.edges[model], key=lambda e: -e[3])[:2]
            via = "; ".join(f"{s}/{lv}: solves {a:.0%}" for s, lv, a, _ in paths)
            out.append(f"Graph: strongest paths to {model}: {via}.")
        return out


class SkillGraph(Predictor):
    name = "skill_graph"

    def __init__(self, shrink: float = 20.0, C: float = 1.0, min_skill_prompts: int = 50) -> None:
        super().__init__()
        self.shrink = shrink  # pseudo-count pulling small cells toward the model's mean
        self.C = C
        self.min_skill_prompts = min_skill_prompts

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
        return self

    def _edges(self, X):
        ps = self.skill_clf_.predict_proba(X)  # (n, skills)
        pl = np.zeros((len(X), len(LEVELS)))
        pl[:, self.level_clf_.classes_] = self.level_clf_.predict_proba(X)
        return ps, pl

    def predict_proba(self, X):
        ps, pl = self._edges(X)
        return np.einsum("ns,nb,sbm->nm", ps, pl, self.acc_)

    def explain(self, x: np.ndarray) -> GraphExplanation:
        ps, pl = self._edges(np.asarray(x, dtype=np.float32).reshape(1, -1))
        ps, pl = ps[0], pl[0]
        order = np.argsort(-ps)
        skills = [(str(self.skill_clf_.classes_[i]), float(ps[i])) for i in order]
        edges = {}
        for j, model in enumerate(self.models):
            edges[model] = [
                (str(self.skill_clf_.classes_[i]), LEVELS[b], float(self.acc_[i, b, j]),
                 float(ps[i] * pl[b] * self.acc_[i, b, j]))
                for i in range(len(ps)) for b in range(len(LEVELS))
            ]  # fmt: skip
        return GraphExplanation(skills, dict(zip(LEVELS, map(float, pl))), edges)
