"""Players: random, the linear heuristic, a Kev checkpoint in-process, or any System One endpoint over HTTP.

Every agent's `act(game, placements)` returns (index of the chosen placement, info dict). The info dict carries the
probabilities when there are any, so the viewer and the logs can show them.
"""
import math
import random

from .env import heuristic_score
from .encode import MAX_BRANCH_TOKENS, request_of


class RandomAgent:
    name = "random"

    def __init__(self, seed=0):
        self.rng = random.Random(seed)

    def act(self, game, placements):
        return self.rng.randrange(len(placements)), {}


class HeuristicAgent:
    name = "heuristic"

    def __init__(self, temperature=0.0, seed=0):
        self.temperature, self.rng = temperature, random.Random(seed)

    def act(self, game, placements):
        scores = [heuristic_score(p) for p in placements]
        if self.temperature <= 0:
            return max(range(len(scores)), key=scores.__getitem__), {"scores": scores}
        m = max(scores); w = [math.exp((s - m) / self.temperature) for s in scores]; z = sum(w)
        probs = [x / z for x in w]
        return self.rng.choices(range(len(scores)), probs)[0], {"probs": probs}


class RemoteAgent:
    """Any System One server (kev.serve, or Jev through its API)."""
    name = "remote"

    def __init__(self, url="http://127.0.0.1:8009", encoding="rich", sample=False, seed=0, api_key=None):
        import httpx
        self.url, self.encoding, self.sample, self.rng = url.rstrip("/"), encoding, sample, random.Random(seed)
        self.client = httpx.Client(timeout=120, headers={"authorization": f"Bearer {api_key}"} if api_key else {})

    def act(self, game, placements):
        req, keys = request_of(game, placements, self.encoding)
        r = self.client.post(f"{self.url}/v1/systemone", json=req); r.raise_for_status()
        body = r.json(); a = body["answers"]["placement"]
        probs = [a["probabilities"][k] for k in keys]
        info = {"probs": probs, "survive": body["answers"].get("survive", {}).get("noul"), "latency_ms": body.get("latency_ms"), "input_tokens": body["usage"]["input_tokens"]}
        return _pick(probs, self.sample, self.rng), info


class ModelAgent:
    """A Kev checkpoint loaded in this process (kev.checkpoint.load): the same pass the server runs, without HTTP."""
    name = "model"

    def __init__(self, run, device="cpu", encoding="rich", sample=False, seed=0, temperature=None):
        import torch
        from kev.api import SystemOneRequest, to_record
        from kev.checkpoint import LoadOptions, load
        self.torch, self.SystemOneRequest, self.to_record = torch, SystemOneRequest, to_record
        self.tok, self.model = load(run, device, LoadOptions(dtype=torch.bfloat16 if str(device).startswith("cuda") else None, temperature=temperature))
        self.encoding, self.sample, self.rng = encoding, sample, random.Random(seed)

    def probs_of(self, game, placements):
        req, keys = request_of(game, placements, self.encoding)
        rec, _ = self.to_record(self.SystemOneRequest.model_validate(req))
        enc = self.model.encode(self.tok, rec, max_branch=MAX_BRANCH_TOKENS)
        ps = self.model.probs(enc)
        return [float(x) for x in ps[0]], float(ps[1][1]) if len(ps) > 1 else None, enc["state_tokens"]

    def act(self, game, placements):
        probs, survive, state_tokens = self.probs_of(game, placements)
        return _pick(probs, self.sample, self.rng), {"probs": probs, "survive": survive, "state_tokens": state_tokens}


def _pick(probs, sample, rng):
    if sample: return rng.choices(range(len(probs)), probs)[0]
    return max(range(len(probs)), key=probs.__getitem__)


def make_agent(kind, **kw):
    if kind == "random": return RandomAgent(seed=kw.get("seed", 0))
    if kind == "heuristic": return HeuristicAgent(temperature=kw.get("temperature", 0.0), seed=kw.get("seed", 0))
    if kind == "remote": return RemoteAgent(url=kw.get("url", "http://127.0.0.1:8009"), encoding=kw.get("encoding", "rich"), sample=kw.get("sample", False), seed=kw.get("seed", 0))
    if kind == "model": return ModelAgent(kw["run"], device=kw.get("device", "cpu"), encoding=kw.get("encoding", "rich"), sample=kw.get("sample", False), seed=kw.get("seed", 0))
    raise ValueError(f"unknown agent {kind!r}")
