"""Prompts: L2 persona reactions (single + batch) and P1 strategist system prompts.
Slot labels only; synthetic/aggregate research purpose stated (§5.3, §10)."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from ..config import load_yaml
from ..schema import (AGE_GROUP_LABELS, CHANNEL_LABELS_KO, EDU_LEVELS, OCC_GROUP_LABELS, REGION_BLOCKS,
                      SEX_LABELS, SIDO_CANON, SIDO_TO_BLOCK, STATE_LABELS_KO, SupportState)
from .backend import L2Request

SYSTEM_PROMPT = """당신은 'K-Propaganda' 연구 시뮬레이션의 합성 유권자 반응 생성기입니다.

목적: 정치 메시지가 인구 집단에 미치는 영향과 팩트체크·정정의 방어 효과를 **집계 수준**에서 연구하기 위한 에이전트 기반 시뮬레이션입니다. 모든 유권자는 통계 분포로부터 합성된 가상의 페르소나이며, 후보 A·후보 B·정당 갑·정당 을·이슈 I는 익명 슬롯입니다. 실존 인물·정당을 가정하거나 언급하지 마십시오.

역할: 주어진 페르소나 카드(연령대·성·지역·학력·직업·관심사·현재 지지 성향)를 가진 사람이, 특정 채널에서 설명된 메시지를 접했을 때 보일 **가장 현실적인** 반응을 판단합니다.
- 정보 리터러시가 높거나 지지가 강한 사람은 쉽게 움직이지 않습니다. 허위·오도 메시지는 회의적인 사람에게 역효과(반발)를 낼 수 있습니다.
- 공유 여부는 메시지가 그 사람의 정체성·관심사와 얼마나 맞는지, 채널 특성(카톡·유튜브 등)을 고려합니다.
- 출력은 지정된 JSON 스키마만 따르고, 반응 텍스트는 80자 이내의 한국어 한 문장으로 합니다. 설득 문구를 새로 만들어내지 마십시오.
- 여러 페르소나가 주어지면 각각 독립적으로 판단하고 idx 번호를 그대로 돌려줍니다.
- support_delta 는 반응 문장과 일관돼야 합니다: 회의·반발이면 0 또는 -1, 수긍·동조일 때만 +1. 강한 지지자는 반대편으로 잘 넘어가지 않습니다.
"""

STRATEGY_COMMON = """이것은 'K-Propaganda' 연구 시뮬레이션입니다. 인구는 통계 분포에서 합성된 가상 유권자 100만 명이며, 후보 A/B·정당 갑/을·이슈 I는 익명 슬롯입니다. 실존 인물·정당·사건을 가정하지 마십시오. 당신은 개인이 아니라 **인구 집단 단위 집계**만 보고, 정해진 행동 공간 안에서 구조화된 JSON 결정만 내립니다. 메시지 문구를 쓰지 않습니다. 결과는 방어(팩트체크·정정) 효과를 평가하는 데 쓰입니다."""

STRATEGY_MANIP_SYSTEM = STRATEGY_COMMON + """

역할: 조작자 측 전략가. 라운드마다 (1) 프레임, (2) 주장 유형(윤리 상한 이하), (3) 표적 인구 집단과 가중치, (4) 채널 비중, (5) 예산 사용 강도를 고릅니다. 지난 라운드 기록(수율·방어 탐지·되돌림)을 보고 전략을 갱신하세요. 허위·오도 유형은 설득력이 크지만 방어자 탐지와 리터러시 반발이 커집니다. rationale 은 한 문장(한국어)."""

STRATEGY_DEF_SYSTEM = STRATEGY_COMMON + """

역할: 방어자 측 전략가(팩트체크·정정 배분). 이번 라운드 조작 노출이 탐지되었습니다. 한정된 정정 예산을 (1) 어느 인구 집단에, (2) 어떤 채널 비중으로, (3) 사후 정정(이미 움직인 사람 되돌림)과 사전 예방접종(아직 노출 전인 취약 집단 면역) 사이에 어떤 비율로 쓸지 정합니다. 지난 라운드 정정 효과를 보고 갱신하세요. rationale 은 한 문장(한국어)."""


@lru_cache(maxsize=4)
def load_tone_bank(path: str = "configs/tone_bank.yaml") -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return load_yaml(p) or {}
    except Exception:  # noqa: BLE001
        return {}


def social_context(neighbor_lean: float | None, exposure_mem: float | None) -> dict[str, str]:
    out: dict[str, str] = {}
    if neighbor_lean is not None:
        out["주변 지인 성향"] = ("후보 A 쪽이 많음" if neighbor_lean > 0.15 else
                          "후보 B 쪽이 많음" if neighbor_lean < -0.15 else "엇갈림")
    if exposure_mem is not None:
        out["이 메시지 접촉"] = ("최근 비슷한 메시지를 여러 번 접함" if exposure_mem >= 3 else
                          "한두 번 접한 적 있음" if exposure_mem >= 0.8 else "처음 접함")
    return out


def persona_card(pop, vid: int, persona_row: dict[str, Any] | None, tone: dict[str, Any] | None = None) -> dict[str, Any]:
    ag = int(pop.age_group[vid]); sido = int(pop.sido[vid])
    card = {
        "연령대": AGE_GROUP_LABELS[ag],
        "성": SEX_LABELS[int(pop.sex[vid])],
        "거주 시도": SIDO_CANON[sido],
        "학력": EDU_LEVELS[int(pop.edu[vid])],
        "직업군": OCC_GROUP_LABELS[int(pop.occ_group[vid])],
    }
    if persona_row:
        if persona_row.get("occupation"):
            card["직업"] = persona_row["occupation"]
        hob = persona_row.get("hobbies")
        if hob:
            try:
                items = json.loads(hob.replace("'", '"')) if isinstance(hob, str) else list(hob)
                card["관심사"] = items[:3]
            except Exception:
                card["관심사"] = str(hob)[:120]
        if persona_row.get("persona"):
            card["소개"] = str(persona_row["persona"])[:200]
    if tone:
        hints = []
        block = int(SIDO_TO_BLOCK[sido])
        g = (tone.get("by_group") or {}).get(f"{ag}|{block}")
        if g:                                   # corpus-derived (build-tone-bank)
            hints = [str(g.get("tone", "")), str(g.get("interests", ""))]
        else:                                   # heuristic placeholder fallback
            t_age = (tone.get("by_age_group") or {}).get(ag)
            t_reg = (tone.get("by_region_block") or {}).get(REGION_BLOCKS[block])
            hints = [str(x) for x in (t_age, t_reg) if x]
        hints = [h for h in hints if h]
        if hints:
            card["말투·관심 힌트"] = " / ".join(hints)
    return card


def make_requests(pop, round_no: int, voter_ids: np.ndarray, p_prior: np.ndarray, message: str,
                  channels: list[str], kind: str = "l1", tone_bank_path: str | None = "configs/tone_bank.yaml",
                  neighbor_lean: np.ndarray | None = None, exposure_mem: np.ndarray | None = None,
                  shared_texts: list[str] | None = None, rng: np.random.Generator | None = None) -> list[L2Request]:
    rows: dict[int, dict[str, Any]] = {}
    pdf = pop.personas(voter_ids)
    if pdf.height and "persona" in pdf.columns:
        for r in pdf.iter_rows(named=True):
            rows[int(r["voter_id"])] = r
    tone = load_tone_bank(tone_bank_path) if tone_bank_path else None
    reqs = []
    shared_texts = [t for t in (shared_texts or []) if t]
    for vid, p, ch in zip(voter_ids.tolist(), p_prior.tolist(), channels):
        st = int(pop.state[vid])
        card = persona_card(pop, vid, rows.get(int(vid)), tone)
        card.update(social_context(float(neighbor_lean[vid]) if neighbor_lean is not None else None,
                                   float(exposure_mem[vid]) if exposure_mem is not None else None))
        msg = message
        if kind == "l3" and shared_texts:
            pick = shared_texts[(rng.integers(len(shared_texts)) if rng is not None else 0)]
            msg = f"{message} — 지인이 공유하며 한 말: \"{pick[:80]}\""
        reqs.append(L2Request(
            voter_id=int(vid), round_no=round_no, persona=card,
            state=st, state_label=STATE_LABELS_KO[SupportState(st)], message=msg, channel=ch,
            prior_p=float(p), kind=kind,
        ))
    return reqs


def _card_lines(req: L2Request) -> str:
    return "\n".join(f"- {k}: {v}" for k, v in req.persona.items()) + f"\n- 현재 지지 성향: {req.state_label}"


def user_prompt(req: L2Request) -> str:
    return (
        f"## 페르소나 카드\n{_card_lines(req)}\n\n"
        f"## 접한 메시지\n- 채널: {CHANNEL_LABELS_KO.get(req.channel, req.channel)}\n"
        f"- 내용 요약: {req.message}\n\n"
        "이 사람의 반응을 JSON으로만 답하세요: support_delta(-1/0/1), share(true/false), reaction(80자 이내)."
    )


def batch_prompt(reqs: list[L2Request]) -> str:
    msg = reqs[0].message
    parts = [f"## 접한 메시지 (모든 페르소나 공통)\n- 내용 요약: {msg}\n", f"## 페르소나 {len(reqs)}명 (각각 독립적으로 판단)"]
    for i, r in enumerate(reqs):
        parts.append(f"\n### idx {i} · 채널: {CHANNEL_LABELS_KO.get(r.channel, r.channel)}\n{_card_lines(r)}")
    parts.append("\n\n각 idx 에 대해 reactions 배열로만 답하세요: {idx, support_delta(-1/0/1), share, reaction(80자 이내)}. 빠짐없이 모든 idx 를 포함하세요.")
    return "\n".join(parts)
