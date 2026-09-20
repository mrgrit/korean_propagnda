"""L2 prompt construction. Slot labels only; synthetic/aggregate research purpose stated (§5.3, §10)."""
from __future__ import annotations

import json
from typing import Any

import numpy as np
import polars as pl

from ..schema import (AGE_GROUP_LABELS, CHANNEL_LABELS_KO, EDU_LEVELS, OCC_GROUP_LABELS, SEX_LABELS,
                      SIDO_CANON, STATE_LABELS_KO, SupportState)
from .backend import L2Request

SYSTEM_PROMPT = """당신은 'K-Propaganda' 연구 시뮬레이션의 합성 유권자 반응 생성기입니다.

목적: 정치 메시지가 인구 집단에 미치는 영향과 팩트체크·정정의 방어 효과를 **집계 수준**에서 연구하기 위한 에이전트 기반 시뮬레이션입니다. 모든 유권자는 통계 분포로부터 합성된 가상의 페르소나이며, 후보 A·후보 B·정당 갑·정당 을·이슈 I는 익명 슬롯입니다. 실존 인물·정당을 가정하거나 언급하지 마십시오.

역할: 주어진 페르소나 카드(연령대·성·지역·학력·직업·관심사·현재 지지 성향)를 가진 사람이, 특정 채널에서 설명된 메시지를 접했을 때 보일 **가장 현실적인** 반응을 판단합니다.
- 정보 리터러시가 높거나 지지가 강한 사람은 쉽게 움직이지 않습니다. 허위·오도 메시지는 회의적인 사람에게 역효과(반발)를 낼 수 있습니다.
- 공유 여부는 메시지가 그 사람의 정체성·관심사와 얼마나 맞는지, 채널 특성(카톡·유튜브 등)을 고려합니다.
- 출력은 지정된 JSON 스키마만 따르고, 반응 텍스트는 80자 이내의 한국어 한 문장으로 합니다. 설득 문구를 새로 만들어내지 마십시오.
"""


def persona_card(pop, vid: int, persona_row: dict[str, Any] | None) -> dict[str, Any]:
    card = {
        "연령대": AGE_GROUP_LABELS[int(pop.age_group[vid])],
        "성": SEX_LABELS[int(pop.sex[vid])],
        "거주 시도": SIDO_CANON[int(pop.sido[vid])],
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
    return card


def make_requests(pop, round_no: int, voter_ids: np.ndarray, p_prior: np.ndarray, message: str,
                  channels: list[str], kind: str = "l1") -> list[L2Request]:
    rows: dict[int, dict[str, Any]] = {}
    pdf = pop.personas(voter_ids)
    if pdf.height and "persona" in pdf.columns:
        for r in pdf.iter_rows(named=True):
            rows[int(r["voter_id"])] = r
    reqs = []
    for vid, p, ch in zip(voter_ids.tolist(), p_prior.tolist(), channels):
        st = int(pop.state[vid])
        reqs.append(L2Request(
            voter_id=int(vid), round_no=round_no, persona=persona_card(pop, vid, rows.get(int(vid))),
            state=st, state_label=STATE_LABELS_KO[SupportState(st)], message=message, channel=ch,
            prior_p=float(p), kind=kind,
        ))
    return reqs


def user_prompt(req: L2Request) -> str:
    card = "\n".join(f"- {k}: {v}" for k, v in req.persona.items())
    return (
        f"## 페르소나 카드\n{card}\n- 현재 지지 성향: {req.state_label}\n\n"
        f"## 접한 메시지\n- 채널: {CHANNEL_LABELS_KO.get(req.channel, req.channel)}\n"
        f"- 내용 요약: {req.message}\n\n"
        "이 사람의 반응을 JSON으로만 답하세요: support_delta(-1/0/1), share(true/false), reaction(80자 이내)."
    )
