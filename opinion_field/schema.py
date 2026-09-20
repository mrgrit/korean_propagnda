"""Shared constants: support-state ladder, demographic codings, channel list, cell ids.

All arrays are numpy struct-of-arrays; see engine/population.py.
"""
from __future__ import annotations

from enum import IntEnum

import numpy as np


class SupportState(IntEnum):
    """Support ladder (§5.1). Codes 1..5 form an ordinal axis B→A; ABSTAIN sits
    beside UNDECIDED (mobilisation moves ABSTAIN→UNDECIDED)."""
    ABSTAIN = 0
    B_STRONG = 1
    B_WEAK = 2
    UNDECIDED = 3
    A_WEAK = 4
    A_STRONG = 5


STATE_LABELS_KO = {
    SupportState.ABSTAIN: "기권 성향",
    SupportState.B_STRONG: "후보 B 강한 지지",
    SupportState.B_WEAK: "후보 B 약한 지지",
    SupportState.UNDECIDED: "부동층",
    SupportState.A_WEAK: "후보 A 약한 지지",
    SupportState.A_STRONG: "후보 A 강한 지지",
}
STATE_NAMES = ["abstain", "B_strong", "B_weak", "undecided", "A_weak", "A_strong"]
N_STATES = 6

# Slot labels (§4.4 / §10): the engine never sees display names.
SLOTS = {"A": "후보 A", "B": "후보 B", "갑": "정당 갑", "을": "정당 을", "I": "이슈 I"}

# --- demographics -----------------------------------------------------------
SIDO_CANON = ["서울", "부산", "대구", "인천", "광주", "대전", "울산", "세종",
              "경기", "강원", "충북", "충남", "전북", "전남", "경북", "경남", "제주"]
# Nemotron-Personas-Korea province spellings → canonical
SIDO_ALIASES = {"충청북": "충북", "충청남": "충남", "전라남": "전남", "전라북": "전북",
                "경상북": "경북", "경상남": "경남"}
N_SIDO = len(SIDO_CANON)

AGE_GROUP_LABELS = ["19~29세", "30대", "40대", "50대", "60대", "70세 이상"]
AGE_GROUP_EDGES = np.array([30, 40, 50, 60, 70], dtype=np.int16)  # np.searchsorted
N_AGE_GROUPS = 6

SEX_LABELS = ["남성", "여성"]
SEX_CODES = {"남자": 0, "여자": 1}

EDU_LEVELS = ["무학", "초등학교", "중학교", "고등학교", "2~3년제 전문대학", "4년제 대학교", "대학원"]
EDU_CODES = {name: i for i, name in enumerate(EDU_LEVELS)}
EDU_BUCKET_LABELS = ["중졸 이하", "고졸·전문대", "대졸 이상"]
EDU_TO_BUCKET = np.array([0, 0, 0, 1, 1, 2, 2], dtype=np.uint8)
N_EDU_BUCKETS = 3

OCC_GROUP_LABELS = ["무직·비경활", "노무·서비스·판매·기능", "사무", "전문·관리", "기타"]
N_OCC_GROUPS = 5

# Population cell = sido × age_group × sex × edu_bucket  (17*6*2*3 = 612)
N_CELLS = N_SIDO * N_AGE_GROUPS * 2 * N_EDU_BUCKETS


def cell_id(sido: np.ndarray, age_group: np.ndarray, sex: np.ndarray, edu_bucket: np.ndarray) -> np.ndarray:
    return (((sido.astype(np.int32) * N_AGE_GROUPS + age_group) * 2 + sex) * N_EDU_BUCKETS
            + edu_bucket).astype(np.uint16)


def decode_cell(cell: int) -> tuple[int, int, int, int]:
    edu_bucket = cell % N_EDU_BUCKETS
    cell //= N_EDU_BUCKETS
    sex = cell % 2
    cell //= 2
    age_group = cell % N_AGE_GROUPS
    sido = cell // N_AGE_GROUPS
    return sido, age_group, sex, edu_bucket


def cell_label(cell: int) -> str:
    s, a, x, e = decode_cell(cell)
    return f"{SIDO_CANON[s]}/{AGE_GROUP_LABELS[a]}/{SEX_LABELS[x]}/{EDU_BUCKET_LABELS[e]}"


# --- channels ---------------------------------------------------------------
CHANNELS = ["youtube", "portal", "kakao", "instagram", "tv", "wom"]
CHANNEL_LABELS_KO = {"youtube": "유튜브", "portal": "포털 뉴스", "kakao": "카카오톡",
                     "instagram": "인스타그램", "tv": "TV 뉴스", "wom": "대면 입소문"}
N_CHANNELS = len(CHANNELS)

# Message frames the rule-based manipulator can pick (slot-labelled, abstract).
FRAMES = ["economy", "security", "fairness", "welfare", "competence"]
FRAME_LABELS_KO = {"economy": "경제·민생", "security": "안보·질서", "fairness": "공정·부패",
                   "welfare": "복지·돌봄", "competence": "능력·경험"}

# Falsehood tier of a message (ethics_level maps onto this, §5.4).
CLAIM_TIERS = ["factual", "misleading", "fabricated"]
CLAIM_TIER_LABELS_KO = {"factual": "사실 기반 긍정 프레임", "misleading": "선택적·오도적 강조",
                        "fabricated": "근거 없는 허위 의혹"}
