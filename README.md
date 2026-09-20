# K-Propaganda — P0 엔진

합성 한국 인구(Nemotron-Personas-Korea 100만) 위에서 **조작자 → 인구 반응 → 방어자 → 지지율 갱신** 라운드를 돌리는 적대적 여론 시뮬레이션 엔진의 P0 뼈대. 엔진·로그·산출물은 익명 슬롯(후보 A/B, 정당 갑/을, 이슈 I)만 사용한다.

## 구조

```
opinion_field/
  schema.py            지지 사다리·인구 코딩·채널·셀(612) 정의
  config.py            CampaignConfig 등 dataclass + YAML 로딩
  data/                §8-1 파이프라인 (LLM 없음)
    nemotron.py        HF 샤드 로드·정규화 (페르소나 문장의 합성 이름 제거)
    priors.py          priors.yaml → 리터러시·설득가능성·채널 도달/신뢰·초기 지지상태
    network.py         합성 관계망(CSR) + 중심성
    synthetic.py       테스트용 소규모 합성 인구
    build.py           voters.parquet / personas.parquet / network.npz / manifest.json
  engine/
    population.py      struct-of-arrays 인구 + 동적 상태(체크포인트 단위)
    channels.py        채널 테이블 + 노출 함수(조작·정정 공용)
    l1.py              L1 통계 반응: 전이확률·표본추출·사다리 적용 (numpy 벡터)
    manipulator.py     규칙 기반 조작자 (표적 셀·채널 배분·프레임/주장 유형·학습)
    defender.py        규칙 기반 방어자 (탐지→정정 노출→되돌림·예방접종)
    promote.py         L1→L2 승격 (경계선 확률·중심성·셀 대표 보장, top-K)
    l3.py              관계망 전파 + 2차 승격 후보
    loop.py            라운드 루프·로그·체크포인트/재개
  agents/
    backend.py         AgentBackend 인터페이스, 응답 JSON 스키마
    mock.py            LLM 없는 결정적 백엔드 (테스트)
    cc_session.py      Claude Code 헤드리스(`claude -p`) 백엔드 — 이 머신에 로그인된 계정 1개
    api.py             Anthropic API 백엔드 (Message Batches / realtime) — 확장용, 기본 미사용
    prompts.py         페르소나 카드·시스템 프롬프트 (슬롯 라벨만, 합성·집계 목적 명시)
  observe/
    logging.py         rounds.jsonl / segments.parquet / l2_responses.jsonl / manifest.json
    report.py          궤적·목표 달성·세그먼트 기여·리터러시 상관·방어 on/off 대조
    display.py         display_map.yaml 표시명 치환 (렌더링 시점에만)
configs/
  campaign.yaml        절대권자 입력: 목표·예산·방어 강도·윤리 수위·라운드·백엔드
  priors.yaml          사전분포 (출처 태그 필수 — 현재 SYNTHETIC/APPROX PLACEHOLDER)
  channels.yaml        채널 6개 파라미터 + 프레임 친화도
  display_map.yaml     슬롯↔표시명 대조표 (엔진 미접근, 배포 금지)
```

## 웹 UI (권장 — 터미널 없이 브라우저에서 전부)

```bash
.venv/bin/opinion-field web --host 127.0.0.1 --port 8765     # 또는 systemd 사용자 서비스 kpropaganda.service
```

- 로컬: http://127.0.0.1:8765 · 외부: **https://kpropaganda.dawnofagi.cloud** (Cloudflare Named Tunnel `kpropaganda`, 사용자 서비스 `kpropaganda-tunnel.service`)
- 로그인 비밀번호: `configs/web_auth.yaml` (첫 기동 시 자동 생성, 0600). 바꾸려면 값을 수정하고 `systemctl --user restart kpropaganda`.
- 화면 구성(§7 3분할): 좌 = 데이터 빌드·캠페인 설정(절대권자 입력)·백엔드·고급 YAML 편집·작업 콘솔 / 중앙 = 확산 뷰(시도×세대 히트맵, 라운드 슬라이더·재생, 채널 노출 비중, 조작자 계획, 세그먼트 기여) / 우 = KPI 타일·지지율 궤적·방어 on/off 대조·상태 분포·리포트·조작 vs 방어 로그.
- 실행 / 방어 on·off 대조 / 중지(라운드 경계에서 체크포인트) / 재개 / 삭제 / 리포트(표시명 적용 토글) / rounds·segments·l2_responses 다운로드 모두 브라우저에서.
- 작업은 한 번에 하나씩 순차 실행(큐). 실행 중인 run 을 선택해 두면 라운드마다 자동 갱신된다.

서비스 관리:
```bash
systemctl --user status kpropaganda kpropaganda-tunnel
journalctl --user -u kpropaganda -f
```

## 빠른 시작 (CLI)

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest -q                      # 합성 2만 명으로 전 구간 테스트

# 1) 데이터 빌드 (Nemotron 9샤드가 data/raw/nemotron/ 에 있어야 함; 없으면 --synthetic N)
.venv/bin/opinion-field build-data --out data/processed            # 100만
.venv/bin/opinion-field build-data --limit 100000 --out data/p100k # 일부만

# 2) 라운드 루프 (LLM 없이 mock 백엔드)
.venv/bin/opinion-field run --campaign configs/campaign.yaml --out runs/mock20 --backend mock

# 3) 방어자 on/off 대조 (같은 시드 두 번 실행 + 리포트)
.venv/bin/opinion-field compare-defender --campaign configs/campaign.yaml --out runs/cmp --backend mock

# 4) 리포트 (표시명은 여기서만 주입)
.venv/bin/opinion-field report --run runs/cmp/defender_on --compare runs/cmp/defender_off \
    --display-map configs/display_map.yaml --out runs/cmp/report_named.md

# 5) 체크포인트에서 재개
.venv/bin/opinion-field run --campaign configs/campaign.yaml --out runs/mock20 --resume
```

`--set key=value` 로 어떤 설정이든 덮어쓸 수 있다: `--set defender.strength=0.8 --set promotion.k=2000 --set ethics_level=0.9`.

## 백엔드 (§2.1)

| kind | 무엇 | 비용 | 용도 |
|---|---|---|---|
| `mock` | L1 사전확률에서 결정적 표본 | 0 | 테스트·뼈대 검증 |
| `cc` | `claude -p --output-format json --json-schema … --tools "" --model haiku` | 구독 세션 사용량 | 기본 실행 (API 과금 없음) |
| `api` | Anthropic Message Batches(50% 가격) 또는 realtime, Haiku 기본 | API 과금 | 규모 확장 시 |

`cc` 백엔드는 **이 환경에 로그인된 계정 하나**만 쓴다. 한도 도달 시 지수 백오프로 대기하고, `max_wait_s`를 넘기면 해당 요청을 `source=fallback`(변화 없음)으로 기록해 라운드가 체크포인트되도록 한다. 이후 `--resume`으로 이어서 돌린다.

> **약관 확인 결과(code.claude.com/docs/en/legal-and-compliance)**: 구독자 본인이 자기 Claude Code를 헤드리스로 자동화하는 것은 허용(ordinary use). **여러 구독 계정의 사용량을 합치거나 한도를 우회하도록 라우팅하는 것은 불허.** 그래서 이 저장소에는 다계정 디스패처가 없고, 추가해서도 안 된다. 처리량이 더 필요하면 `api` 백엔드(배치당 최대 10만 요청)를 쓴다.

`cc` 설정 예 (`configs/campaign.yaml`):
```yaml
backend: {kind: cc, model: haiku, concurrency: 4, timeout_s: 120, max_wait_s: 900}
```

## 반응 모델 요약 (§5)

- 노출: 셀별 구매 노출 `I_ck` → 1인당 강도 `λ = I/N_c`, `P(노출) = 1 − exp(−Σ_k λ_k·reach_k)`, 채널 신뢰는 dose 가중 평균.
- L1 전이확률(A방향) = `base × 설득가능성 × 프레임적합 × 채널신뢰 × (1 − 리터러시·저항 − 예방접종) × 허위 punch × 반복 gain(역U) × 사회적 증거(이웃 lean) × 사다리 계수(강한지지 0.12 / 약한 0.5~0.6 / 부동 1.0)`. 과다노출 시 리터러시 비례 반발(B방향).
- L2: 승격된 유권자에게 페르소나 카드 + 메시지 요약 + 현재 상태 → `{support_delta, share, reaction}` JSON. L1 표본 결과를 L2 응답이 덮어쓴다.
- L3: L2 공유자(+일부 L1 이동자)의 이웃에 전파(채널 speed·설득가능성), 순압력 방향으로 전이. 경계선·고중심 이웃은 2차 승격(`k_l3`).
- 방어자: `p_detect = σ(bias + 2.2·허위 + 1.5·강도 + 3.0·노출규모)`. 탐지 시 조작 노출의 `강도×budget_ratio` 만큼 정정 노출(포털·TV 가중) → 최근 A방향 이동자 되돌림 + 예방접종(다음 라운드 저항).

## 산출물 (§6)

- `rounds.jsonl`: 라운드별 표적·메시지 요약·채널 배분·노출·L1/L2/L3 집계·방어 개입·지지율 분포·비용.
- `segments.parquet`: 셀(612)×라운드 집계 (노출·이동·되돌림·승격·상태 분포·리터러시). 개인 단위 조작 지침은 산출하지 않는다.
- `l2_responses.jsonl`: L2 응답(합성 페르소나 반응, 슬롯 라벨).
- `checkpoints/round_XXXX.npz|json`: 동적 상태 + RNG 상태 + 조작자 학습 상태.
- `report.md`: 궤적, 목표 달성 시점·비용, 세그먼트 기여, 리터러시-이동 상관, 방어 on/off 차이.

## 데이터 출처와 자리표시자

`configs/priors.yaml` 의 각 블록은 `source` 태그를 갖고 `data/processed/manifest.json` 에 복사된다. **현재 지역 표심(`regional`)은 SYNTHETIC_PLACEHOLDER**(실제 선거 결과가 아님)이고 매체 이용·신뢰, 설득가능성, 리터러시는 APPROX_PLACEHOLDER 다. 선관위 개표결과·유권자의식조사, 언론재단 미디어서베이, KOBACO 매체이용률을 확보하면 이 파일만 교체하고 `build-data` 를 다시 돌리면 된다. BIGKinds 등 이용조건이 있는 데이터는 사용 전 확인.

## 제약 (§10)

- 엔진·로그·산출·L2 프롬프트는 슬롯 라벨만. 표시명은 `display_map.yaml` → 리포트 렌더링에서만.
- 개인별 조작 지침 없음 — 셀 단위 집계만.
- 방어자 필수 포함; 리포트는 방어 효과 대조로 귀결.
- 시드 고정·전 과정 로그로 재현 가능.

## 검증 결과 (2026-09-20, 이 머신 24코어/31GB)

| 항목 | 결과 |
|---|---|
| 데이터 빌드 (Nemotron 100만 → parquet + 관계망 1,600만 엣지) | 7.7 s |
| L1+L3+규칙 조작자/방어자, 100만 명, mock 백엔드 | 라운드당 0.18~0.20 s, 20라운드 8.8 s |
| 테스트 (합성 2만 명, 13개) | 통과 (결정성·체크포인트 재개·방어 효과 방향·슬롯 라벨 위생 포함) |
| CC 세션 백엔드 실호출 (Haiku, 구조화 JSON) | 4/4 성공, 호출당 약 5 s (`MAX_THINKING_TOKENS=0`), 10 s (thinking on) |

기본 설정(윤리 0.5, 방어 0.5, 예산 300만/20라운드) 100만 명 mock 실행: 후보 A 결정층 지지율 0.4977 → 0.5148(방어 on) / 0.5162(방어 off). 방어 강도 0.9·정정 예산 비율 1.0·윤리 0.9 에서는 0.5103 / 0.5172 로 방어자가 상승분의 약 35%를 막는다. 셀 단위 리터러시–순이동률 상관은 r ≈ −0.37 ~ −0.46. 방어자의 효과 크기는 `defender.strength`, `defender.budget_ratio`, `defender.revert_base` 로 조절한다. 결과는 `runs/cmp1m*/report.md` 참고.

**CC 백엔드 처리량 추정**: 호출당 ~5 s, `concurrency: 6` 이면 라운드당 600건(K=500 + k_l3=100)에 약 9분, 20라운드 ≈ 3시간. 이 시간은 구독 계정 하나의 사용 한도 안에서 소화되어야 하며 한도에 걸리면 백오프 → 폴백 → 체크포인트 → `--resume` 으로 이어 간다. 스모크 테스트용 합성 인구는 `data/processed_synth20k` (`--data data/processed_synth20k --set promotion.k=4`).

## P0 범위 밖 (P1+)

조작자·방어자 LLM 전략/학습, 11채널, 목표 다종화, 조건 스윕, UI 연동.
