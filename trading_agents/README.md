# TradingAgents (Claude 기반 구현)

[TradingAgents: Multi-Agents LLM Financial Trading Framework](https://arxiv.org/abs/2412.20138) (Xiao et al., 2024) 논문의 구조를 따라 구현한 멀티 에이전트 AI 트레이딩 프레임워크입니다. LLM은 Anthropic Claude API(`claude-opus-5-5`)를 사용합니다.

> ⚠️ 연구·교육용 코드입니다. 투자 조언이 아니며, 실거래에 사용하기 전에 충분한 검증이 필요합니다.

## 구조 (논문의 조직 구성)

```
           ┌──────────── I. 애널리스트 팀 (병렬) ────────────┐
           │ 펀더멘털 │ 센티먼트 │ 뉴스 │ 기술적 분석           │
           └───────────────────────┬──────────────────────────┘
                                   ▼  구조화된 리포트
           II. 리서처 팀: 강세(Bull) ⇄ 약세(Bear) 토론 n라운드
                → 리서치 매니저(진행자)가 우세한 관점 + 투자 계획 결정
                                   ▼
           III. 트레이더: BUY / SELL / HOLD + 포지션 크기 + 확신도
                                   ▼
           IV. 리스크 관리 팀: 공격적 ⇄ 중립 ⇄ 보수적 토론 n라운드
                                   ▼
           V. 펀드 매니저: 최종 승인/조정 → 최종 결정
                                   ▼
           (백테스트 시) 실현 수익률로 반성(Reflection) → 메모리에 교훈 저장
```

논문의 핵심 설계를 다음과 같이 반영했습니다.

| 논문 개념 | 구현 |
|---|---|
| 구조화된 커뮤니케이션 프로토콜 (자연어 대화 대신 문서/상태 공유) | `AgentState`에 각 단계의 리포트·토론·결정을 누적 (`state.py`) |
| Quick-thinking / Deep-thinking 모델 분리 | 같은 모델에 `effort` 수준으로 분리: 분석·토론은 `low`, 의사결정(리서치 매니저·트레이더·펀드 매니저)은 `high` |
| 의사결정의 정형 출력 | JSON 스키마 구조화 출력(Structured Outputs)으로 BUY/SELL/HOLD, 크기, 확신도를 보장 |
| 토론 라운드 수 조절 | `--rounds`, `--risk-rounds` |
| 평가 지표 CR, ARR, SR, MDD 및 규칙 기반 베이스라인 | `backtest.py` (Buy&Hold, SMA 교차, MACD 비교) |
| 과거 결정에서의 학습 | `ReflectionMemory`: 실현 수익률을 바탕으로 교훈을 생성, 이후 결정에 제공 |

## 설치

```bash
cd trading_agents
pip install -r requirements.txt     # 또는 pip install -e ".[dev]"
export ANTHROPIC_API_KEY=sk-ant-...
```

## 사용법

```bash
# 단일 의사결정 (Yahoo Finance 가격 + 현재 뉴스/펀더멘털)
python -m tradingagents decide --ticker NVDA --provider yfinance --live-extras --language ko

# 워크포워드 백테스트 (5 거래일마다 결정, 결정 1회당 약 12회 API 호출)
python -m tradingagents backtest --ticker AAPL --provider yfinance \
    --start 2024-01-02 --end 2024-03-29 --interval 5

# API 키 없이 오프라인 동작 확인 (합성 데이터 + 규칙 기반 Mock LLM)
python -m tradingagents backtest --ticker DEMO --llm mock --start 2024-01-02 --end 2024-12-31 --interval 5
```

결과는 `results/`에 저장됩니다: 결정별 Markdown 리포트(애널리스트 리포트, 토론 기록, 최종 결정)와 JSON, 백테스트 지표 CSV/JSON.

### 파이썬 API

```python
from tradingagents import TradingAgentsGraph, TradingConfig, ClaudeLLM, YFinanceProvider, Backtester

cfg = TradingConfig(max_debate_rounds=2, language="ko")
graph = TradingAgentsGraph(ClaudeLLM(), YFinanceProvider(), cfg)
state = graph.propagate("NVDA", "2024-06-03")
print(state.final_decision)

result = Backtester(graph, graph.data, decision_interval=5).run("NVDA", "2024-01-02", "2024-03-29")
print(result.summary())
```

## 데이터 제공자

| Provider | 설명 |
|---|---|
| `YFinanceProvider` | 가격은 시점 기준(point-in-time). 뉴스·펀더멘털은 Yahoo의 **현재** 스냅샷이므로 과거 백테스트에서는 미래 정보 누수가 생깁니다. 그래서 기본값으로 꺼져 있고 `--live-extras`로 실시간 사용 시에만 켭니다. |
| `CSVProvider` | `{dir}/{TICKER}.csv`(가격), 선택적으로 `{TICKER}_news.csv`(date,title,summary,source), `{TICKER}_fundamentals.csv`(date + 지표 열). 날짜가 있는 뉴스/펀더멘털 데이터를 갖고 있다면 이 방식이 백테스트에 가장 적합합니다. |
| `SyntheticProvider` | 결정적 레짐 전환 랜덤워크 + 과거 수익률만으로 만든 합성 뉴스. 오프라인 개발·테스트용. |

모든 제공자는 결정일 이후 데이터를 반환하지 않도록 강제합니다(look-ahead 방지). 백테스트에서는 t일 종가에 정한 포지션이 t→t+1 수익률을 얻고, 거래비용(`--cost-bps`, 기본 5bp)이 회전율에 부과됩니다.

## Claude API 설정

- 모델: `claude-opus-5-5` (`--model`로 변경 가능)
- 적응형 사고(adaptive thinking), 깊이는 `--quick-effort` / `--deep-effort`로 조절
- **서버 측 거절 폴백(refusal fallback)이 기본으로 켜져 있습니다** (`fallbacks: "default"`). 모델이 요청을 거절하면 API가 같은 호출 안에서 다른 모델로 재시도합니다. 끄려면 `--no-fallbacks`. 최종적으로 거절되면 해당 결정은 HOLD로 처리되고 오류가 기록됩니다.
- 에이전트별 시스템 프롬프트에 프롬프트 캐싱 적용
- 비용 참고: 기본 설정에서 결정 1회당 약 12회 호출(애널리스트 4 + 토론 2 + 매니저 1 + 트레이더 1 + 리스크 3 + 펀드 매니저 1) + 백테스트 시 반성 1회. 긴 백테스트는 `--interval`을 늘리거나 `--analysts technical,news`처럼 애널리스트 수를 줄이세요.

## 논문과의 차이 / 한계

- 논문의 센티먼트 애널리스트는 소셜미디어(Reddit, X 등) 데이터를 사용하지만, 여기서는 뉴스 헤드라인 톤과 가격·거래량 행태를 대리 지표로 사용합니다. `CSVProvider`의 뉴스 파일에 소셜 데이터를 넣어 대체할 수 있습니다.
- 논문은 LangGraph 기반이지만, 여기서는 의존성을 줄이기 위해 순수 파이썬으로 동일한 파이프라인을 구성했습니다.
- `MockLLM`은 파이프라인 테스트용 규칙 기반 대체물이며, 그 백테스트 성과는 LLM 에이전트의 성과를 의미하지 않습니다.

## 테스트

```bash
python -m pytest -q
```

테스트는 시점 기준 데이터 접근, 지표, 백테스트의 미래 정보 미사용, 전체 파이프라인(Mock), 반성 메모리, Claude 요청 형식(가짜 HTTP 전송)과 거절 처리 등을 확인합니다. 실제 API 호출은 하지 않습니다.
