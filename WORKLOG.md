# Work Log

숫자는 그때그때의 `artifacts/results/schedules.pkl` 기준이라 재측정하면 조금씩 움직입니다.
결론(부등호·배수 관계)은 유지되지만, 논문에 넣을 값은 항상 최신 런에서 다시 뽑으세요.

---

## 2026-08-10 — exit-class latency 분석 + plot 2c / 11a-b / 13a-f / 14a-f

### 발단: "naive의 latency KDE에서 왜 bimodality가 안 보이는가"

exit / non-exit 두 population이 분명히 존재하는데 plot2의 naive 곡선은 unimodal.
원인을 수치로 분해한 결과:

**mode 간격은 작고 고정, 공유 분산은 크다.**

naive에서 non-exit이 더 지는 비용은 자기 batch의 seg2 op **하나뿐**입니다
(`seg1_batch`=16, batch당 non-exit 평균 7.1개 → seg2 2.71 ms).

λ=1500(capacity×0.90) 기준:

| | mean | sd |
|---|---|---|
| exit | 13.78 ms | 4.08 |
| non-exit | 16.65 ms | 4.17 |

- mode gap **2.88 ms**, pooled sd **4.12 ms** → **Cohen's d = 0.70**
- 등분산·유사비율 2-component 혼합이 bimodal로 보이려면 gap > 2σ, 즉 **8.2 ms** 필요 → 2.9배 부족

**분산의 출처는 Poisson 그 자체가 아니라 두 class가 공유하는 대기 항입니다.**

| 항 | mean | sd | 공유 여부 |
|---|---|---|---|
| formation wait | 5.01 | 3.6 | 공유 |
| GPU queue wait | 1.86 | 2.7 | 공유 |
| stage-1 compute | 6.91 | 0.1 | 공유 (사실상 상수) |
| stage-2 compute | 2.88 | — | non-exit만 |

공유 3항만으로 **전체 분산의 88%**. 즉 두 조건부 분포는 같은 분포를 2.88 ms 평행이동한 것.

formation wait는 닫힌 형태로 검증됨 — 배치 내 임의 위치 샘플의 대기는 Erlang이라

```
sd = (1/λ)·sqrt((S-1)/2 + (S²-1)/12) = 5.36/λ
```

λ=1485에서 3.61 ms, 측정치와 일치. **S에 비례해 커지는데 mode 간격 seg2(S/2)는 커널 런치
오버헤드 때문에 거의 안 자람** → 구조적으로 분리가 안 됨.

**λ를 바꿔도 해결 안 됨.** 분산이 U자 — λ↓면 formation wait(∝1/λ)가, λ↑면 큐 대기(ρ→1)가
발산. 전 구간 스윕해도 d의 최대가 **0.75** (λ≈1450 부근).

**proposed는 다름.** non-exit이 seg2 큐 대기까지 지므로 gap 15.04 ms, d **2.33** → 실제로 갈라짐.
plot2에서 proposed만 봉우리가 두 개로 보이는 이유가 이것.

**waits를 제거하면 갈라짐** — `exit_split_lambda: 0`(saturated, stage-1 시작 기준)이면
naive도 d가 0.70 → **4.62**. "두 population은 실재한다"는 companion figure로 사용 가능.

### 파생 질문: proposed histogram에서 non-exit 앞부분이 exit과 겹치는 이유

겹침은 렌더링 착시가 아니라 **실재** — overlap coefficient **0.26**.

- 원인: proposed의 seg2 queue wait이 min 0 / **p10 = 0 ms** / p50 7.0 / max 36.8.
  매 flush의 마지막 진입 샘플들은 큐에서 안 기다림 → 추가 비용이 seg2 compute뿐
- non-exit min 12.2 ms **<** exit p50 15.4 ms, exit max 35.4 ms **>** non-exit p50 29.7 ms
- exit p90 아래에 non-exit의 15.9%가 들어옴

KDE에서 안 보였던 이유는 세 가지: ① mixture 가중으로 non-exit 왼쪽 어깨가 눌림
② 곡선은 peak 위치로 읽게 되어 겹침 면적이 무시됨 ③ bandwidth 0.3×sd ≈ 2.3 ms가
non-exit의 하드 컷오프(12.2 ms)를 완만한 어깨로 뭉갬. **histogram 쪽이 정직함.**

---

### 추가한 figure

| figure | 내용 | λ 출처 |
|---|---|---|
| `plot2c_latency_kde_per_runtime_lambda` | plot2의 수동-λ 판. plot11b의 KDE 짝 | `arrivals.lambda` |
| `plot11a_latency_cdf_common_lambda` | 기존 plot11 (iso-load CDF). **파일명 `plot11_`→`plot11a_` 변경** | `plots.cdf_common_lambda` |
| `plot11b_latency_cdf_per_runtime_lambda` | 런타임별 독립 λ CDF | `arrivals.lambda` |
| `plot13a/13b` | naive latency를 exit(보라)/non-exit(빨강)으로 분리, KDE / histogram | 각자 capacity×margin |
| `plot13c/13d` | 같은 그림의 proposed@`seg2_batch` 판 | 각자 capacity×margin |
| `plot13e/13f` | 같은 histogram인데 각 bin을 **평균 latency 구성비**로 색분할, exit\|non-exit 2패널 | 13과 동일 |
| `plot14a`–`plot14d` | 13a–13d를 **하나의 공통 λ**에 고정 + SLO 빨간 수직선 + x축 고정 | `plots.exit_split_common_lambda` |
| `plot14e/14f` | 13e/13f에 같은 처리 (공통 λ + SLO선 + x축 고정) | `plots.exit_split_common_lambda` |

설계 원칙:

- 13/14의 pooled 곡선은 plot2/3/12b의 해당 런타임 곡선과 **정확히 동일**하도록 common set과
  λ 규칙을 맞춤
- KDE 기본값 `"mixture"` — class 곡선을 표본 비율로 스케일해 **합 = pooled density**.
  "겹치는 두 성분의 합에는 골이 없다"가 그림 자체로 보임
- histogram 기본값 stacked — disjoint subset이라 쌓으면 pooled histogram이 재현됨
- 13e/13f/14e/14f에 `gpu_wait` 포함 필수. 빼면 5개 성분의 합이 latency와 안 맞아 막대가 거짓말함
- 14는 λ 자동 유도 금지 (그건 13의 역할). x축도 percentile 클립이 아니라 고정 —
  여섯 패널이 같은 스케일을 공유해야 하고, SLO 오른쪽 tail이 논증의 증거이므로

### config 키

```yaml
plots:
  # 13a-13f
  exit_split_lambda: null          # null/"auto"=capacity×margin, 숫자, 0=saturated, 런타임별 매핑 가능
  exit_split_normalize: "mixture"  # | "each"
  exit_split_show_pooled: true
  exit_split_hist_stacked: true
  composition_bins: 40             # 13e/13f/14e/14f 전용, hist_bins(80)보다 성기게
  # 14a-14f
  exit_split_common_lambda: 1650   # 공통 λ. null이면 여섯 장 전부 스킵
  exit_split_slo_ms: 50            # 숫자 or 리스트 → 빨간 수직선
  exit_split_xlim_ms: 100          # 고정 x 상한
arrivals:
  lambda: {plain: …, naive: …, proposed: …}   # plot2c / plot11b 전용
```

`arrivals.lambda`는 "unused" 상태였다가 plot2c/11b 전용 입력으로 되살림.
`naive_exit_*` → `exit_split_*`로 키 이름 통일. class 평균 점선(`exit_split_marks`)은 제거.

### 측정값 스냅샷 (08/10 15:30 pkl)

| | capacity (req/s) | ×0.90 | op 시간 |
|---|---|---|---|
| plain | 1606 | 1445 | whole 9.96 ms |
| naive | 1664 | 1500 | seg1 6.91 + seg2 2.71 ms |
| proposed@16 | 1725 | 1550 | seg1 6.91 + seg2 5.30 ms |

exit rate 55.4% (`confidence_threshold` 0.7), common set n=16379.

λ=1650(현재 `exit_split_common_lambda`), SLO 50 ms에서의 클래스별 위반율:

| | 전체 | exit | non-exit |
|---|---|---|---|
| naive | 41.89% | 39.47% | 44.89% |
| proposed@16 | 1.12% | 0.00% | 2.52% |

같은 λ에서의 평균 latency 구성 (14e/14f):

- naive exit 28.96 ms 중 **GPU wait 60%** ← ρ≈0.99의 증거
- proposed non-exit 30.82 ms 중 stage-2 queue wait 29%, stage-2 compute 17%

---

### 열린 이슈 / 다음 작업

**1. `arrivals.lambda`와 `exit_split_common_lambda`가 near-critical에 있음 (중요)**

`arrivals.lambda`가 1600/1650/1650인데 plain capacity 1606, naive 1664 → **ρ = 0.996 / 0.991**.
`exit_split_common_lambda: 1650`도 naive 기준 ρ=0.991.

이 상태의 plot2c에서 plain·naive가 bimodal하게 나오는데 이건 batching이 아니라
**큐 발산 직전의 burstiness**입니다. 14e에서 naive exit의 GPU wait이 60%를 차지하는 것도 같은 증상.
`capacity_margin_frac: 0.90`이 존재하는 이유이기도 함 (config 주석에 "capacity−step에서
plain KDE가 bimodal로 나와 0.90으로 바꿨다"고 기록됨). 1445 / 1500 / 1550 근처를 권장.

**2. intro에 쓸 "고부하에서 naive의 SLO 위반이 늘어난다"는 주장은 그대로 쓰면 위험**

λ 스윕 결과 (SLO 50 ms, 위반율 %):

| λ | plain | naive | proposed |
|---|---|---|---|
| 1400 | 0.00 | 0.00 | 0.98 |
| 1500 | 0.00 | 0.00 | 0.51 |
| 1600 | 45.98 | 0.00 | 0.35 |
| 1650 | 77.30 | 41.89 | 1.12 |
| 1700 | 93.36 | 87.44 | 27.57 |

λ ≤ 1550에서는 **naive 0.00%, proposed 0.4~1%로 오히려 proposed가 나쁨.**
SLO를 30 ms로 잡으면 proposed가 전 구간 20~29% 위반이라 λ=1650 전까지 계속 짐.
→ 단일 λ 표는 "1500에서는 반대던데요" 반박에 취약.

데이터가 실제로 말하는 것: 위반율이 각자 capacity에서 절벽처럼 치솟고 그 위치가
**1606 → 1664 → 1725**로 밀림. proposed의 기여는 "같은 부하에서 위반이 적다"가 아니라
**"위반이 시작되는 부하를 뒤로 민다"**. goodput 논문의 주장과 일치하고 반박도 어려움.

- **TODO: `plot15_slo_violation_vs_load`** — λ 스윕 × 런타임별 위반율 곡선,
  capacity 세로 점선, SLO는 리스트로 받아 다중 패널. intro figure 후보.
- SLO는 50 ms 이상이어야 이 서사가 성립 (proposed non-exit tail이 40 ms 근처까지 뻗음).
  SLO 선택 근거를 캡션에 명시할 것.

**3. tail 잘림 — 캡션에 명시 필요**

`kde_clip_percentile: 99`라 x축이 "그려지는 배열들의 p99 중 최댓값"에서 잘립니다.
컷이 가장 넓은 런타임 기준이라 **사실상 proposed만** 잘림 (plain/naive는 손실 0%).

- plot2/2b/12b: 컷 46.8 ms, proposed 1.00% 손실 (max 62.0)
- plot13c/13d: pooled 1.00% — 그중 **non-exit 2.24%** 손실
- plot3/3b/11: 클립 없음
- plot14a-f: 고정 100 ms라 손실 없음

주의할 차이 — **KDE는 전체 데이터로 적합하고 그리는 grid만 멈추므로 곡선은 정확**하지만,
**histogram은 bin edge 밖 샘플을 카운트에서 아예 버림** (진짜 손실).
또 bandwidth가 `0.3 × 전체 sd`라 **잘라낸 꼬리가 bandwidth를 부풀려** 보이는 구간이
필요 이상으로 매끄러워짐. 13c/13d에서 잘리는 2.24%가 전부 seg2 큐 대기 최악 샘플이므로,
"proposed는 non-exit에 무거운 꼬리가 생긴다"를 주장하려면 근거가 그림 밖에 있는 셈.
→ `kde_clip_percentile: 99.9`(컷 53.0 ms, 손실 0.1%)로 올리거나 캡션에 truncation 명시.

**4. plot3의 λ는 런타임마다 다름 (해석 주의)**

plot3은 각 런타임의 capacity×0.90이라 곡선 간 격차에 **batching 구조 + offered load 차이가
섞여 있음.** 동일 λ 비교는 plot11a가 담당하는데 `cdf_common_lambda: null`이라 현재 스킵 중.
켜려면 가장 작은 capacity(plain) 아래로, 1445 정도.

**5. 운영 이슈**

- `artifacts/plots`, `artifacts/results`가 추적 대상이라 양쪽 머신에서 그림을 생성하면
  매번 바이너리 충돌. **렌더는 서버에서만** 하는 게 간단함
- Windows/WSL 혼용으로 `config.yaml`이 CRLF/LF 차이만으로 modified가 뜸.
  `.gitattributes`에 `* text=auto eol=lf` 한 줄이면 해결 (미적용)
- 이 파일은 추적 대상이 아니면 `git clean -fd`에 지워집니다. 커밋해두세요.
