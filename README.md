# SeeThrough Portrait (ComfyUI Fork)

[English](README_EN.md)

![Preview](https://raw.githubusercontent.com/tackcrypto1031/tk_seethrough/main/workflows/sample.png)

[@jtydhr88](https://github.com/jtydhr88)의 [ComfyUI-See-through](https://github.com/jtydhr88/ComfyUI-See-through)를
포크해, 애니메이션풍 일러스트를 투명 시맨틱 레이어(머리카락, 얼굴, 상의 등)로
분해하는 기능에 더해 상반신 인물화를 위한 **Portrait Mode**를 얹은 프로젝트입니다.
Portrait Mode는 모델이 놓친 픽셀(빠진 소매, 어깨, 팔 등)을 복구하는 Silhouette
Guard를 갖추고 있어, 최종 합성 결과가 모델이 실제로 생성한 것보다 더 많이 피사체를
잃어버리는 일이 없도록 합니다.

실행 방법은 두 가지입니다:

| | ComfyUI 노드 그래프 | 독립 실행 웹UI |
| --- | --- | --- |
| ComfyUI 필요 여부 | 필요 | 불필요 |
| 적합한 용도 | 전체 파이프라인: depth, 머리카락 좌/우 분리, PSD/Spine 내보내기 | 이미지 한 장으로 Portrait Mode를 빠르게 실행하고 verdict(판정) 확인 |
| 시작하기 | 아래 [설치 (ComfyUI)](#설치-comfyui) | [`webui/README.md`](webui/README.md) |

두 경로 모두 동일한 모델 로딩/레이어 생성 코드(`seethrough_engine/`)를
공유하므로, 같은 seed와 설정이면 어느 쪽에서 얻은 결과든 서로 직접 비교할 수
있습니다.

## Portrait Mode

Portrait Mode는 정면을 바라보는 상반신 캐릭터 인물화를 위한 복구 프로파일입니다.
모델을 재학습하거나 시맨틱 팔 레이어의 존재를 보장하지는 않습니다. 대신 피사체의
실제 실루엣을 생성된 레이어들의 합집합과 비교해, 빠진 픽셀을 `body_remainder`
안전 레이어로 복구합니다. 즉 합성 결과가 모델이 실제로 놓친 것보다 더 많이
피사체를 잃어버리지 않습니다.

- **Silhouette Guard** — 신뢰할 수 있는 피사체 마스크 밖으로 삐져나온 레이어를
  잘라내고, 설명되지 않은 원본 픽셀을 `body_remainder`로 복구합니다.
- **크롭을 고려한 필수 그룹(critical groups)** — 다리나 손 착용물(handwear)이
  없다는 이유만으로, 그 외에는 완전한 상반신 분해가 실패 처리되지 않습니다.
- **실루엣 기반 자동 보완(auto-fill)** — 추론을 최대 5회까지 재실행하며, 단순히
  첫 결과가 아니라 coverage/유사도가 가장 좋은 레이어 조합을 채택합니다.
- **두 개로 분리된 verdict** — Silhouette Guard가 전체 실루엣을 복원했는지를
  나타내는 `recovery_verdict`와, 여기에 시맨틱 완성도까지 포함한 종합
  `verdict`(PASS / SOFT_PASS / SOFT_PASS_LOW_CONFIDENCE / REWORK / FAIL)로
  나뉩니다. 실루엣이 100% 복원됐다고 해서 FACE/HAIR/BODY가 올바르게
  분리됐다는 근거가 되지는 않습니다.

모든 실행은 `body_remainder`, coverage/missing/spill 마스크 PNG, 재구성
미리보기, 전체 진단이 담긴 `*_portrait_report.json`을 생성합니다. 정확한
계약(contract)과 verdict 규칙은 [`docs/M1_IMPLEMENTATION_SPEC.md`](docs/M1_IMPLEMENTATION_SPEC.md)를,
두 진입점 모두가 검증받는 절차(A-001)는
[`docs/TEST_PROTOCOL_A001.md`](docs/TEST_PROTOCOL_A001.md)를 참고하세요.

**ComfyUI에서:** 투명 배경 PNG 입력에는 **SeeThrough Load Source**를 사용하고,
그 `subject_mask` 출력을 **Generate Layers (Custom)**에 연결한 뒤
`portrait_mode`를 켜세요(그리고 `silhouette_guard`도 켜진 상태를 유지하세요).
불투명 배경이라면 별도의 foreground 마스크를 제공해야 합니다. 전체 파라미터
표는 아래 [SeeThrough Generate Layers (Custom)](#seethrough-generate-layers-custom)를
참고하세요.

**독립 실행 웹UI에서:** 투명 배경 PNG를 업로드하기만 하면 됩니다 — Portrait
Mode는 항상 켜져 있고, 실제 alpha 채널을 그대로 신뢰할 수 있는 피사체 마스크로
읽어들입니다. 자세한 내용은 [`webui/README.md`](webui/README.md)를 참고하세요.

## 설치 (ComfyUI)

이 저장소를 ComfyUI의 `custom_nodes` 디렉터리에 클론하세요:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/tackcrypto1031/tk_seethrough.git
```

의존성을 설치하세요:

```bash
cd tk_seethrough
pip install -r requirements.txt
```

ComfyUI를 재시작하면 `SeeThrough` 카테고리 아래에 노드들이 나타납니다.

### 모델

모델은 처음 사용할 때 HuggingFace에서 자동으로 다운로드됩니다:

| 모델 | HuggingFace 저장소 | 용도 |
|-------|-------------------|---------|
| LayerDiff 3D | `layerdifforg/seethroughv0.0.2_layerdiff3d` | SDXL 기반 투명 레이어 생성 |
| Marigold Depth | `layerdifforg/seethroughv0.0.1_marigold` | 애니메이션에 맞게 파인튜닝된 단안(monocular) depth |

직접 다운로드해서 `ComfyUI/models/SeeThrough/`에 넣어도 됩니다.

## 사용법 (ComfyUI)

1. **SeeThrough Load LayerDiff Model**과 **SeeThrough Load Depth Model** 추가
2. **SeeThrough Generate Layers (Custom)** 추가 — 두 모델과 **Load Image** 노드를 연결
3. head detail 없이 더 빠르게 처리하고 싶다면 `enable_head_detail` 체크 해제
4. **SeeThrough Generate Depth** → **SeeThrough Post Process** → **SeeThrough Save PSD** 순으로 연결
5. 워크플로우를 실행하고 **Download PSD**를 눌러 내보내기

**상반신 인물화라면:** **SeeThrough Load Source**를 사용해 `subject_mask`를
연결하고, `portrait_mode`를 켜고, `silhouette_guard`도 켜진 상태로 두세요.

**Spine 내보내기라면:** 4단계의 **Save PSD** 대신 **Layer Rename** →
**Layer Filter** → **Export Spine**을 사용하세요. 출력된 JSON을 Spine
에디터에서 열면 됩니다.

## 전체 노드 목록

| 노드 | 설명 |
|------|-------------|
| **SeeThrough Load LayerDiff Model** | LayerDiff SDXL 파이프라인 로드 |
| **SeeThrough Load Depth Model** | Marigold depth 추정 파이프라인 로드 |
| **SeeThrough Generate Layers** | 원본 레이어 생성 (모든 스테이지, 모든 레이어) |
| **SeeThrough Generate Layers (Custom)** | `enable_head_detail` 토글, `auto_fill`, Portrait Mode가 추가된 레이어 생성 |
| **SeeThrough Generate Depth** | 레이어별 depth map 추정 |
| **SeeThrough Post Process** | 좌/우 분리, 머리카락 클러스터링, 색 복원 |
| **SeeThrough Save PSD** | 레이어를 PNG + 메타데이터로 내보내기; Best PSD / Depth PSD / All Runs PSD를 브라우저에서 다운로드 |
| **SeeThrough Layer Rename** | 레이어 태그를 Spine 친화적인 이름으로 변경(커스터마이즈 가능) |
| **SeeThrough Layer Filter** | 내보내기 전 특정 레이어를 포함/제외 |
| **SeeThrough Export Spine** | 레이어를 Spine 2D 스켈레톤 프로젝트(JSON + 이미지)로 내보내기 |

### SeeThrough Generate Layers (Custom)

원본 `SeeThrough Generate Layers`와 비교해 파라미터가 추가된 새 노드
`SeeThrough_GenerateLayers_Custom`입니다:

| 파라미터 | 기본값 | 설명 |
|-----------|---------|-------------|
| `enable_head_detail` | true | v3 모델 전용: head detail 추론 스테이지 켜기/끄기 |
| `auto_fill` | false | 누락된 레이어 자동 보완: 기대되는 레이어가 모두 생성될 때까지 추론을 최대 5회 재실행 (v3+head=24, v3 body=13, v2=19) |
| `min_alpha_coverage` | 0.01 | 레이어를 유효하다고 볼 최소 alpha coverage 비율. `auto_fill`이 켜졌을 때만 사용 |
| `portrait_mode` | false | 전신 파츠를 모두 요구하는 대신, 상반신 필수 그룹과 실루엣 기반 run 선택을 사용 |
| `silhouette_guard` | true | Portrait Mode에서, 삐져나온 부분을 잘라내고 설명되지 않은 원본 픽셀을 `body_remainder`로 복구 |
| `subject_mask` | 선택 | Foreground-positive 마스크(`흰색 = 피사체`). 불투명 배경 인물화에서는 강력히 권장 |

#### 동작 방식

v3 See-through 모델은 **두 단계의 추론 스테이지**로 동작합니다:

1. **Body 스테이지** — 13개의 body 레벨 레이어 생성 (front hair, back hair, head, neck, neckwear, topwear, handwear, bottomwear, legwear, footwear, tail, wings, objects)
2. **Head 스테이지** — 1단계에서 head 영역을 잘라내 업스케일한 뒤, 두 번째 추론을 돌려 11개의 세밀한 head 레이어를 생성 (headwear, face, irides, eyebrow, eyewhite, eyelash, eyewear, ears, earwear, nose, mouth)

각 스테이지는 하나의 완전한 diffusion 파이프라인 호출입니다. `enable_head_detail = false`로
설정하면 head 스테이지 전체가 **생략**되어(GPU 연산 없음) 전체 추론 시간의
**약 50%**를 절약합니다.

body 레벨 분해만 필요하고 세밀한 얼굴 특징이 필요 없다면 유용합니다.

#### Multi-Run Auto-Fill

diffusion 모델은 확률적(stochastic)이라 매 실행마다 결과가 조금씩 달라질 수
있습니다. 어떤 레이어(예: 얼굴이나 손)는 한 번의 실행에서는 없거나 거의
비어있지만, 다른 실행에서는 존재할 수 있습니다.

`auto_fill`을 켜면 기대되는 모든 레이어가 좋은 품질로 생성될 때까지 추론을
자동으로 재실행합니다:

1. **Run 1**은 원래 seed를 사용 — 이것이 기본 결과입니다
2. 각 레이어를 원본 이미지와 비교해 **유사도 점수**(0~1)를 계산합니다
3. **누락**되었거나(alpha coverage가 임계값 미만) **유사도가 낮은**(< 0.85) 레이어가 있으면 추가 실행을 트리거합니다
4. **Run 2**는 `seed + 1`, **Run 3**은 `seed + 2`, ...
5. 각 레이어마다 **원본과 유사도가 가장 높은** 버전을 채택합니다
6. **5회**까지, 또는 모든 레이어의 유사도가 충분해질 때까지 반복합니다

즉 Run 1이 얼굴 레이어를 생성했더라도, Run 2가 원본과 더 잘 맞는 얼굴을
생성했다면 Run 2 쪽이 자동으로 채택됩니다.

Portrait Mode에서는 auto-fill이 대신 실루엣 coverage와 필수 시맨틱 그룹이
모두 통과할 때까지 재실행되며, 레이어별 유사도만이 아니라 (유사도 + 실루엣
coverage + 어깨/팔 영역 - spill)을 종합한 점수가 가장 좋은 레이어 *조합*을
채택합니다.

모델은 전체 run에 걸쳐 GPU에 한 번만 로드됩니다 — 추가되는 오버헤드는 모델
로딩이 아니라 추가 diffusion 시간뿐입니다.

> **참고:** v2 모델은 단일 스테이지 추론을 사용하므로 이 토글이 영향을 주지 않습니다.

### Spine 내보내기 워크플로우

[Spine](http://esotericsoftware.com/) 애니메이션 준비를 위해 다음과 같이 연결하세요:

```
PostProcess → Layer Rename (선택) → Layer Filter (선택) → Export Spine
```

#### Layer Rename

내부 태그를 Spine 친화적인 이름으로 매핑합니다. 모든 태그에 기본 매핑이
내장되어 있습니다. `custom_mapping_json` 필드는 **선택 사항**입니다 — 비워두면
기본값을 사용합니다.

**언제 사용하나요:**
- Spine에서 깔끔하고 읽기 쉬운 이름을 쓰고 싶을 때 (예: `hairf` 대신 `front-hair`)
- 팀에서 쓰는 네이밍 컨벤션이 있어서 커스텀 이름이 필요할 때

**내장 기본 매핑 (일부):**

| 원본 태그 | → 변경된 이름 |
|-------------|-------------|
| `hairf` | `front-hair` |
| `hairb` | `back-hair` |
| `eyel` | `eye-left` |
| `eyer` | `eye-right` |
| `handwearl` | `handwear-left` |
| `handwearr` | `handwear-right` |
| `earl` | `ear-left` |
| `earr` | `ear-right` |
| `topwear` | `topwear` (변경 없음) |
| `face` | `face` (변경 없음) |

> `face`, `head`, `nose`처럼 이미 깔끔한 이름을 가진 태그는 그대로 유지됩니다.

**커스텀 매핑 예시:** 특정 이름만 덮어쓰려면 `custom_mapping_json`에 JSON
객체를 입력하세요:

```json
{
  "hairf": "bangs",
  "hairb": "back-hair",
  "topwear": "shirt",
  "bottomwear": "skirt",
  "handwearl": "left-glove",
  "handwearr": "right-glove"
}
```

JSON에 지정한 태그만 덮어쓰이고, 나머지 태그는 계속 기본 매핑을 사용합니다.
잘못된 JSON은 경고와 함께 무시됩니다.

#### Layer Filter

include 또는 exclude 모드로 원하지 않는 레이어를 제거합니다. 사용 가능한
모든 태그가 기본으로 채워져 있으므로, 필요 없는 것만 지우면 됩니다. 한 줄에
태그 하나씩 입력하세요.

> **팁:** Layer Rename을 Layer Filter 앞에 연결했다면 **변경된** 태그 이름
> (예: `front-hair`)을 사용하세요. Layer Rename을 쓰지 않는다면 원본 태그
> (예: `hairf`)를 사용하세요.

#### Export Spine

설정 가능한 출력 경로(기본값은 ComfyUI 출력 디렉터리)로 폴더를 출력합니다:

- `{prefix}.json` — Spine 스켈레톤 파일 (Spine 에디터에서 바로 열기)
- `images/` — 레이어별로 크롭된 PNG 파일
- 커스텀 디렉터리로 내보내려면 `output_path`를 설정하세요 (예: `D:/my_project/spine_assets`)

좌표는 이미지 공간(Y축 아래 방향)에서 Spine 공간(Y축 위 방향, 원점은
하단 중앙)으로 자동 변환됩니다. 그리기 순서는 PostProcess에서 정해진 depth
순서를 따릅니다.

#### PSD Import vs JSON Export — 뭘 써야 하나요?

Spine Professional(3.6+)은 PSD 파일을 직접 가져올 수 있어서, 이 JSON
내보내기가 왜 필요한지 궁금할 수 있습니다. 비교하면 다음과 같습니다:

| | Save PSD → Spine PSD Import | Export Spine (JSON + 이미지) |
|---|---|---|
| **Spine 버전** | Professional 3.6+ 전용 | **모든 버전** (Essential + Professional) |
| **레이어 배치** | 자동 | 자동 (좌표 사전 변환됨) |
| **레이어 이름** | PSD 레이어 이름에 의존 | LayerRename으로 제어 가능 |
| **레이어 필터링** | PSD에서 먼저 숨기거나 삭제해야 함 | 내장 LayerFilter 노드 사용 |
| **반복 작업** | 이미지 갱신마다 PSD 재-임포트 | 재-내보내기만 하면 됨 |
| **본(bone) 계층 구조** | 자동 생성 안 됨 | 자동 생성 안 됨 |
| **적합한 대상** | 빠르게 시작하고 싶은 Spine Professional 사용자 | Spine Essential 사용자, 또는 자동화 파이프라인에 사전 필터링/이름 변경된 레이어가 필요한 팀 |

**추천:**
- **Spine Professional 사용자** → **Save PSD**를 쓰고 Spine 내장 PSD
  import를 사용하세요. 가장 간단한 워크플로우입니다.
- **Spine Essential 사용자** → Essential은 PSD import를 지원하지 않으므로
  **Export Spine**을 사용하세요.
- **자동화 파이프라인** → 일관되고 사전 처리된 출력을 위해 LayerRename +
  LayerFilter와 함께 **Export Spine**을 사용하세요.

<details>
<summary>사용 가능한 레이어 태그 (LayerRename 적용 후, 38개)</summary>

| 분류 | 태그 |
|----------|------|
| 머리카락 | `front-hair`, `back-hair` |
| 머리 | `head`, `headwear` |
| 얼굴 | `face`, `nose`, `mouth` |
| 눈 | `eye-left`, `eye-right`, `eyewear` |
| 눈 세부 | `irides`, `irides-left`, `irides-right`, `eyebrow`, `eyebrow-left`, `eyebrow-right`, `eye-white`, `eye-white-left`, `eye-white-right`, `eyelash`, `eyelash-left`, `eyelash-right` |
| 귀 | `ears`, `ear-left`, `ear-right`, `earwear` |
| 몸통 | `neck`, `neckwear`, `topwear`, `bottomwear` |
| 팔다리 | `handwear`, `handwear-left`, `handwear-right`, `legwear`, `footwear` |
| 기타 | `tail`, `wings`, `objects` |

LayerRename을 쓰지 않는다면 원본 태그를 사용하세요: `hairf`, `hairb`,
`eyel`, `eyer`, `handwearl`, `handwearr`, `earl`, `earr` 등.

</details>

## 독립 실행 웹UI

`webui/app.py`는 ComfyUI 프로세스 없이 이미지 한 장으로 Portrait Mode를
실행합니다 -- 업로드, 실행, verdict 확인, 레이어와 리포트를 zip으로 다운로드까지.
위의 ComfyUI 노드와 모델 로딩/레이어 생성 코드(`seethrough_engine/`)를
공유하므로, 서로 다른 방향으로 갈라져 버리는 두 번째 구현이 아닙니다.

```bash
pip install -r webui/requirements.txt
python webui/app.py
# http://127.0.0.1:7860 열기
```

아직 depth 추정, 머리카락 좌/우 분리, PSD, Spine 내보내기는 하지 않습니다 --
그 부분은 당분간 ComfyUI 전용으로 남아 있습니다. 전체 설치/사용 방법과 알려진
한계, 구현 계약은
[`webui/README.md`](webui/README.md)와
[`docs/M2_IMPLEMENTATION_SPEC.md`](docs/M2_IMPLEMENTATION_SPEC.md)에 있습니다.

## 이 Fork의 변경 이력

### v1.4.0.dev1 — Portrait Mode M2: 독립 실행 웹UI

- 새로운 `webui/app.py`: ComfyUI 없이 실행되는 이미지 한 장짜리 Portrait
  Mode 웹UI (위의 [독립 실행 웹UI](#독립-실행-웹ui) 참고).
- 새로운 `seethrough_engine/` 패키지: ComfyUI에 종속되지 않은 모델 로딩/레이어
  생성 코어로, `nodes.py`와 웹UI가 함께 공유합니다 — GPU를 다루는 파이프라인
  구현이 하나만 존재하게 됐습니다. `nodes.py`의 노드 동작 자체는 변경되지
  않았습니다; 무엇이 어디로 옮겨졌고 왜인지는 `docs/M2_IMPLEMENTATION_SPEC.md`를
  참고하세요.
- M1의 스펙/설정 불일치를 수정: 하드 `FAIL` verdict를 가르는 post-recovery
  coverage 최소값이 `config/portrait_defaults.json`에서는 `0.995`였지만
  `M1_IMPLEMENTATION_SPEC.md` / `TEST_PROTOCOL_A001.md`에는 `0.999`로
  문서화(이자 의도)되어 있었습니다. 99.5%~99.9% 사이의 post-recovery
  coverage를 가진 실행이 이제는 통과가 아니라 올바르게 실패 처리됩니다.

### v1.3.0.dev1 — Portrait Mode M1

- 크롭을 고려한 시맨틱 그룹을 갖춘 상반신 Portrait Mode
- 신뢰할 수 있는 피사체 마스크 밖으로 삐져나온 레이어를 잘라내는 Silhouette Guard
- 설명되지 않은 원본 픽셀을 위한 `body_remainder` 복구 레이어
- 실루엣 기반 multi-run 선택 및 JSON 진단
- 분리된 recovery/semantic verdict

### v1.2.8 — Issue #5

- 새 노드 **SeeThrough Load Source**: ComfyUI LoadImage와 같은 드롭다운에 더해, PSD 출력에서 원본 파일명을 보존하기 위한 `source_filename`을 출력합니다.
- **SeeThrough Save PSD**가 이제 선택적으로 `original_image` + `source_filename` 입력을 받아, 생성된 PSD에 원본 입력 이미지를 보이는(visible) 베이스 레이어로 자동 포함시키고, 가능하면 출력 파일명에도 원본 파일명을 사용합니다.
- PSD 레이어 구조가 그룹화되었습니다: `Original`(보임, 맨 아래), `Parts`(숨김), `Runs`(숨김, grouped-PSD 모드에서만) — 원본 위에서 PSD를 열어, 필요한 부분만 그룹을 펼쳐 편집할 수 있습니다.

### v1.2.4 — 버그 수정

- **수정: depth 모델 다운로드 실패 — HuggingFace 저장소 ID 변경** — 기본 Marigold depth 모델 저장소 `24yearsold/seethroughv0.0.1_marigold`가 `layerdifforg/seethroughv0.0.1_marigold`로 이전되었습니다. 기본 저장소 ID를 갱신해 depth 모델 자동 다운로드가 다시 동작합니다. ([#3](https://github.com/tackcrypto1031/tk_seethrough/issues/3))

### v1.2.3 — 버그 수정

- **수정: "Failed to load ag-psd bundle from any path" 오류로 PSD 다운로드 실패** — ComfyUI의 새 프론트엔드가 ES module `import()`로 확장 기능을 로드하면서 `document.currentScript`가 `null`이 되는 문제였습니다. `import.meta.url`을 사용하도록 바꿔, 설치 폴더 이름과 무관하게 bundle 경로를 안정적으로 찾도록 했습니다. ([#1](https://github.com/tackcrypto1031/tk_seethrough/issues/1))
- **수정: 새 환경에서 노드 로드 실패** — 업스트림 Marigold 모듈이 선택적 시각화 함수를 위해 모듈 최상단에서 `matplotlib`을 임포트하면서, `matplotlib`이 없는 환경에서 `ModuleNotFoundError`가 발생했습니다. 지연 임포트(lazy import)로 바꿔, `matplotlib` 없이도 노드가 로드되도록 했습니다.

### v1.2 — Spine 내보내기, Auto-Fill & All Runs PSD

- **Spine 내보내기 노드** — Spine 2D 애니메이션 준비를 위한 `Layer Rename`, `Layer Filter`, `Export Spine` 노드 추가.
- **누락 레이어 자동 보완(Auto-Fill)** — GenerateLayers (Custom)에서 `auto_fill`을 켜면 추론을 최대 5회 재실행해, 누락된 레이어를 채우고 원본 이미지와의 비교를 통해 품질이 낮은 레이어를 업그레이드합니다.
- **All Runs PSD** — `auto_fill`이 켜져 있으면 Save PSD에 "Download All Runs PSD" 버튼이 새로 생깁니다. 모든 run을 그룹 폴더 형태로 하나의 PSD에 담아, 수동으로 비교하며 고를 수 있습니다.
- **PSD 다운로드 버튼** — Save PSD에 이제 버튼이 3개 있습니다:
  - **Download PSD** (녹색) — auto-fill로 선택된 최적 레이어
  - **Download Depth PSD** (보라색) — depth map
  - **Download All Runs PSD** (주황색) — 모든 run을 그룹별로 (`auto_fill` 필요)

### v1.1 — 업스트림과 동기화

이 fork는 [업스트림 v0.2.2](https://github.com/jtydhr88/ComfyUI-See-through)와
동기화되어 다음 개선 사항을 포함합니다:

- **VRAM 오프로딩** — 모델이 이제 CPU에 머물다가 추론 중에만 GPU로 이동하고, 끝나면 다시 오프로드됩니다. VRAM 사용량이 크게 줄어 8GB 이하 GPU에서도 실행할 수 있습니다.
- **텍스트 인코더 오프로딩** — 텍스트 인코더는 프롬프트 인코딩을 위해 GPU에 로드된 뒤, UNet+VAE를 로드하기 전에 오프로드되어 VRAM을 동시에 두고 경쟁하지 않습니다.
- **Marigold 호환성 수정** — torchvision >= 0.23에서 더 엄격해진 `InterpolationMode` 검사에 맞춰 `resize` 호출을 수정했습니다.
- **웹 관련 수정** — 서브패스 배포 지원; 대소문자를 구분하는 파일 시스템에서 발생하던 ag-psd bundle 404 오류 수정.
- **커스텀 노드 최적화** — `enable_head_detail = false`일 때 diffusion뿐 아니라 head 텍스트 인코딩도 생략되어, GPU 메모리 사용량이 더 줄었습니다.

## 프로젝트 문서

Portrait Mode의 전체 설계/구현 기록:

- [`docs/PORTRAIT_MODE_FORK_PLAN_v0.1.md`](docs/PORTRAIT_MODE_FORK_PLAN_v0.1.md) — 전체 범위와 마일스톤 (M1–M4)
- [`docs/M1_IMPLEMENTATION_SPEC.md`](docs/M1_IMPLEMENTATION_SPEC.md) — Portrait Mode 핵심 계약과 verdict 규칙
- [`docs/TEST_PROTOCOL_A001.md`](docs/TEST_PROTOCOL_A001.md) — 두 진입점 모두가 검증받는 A-001 검증 절차
- [`docs/M2_IMPLEMENTATION_SPEC.md`](docs/M2_IMPLEMENTATION_SPEC.md) — 독립 실행 웹UI 계약, `nodes.py`와 무엇을 왜 공유하는지, GPU 없이는 아직 검증하지 못한 부분

## 감사의 말

이 프로젝트는 [@jtydhr88](https://github.com/jtydhr88)의
[ComfyUI-See-through](https://github.com/jtydhr88/ComfyUI-See-through)를
포크한 것입니다. 원본 ComfyUI 통합을 만들어주신 것에 깊이 감사드립니다.

기반이 되는 연구는 [shitagaki-lab](https://github.com/shitagaki-lab)의
[See-through](https://github.com/shitagaki-lab/see-through)입니다.
논문: [arxiv:2602.03749](https://arxiv.org/abs/2602.03749) (ACM SIGGRAPH 2026 조건부 승인)

PSD 생성에는 브라우저에서 동작하는 [ag-psd](https://github.com/nicasiomg/ag-psd)를 사용합니다.

## 라이선스

ComfyUI fork 코드는 MIT 라이선스입니다. 함께 포함된 업스트림 `see-through`
연구 코드는 `see-through/LICENSE`에 명시된 대로 Apache-2.0 라이선스를
유지합니다.
