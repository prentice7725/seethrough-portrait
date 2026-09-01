# SeeThrough Portrait

[English](README_EN.md)

한 장의 애니메이션풍 인물화를 **검증된 production-ready semantic portrait
bundle**로 만드는 ComfyUI 확장 및 독립 실행 WebUI입니다.

[@jtydhr88](https://github.com/jtydhr88)의
[ComfyUI-See-through](https://github.com/jtydhr88/ComfyUI-See-through)를 기반으로
Portrait Mode, Silhouette Guard, fidelity repair와 정적 품질 검증을 추가했습니다.

## 프로젝트 책임

```text
portrait.png
    ↓
seethrough-portrait
    ↓
Portrait Bundle v1
    ↓
portrait-autorig
    ↓
Rig Bundle
```

이 저장소의 책임은 `Image → validated, production-ready semantic portrait
bundle`에서 끝납니다. 자동 리깅, 눈 좌우 분할, anchor·mesh·weight, 표정 팩,
Spine exporter와 브라우저 runtime은 별도
[`portrait-autorig`](https://github.com/prentice7725/portrait-autorig) 저장소에
있습니다. 두 프로젝트 사이에는 Python import 의존성이 없으며 파일 계약만
공유합니다.

## Portrait Mode

- **Silhouette Guard** — 피사체 밖의 spill을 자르고 누락된 원본 픽셀을
  `body_remainder`로 복구합니다.
- **Portrait-aware auto-fill** — coverage와 필수 semantic group을 기준으로 최대
  5회 실행 중 최선의 레이어 조합을 선택합니다.
- **Fidelity repair** — `reclaim_occluded → fit_layer_tone → fit_edge_alpha →
  fit_seam_residual` 순서로 정지화면을 원본과 맞춥니다.
- **Static validation** — 전체 composite fidelity와 가늘고 긴 seam을 별도로
  측정합니다.
- **두 verdict** — 실루엣 복구 상태와 semantic 완성도를 분리해 보고합니다.

## Portrait Bundle v1

독립 WebUI의 정본 출력입니다.

```text
A001.portrait/
├─ manifest.json
├─ original.png
├─ layers/                 # canonical production-repaired assets
├─ raw_layers/             # optional forensic output
└─ diagnostics/
   ├─ portrait_report.json
   ├─ fidelity.json
   ├─ seams.json
   └─ coverage/missing/spill/composite PNGs
```

`layers/`만 downstream consumer가 사용합니다. `raw_layers/`는 모델 출력을 추적하기
위한 진단 자료이며 소비 금지입니다. 전체 불변식과 JSON Schema는
[`docs/PORTRAIT_BUNDLE_V1.md`](docs/PORTRAIT_BUNDLE_V1.md)를 참고하세요.

## 설치 (ComfyUI)

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/prentice7725/seethrough-portrait.git
cd seethrough-portrait
pip install -r requirements.txt
```

모델은 처음 사용할 때 Hugging Face에서 `models/SeeThrough/`로 자동 다운로드됩니다.

| 모델 | 저장소 | 용도 |
| --- | --- | --- |
| LayerDiff 3D | `layerdifforg/seethroughv0.0.2_layerdiff3d` | semantic layer 생성 |
| Marigold Depth | `layerdifforg/seethroughv0.0.1_marigold` | depth 추정 및 PostProcess |

## 8GB GPU 및 처리 시간

독립 WebUI는 UNet의 남은 VRAM을 측정해 통째로 올릴 수 없을 때 **leaf-level
block streaming**으로 전환합니다. 따라서 8GB급 GPU에서도 실행할 수 있습니다.
이는 단순히 `resolution`이나 `steps`를 낮추는 방식이 아닙니다. 이 모델의 UNet은
bf16에서도 약 7.58 GiB이므로, 가중치가 통째로 들어가지 않는 카드에서는 연산 전에
OOM이 날 수 있기 때문입니다. VAE는 512px 초과 해상도에서 타일링됩니다.

다음은 RTX 5060 Laptop 8GB (A-001, seed 42, 30 steps)에서 기록한 단일 생성
패스의 참고 측정치입니다. 실제 시간은 GPU, 여유 VRAM, 입력과 Portrait Mode의
auto-fill 횟수에 따라 달라집니다.

| 설정 | 처리 시간 | 피크 VRAM | 레이어 |
| --- | ---: | ---: | ---: |
| res 512, head off | 60.7초 | 2.03 GiB | 13 |
| res 512, head on | 109.6초 | 2.03 GiB | 24 |
| res 1280, head off | 333.9초 | 3.66 GiB | 13 |
| res 1280, head on | 609.9초 | 3.66 GiB | 24 |

head 단계는 같은 해상도에서 추가 프레임만 생성하므로 위 측정에서는 VRAM보다
시간에 영향을 줍니다.

### ComfyUI 노드

| 노드 | 설명 |
| --- | --- |
| SeeThrough Load Source | 이미지, alpha mask, 원본 이름 로드 |
| SeeThrough Load LayerDiff Model | LayerDiff 모델 로드 |
| SeeThrough Load Depth Model | Marigold 모델 로드 |
| SeeThrough Generate Layers | 기본 semantic layer 생성 |
| SeeThrough Generate Layers (Custom) | Portrait Mode, head detail, auto-fill |
| SeeThrough Generate Depth | 레이어별 depth map 생성 |
| SeeThrough Post Process | crop, hair/part 분할, 색 복원 |
| SeeThrough Save PSD | PNG·메타데이터·PSD 다운로드 자료 저장 |

Portrait Mode에서는 투명 PNG를 **Load Source**로 읽고 `subject_mask`를 Custom
노드에 연결하세요. 불투명 배경이라면 foreground-positive mask가 필요합니다.

## 독립 실행 WebUI

```bash
pip install -r webui/requirements.txt
python webui/app.py
# http://127.0.0.1:7860
```

이미지 한 장을 업로드해 Portrait Mode를 실행하고 verdict, canonical layers와
diagnostics가 담긴 Portrait Bundle zip을 받습니다. 자세한 내용은
[`webui/README.md`](webui/README.md)를 참고하세요.

## 테스트

```bash
python -m pytest tests -q
```

vendored `see-through/ui`는 별도 선택 의존성을 가진 연구 UI이므로 이 프로젝트의
migration gate는 루트 `tests/`를 대상으로 합니다.

## 문서

- [`docs/PORTRAIT_BUNDLE_V1.md`](docs/PORTRAIT_BUNDLE_V1.md) — 두 저장소 사이 파일 계약
- [`docs/M1_IMPLEMENTATION_SPEC.md`](docs/M1_IMPLEMENTATION_SPEC.md) — Portrait Mode 계약
- [`docs/M2_IMPLEMENTATION_SPEC.md`](docs/M2_IMPLEMENTATION_SPEC.md) — standalone WebUI
- [`docs/TEST_PROTOCOL_A001.md`](docs/TEST_PROTOCOL_A001.md) — A-001 검증 절차
- 자동 리깅 설계와 실험 기록은
  [`portrait-autorig`](https://github.com/prentice7725/portrait-autorig)에 있습니다.

## 감사의 말

원본 ComfyUI 통합을 만든 [@jtydhr88](https://github.com/jtydhr88)와 See-Through
연구 프로젝트 기여자들에게 감사드립니다.

## 라이선스

MIT
