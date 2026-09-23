# SeeThrough Portrait

[English](README_EN.md)

한 장의 애니메이션풍 인물화를 검증 가능한 **Portrait Bundle v1**로 변환하는
독립형 producer입니다. 지원 인터페이스는 standalone WebUI와 Python 엔진이며,
이 저장소에는 ComfyUI 커스텀 노드가 포함되지 않습니다.

```text
source portrait
      ↓
SeeThrough Portrait
      ↓
Portrait Bundle v1
      ↓
Portrait Composer → portrait-autorig
```

이 저장소는 이미지 분해, canonical semantic layer 복구, 정적 품질 검증과
Bundle 발행까지 담당합니다. Composer와 AutoRig는 별도 프로젝트이며,
프로젝트 간 경계는 Python import가 아닌 Portrait Bundle 파일 계약입니다.

## 주요 기능

- **Portrait Mode 및 Silhouette Guard** — 누락된 피사체 영역을 검사하고,
  설명되지 않는 잔여 픽셀은 진단용 `body_remainder`로 보존합니다.
- **생성 프로파일** — `NORMAL`은 1회, `QUALITY`는 3회, `HARVEST`는 5회
  deterministic 후보를 비교합니다. HARVEST는 SeeThrough 내부 후보 생성 모드입니다.
- **Fidelity repair와 품질 진단** — canonical layer를 원본과 맞춘 뒤
  Static Reconstruction, Seams, Local Fidelity를 각각 보고합니다.
- **선택적 좌우 파생물** — 손/팔(`handwear`), 다리(`legwear`), 신발(`footwear`)을
  포함한 지원 semantic을 좌우로 분할해 `derived/left_right/`에 저장할 수 있습니다.
  분할은 best-effort이며 alpha 기준을 넘는 연결 component가 두 개 미만이면
  해당 파생물을 생략합니다. `layers/`의 canonical PNG는 바뀌지 않습니다.
- **선택적 depth 파생물** — 검증된 depth map을 Python API로 전달할 때만 저장합니다.
  WebUI는 Marigold depth를 자동 실행하지 않습니다.

## 설치 및 실행

Python 3.10 이상이 필요합니다. CUDA를 사용할 경우 환경에 맞는 PyTorch와
torchvision을 먼저 설치하세요.

```bash
python -m pip install -r webui/requirements.txt
python webui/app.py
```

브라우저에서 [http://127.0.0.1:7860](http://127.0.0.1:7860)을 열고 이미지를
업로드하세요. 모델은 첫 실행 시 Hugging Face에서 `models/SeeThrough/`로
다운로드됩니다. 자세한 설치, 옵션, 문제 해결은
[`webui/README.md`](webui/README.md)를 참고하세요.

지원 모델:

| 모델 | Hugging Face 저장소 | 용도 |
| --- | --- | --- |
| LayerDiff 3D | `layerdifforg/seethroughv0.0.2_layerdiff3d` | semantic layer 생성 |
| Marigold Depth | `layerdifforg/seethroughv0.0.1_marigold` | 선택적 depth 추정 |

8GB급 GPU는 UNet block streaming 경로를 사용할 수 있습니다. VAE tiling은
확산 단계의 해상도와 여유 VRAM에 따라 선택하며, untiled CUDA OOM이면 tiled로
한 번 재시도합니다. 측정 방법과 결과는
[`docs/VAE_RUNTIME_POLICY.md`](docs/VAE_RUNTIME_POLICY.md)에 있습니다.

## Portrait Bundle v1

Bundle의 `layers/`가 downstream에서 사용할 canonical production-repaired
레이어입니다. `raw_layers/`는 forensic 자료이므로 authoring이나 downstream
레이어 입력으로 사용하지 마세요. `body_remainder`도
`diagnostics/body_remainder.png`에만 기록되는 재구성 진단물입니다.

```text
A001.portrait/
├─ manifest.json
├─ original.png
├─ layers/                 # canonical semantic layers
├─ raw_layers/             # optional forensic output; consumer input 금지
├─ derived/                # optional 좌우 / depth 파생물
└─ diagnostics/            # report, fidelity, seams, masks, composites
```

좌우 파생물은 manifest에서 `left`와 `right`로 표기하며, 이는 인물의 해부학적
좌우가 아니라 **이미지 캔버스 기준 geometric left/right**입니다.

```json
{
  "derived": {
    "source_stage": "production_repaired",
    "left_right": {
      "status": "computed",
      "paths": {
        "handwear": {
          "left": "derived/left_right/handwear_left.png",
          "right": "derived/left_right/handwear_right.png"
        }
      }
    },
    "depth": { "status": "not_computed", "paths": {} }
  }
}
```

파생물은 명시적으로 요청한 경우에만 생성되며 canonical `layers/`를 교체하지
않습니다. 좌우 분할 API는 `handwear`, `legwear`, `footwear`와 눈·귀 semantic을
지원합니다. 전체 파일 불변식, manifest 검증 규칙과 JSON Schema는
[`docs/PORTRAIT_BUNDLE_V1.md`](docs/PORTRAIT_BUNDLE_V1.md)를 참고하세요.

Python 호출에서는 `save_portrait_bundle(..., stratify_left_right=True)`로
좌우 분할을 요청하고, 선택적으로 `depth_maps={tag: float32_HxW}`를 전달할 수
있습니다. 파생물은 downstream에서 별도로 검토할 수 있는 producer output이며,
canonical layer를 자동으로 대체하지 않습니다.

## 다음 단계와 개발 문서

- [Portrait Bundle v1 계약](docs/PORTRAIT_BUNDLE_V1.md) — 소비자 간 파일 계약과 schema
- [WebUI 사용법](webui/README.md) — 설치, 생성 옵션, 결과 확인, 문제 해결
- [Portrait Mode 사양](docs/M1_IMPLEMENTATION_SPEC.md)
- [회귀 검증 절차](docs/TEST_PROTOCOL_A001.md)
- [VAE runtime policy](docs/VAE_RUNTIME_POLICY.md)
- [P0 closeout](docs/P0_CLOSEOUT_V0.2.md)

자동 리깅은 별도 [`portrait-autorig`](https://github.com/prentice7725/portrait-autorig)
저장소에서 관리합니다.

## 테스트

```bash
python -m pytest tests -q
```

vendored `see-through/ui`는 별도의 선택 의존성을 가진 연구 UI입니다. 이
저장소의 테스트는 루트 `tests/`에 있습니다.

## 라이선스

MIT
