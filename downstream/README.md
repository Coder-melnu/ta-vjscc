# Downstream Task Evaluation

```
ta-vjscc/
└── downstream/
    ├── action_recognition/    
    │   ├── models/
    │   │   ├── c3d_recognizer.py       # C3D: Sports1M→UCF101, no fine-tuning
    │   │   └── slowfast_recognizer.py  # SlowFast R50: head fine-tuned on UCF101
    │   ├── pipeline/
    │   │   ├── video_utils.py          # Frame loading, GoP↔numpy
    │   │   └── encode_decode.py        # VideoJSCC checkpoint wrapper
    │   ├── evaluate/
    │   │   ├── metrics.py              # PSNR, MS-SSIM
    │   │   └── action_eval.py          # Eval loop + PSNR≠accuracy gap finder
    │   ├── results/
    │   │   └── export_table.py         # CSV + formatted table output
    │   ├── weights/                    # Downloaded/trained model weights
    │   ├── logs/                       # Training logs
    │   ├── finetune_slowfast.py        # SlowFast head fine-tuning (run overnight)
    │   └── run_pipeline.py             # Master runner
    ├── object_detection/       ← future
    └── segmentation/           ← future
```

## Setup

```bash
# C3D (MMAction2)
pip install openmim
mim install mmengine mmcv mmaction2

# SlowFast
pip install pytorchvideo av

# General
pip install opencv-python tqdm
```

## Step 1 — Fine-tune SlowFast head  (3060 server)

```bash
nohup python downstream/action_recognition/finetune_slowfast.py \
    --device cuda:1 \
    --epochs 10 \
    --batch_size 4 \
    > downstream/action_recognition/logs/finetune_slowfast.log 2>&1 &

tail -f downstream/action_recognition/logs/finetune_slowfast.log
```

## Step 2 — Smoke test (C3D, 20 videos)

```bash
python downstream/action_recognition/run_pipeline.py \
    --snr_list 13 7 \
    --max_videos 20 \
    --recognizers c3d \
    --device cuda:0
```

## Step 3 — Full evaluation

```bash
python downstream/action_recognition/run_pipeline.py \
    --snr_list 19 13 7 4 1 \
    --max_videos 200 \
    --recognizers c3d slowfast \
    --slowfast_head downstream/action_recognition/weights/slowfast_ucf101_head.pth \
    --device cuda:0
```

## Expected results directory

```
downstream/action_recognition/results/
├── raw_results_AWGN_YYYYMMDD_HHMMSS.csv
├── summary_table_AWGN_YYYYMMDD_HHMMSS.csv
└── gap_examples_AWGN_YYYYMMDD_HHMMSS.json
```

## Design notes

- **C3D**: CNN (3D conv), direct UCF101 pretrained weights via MMAction2, ~52.8% Top-1.
  Architecturally consistent with DeepJSCC. Used as lightweight CNN baseline.

- **SlowFast R50**: CNN (3D ResNet backbone, fully frozen).
  Head-only fine-tuned on UCF101 (~60–75% Top-1). Primary evaluator in paper.

- **Structure**: `downstream/` namespace allows adding `object_detection/` and
  `segmentation/` later without renaming anything. Each downstream task shares
  the same VideoJSCC pipeline but has its own recognizer and eval loop.

- **Later**: when testing  proposed TA-VJSCC method, only `encode_decode.py`
  changes — everything else (recognizers, metrics, export) stays identical.