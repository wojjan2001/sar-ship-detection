# Results

All models evaluated with imgsz=640. Validation = 10% of the official train split (seed 42).

| Model | Epochs | Split | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|
| YOLOv8n (baseline) | 100 | val | 0.974 | 0.968 | 0.988 | 0.746 |
| YOLOv8n (baseline) | 100 | test | 0.972 | 0.956 | 0.990 | 0.711 |
| YOLOv8n (baseline) | 100 | test_inshore | 0.933 | 0.894 | 0.963 | 0.661 |
| YOLOv8n (baseline) | 100 | test_offshore | 0.982 | 0.989 | 0.994 | 0.733 |