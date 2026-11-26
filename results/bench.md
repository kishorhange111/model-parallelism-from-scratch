p = 4 stages × 2 Linear(1024, 1024) + GELU, global batch 512, float32, gloo, one CPU thread per stage. One process, whole model and batch: 0.243 s per step.

| m | schedule | step (s) | measured bubble | (p − 1)/(m + p − 1) | speed-up vs 1 process | micro-batches held, stage 0 |
|---|---|---|---|---|---|---|
| 1 | gpipe | 0.352 | 0.771 | 0.750 | 0.69× | 1 |
| 1 | 1f1b | 0.366 | 0.767 | 0.750 | 0.66× | 1 |
| 2 | gpipe | 0.213 | 0.621 | 0.600 | 1.14× | 2 |
| 2 | 1f1b | 0.246 | 0.616 | 0.600 | 0.99× | 2 |
| 4 | gpipe | 0.192 | 0.466 | 0.429 | 1.27× | 4 |
| 4 | 1f1b | 0.204 | 0.459 | 0.429 | 1.19× | 4 |
| 8 | gpipe | 0.206 | 0.380 | 0.273 | 1.18× | 8 |
| 8 | 1f1b | 0.198 | 0.367 | 0.273 | 1.23× | 4 |
| 16 | gpipe | 0.252 | 0.264 | 0.158 | 0.96× | 16 |
| 16 | 1f1b | 0.251 | 0.266 | 0.158 | 0.97× | 4 |
| 32 | gpipe | 0.363 | 0.209 | 0.086 | 0.67× | 32 |
| 32 | 1f1b | 0.368 | 0.244 | 0.086 | 0.66× | 4 |
