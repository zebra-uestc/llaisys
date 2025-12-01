# NVIDIA 算子性能测评
## Add
```bash
Testing Ops.add on nvidia
   shape (1, 1536) dtype <f32>
        Torch time: 0.00601 ms
        LLAISYS time: 0.00451 ms
   shape (1, 1536) dtype <f16>
        Torch time: 0.00660 ms
        LLAISYS time: 0.00407 ms
   shape (1, 1536) dtype <bf16>
        Torch time: 0.00613 ms
        LLAISYS time: 0.00407 ms
   shape (128, 1536) dtype <f32>
        Torch time: 0.00610 ms
        LLAISYS time: 0.00447 ms
   shape (128, 1536) dtype <f16>
        Torch time: 0.00635 ms
        LLAISYS time: 0.00435 ms
   shape (128, 1536) dtype <bf16>
        Torch time: 0.00632 ms
        LLAISYS time: 0.00407 ms
   shape (1, 4096) dtype <f32>
        Torch time: 0.00638 ms
        LLAISYS time: 0.00405 ms
   shape (1, 4096) dtype <f16>
        Torch time: 0.00594 ms
        LLAISYS time: 0.00441 ms
   shape (1, 4096) dtype <bf16>
        Torch time: 0.00633 ms
        LLAISYS time: 0.00437 ms
   shape (512, 4096) dtype <f32>
        Torch time: 0.00735 ms
        LLAISYS time: 0.00684 ms
   shape (512, 4096) dtype <f16>
        Torch time: 0.00636 ms
        LLAISYS time: 0.00448 ms
   shape (512, 4096) dtype <bf16>
        Torch time: 0.00604 ms
        LLAISYS time: 0.00477 ms
   shape (1, 5120) dtype <f32>
        Torch time: 0.00605 ms
        LLAISYS time: 0.00443 ms
   shape (1, 5120) dtype <f16>
        Torch time: 0.00654 ms
        LLAISYS time: 0.00446 ms
   shape (1, 5120) dtype <bf16>
        Torch time: 0.00617 ms
        LLAISYS time: 0.00403 ms
   shape (1024, 5120) dtype <f32>
        Torch time: 0.01521 ms
        LLAISYS time: 0.01394 ms
   shape (1024, 5120) dtype <f16>
        Torch time: 0.00787 ms
        LLAISYS time: 0.00798 ms
   shape (1024, 5120) dtype <bf16>
        Torch time: 0.00784 ms
        LLAISYS time: 0.00803 ms
Test passed!
```

## Argmax
```bash
Testing Ops.argmax on nvidia
   shape (151936,) dtype <f32>
        Torch time: 0.01250 ms
        LLAISYS time: 0.01077 ms
   shape (151936,) dtype <f16>
        Torch time: 0.01247 ms
        LLAISYS time: 0.01061 ms
   shape (151936,) dtype <bf16>
        Torch time: 0.01216 ms
        LLAISYS time: 0.01066 ms
   shape (152064,) dtype <f32>
        Torch time: 0.01263 ms
        LLAISYS time: 0.01072 ms
   shape (152064,) dtype <f16>
        Torch time: 0.01235 ms
        LLAISYS time: 0.01047 ms
   shape (152064,) dtype <bf16>
        Torch time: 0.01221 ms
        LLAISYS time: 0.01080 ms
Test passed!
```

## Embedding
```bash
Testing Ops.embedding on nvidia
   idx_shape (1,) embd_shape (151936, 1536) dtype <f32>
        Torch time: 0.02439 ms
        LLAISYS time: 0.00401 ms
   idx_shape (1,) embd_shape (151936, 1536) dtype <f16>
        Torch time: 0.02388 ms
        LLAISYS time: 0.00395 ms
   idx_shape (1,) embd_shape (151936, 1536) dtype <bf16>
        Torch time: 0.02367 ms
        LLAISYS time: 0.00400 ms
   idx_shape (128,) embd_shape (151936, 1536) dtype <f32>
        Torch time: 0.02372 ms
        LLAISYS time: 0.00390 ms
   idx_shape (128,) embd_shape (151936, 1536) dtype <f16>
        Torch time: 0.02389 ms
        LLAISYS time: 0.00396 ms
   idx_shape (128,) embd_shape (151936, 1536) dtype <bf16>
        Torch time: 0.02406 ms
        LLAISYS time: 0.00394 ms
   idx_shape (1,) embd_shape (151936, 4096) dtype <f32>
        Torch time: 0.02390 ms
        LLAISYS time: 0.00390 ms
   idx_shape (1,) embd_shape (151936, 4096) dtype <f16>
        Torch time: 0.02368 ms
        LLAISYS time: 0.00389 ms
   idx_shape (1,) embd_shape (151936, 4096) dtype <bf16>
        Torch time: 0.03043 ms
        LLAISYS time: 0.00596 ms
   idx_shape (512,) embd_shape (151936, 4096) dtype <f32>
        Torch time: 0.03046 ms
        LLAISYS time: 0.00601 ms
   idx_shape (512,) embd_shape (151936, 4096) dtype <f16>
        Torch time: 0.03191 ms
        LLAISYS time: 0.00611 ms
   idx_shape (512,) embd_shape (151936, 4096) dtype <bf16>
        Torch time: 0.03066 ms
        LLAISYS time: 0.00609 ms
   idx_shape (1,) embd_shape (152064, 5120) dtype <f32>
        Torch time: 0.03077 ms
        LLAISYS time: 0.00605 ms
   idx_shape (1,) embd_shape (152064, 5120) dtype <f16>
        Torch time: 0.03210 ms
        LLAISYS time: 0.00593 ms
   idx_shape (1,) embd_shape (152064, 5120) dtype <bf16>
        Torch time: 0.03061 ms
        LLAISYS time: 0.00590 ms
   idx_shape (1024,) embd_shape (152064, 5120) dtype <f32>
        Torch time: 0.03076 ms
        LLAISYS time: 0.01198 ms
   idx_shape (1024,) embd_shape (152064, 5120) dtype <f16>
        Torch time: 0.03053 ms
        LLAISYS time: 0.00659 ms
   idx_shape (1024,) embd_shape (152064, 5120) dtype <bf16>
        Torch time: 0.03052 ms
        LLAISYS time: 0.00656 ms
Test passed!
```

## Linear
```bash
Testing Ops.linear on nvidia
   out (1, 1536), x (1, 1536), w (1536, 1536), bias True, dtype <f32>
        Torch time: 0.01348 ms
        LLAISYS time: 0.00514 ms
   out (1, 1536), x (1, 1536), w (1536, 1536), bias True, dtype <f16>
        Torch time: 0.01355 ms
        LLAISYS time: 0.00494 ms
   out (1, 1536), x (1, 1536), w (1536, 1536), bias True, dtype <bf16>
        Torch time: 0.01335 ms
        LLAISYS time: 0.00497 ms
   out (1, 8960), x (1, 1536), w (8960, 1536), bias True, dtype <f32>
        Torch time: 0.01721 ms
        LLAISYS time: 0.01516 ms
   out (1, 8960), x (1, 1536), w (8960, 1536), bias True, dtype <f16>
        Torch time: 0.01417 ms
        LLAISYS time: 0.01058 ms
   out (1, 8960), x (1, 1536), w (8960, 1536), bias True, dtype <bf16>
        Torch time: 0.01407 ms
        LLAISYS time: 0.01063 ms
   out (1, 1536), x (1, 8960), w (1536, 8960), bias True, dtype <f32>
        Torch time: 0.06036 ms
        LLAISYS time: 0.01616 ms
   out (1, 1536), x (1, 8960), w (1536, 8960), bias True, dtype <f16>
        Torch time: 0.02144 ms
        LLAISYS time: 0.00953 ms
   out (1, 1536), x (1, 8960), w (1536, 8960), bias True, dtype <bf16>
        Torch time: 0.02142 ms
        LLAISYS time: 0.00938 ms
   out (128, 1536), x (128, 1536), w (1536, 1536), bias True, dtype <f32>
        Torch time: 0.02972 ms
        LLAISYS time: 0.12685 ms
   out (128, 1536), x (128, 1536), w (1536, 1536), bias True, dtype <f16>
        Torch time: 0.01788 ms
        LLAISYS time: 0.10841 ms
   out (128, 1536), x (128, 1536), w (1536, 1536), bias True, dtype <bf16>
        Torch time: 0.01815 ms
        LLAISYS time: 0.12291 ms
   out (128, 8960), x (128, 1536), w (8960, 1536), bias True, dtype <f32>
        Torch time: 0.11554 ms
        LLAISYS time: 0.12955 ms
   out (128, 8960), x (128, 1536), w (8960, 1536), bias True, dtype <f16>
        Torch time: 0.03631 ms
        LLAISYS time: 0.11285 ms
   out (128, 8960), x (128, 1536), w (8960, 1536), bias True, dtype <bf16>
        Torch time: 0.03560 ms
        LLAISYS time: 0.11753 ms
   out (128, 1536), x (128, 8960), w (1536, 8960), bias True, dtype <f32>
        Torch time: 0.14444 ms
        LLAISYS time: 0.64426 ms
   out (128, 1536), x (128, 8960), w (1536, 8960), bias True, dtype <f16>
        Torch time: 0.03368 ms
        LLAISYS time: 0.54038 ms
   out (128, 1536), x (128, 8960), w (1536, 8960), bias True, dtype <bf16>
        Torch time: 0.03427 ms
        LLAISYS time: 0.62360 ms
   out (1, 4096), x (1, 4096), w (4096, 4096), bias True, dtype <f32>
        Torch time: 0.01778 ms
        LLAISYS time: 0.01610 ms
   out (1, 4096), x (1, 4096), w (4096, 4096), bias True, dtype <f16>
        Torch time: 0.01648 ms
        LLAISYS time: 0.00998 ms
   out (1, 4096), x (1, 4096), w (4096, 4096), bias True, dtype <bf16>
        Torch time: 0.01656 ms
        LLAISYS time: 0.00974 ms
   out (1, 12288), x (1, 4096), w (12288, 4096), bias True, dtype <f32>
        Torch time: 0.21247 ms
        LLAISYS time: 0.21262 ms
   out (1, 12288), x (1, 4096), w (12288, 4096), bias True, dtype <f16>
        Torch time: 0.12607 ms
        LLAISYS time: 0.10752 ms
   out (1, 12288), x (1, 4096), w (12288, 4096), bias True, dtype <bf16>
        Torch time: 0.10765 ms
        LLAISYS time: 0.10747 ms
   out (1, 4096), x (1, 12288), w (4096, 12288), bias True, dtype <f32>
        Torch time: 0.24006 ms
        LLAISYS time: 0.21333 ms
   out (1, 4096), x (1, 12288), w (4096, 12288), bias True, dtype <f16>
        Torch time: 0.11025 ms
        LLAISYS time: 0.10743 ms
   out (1, 4096), x (1, 12288), w (4096, 12288), bias True, dtype <bf16>
        Torch time: 0.10960 ms
        LLAISYS time: 0.10746 ms
   out (512, 4096), x (512, 4096), w (4096, 4096), bias True, dtype <f32>
        Torch time: 0.51945 ms
        LLAISYS time: 0.55372 ms
   out (512, 4096), x (512, 4096), w (4096, 4096), bias True, dtype <f16>
        Torch time: 0.14940 ms
        LLAISYS time: 0.30586 ms
   out (512, 4096), x (512, 4096), w (4096, 4096), bias True, dtype <bf16>
        Torch time: 0.12349 ms
        LLAISYS time: 0.28948 ms
   out (512, 12288), x (512, 4096), w (12288, 4096), bias True, dtype <f32>
        Torch time: 1.49008 ms
        LLAISYS time: 2.18085 ms
   out (512, 12288), x (512, 4096), w (12288, 4096), bias True, dtype <f16>
        Torch time: 0.38936 ms
        LLAISYS time: 0.62070 ms
   out (512, 12288), x (512, 4096), w (12288, 4096), bias True, dtype <bf16>
        Torch time: 0.37691 ms
        LLAISYS time: 0.67241 ms
   out (512, 4096), x (512, 12288), w (4096, 12288), bias True, dtype <f32>
        Torch time: 1.31109 ms
        LLAISYS time: 1.58544 ms
   out (512, 4096), x (512, 12288), w (4096, 12288), bias True, dtype <f16>
        Torch time: 0.37869 ms
        LLAISYS time: 0.83337 ms
   out (512, 4096), x (512, 12288), w (4096, 12288), bias True, dtype <bf16>
        Torch time: 0.39339 ms
        LLAISYS time: 0.96134 ms
   out (1, 5120), x (1, 5120), w (5120, 5120), bias True, dtype <f32>
        Torch time: 0.11283 ms
        LLAISYS time: 0.11176 ms
   out (1, 5120), x (1, 5120), w (5120, 5120), bias True, dtype <f16>
        Torch time: 0.05745 ms
        LLAISYS time: 0.01466 ms
   out (1, 5120), x (1, 5120), w (5120, 5120), bias True, dtype <bf16>
        Torch time: 0.05745 ms
        LLAISYS time: 0.01442 ms
   out (1, 27648), x (1, 5120), w (27648, 5120), bias True, dtype <f32>
        Torch time: 0.59248 ms
        LLAISYS time: 0.59327 ms
   out (1, 27648), x (1, 5120), w (27648, 5120), bias True, dtype <f16>
        Torch time: 0.30158 ms
        LLAISYS time: 0.29777 ms
   out (1, 27648), x (1, 5120), w (27648, 5120), bias True, dtype <bf16>
        Torch time: 0.29766 ms
        LLAISYS time: 0.29775 ms
   out (1, 5120), x (1, 27648), w (5120, 27648), bias True, dtype <f32>
        Torch time: 0.65929 ms
        LLAISYS time: 0.59280 ms
   out (1, 5120), x (1, 27648), w (5120, 27648), bias True, dtype <f16>
        Torch time: 0.30365 ms
        LLAISYS time: 0.29755 ms
   out (1, 5120), x (1, 27648), w (5120, 27648), bias True, dtype <bf16>
        Torch time: 0.30002 ms
        LLAISYS time: 0.29766 ms
   out (1024, 5120), x (1024, 5120), w (5120, 5120), bias True, dtype <f32>
        Torch time: 1.49345 ms
        LLAISYS time: 1.81029 ms
   out (1024, 5120), x (1024, 5120), w (5120, 5120), bias True, dtype <f16>
        Torch time: 0.36449 ms
        LLAISYS time: 0.67310 ms
   out (1024, 5120), x (1024, 5120), w (5120, 5120), bias True, dtype <bf16>
        Torch time: 0.37387 ms
        LLAISYS time: 0.75395 ms
   out (1024, 27648), x (1024, 5120), w (27648, 5120), bias True, dtype <f32>
        Torch time: 8.68043 ms
        LLAISYS time: 19.29609 ms
   out (1024, 27648), x (1024, 5120), w (27648, 5120), bias True, dtype <f16>
        Torch time: 1.98788 ms
        LLAISYS time: 4.22887 ms
   out (1024, 27648), x (1024, 5120), w (27648, 5120), bias True, dtype <bf16>
        Torch time: 1.91801 ms
        LLAISYS time: 4.30433 ms
   out (1024, 5120), x (1024, 27648), w (5120, 27648), bias True, dtype <f32>
        Torch time: 7.54219 ms
        LLAISYS time: 11.05026 ms
   out (1024, 5120), x (1024, 27648), w (5120, 27648), bias True, dtype <f16>
        Torch time: 1.96672 ms
        LLAISYS time: 3.85739 ms
   out (1024, 5120), x (1024, 27648), w (5120, 27648), bias True, dtype <bf16>
        Torch time: 1.92490 ms
        LLAISYS time: 4.12456 ms
Test passed!
```

## RMS Norm
```bash
Testing Ops.rms_norm on nvidia
   shape (1, 1536) dtype <f32>
        Torch time: 0.04422 ms
        LLAISYS time: 0.00435 ms
   shape (1, 1536) dtype <f16>
        Torch time: 0.04477 ms
        LLAISYS time: 0.00451 ms
   shape (1, 1536) dtype <bf16>
        Torch time: 0.04448 ms
        LLAISYS time: 0.00437 ms
   shape (128, 1536) dtype <f32>
        Torch time: 0.04466 ms
        LLAISYS time: 0.00444 ms
   shape (128, 1536) dtype <f16>
        Torch time: 0.04488 ms
        LLAISYS time: 0.00440 ms
   shape (128, 1536) dtype <bf16>
        Torch time: 0.04485 ms
        LLAISYS time: 0.00440 ms
   shape (1, 4096) dtype <f32>
        Torch time: 0.04429 ms
        LLAISYS time: 0.00435 ms
   shape (1, 4096) dtype <f16>
        Torch time: 0.04460 ms
        LLAISYS time: 0.00444 ms
   shape (1, 4096) dtype <bf16>
        Torch time: 0.04481 ms
        LLAISYS time: 0.00435 ms
   shape (512, 4096) dtype <f32>
        Torch time: 0.04508 ms
        LLAISYS time: 0.00736 ms
   shape (512, 4096) dtype <f16>
        Torch time: 0.04529 ms
        LLAISYS time: 0.00458 ms
   shape (512, 4096) dtype <bf16>
        Torch time: 0.04517 ms
        LLAISYS time: 0.00449 ms
   shape (1, 5120) dtype <f32>
        Torch time: 0.04412 ms
        LLAISYS time: 0.00438 ms
   shape (1, 5120) dtype <f16>
        Torch time: 0.04475 ms
        LLAISYS time: 0.00447 ms
   shape (1, 5120) dtype <bf16>
        Torch time: 0.04469 ms
        LLAISYS time: 0.00439 ms
   shape (1024, 5120) dtype <f32>
        Torch time: 0.05101 ms
        LLAISYS time: 0.01690 ms
   shape (1024, 5120) dtype <f16>
        Torch time: 0.04646 ms
        LLAISYS time: 0.00891 ms
   shape (1024, 5120) dtype <bf16>
        Torch time: 0.04534 ms
        LLAISYS time: 0.00885 ms
Test passed!
```

## RoPE
```bash
Testing Ops.rope on nvidia
   shape (1, 12, 128) range (128, 129) dtype <f32>
        Torch time: 0.18430 ms
        LLAISYS time: 0.00432 ms
   shape (1, 12, 128) range (128, 129) dtype <f16>
        Torch time: 0.18480 ms
        LLAISYS time: 0.00430 ms
   shape (1, 12, 128) range (128, 129) dtype <bf16>
        Torch time: 0.18376 ms
        LLAISYS time: 0.00445 ms
   shape (1, 2, 128) range (128, 129) dtype <f32>
        Torch time: 0.18296 ms
        LLAISYS time: 0.00433 ms
   shape (1, 2, 128) range (128, 129) dtype <f16>
        Torch time: 0.18336 ms
        LLAISYS time: 0.00434 ms
   shape (1, 2, 128) range (128, 129) dtype <bf16>
        Torch time: 0.18369 ms
        LLAISYS time: 0.00432 ms
   shape (128, 12, 128) range (0, 128) dtype <f32>
        Torch time: 0.18474 ms
        LLAISYS time: 0.00427 ms
   shape (128, 12, 128) range (0, 128) dtype <f16>
        Torch time: 0.18552 ms
        LLAISYS time: 0.00427 ms
   shape (128, 12, 128) range (0, 128) dtype <bf16>
        Torch time: 0.18537 ms
        LLAISYS time: 0.00436 ms
   shape (128, 2, 128) range (0, 128) dtype <f32>
        Torch time: 0.18454 ms
        LLAISYS time: 0.00441 ms
   shape (128, 2, 128) range (0, 128) dtype <f16>
        Torch time: 0.18626 ms
        LLAISYS time: 0.00429 ms
   shape (128, 2, 128) range (0, 128) dtype <bf16>
        Torch time: 0.18570 ms
        LLAISYS time: 0.00432 ms
   shape (1, 32, 128) range (512, 513) dtype <f32>
        Torch time: 0.18336 ms
        LLAISYS time: 0.00434 ms
   shape (1, 32, 128) range (512, 513) dtype <f16>
        Torch time: 0.18426 ms
        LLAISYS time: 0.00425 ms
   shape (1, 32, 128) range (512, 513) dtype <bf16>
        Torch time: 0.18399 ms
        LLAISYS time: 0.00435 ms
   shape (1, 8, 128) range (512, 513) dtype <f32>
        Torch time: 0.18283 ms
        LLAISYS time: 0.00433 ms
   shape (1, 8, 128) range (512, 513) dtype <f16>
        Torch time: 0.18466 ms
        LLAISYS time: 0.00439 ms
   shape (1, 8, 128) range (512, 513) dtype <bf16>
        Torch time: 0.18529 ms
        LLAISYS time: 0.00430 ms
   shape (512, 32, 128) range (0, 512) dtype <f32>
        Torch time: 0.18410 ms
        LLAISYS time: 0.01078 ms
   shape (512, 32, 128) range (0, 512) dtype <f16>
        Torch time: 0.18524 ms
        LLAISYS time: 0.01091 ms
   shape (512, 32, 128) range (0, 512) dtype <bf16>
        Torch time: 0.18621 ms
        LLAISYS time: 0.01093 ms
   shape (512, 8, 128) range (0, 512) dtype <f32>
        Torch time: 0.18407 ms
        LLAISYS time: 0.00441 ms
   shape (512, 8, 128) range (0, 512) dtype <f16>
        Torch time: 0.18584 ms
        LLAISYS time: 0.00432 ms
   shape (512, 8, 128) range (0, 512) dtype <bf16>
        Torch time: 0.18505 ms
        LLAISYS time: 0.00439 ms
   shape (1, 40, 128) range (1024, 1025) dtype <f32>
        Torch time: 0.18422 ms
        LLAISYS time: 0.00426 ms
   shape (1, 40, 128) range (1024, 1025) dtype <f16>
        Torch time: 0.18528 ms
        LLAISYS time: 0.00427 ms
   shape (1, 40, 128) range (1024, 1025) dtype <bf16>
        Torch time: 0.18453 ms
        LLAISYS time: 0.00441 ms
   shape (1, 8, 128) range (1024, 1025) dtype <f32>
        Torch time: 0.18349 ms
        LLAISYS time: 0.00431 ms
   shape (1, 8, 128) range (1024, 1025) dtype <f16>
        Torch time: 0.18466 ms
        LLAISYS time: 0.00432 ms
   shape (1, 8, 128) range (1024, 1025) dtype <bf16>
        Torch time: 0.18458 ms
        LLAISYS time: 0.00442 ms
   shape (1024, 32, 128) range (0, 1024) dtype <f32>
        Torch time: 0.18489 ms
        LLAISYS time: 0.01979 ms
   shape (1024, 32, 128) range (0, 1024) dtype <f16>
        Torch time: 0.18714 ms
        LLAISYS time: 0.01998 ms
   shape (1024, 32, 128) range (0, 1024) dtype <bf16>
        Torch time: 0.18619 ms
        LLAISYS time: 0.01993 ms
   shape (1024, 8, 128) range (0, 1024) dtype <f32>
        Torch time: 0.18551 ms
        LLAISYS time: 0.00634 ms
   shape (1024, 8, 128) range (0, 1024) dtype <f16>
        Torch time: 0.18614 ms
        LLAISYS time: 0.00632 ms
   shape (1024, 8, 128) range (0, 1024) dtype <bf16>
        Torch time: 0.18625 ms
        LLAISYS time: 0.00631 ms
Test passed!
```

## Self-Attention
```bash
Testing Ops.self_attention on nvidia
   qlen=1 kvlen=128 nh=12 nkvh=2 hd=128 dtype <f32>
        Torch time: 0.20475 ms
        LLAISYS time: 0.01203 ms
   qlen=1 kvlen=128 nh=12 nkvh=2 hd=128 dtype <f16>
        Torch time: 0.21212 ms
        LLAISYS time: 0.01208 ms
   qlen=1 kvlen=128 nh=12 nkvh=2 hd=128 dtype <bf16>
        Torch time: 0.21083 ms
        LLAISYS time: 0.01208 ms
   qlen=128 kvlen=128 nh=12 nkvh=2 hd=128 dtype <f32>
        Torch time: 0.20639 ms
        LLAISYS time: 0.02627 ms
   qlen=128 kvlen=128 nh=12 nkvh=2 hd=128 dtype <f16>
        Torch time: 0.21229 ms
        LLAISYS time: 0.02829 ms
   qlen=128 kvlen=128 nh=12 nkvh=2 hd=128 dtype <bf16>
        Torch time: 0.22925 ms
        LLAISYS time: 0.02871 ms
   qlen=1 kvlen=512 nh=32 nkvh=8 hd=128 dtype <f32>
        Torch time: 0.21123 ms
        LLAISYS time: 0.04198 ms
   qlen=1 kvlen=512 nh=32 nkvh=8 hd=128 dtype <f16>
        Torch time: 0.21233 ms
        LLAISYS time: 0.04233 ms
   qlen=1 kvlen=512 nh=32 nkvh=8 hd=128 dtype <bf16>
        Torch time: 0.20979 ms
        LLAISYS time: 0.04233 ms
   qlen=512 kvlen=512 nh=32 nkvh=8 hd=128 dtype <f32>
        Torch time: 0.27158 ms
        LLAISYS time: 0.83619 ms
   qlen=512 kvlen=512 nh=32 nkvh=8 hd=128 dtype <f16>
        Torch time: 0.21210 ms
        LLAISYS time: 0.82933 ms
   qlen=512 kvlen=512 nh=32 nkvh=8 hd=128 dtype <bf16>
        Torch time: 0.20683 ms
        LLAISYS time: 0.81187 ms
   qlen=1 kvlen=1024 nh=40 nkvh=8 hd=128 dtype <f32>
        Torch time: 0.21107 ms
        LLAISYS time: 0.08164 ms
   qlen=1 kvlen=1024 nh=40 nkvh=8 hd=128 dtype <f16>
        Torch time: 0.21887 ms
        LLAISYS time: 0.07632 ms
   qlen=1 kvlen=1024 nh=40 nkvh=8 hd=128 dtype <bf16>
        Torch time: 0.27532 ms
        LLAISYS time: 0.07499 ms
   qlen=1024 kvlen=1024 nh=40 nkvh=8 hd=128 dtype <f32>
        Torch time: 1.83715 ms
        LLAISYS time: 4.12091 ms
   qlen=1024 kvlen=1024 nh=40 nkvh=8 hd=128 dtype <f16>
        Torch time: 0.87240 ms
        LLAISYS time: 4.44587 ms
   qlen=1024 kvlen=1024 nh=40 nkvh=8 hd=128 dtype <bf16>
        Torch time: 0.87514 ms
        LLAISYS time: 4.03622 ms
Test passed!
```

## SwiGLU
```bash
Testing Ops.swiglu on nvidia
   shape (1, 8960) dtype <f32>
        Torch time: 0.04037 ms
        LLAISYS time: 0.00391 ms
   shape (1, 8960) dtype <f16>
        Torch time: 0.05835 ms
        LLAISYS time: 0.00400 ms
   shape (1, 8960) dtype <bf16>
        Torch time: 0.05849 ms
        LLAISYS time: 0.00397 ms
   shape (128, 8960) dtype <f32>
        Torch time: 0.04098 ms
        LLAISYS time: 0.00480 ms
   shape (128, 8960) dtype <f16>
        Torch time: 0.05893 ms
        LLAISYS time: 0.00388 ms
   shape (128, 8960) dtype <bf16>
        Torch time: 0.05868 ms
        LLAISYS time: 0.00404 ms
   shape (1, 12288) dtype <f32>
        Torch time: 0.04001 ms
        LLAISYS time: 0.00395 ms
   shape (1, 12288) dtype <f16>
        Torch time: 0.05830 ms
        LLAISYS time: 0.00390 ms
   shape (1, 12288) dtype <bf16>
        Torch time: 0.05806 ms
        LLAISYS time: 0.00393 ms
   shape (512, 12288) dtype <f32>
        Torch time: 0.15680 ms
        LLAISYS time: 0.02872 ms
   shape (512, 12288) dtype <f16>
        Torch time: 0.10040 ms
        LLAISYS time: 0.00955 ms
   shape (512, 12288) dtype <bf16>
        Torch time: 0.10050 ms
        LLAISYS time: 0.00963 ms
   shape (1, 27648) dtype <f32>
        Torch time: 0.04011 ms
        LLAISYS time: 0.00386 ms
   shape (1, 27648) dtype <f16>
        Torch time: 0.05859 ms
        LLAISYS time: 0.00387 ms
   shape (1, 27648) dtype <bf16>
        Torch time: 0.05822 ms
        LLAISYS time: 0.00391 ms
   shape (1024, 27648) dtype <f32>
        Torch time: 1.47460 ms
        LLAISYS time: 0.36575 ms
   shape (1024, 27648) dtype <f16>
        Torch time: 1.33505 ms
        LLAISYS time: 0.18278 ms
   shape (1024, 27648) dtype <bf16>
        Torch time: 1.33486 ms
        LLAISYS time: 0.18311 ms
Test passed!
```