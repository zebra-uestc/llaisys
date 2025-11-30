# CPU 算子性能测评
## Add
```bash
Testing Ops.add on cpu
   shape (1, 1536) dtype <f32>
        Torch time: 0.00262 ms
        LLAISYS time: 0.00125 ms
   shape (1, 1536) dtype <f16>
        Torch time: 0.00374 ms
        LLAISYS time: 0.00221 ms
   shape (1, 1536) dtype <bf16>
        Torch time: 0.00395 ms
        LLAISYS time: 0.00249 ms
   shape (128, 1536) dtype <f32>
        Torch time: 0.02013 ms
        LLAISYS time: 0.01101 ms
   shape (128, 1536) dtype <f16>
        Torch time: 0.02257 ms
        LLAISYS time: 0.00874 ms
   shape (128, 1536) dtype <bf16>
        Torch time: 0.02727 ms
        LLAISYS time: 0.01166 ms
   shape (1, 4096) dtype <f32>
        Torch time: 0.00378 ms
        LLAISYS time: 0.00219 ms
   shape (1, 4096) dtype <f16>
        Torch time: 0.00390 ms
        LLAISYS time: 0.00284 ms
   shape (1, 4096) dtype <bf16>
        Torch time: 0.00320 ms
        LLAISYS time: 0.00145 ms
   shape (512, 4096) dtype <f32>
        Torch time: 0.11454 ms
        LLAISYS time: 0.11411 ms
   shape (512, 4096) dtype <f16>
        Torch time: 0.05422 ms
        LLAISYS time: 0.02505 ms
   shape (512, 4096) dtype <bf16>
        Torch time: 0.06755 ms
        LLAISYS time: 0.02741 ms
   shape (1, 5120) dtype <f32>
        Torch time: 0.00294 ms
        LLAISYS time: 0.00197 ms
   shape (1, 5120) dtype <f16>
        Torch time: 0.00324 ms
        LLAISYS time: 0.00160 ms
   shape (1, 5120) dtype <bf16>
        Torch time: 0.00350 ms
        LLAISYS time: 0.00158 ms
   shape (1024, 5120) dtype <f32>
        Torch time: 0.40985 ms
        LLAISYS time: 0.36958 ms
   shape (1024, 5120) dtype <f16>
        Torch time: 0.12914 ms
        LLAISYS time: 0.11051 ms
   shape (1024, 5120) dtype <bf16>
        Torch time: 0.15628 ms
        LLAISYS time: 0.10541 ms
Test passed!
```

## Argmax
```bash
Testing Ops.argmax on cpu
   shape (151936,) dtype <f32>
        Torch time: 0.18724 ms
        LLAISYS time: 0.02910 ms
   shape (151936,) dtype <f16>
        Torch time: 0.24002 ms
        LLAISYS time: 0.03584 ms
   shape (151936,) dtype <bf16>
        Torch time: 0.18334 ms
        LLAISYS time: 0.03125 ms
   shape (152064,) dtype <f32>
        Torch time: 0.17431 ms
        LLAISYS time: 0.02723 ms
   shape (152064,) dtype <f16>
        Torch time: 0.23121 ms
        LLAISYS time: 0.03022 ms
   shape (152064,) dtype <bf16>
        Torch time: 0.17427 ms
        LLAISYS time: 0.03281 ms
Test passed!
```

## Embedding
```bash
Testing Ops.embedding on cpu
   idx_shape (1,) embd_shape (151936, 1536) dtype <f32>
        Torch time: 0.01091 ms
        LLAISYS time: 0.00142 ms
   idx_shape (1,) embd_shape (151936, 1536) dtype <f16>
        Torch time: 0.01069 ms
        LLAISYS time: 0.00138 ms
   idx_shape (1,) embd_shape (151936, 1536) dtype <bf16>
        Torch time: 0.01073 ms
        LLAISYS time: 0.00139 ms
   idx_shape (128,) embd_shape (151936, 1536) dtype <f32>
        Torch time: 0.04728 ms
        LLAISYS time: 0.00686 ms
   idx_shape (128,) embd_shape (151936, 1536) dtype <f16>
        Torch time: 0.06468 ms
        LLAISYS time: 0.00584 ms
   idx_shape (128,) embd_shape (151936, 1536) dtype <bf16>
        Torch time: 0.06009 ms
        LLAISYS time: 0.00586 ms
   idx_shape (1,) embd_shape (151936, 4096) dtype <f32>
        Torch time: 0.02345 ms
        LLAISYS time: 0.00228 ms
   idx_shape (1,) embd_shape (151936, 4096) dtype <f16>
        Torch time: 0.02479 ms
        LLAISYS time: 0.00220 ms
   idx_shape (1,) embd_shape (151936, 4096) dtype <bf16>
        Torch time: 0.02001 ms
        LLAISYS time: 0.00211 ms
   idx_shape (512,) embd_shape (151936, 4096) dtype <f32>
        Torch time: 0.25836 ms
        LLAISYS time: 0.04098 ms
   idx_shape (512,) embd_shape (151936, 4096) dtype <f16>
        Torch time: 0.17668 ms
        LLAISYS time: 0.01430 ms
   idx_shape (512,) embd_shape (151936, 4096) dtype <bf16>
        Torch time: 0.19080 ms
        LLAISYS time: 0.01505 ms
   idx_shape (1,) embd_shape (152064, 5120) dtype <f32>
        Torch time: 0.02132 ms
        LLAISYS time: 0.00256 ms
   idx_shape (1,) embd_shape (152064, 5120) dtype <f16>
        Torch time: 0.02654 ms
        LLAISYS time: 0.00209 ms
   idx_shape (1,) embd_shape (152064, 5120) dtype <bf16>
        Torch time: 0.39996 ms
        LLAISYS time: 0.00208 ms
   idx_shape (1024,) embd_shape (152064, 5120) dtype <f32>
        Torch time: 0.82579 ms
        LLAISYS time: 0.27305 ms
   idx_shape (1024,) embd_shape (152064, 5120) dtype <f16>
        Torch time: 0.50608 ms
        LLAISYS time: 0.08731 ms
   idx_shape (1024,) embd_shape (152064, 5120) dtype <bf16>
        Torch time: 0.50539 ms
        LLAISYS time: 0.08463 ms
Test passed!
```

## Linear
```bash
Testing Ops.linear on cpu
   out (2, 3), x (2, 4), w (3, 4), bias True, dtype <f32>
        Torch time: 0.00648 ms
        LLAISYS time: 0.02585 ms
   out (2, 3), x (2, 4), w (3, 4), bias True, dtype <f16>
        Torch time: 0.01172 ms
        LLAISYS time: 0.02055 ms
   out (2, 3), x (2, 4), w (3, 4), bias True, dtype <bf16>
        Torch time: 0.01145 ms
        LLAISYS time: 0.02000 ms
   out (512, 4096), x (512, 4096), w (4096, 4096), bias True, dtype <f32>
        Torch time: 14.04708 ms
        LLAISYS time: 15.77965 ms
   out (512, 4096), x (512, 4096), w (4096, 4096), bias True, dtype <f16>
        Torch time: 100.69306 ms
        LLAISYS time: 77.08914 ms
   out (512, 4096), x (512, 4096), w (4096, 4096), bias True, dtype <bf16>
        Torch time: 60.30512 ms
        LLAISYS time: 81.89152 ms
   out (1, 1536), x (1, 1536), w (1536, 1536), bias True, dtype <f32>
        Torch time: 0.03150 ms
        LLAISYS time: 0.02055 ms
   out (1, 1536), x (1, 1536), w (1536, 1536), bias True, dtype <f16>
        Torch time: 0.05552 ms
        LLAISYS time: 1.97409 ms
   out (1, 1536), x (1, 1536), w (1536, 1536), bias True, dtype <bf16>
        Torch time: 5.52066 ms
        LLAISYS time: 1.74603 ms
   out (1, 8960), x (1, 1536), w (8960, 1536), bias True, dtype <f32>
        Torch time: 0.33603 ms
        LLAISYS time: 0.32509 ms
   out (1, 8960), x (1, 1536), w (8960, 1536), bias True, dtype <f16>
        Torch time: 0.24503 ms
        LLAISYS time: 39.74828 ms
   out (1, 8960), x (1, 1536), w (8960, 1536), bias True, dtype <bf16>
        Torch time: 0.19489 ms
        LLAISYS time: 39.73993 ms
   out (1, 1536), x (1, 8960), w (1536, 8960), bias True, dtype <f32>
        Torch time: 0.30131 ms
        LLAISYS time: 0.29183 ms
   out (1, 1536), x (1, 8960), w (1536, 8960), bias True, dtype <f16>
        Torch time: 0.25232 ms
        LLAISYS time: 37.83597 ms
   out (1, 1536), x (1, 8960), w (1536, 8960), bias True, dtype <bf16>
        Torch time: 4.24978 ms
        LLAISYS time: 39.58697 ms
   out (128, 1536), x (128, 1536), w (1536, 1536), bias True, dtype <f32>
        Torch time: 0.59083 ms
        LLAISYS time: 0.78103 ms
   out (128, 1536), x (128, 1536), w (1536, 1536), bias True, dtype <f16>
        Torch time: 3.91532 ms
        LLAISYS time: 2.73151 ms
   out (128, 1536), x (128, 1536), w (1536, 1536), bias True, dtype <bf16>
        Torch time: 2.17678 ms
        LLAISYS time: 2.68255 ms
   out (128, 8960), x (128, 1536), w (8960, 1536), bias True, dtype <f32>
        Torch time: 9.00348 ms
        LLAISYS time: 4.38465 ms
   out (128, 8960), x (128, 1536), w (8960, 1536), bias True, dtype <f16>
        Torch time: 26.22367 ms
        LLAISYS time: 49.72974 ms
   out (128, 8960), x (128, 1536), w (8960, 1536), bias True, dtype <bf16>
        Torch time: 19.78453 ms
        LLAISYS time: 45.50789 ms
   out (128, 1536), x (128, 8960), w (1536, 8960), bias True, dtype <f32>
        Torch time: 3.00326 ms
        LLAISYS time: 3.68218 ms
   out (128, 1536), x (128, 8960), w (1536, 8960), bias True, dtype <f16>
        Torch time: 20.72079 ms
        LLAISYS time: 41.66163 ms
   out (128, 1536), x (128, 8960), w (1536, 8960), bias True, dtype <bf16>
        Torch time: 12.54653 ms
        LLAISYS time: 39.94967 ms
   out (1, 4096), x (1, 4096), w (4096, 4096), bias True, dtype <f32>
        Torch time: 0.43818 ms
        LLAISYS time: 0.44445 ms
   out (1, 4096), x (1, 4096), w (4096, 4096), bias True, dtype <f16>
        Torch time: 0.21999 ms
        LLAISYS time: 44.09353 ms
   out (1, 4096), x (1, 4096), w (4096, 4096), bias True, dtype <bf16>
        Torch time: 0.22669 ms
        LLAISYS time: 44.41967 ms
   out (1, 12288), x (1, 4096), w (12288, 4096), bias True, dtype <f32>
        Torch time: 2.53126 ms
        LLAISYS time: 2.39524 ms
   out (1, 12288), x (1, 4096), w (12288, 4096), bias True, dtype <f16>
        Torch time: 0.96195 ms
        LLAISYS time: 129.10251 ms
   out (1, 12288), x (1, 4096), w (12288, 4096), bias True, dtype <bf16>
        Torch time: 1.10616 ms
        LLAISYS time: 128.09046 ms
   out (1, 4096), x (1, 12288), w (4096, 12288), bias True, dtype <f32>
        Torch time: 3.16014 ms
        LLAISYS time: 3.27317 ms
   out (1, 4096), x (1, 12288), w (4096, 12288), bias True, dtype <f16>
        Torch time: 1.24868 ms
        LLAISYS time: 132.19068 ms
   out (1, 4096), x (1, 12288), w (4096, 12288), bias True, dtype <bf16>
        Torch time: 2.76618 ms
        LLAISYS time: 132.62828 ms
   out (512, 4096), x (512, 4096), w (4096, 4096), bias True, dtype <f32>
        Torch time: 16.28651 ms
        LLAISYS time: 19.26332 ms
   out (512, 4096), x (512, 4096), w (4096, 4096), bias True, dtype <f16>
        Torch time: 110.08841 ms
        LLAISYS time: 81.61504 ms
   out (512, 4096), x (512, 4096), w (4096, 4096), bias True, dtype <bf16>
        Torch time: 62.56967 ms
        LLAISYS time: 77.58017 ms
   out (512, 12288), x (512, 4096), w (12288, 4096), bias True, dtype <f32>
        Torch time: 68.13789 ms
        LLAISYS time: 45.01108 ms
   out (512, 12288), x (512, 4096), w (12288, 4096), bias True, dtype <f16>
        Torch time: 293.34185 ms
        LLAISYS time: 177.92957 ms
   out (512, 12288), x (512, 4096), w (12288, 4096), bias True, dtype <bf16>
        Torch time: 163.96168 ms
        LLAISYS time: 189.57881 ms
   out (512, 4096), x (512, 12288), w (4096, 12288), bias True, dtype <f32>
        Torch time: 42.86984 ms
        LLAISYS time: 51.26665 ms
   out (512, 4096), x (512, 12288), w (4096, 12288), bias True, dtype <f16>
        Torch time: 324.15250 ms
        LLAISYS time: 177.71885 ms
   out (512, 4096), x (512, 12288), w (4096, 12288), bias True, dtype <bf16>
        Torch time: 153.70608 ms
        LLAISYS time: 179.06548 ms
Test passed!
```

## RMS Norm
```bash
Testing Ops.rms_norm on cpu
   shape (1, 1536) dtype <f32>
        Torch time: 0.03278 ms
        LLAISYS time: 0.00974 ms
   shape (1, 1536) dtype <f16>
        Torch time: 0.04348 ms
        LLAISYS time: 0.00918 ms
   shape (1, 1536) dtype <bf16>
        Torch time: 0.04075 ms
        LLAISYS time: 0.00936 ms
   shape (128, 1536) dtype <f32>
        Torch time: 0.10410 ms
        LLAISYS time: 0.01235 ms
   shape (128, 1536) dtype <f16>
        Torch time: 0.26267 ms
        LLAISYS time: 0.02107 ms
   shape (128, 1536) dtype <bf16>
        Torch time: 0.13555 ms
        LLAISYS time: 0.01920 ms
   shape (1, 4096) dtype <f32>
        Torch time: 0.03412 ms
        LLAISYS time: 0.00970 ms
   shape (1, 4096) dtype <f16>
        Torch time: 0.05613 ms
        LLAISYS time: 0.01235 ms
   shape (1, 4096) dtype <bf16>
        Torch time: 0.04360 ms
        LLAISYS time: 0.01194 ms
   shape (512, 4096) dtype <f32>
        Torch time: 0.30095 ms
        LLAISYS time: 0.06323 ms
   shape (512, 4096) dtype <f16>
        Torch time: 0.80285 ms
        LLAISYS time: 0.16998 ms
   shape (512, 4096) dtype <bf16>
        Torch time: 1.03045 ms
        LLAISYS time: 0.17510 ms
   shape (1, 5120) dtype <f32>
        Torch time: 0.03440 ms
        LLAISYS time: 0.01004 ms
   shape (1, 5120) dtype <f16>
        Torch time: 0.06121 ms
        LLAISYS time: 0.01564 ms
   shape (1, 5120) dtype <bf16>
        Torch time: 0.04420 ms
        LLAISYS time: 0.01323 ms
   shape (1024, 5120) dtype <f32>
        Torch time: 0.60898 ms
        LLAISYS time: 0.21670 ms
   shape (1024, 5120) dtype <f16>
        Torch time: 1.97151 ms
        LLAISYS time: 0.41144 ms
   shape (1024, 5120) dtype <bf16>
        Torch time: 0.58732 ms
        LLAISYS time: 0.42043 ms
Test passed!
```

## RoPE
```bash
Testing Ops.rope on cpu
   shape (1, 12, 128) range (128, 129) dtype <f32>
        Torch time: 0.09013 ms
        LLAISYS time: 0.00822 ms
   shape (1, 12, 128) range (128, 129) dtype <f16>
        Torch time: 0.10628 ms
        LLAISYS time: 0.01259 ms
   shape (1, 12, 128) range (128, 129) dtype <bf16>
        Torch time: 0.10613 ms
        LLAISYS time: 0.01284 ms
   shape (1, 2, 128) range (128, 129) dtype <f32>
        Torch time: 0.08892 ms
        LLAISYS time: 0.00890 ms
   shape (1, 2, 128) range (128, 129) dtype <f16>
        Torch time: 0.10279 ms
        LLAISYS time: 0.00797 ms
   shape (1, 2, 128) range (128, 129) dtype <bf16>
        Torch time: 0.10237 ms
        LLAISYS time: 0.00870 ms
   shape (128, 12, 128) range (0, 128) dtype <f32>
        Torch time: 0.22165 ms
        LLAISYS time: 0.01494 ms
   shape (128, 12, 128) range (0, 128) dtype <f16>
        Torch time: 0.28142 ms
        LLAISYS time: 0.06918 ms
   shape (128, 12, 128) range (0, 128) dtype <bf16>
        Torch time: 0.28675 ms
        LLAISYS time: 0.06317 ms
   shape (128, 2, 128) range (0, 128) dtype <f32>
        Torch time: 0.15358 ms
        LLAISYS time: 0.01226 ms
   shape (128, 2, 128) range (0, 128) dtype <f16>
        Torch time: 0.17949 ms
        LLAISYS time: 0.02157 ms
   shape (128, 2, 128) range (0, 128) dtype <bf16>
        Torch time: 0.18057 ms
        LLAISYS time: 0.02127 ms
   shape (1, 32, 128) range (512, 513) dtype <f32>
        Torch time: 0.09240 ms
        LLAISYS time: 0.00876 ms
   shape (1, 32, 128) range (512, 513) dtype <f16>
        Torch time: 0.11005 ms
        LLAISYS time: 0.01904 ms
   shape (1, 32, 128) range (512, 513) dtype <bf16>
        Torch time: 0.12682 ms
        LLAISYS time: 0.01996 ms
   shape (1, 8, 128) range (512, 513) dtype <f32>
        Torch time: 0.09114 ms
        LLAISYS time: 0.00934 ms
   shape (1, 8, 128) range (512, 513) dtype <f16>
        Torch time: 0.10605 ms
        LLAISYS time: 0.01183 ms
   shape (1, 8, 128) range (512, 513) dtype <bf16>
        Torch time: 0.10570 ms
        LLAISYS time: 0.01154 ms
   shape (512, 32, 128) range (0, 512) dtype <f32>
        Torch time: 0.39850 ms
        LLAISYS time: 0.05862 ms
   shape (512, 32, 128) range (0, 512) dtype <f16>
        Torch time: 6.75481 ms
        LLAISYS time: 0.28545 ms
   shape (512, 32, 128) range (0, 512) dtype <bf16>
        Torch time: 6.68824 ms
        LLAISYS time: 0.28253 ms
   shape (512, 8, 128) range (0, 512) dtype <f32>
        Torch time: 0.28273 ms
        LLAISYS time: 0.02475 ms
   shape (512, 8, 128) range (0, 512) dtype <f16>
        Torch time: 0.32283 ms
        LLAISYS time: 0.08710 ms
   shape (512, 8, 128) range (0, 512) dtype <bf16>
        Torch time: 0.32498 ms
        LLAISYS time: 0.08435 ms
   shape (1, 40, 128) range (1024, 1025) dtype <f32>
        Torch time: 0.09264 ms
        LLAISYS time: 0.00964 ms
   shape (1, 40, 128) range (1024, 1025) dtype <f16>
        Torch time: 0.10945 ms
        LLAISYS time: 0.02222 ms
   shape (1, 40, 128) range (1024, 1025) dtype <bf16>
        Torch time: 0.10943 ms
        LLAISYS time: 0.02223 ms
   shape (1, 8, 128) range (1024, 1025) dtype <f32>
        Torch time: 0.09015 ms
        LLAISYS time: 0.00875 ms
   shape (1, 8, 128) range (1024, 1025) dtype <f16>
        Torch time: 0.10550 ms
        LLAISYS time: 0.01039 ms
   shape (1, 8, 128) range (1024, 1025) dtype <bf16>
        Torch time: 0.10487 ms
        LLAISYS time: 0.01037 ms
   shape (1024, 32, 128) range (0, 1024) dtype <f32>
        Torch time: 0.81286 ms
        LLAISYS time: 0.08377 ms
   shape (1024, 32, 128) range (0, 1024) dtype <f16>
        Torch time: 11.78928 ms
        LLAISYS time: 0.55016 ms
   shape (1024, 32, 128) range (0, 1024) dtype <bf16>
        Torch time: 11.98018 ms
        LLAISYS time: 0.57124 ms
   shape (1024, 8, 128) range (0, 1024) dtype <f32>
        Torch time: 0.26156 ms
        LLAISYS time: 0.04173 ms
   shape (1024, 8, 128) range (0, 1024) dtype <f16>
        Torch time: 0.32697 ms
        LLAISYS time: 0.18694 ms
   shape (1024, 8, 128) range (0, 1024) dtype <bf16>
        Torch time: 0.33184 ms
        LLAISYS time: 0.16735 ms
Test passed!
```

## Self-Attention
```bash
Testing Ops.self_attention on cpu
   qlen=1 kvlen=128 nh=12 nkvh=2 hd=128 dtype <f32>
        Torch time: 0.21772 ms
        LLAISYS time: 0.02118 ms
   qlen=1 kvlen=128 nh=12 nkvh=2 hd=128 dtype <f16>
        Torch time: 0.72310 ms
        LLAISYS time: 0.14649 ms
   qlen=1 kvlen=128 nh=12 nkvh=2 hd=128 dtype <bf16>
        Torch time: 0.29465 ms
        LLAISYS time: 0.13438 ms
   qlen=128 kvlen=128 nh=12 nkvh=2 hd=128 dtype <f32>
        Torch time: 0.50132 ms
        LLAISYS time: 0.35492 ms
   qlen=128 kvlen=128 nh=12 nkvh=2 hd=128 dtype <f16>
        Torch time: 32.30767 ms
        LLAISYS time: 4.52054 ms
   qlen=128 kvlen=128 nh=12 nkvh=2 hd=128 dtype <bf16>
        Torch time: 0.88615 ms
        LLAISYS time: 6.39522 ms
   qlen=1 kvlen=512 nh=32 nkvh=8 hd=128 dtype <f32>
        Torch time: 2.62542 ms
        LLAISYS time: 0.07751 ms
   qlen=1 kvlen=512 nh=32 nkvh=8 hd=128 dtype <f16>
        Torch time: 3.27759 ms
        LLAISYS time: 0.73070 ms
   qlen=1 kvlen=512 nh=32 nkvh=8 hd=128 dtype <bf16>
        Torch time: 0.46103 ms
        LLAISYS time: 0.65565 ms
   qlen=512 kvlen=512 nh=32 nkvh=8 hd=128 dtype <f32>
        Torch time: 29.25550 ms
        LLAISYS time: 10.27990 ms
   qlen=512 kvlen=512 nh=32 nkvh=8 hd=128 dtype <f16>
        Torch time: 1214.56411 ms
        LLAISYS time: 163.98416 ms
   qlen=512 kvlen=512 nh=32 nkvh=8 hd=128 dtype <bf16>
        Torch time: 30.98963 ms
        LLAISYS time: 138.81445 ms
   qlen=1 kvlen=1024 nh=40 nkvh=8 hd=128 dtype <f32>
        Torch time: 1.11639 ms
        LLAISYS time: 0.15972 ms
   qlen=1 kvlen=1024 nh=40 nkvh=8 hd=128 dtype <f16>
        Torch time: 18.02537 ms
        LLAISYS time: 1.50406 ms
   qlen=1 kvlen=1024 nh=40 nkvh=8 hd=128 dtype <bf16>
        Torch time: 0.70819 ms
        LLAISYS time: 1.98549 ms
   qlen=1024 kvlen=1024 nh=40 nkvh=8 hd=128 dtype <f32>
        Torch time: 107.89074 ms
        LLAISYS time: 53.29805 ms
   qlen=1024 kvlen=1024 nh=40 nkvh=8 hd=128 dtype <f16>
        Torch time: 6462.94605 ms
        LLAISYS time: 725.99376 ms
   qlen=1024 kvlen=1024 nh=40 nkvh=8 hd=128 dtype <bf16>
        Torch time: 116.82543 ms
        LLAISYS time: 620.13353 ms
Test passed!
```

## SwiGLU
```bash
Testing Ops.swiglu on cpu
   shape (1, 8960) dtype <f32>
        Torch time: 7.32909 ms
        LLAISYS time: 0.71903 ms
   shape (1, 8960) dtype <f16>
        Torch time: 0.06433 ms
        LLAISYS time: 0.01364 ms
   shape (1, 8960) dtype <bf16>
        Torch time: 0.06350 ms
        LLAISYS time: 0.01543 ms
   shape (128, 8960) dtype <f32>
        Torch time: 2.15325 ms
        LLAISYS time: 0.50715 ms
   shape (128, 8960) dtype <f16>
        Torch time: 2.22173 ms
        LLAISYS time: 0.85925 ms
   shape (128, 8960) dtype <bf16>
        Torch time: 2.41520 ms
        LLAISYS time: 0.79054 ms
   shape (1, 12288) dtype <f32>
        Torch time: 0.04483 ms
        LLAISYS time: 0.01212 ms
   shape (1, 12288) dtype <f16>
        Torch time: 0.05603 ms
        LLAISYS time: 0.01586 ms
   shape (1, 12288) dtype <bf16>
        Torch time: 0.06148 ms
        LLAISYS time: 0.01526 ms
   shape (512, 12288) dtype <f32>
        Torch time: 15.95737 ms
        LLAISYS time: 3.50149 ms
   shape (512, 12288) dtype <f16>
        Torch time: 10.01503 ms
        LLAISYS time: 11.07033 ms
   shape (512, 12288) dtype <bf16>
        Torch time: 9.38210 ms
        LLAISYS time: 6.43708 ms
   shape (1, 27648) dtype <f32>
        Torch time: 6.46970 ms
        LLAISYS time: 0.51859 ms
   shape (1, 27648) dtype <f16>
        Torch time: 0.09058 ms
        LLAISYS time: 0.03122 ms
   shape (1, 27648) dtype <bf16>
        Torch time: 0.10892 ms
        LLAISYS time: 0.03298 ms
   shape (1024, 27648) dtype <f32>
        Torch time: 81.70530 ms
        LLAISYS time: 12.51257 ms
   shape (1024, 27648) dtype <f16>
        Torch time: 64.29865 ms
        LLAISYS time: 21.12508 ms
   shape (1024, 27648) dtype <bf16>
        Torch time: 65.57228 ms
        LLAISYS time: 19.52978 ms
Test passed!
```