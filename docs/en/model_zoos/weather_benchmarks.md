# Weather Prediction Benchmarks

**We provide benchmark results of spatiotemporal prediction learning (STL) methods on the famous weather prediction datasets, WeatherBench. More STL methods will be supported in the future. Issues and PRs are welcome!** Currently, we only provide benchmark results, trained models and logs will be released soon (you can contact us if you require these files).

<details open>
<summary>Currently supported spatiotemporal prediction methods</summary>

- [x] [ConvLSTM](https://arxiv.org/abs/1506.04214) (NeurIPS'2015)
- [x] [PredNet](https://openreview.net/forum?id=B1ewdt9xe) (ICLR'2017)
- [x] [PredRNN](https://dl.acm.org/doi/abs/10.5555/3294771.3294855) (NeurIPS'2017)
- [x] [PredRNN++](https://arxiv.org/abs/1804.06300) (ICML'2018)
- [x] [E3D-LSTM](https://openreview.net/forum?id=B1lKS2AqtX) (ICLR'2018)
- [x] [MIM](https://arxiv.org/abs/1811.07490) (CVPR'2019)
- [x] [CrevNet](https://openreview.net/forum?id=B1lKS2AqtX) (ICLR'2020)
- [x] [PhyDNet](https://arxiv.org/abs/2003.01460) (CVPR'2020)
- [x] [MAU](https://openreview.net/forum?id=qwtfY-3ibt7) (NeurIPS'2021)
- [x] [PredRNN.V2](https://arxiv.org/abs/2103.09504v4) (TPAMI'2022)
- [x] [SimVP](https://arxiv.org/abs/2206.05099) (CVPR'2022)
- [x] [SimVP.V2](https://arxiv.org/abs/2211.12509) (ArXiv'2022)
- [x] [TAU](https://arxiv.org/abs/2206.12126) (CVPR'2023)
- [x] [DMVFN](https://arxiv.org/abs/2303.09875) (CVPR'2023)

</details>

<details open>
<summary>Currently supported MetaFormer models for SimVP</summary>

- [x] [ViT](https://arxiv.org/abs/2010.11929) (ICLR'2021)
- [x] [Swin-Transformer](https://arxiv.org/abs/2103.14030) (ICCV'2021)
- [x] [MLP-Mixer](https://arxiv.org/abs/2105.01601) (NeurIPS'2021)
- [x] [ConvMixer](https://arxiv.org/abs/2201.09792) (Openreview'2021)
- [x] [UniFormer](https://arxiv.org/abs/2201.09450) (ICLR'2022)
- [x] [PoolFormer](https://arxiv.org/abs/2111.11418) (CVPR'2022)
- [x] [ConvNeXt](https://arxiv.org/abs/2201.03545) (CVPR'2022)
- [x] [VAN](https://arxiv.org/abs/2202.09741) (ArXiv'2022)
- [x] [IncepU (SimVP.V1)](https://arxiv.org/abs/2206.05099) (CVPR'2022)
- [x] [gSTA (SimVP.V2)](https://arxiv.org/abs/2211.12509) (ArXiv'2022)
- [x] [HorNet](https://arxiv.org/abs/2207.14284) (NeurIPS'2022)
- [x] [MogaNet](https://arxiv.org/abs/2211.03295) (ArXiv'2022)

</details>


## WeatherBench Benchmarks

We provide temperature prediction benchmark results on the popular [WeatherBench](https://arxiv.org/abs/2002.00469) dataset (temperature prediction `t2m`) using $12\rightarrow 12$ frames prediction setting. Metrics (MSE, MAE, SSIM, pSNR) of the the best models are reported in three trials. Parameters (M), FLOPs (G), and V100 inference FPS (s) are also reported for all methods. All methods are trained by Adam optimizer with Cosine Annealing scheduler (no warmup and min lr is 1e-6).

### **STL Benchmarks on Temperature (t2m)**

Similar to [Moving MNIST Benchmarks](video_benchmarks.md#moving-mnist-benchmarks), we benchmark popular Metaformer architectures on [SimVP](https://arxiv.org/abs/2211.12509) training 50 epochs with **single GPU** on `t2m` (K). We provide config files in [configs/weather/t2m_5_625](https://github.com/chengtan9907/OpenSTL/configs/weather/t2m_5_625/) for `5.625` settings ($32\times 64$ resolutions).

| Method           |  Setting | Params | FLOPs |  FPS |  MSE  |   MAE  |  RMSE |   Download   |
|------------------|:--------:|:------:|:-----:|:----:|:-----:|:------:|:-----:|:------------:|
| ConvLSTM         | 50 epoch | 14.98M |  136G |   46 | 1.521 | 0.7949 | 1.233 | model \| log |
| PhyDNet          | 50 epoch |  3.09M | 36.8G |  177 | 285.9 | 8.7370 | 16.91 | model \| log |
| PredRNN          | 50 epoch | 23.57M |  278G |   22 | 1.331 | 0.7246 | 1.154 | model \| log |
| PredRNN++        | 50 epoch | 38.31M |  413G |   15 | 1.634 | 0.7883 | 1.278 | model \| log |
| MIM              | 50 epoch | 37.75M |  109G |  126 | 1.784 | 0.8716 | 1.336 | model \| log |
| MAU              | 50 epoch |  5.46M | 39.6G |  237 | 1.251 | 0.7036 | 1.119 | model \| log |
| PredRNNv2        | 50 epoch | 23.59M |  279G |   22 | 1.545 | 0.7986 | 1.243 | model \| log |
| IncepU (SimVPv1) | 50 epoch | 14.67M | 8.03G |  160 | 1.238 | 0.7037 | 1.113 | model \| log |
| gSTA (SimVPv2)   | 50 epoch | 12.76M | 7.01G |  504 | 1.105 | 0.6567 | 1.051 | model \| log |
| ViT              | 50 epoch | 12.41M | 7.99G |  432 | 1.146 | 0.6712 | 1.070 | model \| log |
| Swin Transformer | 50 epoch | 12.42M | 6.88G |  581 | 1.143 | 0.6735 | 1.069 | model \| log |
| Uniformer        | 50 epoch | 12.02M | 7.45G |  465 | 1.204 | 0.6885 | 1.097 | model \| log |
| MLP-Mixer        | 50 epoch | 11.10M | 5.92G |  713 | 1.255 | 0.7011 | 1.119 | model \| log |
| ConvMixer        | 50 epoch |  1.13M | 0.95G | 1705 | 1.267 | 0.7073 | 1.126 | model \| log |
| Poolformer       | 50 epoch |  9.98M | 5.61G |  722 | 1.156 | 0.6715 | 1.075 | model \| log |
| ConvNeXt         | 50 epoch | 10.09M | 5.66G |  689 | 1.277 | 0.7220 | 1.130 | model \| log |
| VAN              | 50 epoch | 12.15M | 6.70G |  523 | 1.150 | 0.6803 | 1.072 | model \| log |
| HorNet           | 50 epoch | 12.42M | 6.84G |  517 | 1.201 | 0.6906 | 1.096 | model \| log |
| MogaNet          | 50 epoch | 12.76M | 7.01G |  416 | 1.152 | 0.6665 | 1.073 | model \| log |
| TAU              | 50 epoch | 12.22M | 6.70G |  511 | 1.162 | 0.6707 | 1.078 | model \| log |

Then, we also provide the high-resolution benchmark of `t2m` using the similar training settings with **4GPUs** (4xbs4). The config files are in [configs/weather/t2m_1_40625](https://github.com/chengtan9907/OpenSTL/configs/weather/t2m_1_40625/) for `1.40625` settings ($128\times 256$ resolutions).

| Method           |  Setting | Params | FLOPs | FPS |  MSE  |   MAE  |  RMSE |   Download   |
|------------------|:--------:|:------:|:-----:|:---:|:-----:|:------:|:-----:|:------------:|
| ConvLSTM         | 50 epoch | 15.08M |  550G |  35 | 1.0625 | 0.6517 | 1.031 | model \| log |
| PhyDNet          | 50 epoch |  3.09M |  148G |  41 | 297.34 | 8.9788 | 17.243 | model \| log |
| PredRNN          | 50 epoch | 23.84M | 1123G |   3 | 0.8966 | 0.5869 | 0.9469 | model \| log |
| PredRNN++        | 50 epoch | 38.58M | 1663G |   2 | 0.8538 | 0.5708 | 0.9240 | model \| log |
| MIM              | 50 epoch | 42.17M | 1739G |  11 | 1.2138 | 0.6857 | 1.1017 | model \| log |
| MAU              | 50 epoch | 11.82M |  172G |  17 | 1.0031 | 0.6316 | 1.0016 | model \| log |
| PredRNNv2        | 50 epoch | 23.86M | 1129G |   3 | 1.0451 | 0.6190 | 1.0223 | model \| log |
| IncepU (SimVPv1) | 50 epoch | 14.67M |  128G |  27 | 0.8492 | 0.5636 | 0.9215 | model \| log |
| gSTA (SimVPv2)   | 50 epoch | 12.76M |  112G |  33 | 0.6499 | 0.4909 | 0.8062 | model \| log |
| ViT              | 50 epoch | 12.48M | 36.8G |  50 | 0.8969 | 0.5834 | 0.9470 | model \| log |
| Swin Transformer | 50 epoch | 12.42M |  110G |  38 | 0.7606 | 0.5193 | 0.8721 | model \| log |
| Uniformer        | 50 epoch | 12.09M | 48.8G |  57 | 1.0052 | 0.6294 | 1.0026 | model \| log |
| MLP-Mixer        | 50 epoch | 27.87M | 94.7G |  49 | 1.1865 | 0.6593 | 1.0893 | model \| log |
| ConvMixer        | 50 epoch |  1.14M | 15.1G | 117 | 0.8557 | 0.5669 | 0.9250 | model \| log |
| Poolformer       | 50 epoch |  9.98M | 89.7G |  42 | 0.7983 | 0.5316 | 0.8935 | model \| log |
| ConvNeXt         | 50 epoch | 10.09M | 90.5G |  47 | 0.8058 | 0.5406 | 0.8976 | model \| log |
| VAN              | 50 epoch | 12.15M |  107G |  34 | 0.7110 | 0.5094 | 0.8432 | model \| log |
| HorNet           | 50 epoch | 12.42M |  109G |  34 | 0.8250 | 0.5467 | 0.9083 | model \| log |
| MogaNet          | 50 epoch | 12.76M |  112G |  27 | 0.7517 | 0.5232 | 0.8670 | model \| log |
| TAU              | 50 epoch | 12.29M | 36.1G |  94 | 0.8316 | 0.5615 | 0.9119 | model \| log |

### **STL Benchmarks on Humidity (r)**

Similar to [Moving MNIST Benchmarks](video_benchmarks.md#moving-mnist-benchmarks), we benchmark popular Metaformer architectures on [SimVP](https://arxiv.org/abs/2211.12509) training 50 epochs with **single GPU** on `r` (%). We provide config files in [configs/weather/r_5_625](https://github.com/chengtan9907/OpenSTL/configs/weather/r_5_625/) for `5.625` settings ($32\times 64$ resolutions).

| Method           |  Setting | Params | FLOPs |  FPS |  MSE   |   MAE  |  RMSE |   Download   |
|------------------|:--------:|:------:|:-----:|:----:|:------:|:------:|:-----:|:------------:|
| ConvLSTM         | 50 epoch | 14.98M |  136G |   46 | 35.146 |  4.012 | 5.928 | model \| log |
| PhyDNet          | 50 epoch |  3.09M | 36.8G |  177 | 239.00 |  8.975 | 15.46 | model \| log |
| PredRNN          | 50 epoch | 23.57M |  278G |   22 | 37.611 |  4.096 | 6.133 | model \| log |
| PredRNN++        | 50 epoch | 38.31M |  413G |   15 | 35.146 |  4.012 | 5.928 | model \| log |
| MIM              | 50 epoch | 37.75M |  109G |  126 | | | | model \| log |
| MAU              | 50 epoch |  5.46M | 39.6G |  237 | 34.529 |  4.004 | 5.876 | model \| log |
| PredRNNv2        | 50 epoch | 23.59M |  279G |   22 | 36.508 |  4.087 | 6.042 | model \| log |
| IncepU (SimVPv1) | 50 epoch | 14.67M | 8.03G |  160 | 34.355 |  3.994 | 5.861 | model \| log |
| gSTA (SimVPv2)   | 50 epoch | 12.76M | 7.01G |  504 | 31.426 |  3.765 | 5.606 | model \| log |
| ViT              | 50 epoch | 12.41M | 7.99G |  432 | 32.616 |  3.852 | 5.711 | model \| log |
| Swin Transformer | 50 epoch | 12.42M | 6.88G |  581 | 31.332 |  3.776 | 5.597 | model \| log |
| Uniformer        | 50 epoch | 12.02M | 7.45G |  465 | 32.199 |  3.864 | 5.674 | model \| log |
| MLP-Mixer        | 50 epoch | 11.10M | 5.92G |  713 | 34.467 |  3.950 | 5.871 | model \| log |
| ConvMixer        | 50 epoch |  1.13M | 0.95G | 1705 | 32.829 |  3.909 | 5.730 | model \| log |
| Poolformer       | 50 epoch |  9.98M | 5.61G |  722 | 31.989 |  3.803 | 5.656 | model \| log |
| ConvNeXt         | 50 epoch | 10.09M | 5.66G |  689 | 33.179 |  3.928 | 5.760 | model \| log |
| VAN              | 50 epoch | 12.15M | 6.70G |  523 | 31.712 |  3.812 | 5.631 | model \| log |
| HorNet           | 50 epoch | 12.42M | 6.84G |  517 | 32.081 |  3.826 | 5.664 | model \| log |
| MogaNet          | 50 epoch | 12.76M | 7.01G |  416 | 31.795 |  3.816 | 5.639 | model \| log |
| TAU              | 50 epoch | 12.22M | 6.70G |  511 | 31.831 |  3.818 | 5.642 | model \| log |

### **STL Benchmarks on Wind Component (uv10)**

Similar to [Moving MNIST Benchmarks](video_benchmarks.md#moving-mnist-benchmarks), we benchmark popular Metaformer architectures on [SimVP](https://arxiv.org/abs/2211.12509) training 50 epochs with **single GPU** on `uv10` (ms-1). We provide config files in [configs/weather/uv10_5_625](https://github.com/chengtan9907/OpenSTL/configs/weather/uv10_5_625/) for `5.625` settings ($32\times 64$ resolutions). Notice that the input data of `uv10` has two channels.

| Method           |  Setting | Params | FLOPs |  FPS |   MSE  |   MAE  |  RMSE  |   Download   |
|------------------|:--------:|:------:|:-----:|:----:|:------:|:------:|:------:|:------------:|
| ConvLSTM         | 50 epoch | 14.98M |  136G |   43 | 1.8976 | 0.9215 | 1.3775 | model \| log |
| PhyDNet          | 50 epoch |  3.09M | 36.8G |  172 | 16.798 | 2.9208 | 4.0986 | model \| log |
| PredRNN          | 50 epoch | 23.65M |  279G |   21 | 1.8810 | 0.9068 | 1.3715 | model \| log |
| PredRNN++        | 50 epoch | 38.40M |  414G |   14 | 1.8727 | 0.9019 | 1.3685 | model \| log |
| MIM              | 50 epoch | 37.75M |  109G |  122 | 3.1399 | 1.1837 | 1.7720 | model \| log |
| MAU              | 50 epoch |  5.46M | 39.6G |  233 | 1.9001 | 0.9194 | 1.3784 | model \| log |
| PredRNNv2        | 50 epoch | 23.68M |  280G |   21 | 2.0072 | 0.9413 | 1.4168 | model \| log |
| IncepU (SimVPv1) | 50 epoch | 14.67M | 8.04G |  154 | 1.9993 | 0.9510 | 1.4140 | model \| log |
| gSTA (SimVPv2)   | 50 epoch | 12.76M | 7.02G |  498 | 1.5069 | 0.8142 | 1.2276 | model \| log |
| ViT              | 50 epoch | 12.42M |  8.0G |  427 | 1.6262 | 0.8438 | 1.2752 | model \| log |
| Swin Transformer | 50 epoch | 12.42M | 6.89G |  577 | 1.4996 | 0.8145 | 1.2246 | model \| log |
| Uniformer        | 50 epoch | 12.03M | 7.46G |  459 | 1.4850 | 0.8085 | 1.2186 | model \| log |
| MLP-Mixer        | 50 epoch | 11.10M | 5.93G |  707 | 1.6066 | 0.8395 | 1.2675 | model \| log |
| ConvMixer        | 50 epoch |  1.14M | 0.96G | 1698 | 1.7067 | 0.8714 | 1.3064 | model \| log |
| Poolformer       | 50 epoch |  9.99M | 5.62G |  717 | 1.6123 | 0.8410 | 1.2698 | model \| log |
| ConvNeXt         | 50 epoch | 10.09M | 5.67G |  682 | 1.6914 | 0.8698 | 1.3006 | model \| log |
| VAN              | 50 epoch | 12.15M | 6.71G |  520 | 1.5958 | 0.8371 | 1.2632 | model \| log |
| HorNet           | 50 epoch | 12.42M | 6.85G |  513 | 1.5539 | 0.8254 | 1.2466 | model \| log |
| MogaNet          | 50 epoch | 12.76M | 7.01G |  411 | 1.6072 | 0.8451 | 1.2678 | model \| log |
| TAU              | 50 epoch | 12.22M | 6.70G |  505 | 1.5925 | 0.8426 | 1.2619 | model \| log |

### **STL Benchmarks on Cloud Cover (tcc)**

Similar to [Moving MNIST Benchmarks](video_benchmarks.md#moving-mnist-benchmarks), we benchmark popular Metaformer architectures on [SimVP](https://arxiv.org/abs/2211.12509) training 50 epochs with **single GPU** on `tcc` (%). We provide config files in [configs/weather/tcc_5_625](https://github.com/chengtan9907/OpenSTL/configs/weather/tcc_5_625/) for `5.625` settings ($32\times 64$ resolutions).

| Method           |  Setting | Params | FLOPs |  FPS |   MSE   |    MAE  |   RMSE  |   Download   |
|------------------|:--------:|:------:|:-----:|:----:|:-------:|:-------:|:-------:|:------------:|
| ConvLSTM         | 50 epoch | 14.98M |  136G |   46 | 0.04944 | 0.15419 | 0.22234 | model \| log |
| PhyDNet          | 50 epoch |  3.09M | 36.8G |  172 | 0.09913 | 0.22614 | 0.31485 | model \| log |
| PredRNN          | 50 epoch | 23.57M |  278G |   22 | 0.05504 | 0.15877 | 0.23461 | model \| log |
| PredRNN++        | 50 epoch | 38.31M |  413G |   15 | 0.05479 | 0.15435 | 0.23407 | model \| log |
| MIM              | 50 epoch | 37.75M |  109G |  126 | | | | model \| log |
| MAU              | 50 epoch |  5.46M | 39.6G |  237 | 0.04955 | 0.15158 | 0.22260 | model \| log |
| PredRNNv2        | 50 epoch | 23.59M |  279G |   22 | 0.05051 | 0.15867 | 0.22475 | model \| log |
| IncepU (SimVPv1) | 50 epoch | 14.67M | 8.03G |  160 | 0.04765 | 0.15029 | 0.21829 | model \| log |
| gSTA (SimVPv2)   | 50 epoch | 12.76M | 7.01G |  504 | 0.04657 | 0.14688 | 0.21580 | model \| log |
| ViT              | 50 epoch | 12.41M | 7.99G |  432 | 0.04778 | 0.15026 | 0.21859 | model \| log |
| Swin Transformer | 50 epoch | 12.42M | 6.88G |  581 | 0.04639 | 0.14729 | 0.21539 | model \| log |
| Uniformer        | 50 epoch | 12.02M | 7.45G |  465 | 0.04680 | 0.14777 | 0.21634 | model \| log |
| MLP-Mixer        | 50 epoch | 11.10M | 5.92G |  713 | 0.04925 | 0.15264 | 0.22192 | model \| log |
| ConvMixer        | 50 epoch |  1.13M | 0.95G | 1705 | 0.04717 | 0.14874 | 0.21718 | model \| log |
| Poolformer       | 50 epoch |  9.98M | 5.61G |  722 | 0.04694 | 0.14884 | 0.21667 | model \| log |
| ConvNeXt         | 50 epoch | 10.09M | 5.66G |  689 | 0.04742 | 0.14867 | 0.21775 | model \| log |
| VAN              | 50 epoch | 12.15M | 6.70G |  523 | 0.04694 | 0.14725 | 0.21665 | model \| log |
| HorNet           | 50 epoch | 12.42M | 6.84G |  517 | 0.04692 | 0.14751 | 0.21661 | model \| log |
| MogaNet          | 50 epoch | 12.76M | 7.01G |  416 | 0.04699 | 0.14802 | 0.21676 | model \| log |
| TAU              | 50 epoch | 12.22M | 6.70G |  511 | 0.04723 | 0.14604 | 0.21733 | model \| log |

<p align="right">(<a href="#top">back to top</a>)</p>
