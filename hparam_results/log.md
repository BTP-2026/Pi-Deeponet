# Hyperparameter Search Log

Device: cpu  |  Started: 2026-04-12 15:13:57

## Loading data ...

## Phase 1  (17 configs × 2000 epochs)

  ID         Group        lr    bc_d    bc_n      p   n_f     mean%      max%   min_data_loss    time
----------------------------------------------------------------------------------------------------
  Running config 01 (A_weights) ...
  01     A_weights     1e-03     5.0     1.0    512     8   23.9898   34.8842    1.168400e-02    3.1m
  Running config 02 (A_weights) ...
  02     A_weights     1e-03     5.0     5.0    512     8   27.3277   39.7240    1.529300e-02    3.1m
  Running config 03 (A_weights) ...
  03     A_weights     1e-03     5.0    10.0    512     8   39.8335   46.3555    3.200600e-02    3.1m
  Running config 04 (A_weights) ...
  04     A_weights     1e-03    10.0     1.0    512     8   24.0041   34.9055    1.169900e-02    3.1m
  Running config 05 (A_weights) ...
  05     A_weights     1e-03    10.0     5.0    512     8   28.0883   40.3268    1.616800e-02    3.1m
  Running config 06 (A_weights) ...
  06     A_weights     1e-03    10.0    10.0    512     8   39.4793   46.5869    3.146100e-02    3.2m
  Running config 07 (A_weights) ...
  07     A_weights     1e-03    20.0     1.0    512     8   23.9129   34.9685    1.156900e-02    3.1m
  Running config 08 (A_weights) ...
  08     A_weights     1e-03    20.0     5.0    512     8   28.8695   40.7409    1.714300e-02    3.3m
  Running config 09 (A_weights) ...
  09     A_weights     1e-03    20.0    10.0    512     8   40.5692   47.6911    3.325600e-02    3.9m
  Running config 10 (B_lr) ...
  10          B_lr     5e-04    10.0     5.0    512     8   26.2654   38.8770    1.419700e-02    3.9m
  Running config 11 (B_lr) ...
  11          B_lr     2e-04    10.0     5.0    512     8   26.3256   38.7041    1.429900e-02    4.0m
  Running config 12 (B_lr) ...
  12          B_lr     5e-03    10.0     5.0    512     8  395.6697  503.8395    3.910200e-01    3.9m
  Running config 13 (C_arch) ...
  13        C_arch     1e-03    10.0     5.0    256     8   26.5933   39.2454    1.464600e-02    2.8m
  Running config 14 (C_arch) ...
  14        C_arch     1e-03    10.0     5.0    512     8   25.4979   37.2718    1.343500e-02    3.9m
  Running config 15 (C_arch) ...
  15        C_arch     1e-03    10.0     5.0    512     8   25.5610   37.4674    1.345300e-02    2.1m
  Running config 16 (D_fourier) ...
  16     D_fourier     1e-03    10.0     5.0    512     4   26.1213   32.8583    1.308700e-02    3.2m
  Running config 17 (D_fourier) ...
  17     D_fourier     1e-03    10.0     5.0    512    16   44.7483   55.2031    4.142100e-02    3.3m

### Phase 1 — Top 5 configs by mean rel-L2

  #1  config_07  group=A_weights  lr=1e-03  bc_d=20.0  bc_n=1.0  p=512  nf=8  mean=23.9129%  max=34.9685%
  #2  config_01  group=A_weights  lr=1e-03  bc_d=5.0  bc_n=1.0  p=512  nf=8  mean=23.9898%  max=34.8842%
  #3  config_04  group=A_weights  lr=1e-03  bc_d=10.0  bc_n=1.0  p=512  nf=8  mean=24.0041%  max=34.9055%
  #4  config_14  group=C_arch  lr=1e-03  bc_d=10.0  bc_n=5.0  p=512  nf=8  mean=25.4979%  max=37.2718%
  #5  config_15  group=C_arch  lr=1e-03  bc_d=10.0  bc_n=5.0  p=512  nf=8  mean=25.5610%  max=37.4674%

## Phase 2  (Top-3 configs × 15000 epochs)

  ID         Group        lr    bc_d    bc_n      p   n_f     mean%      max%    time
------------------------------------------------------------------------------------------
  Running config 07 (phase2) ...
  07     A_weights     1e-03    20.0     1.0    512     8   19.4637   25.5387   55.7m
  Running config 01 (phase2) ...
  01     A_weights     1e-03     5.0     1.0    512     8   18.3341   24.7198   54.8m
  Running config 04 (phase2) ...
  04     A_weights     1e-03    10.0     1.0    512     8   18.8358   25.1788   55.4m

## Final Best Config

  config_01  group=A_weights
  lr=0.001  bc_d_weight=5.0  bc_n_weight=1.0
  p_dim=512  trunk_h=[512, 512, 512]  n_fourier=8
  Mean rel-L2: 18.3341%  Max rel-L2: 24.7198%

  Best checkpoint: hparam_results/phase2/config_01/model_best.pth

Search complete.  2026-04-12 18:56:02
