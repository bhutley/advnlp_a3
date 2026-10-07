# Audio acceptance checks

## 1. Counts (must pass)

Failed clips must be exactly the dataset's empty files (1: Ventas Inc_20171027/Robert Probst_1_8.mp3). `n_bad` counts clips that are not `audio_ok` (failed, or implausibly fast for their transcript line); `n_bad_calls` counts calls with `audio_call_ok` false.

| features | n_calls | n_clips | n_failed | n_bad | n_bad_calls | runtime_s | pass |
|---|---|---|---|---|---|---|---|
| praat | 572 | 89722 | 1 | 78 | 2 | 1515.7 | yes |
| hubert | 572 | 89722 | 1 | 78 | 2 | 3197.2 | yes |
| wav2vec2 | 572 | 89722 | 1 | 78 | 2 | 3177.1 | yes |

## 2. Decoded vs ffprobe duration (must pass)

30 random calls × 3 clips: worst within-call range 0.0001 s (limit 0.002), largest |offset| 0.048 s (limit 0.2); per-call offsets seen: [-0.048, -0.024, -0.022, -0.012, -0.011]. Pass: yes. ffprobe's container duration includes the MP3 encoder delay and padding, which the decoder trims; its length in seconds depends on the call's sample rate.

## 3. MPS vs CPU, all 13 layers (must pass, tolerance 0.01)

| model | n | max_abs_diff | pass |
|---|---|---|---|
| hubert | 50 | 7.15e-06 | yes |
| wav2vec2 | 50 | 9.66e-06 | yes |

## 4. Codec noise floor (reported)

A synthetic 120 Hz voiced signal, measured before and after the dataset's lowest-quality common encoding (11,025 Hz, 16 kbps MP3, used by 92 calls; most calls are 44.1 kHz, 64 kbps), so this is a worst case. Expected raw jitter for ±1% uniform period perturbation is about 0.0067.

| signal | stage | f0_mean_hz | jitter_local | shimmer_local | hnr_db |
|---|---|---|---|---|---|
| steady pulses | raw 16 kHz | 119.9999 | 0.0001 | 0.0004 | 39.7589 |
| steady pulses | after MP3 | 119.9991 | 0.0004 | 0.0141 | 27.7036 |
| ±1% period jitter | raw 16 kHz | 119.9194 | 0.0066 | 0.0074 | 21.8780 |
| ±1% period jitter | after MP3 | 119.9188 | 0.0060 | 0.0212 | 18.2319 |

## 5. Clipping (reported)

1000 random clips: 8.5% contain at least one sample at |x| ≥ 0.999; 0.004% of all samples are clipped.

## 5b. Clip and call quality (reported)

78 of 89722 clips are not `audio_ok` (excluded from call aggregates). Calls with `audio_call_ok` false (over 10% of clips bad), to be excluded from audio models:

| event_id | n_sentences | n_bad |
|---|---|---|
| BorgWarner_20170727 | 49 | 42 |
| eBay Inc._20170419 | 5 | 20 |

## 6. Plausibility (reported)

Call-level F0: min 95, median 137, max 232 Hz; 0 calls outside 85–255 Hz. Median speech rate 2.52 words/s (pauses included); median voiced fraction 0.53.

## 7. Cross-model similarity (reported)

RSA between HuBERT and wav2vec2 call vectors at layer 6: Spearman 0.746 over 572 calls.

## 8. First look at the targets (reported only)

Spearman correlation of Praat call features with event-study targets over `ok` events. A sanity check, not a basis for choosing features or models.

| feature | target | n | spearman | p |
|---|---|---|---|---|
| f0_mean_hz | vol_3 | 490 | -0.005 | 0.907 |
| f0_mean_hz | vol_30 | 490 | -0.077 | 0.089 |
| f0_mean_hz | car_mm_0_1 | 490 | 0.041 | 0.367 |
| f0_mean_hz | car_mm_0_5 | 490 | 0.035 | 0.433 |
| f0_sd_st | vol_3 | 490 | -0.044 | 0.328 |
| f0_sd_st | vol_30 | 490 | 0.010 | 0.822 |
| f0_sd_st | car_mm_0_1 | 490 | -0.036 | 0.433 |
| f0_sd_st | car_mm_0_5 | 490 | -0.033 | 0.460 |
| intensity_sd_db | vol_3 | 490 | -0.031 | 0.495 |
| intensity_sd_db | vol_30 | 490 | -0.036 | 0.422 |
| intensity_sd_db | car_mm_0_1 | 490 | 0.011 | 0.810 |
| intensity_sd_db | car_mm_0_5 | 490 | 0.010 | 0.830 |
| speech_rate | vol_3 | 490 | 0.009 | 0.848 |
| speech_rate | vol_30 | 490 | 0.021 | 0.641 |
| speech_rate | car_mm_0_1 | 490 | -0.023 | 0.618 |
| speech_rate | car_mm_0_5 | 490 | -0.051 | 0.256 |
| voiced_frac | vol_3 | 490 | -0.005 | 0.915 |
| voiced_frac | vol_30 | 490 | -0.005 | 0.918 |
| voiced_frac | car_mm_0_1 | 490 | -0.048 | 0.289 |
| voiced_frac | car_mm_0_5 | 490 | -0.067 | 0.141 |
| jitter_local | vol_3 | 490 | 0.037 | 0.409 |
| jitter_local | vol_30 | 490 | 0.085 | 0.060 |
| jitter_local | car_mm_0_1 | 490 | 0.010 | 0.825 |
| jitter_local | car_mm_0_5 | 490 | 0.007 | 0.877 |
| shimmer_local | vol_3 | 490 | -0.045 | 0.325 |
| shimmer_local | vol_30 | 490 | -0.025 | 0.580 |
| shimmer_local | car_mm_0_1 | 490 | -0.003 | 0.953 |
| shimmer_local | car_mm_0_5 | 490 | 0.004 | 0.925 |
| hnr_db | vol_3 | 490 | -0.003 | 0.944 |
| hnr_db | vol_30 | 490 | -0.045 | 0.324 |
| hnr_db | car_mm_0_1 | 490 | -0.008 | 0.864 |
| hnr_db | car_mm_0_5 | 490 | -0.014 | 0.762 |

## 9. Recording format (reported)

Clips come in several formats; format is constant within a call (2 calls contain a clip in another format).

| format | calls |
|---|---|
| 44100 Hz / 64 kbps | 336 |
| 22050 Hz / 32 kbps | 104 |
| 11025 Hz / 16 kbps | 92 |
| 48000 Hz / 64 kbps | 16 |
| 32000 Hz / 48 kbps | 11 |
| 24000 Hz / 32 kbps | 8 |
| 44100 Hz / 128 kbps stereo | 4 |
| 8000 Hz / 8 kbps | 1 |

Praat call features and targets by format, for the 3 largest groups (Kruskal–Wallis p across them; `audio_call_ok` calls only):

| feature | 44100 Hz / 64 kbps | 22050 Hz / 32 kbps | 11025 Hz / 16 kbps | p |
|---|---|---|---|---|
| f0_mean_hz | 137.419 | 140.089 | 135.979 | 0.296 |
| f0_sd_st | 3.791 | 3.966 | 4.027 | 0.256 |
| intensity_sd_db | 13.210 | 13.436 | 13.191 | 0.890 |
| speech_rate | 2.527 | 2.513 | 2.477 | 0.140 |
| voiced_frac | 0.535 | 0.549 | 0.521 | 0.040 |
| jitter_local | 0.024 | 0.025 | 0.025 | 0.002 |
| shimmer_local | 0.142 | 0.142 | 0.154 | 0.000 |
| hnr_db | 8.177 | 8.414 | 6.859 | 0.000 |
| target: vol_3 | -3.984 | -4.038 | -4.064 | 0.092 |
| target: vol_30 | -4.385 | -4.470 | -4.494 | 0.026 |
| target: car_mm_0_1 | 0.001 | 0.006 | 0.007 | 0.168 |
| target: car_mm_0_5 | -0.002 | 0.006 | 0.002 | 0.388 |

How well the format can be predicted from SSL call vectors (leave-one-out nearest centroid on standardised vectors, same groups):

| model | layer | n | loo_accuracy | majority_rate |
|---|---|---|---|---|
| hubert | 6 | 530 | 0.84 | 0.63 |
| wav2vec2 | 6 | 530 | 0.86 | 0.63 |

Format therefore affects voice-quality features and SSL embeddings. Include it as a covariate (or fixed effect) in audio models and report results by format.
