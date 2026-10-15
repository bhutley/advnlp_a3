# Round 3 features: acceptance checks

| check | pass | details |
|---|---|---|
| text finbert_tone | yes | sentences_matched=89722, sentences_expected=89722, max_sentence_prob_diff=0.0, call_vectors_recombine=True, max_call_prob_diff=3.650948241507024e-07 |
| text prosus_finbert | yes | sentences_matched=89722, sentences_expected=89722, max_sentence_prob_diff=0.0, call_vectors_recombine=True, max_call_prob_diff=3.509926472800018e-07 |
| prosody | yes | max_rel_diff=4.455774280709474e-16 |
| hubert | yes | build_check=True, full_run=True, stored_means_recombine=True |
| sample | yes | calls=457, firms=232 |
| no_missing_values | yes | columns_with_nan=[] |
