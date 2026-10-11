# Third-party notices

The baseline models in `src/models/baselines/` and the backbones in `src/models/backbones/` were re-implemented
for a common forecasting protocol (same splits, inputs, data-quality mask and quantile head). The files below
follow, or are adapted from, the official implementations listed here. Each file states in its header how it
differs from the original. The licenses of the original projects apply to the adapted parts.

| File | Original project | License |
|---|---|---|
| `src/models/baselines/pag.py`, `src/models/pseudo_samples.py` | PAG, https://github.com/IntelligentSystemsLab/ST-EVCDP | MIT License, Copyright (c) 2023 IntelligentSystemsLab |
| `src/models/baselines/piast.py` | PIAST, https://github.com/kuanghx3/PIAST | Apache License 2.0 (modified: ported to the common protocol; changes are described in the file header) |
| `src/models/baselines/urbanev.py` (FCNN, LSTM, GCN, GCN-LSTM) | UrbanEV, https://github.com/IntelligentSystemsLab/UrbanEV | CC0 1.0 Universal |
| `src/models/baselines/stgcn.py` | STGCN, https://github.com/VeritasYin/STGCN_IJCAI-18 | BSD 2-Clause License, Copyright (c) 2017-Present, Haoteng Yin and PKUAI26@PKU |
| `src/models/baselines/agcrn.py` | AGCRN, https://github.com/LeiBAI/AGCRN | MIT License, Copyright (c) 2024 LeiBAI |
| `src/models/backbones/graph_wavenet.py` | Graph WaveNet, https://github.com/nnzhan/Graph-WaveNet | MIT License, Copyright (c) 2019 Zonghan Wu (re-implemented as an encoder; tensor layout as in the original) |
| `src/models/baselines/astgcn.py` | ASTGCN (Guo et al., AAAI 2019) | re-implemented from the paper (parameter initialisation as in the official code); no code copied |
| `src/models/backbones/staeformer.py` | STAEformer (Liu et al., CIKM 2023) | simplified re-implementation from the paper; no code copied |

The full texts of the MIT, BSD 2-Clause and Apache 2.0 licenses are available at https://opensource.org/licenses.

## MIT License (ST-EVCDP, AGCRN, Graph WaveNet)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above
copyright notice and this permission notice shall be included in all copies or substantial portions of the
Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.

## BSD 2-Clause License (STGCN)

Redistribution and use in source and binary forms, with or without modification, are permitted provided that
the following conditions are met: (1) Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer. (2) Redistributions in binary form must reproduce the
above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other
materials provided with the distribution. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES ARE DISCLAIMED.

## Data

UrbanEV (Li et al., Scientific Data 12:523, 2025) is dedicated to the public domain under CC0 1.0 Universal.
The data are not included in this repository.
