# Third-party notices

This repository is licensed under the Apache License 2.0 (see `LICENSE`). It contains code derived from the
following projects, which keep their own licenses.

## SynMVCrowd

- Source: https://github.com/zqyq/SynMVCrowd
- License: Apache License 2.0 (same text as `LICENSE`)
- Reference: Q. Zhang, D. Chen, Y. Gong, H. Huang. *SynMVCrowd: A Large Synthetic Benchmark for Multi-view Crowd
  Counting and Localization.* International Journal of Computer Vision 134, 191 (2026).
  https://doi.org/10.1007/s11263-026-02778-3

Derived files, modified for this thesis (reformatted, adapted to this code base's data pipeline and training loop):

- `losses/ot_loss/genloss_synmvcrowd.py` (port of the baseline's OT loss), `losses/ot_loss/genloss.py`,
  `losses/ot_loss/Cost.py`
- the baseline re-implementation (view selection, homography projector, decoder) in `models/neck/baseline/`,
  `models/head/baseline/` and `training_scripts/train_baseline.py`, and the evaluation protocol in
  `evaluation/evaluate_paper_protocol.py`, which follow the released SynMVCrowd code

## GeomLoss

- Source: https://github.com/jeanfeydy/geomloss (vendored copy in `losses/ot_loss/geomloss/`, as shipped with the
  SynMVCrowd code; reformatted)
- License: MIT

```
MIT License

Copyright (c) 2019 Jean Feydy

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## torchvision

- Source: https://github.com/pytorch/vision (the ResNet definition in `models/backbone/resnet.py`, as shipped with
  the SynMVCrowd code; trimmed to ResNet-18 and reformatted)
- License: BSD 3-Clause

```
BSD 3-Clause License

Copyright (c) Soumith Chintala 2016,
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## Datasets

No dataset is included. SynMVCrowd, Wildtrack and MultiviewX are downloaded separately (README section 2) and are
subject to their own terms of use.
