Dataset provenance
==================

MVTec AD
--------
Official provider: MVTec Software GmbH
Page: https://www.mvtec.com/research-teaching/datasets/mvtec-ad
License: CC BY-NC-SA 4.0 (per the official dataset page).
Used categories: all 15 official categories.
Expected layout (MVTec-style):
  <root>/<category>/train/good/*.png
  <root>/<category>/test/<defect>/*.png
  <root>/<category>/ground_truth/<defect>/*.png
The dataset images are not redistributed in this package. Obtain them from the official
provider and follow its license.

VisA
----
Official provider: Amazon Science, SPot-the-Difference (VisA) release.
Page: https://github.com/amazon-science/spot-diff
Dataset license: CC BY 4.0 (per the official project README).
The project's source code is separately Apache-2.0; the dataset license is not the source-code
license. The dataset images are not redistributed here.
Used categories: all 12 official one-class categories.
The experiments used a read-only MVTec-style adapter that preserved the official train/test
assignments and binary defect masks. The adapter is part of the project code, not of this
minimal package; the runner documents the expected paths and split rule.
The dataset images are not redistributed in this package. Obtain them from the official
provider and follow its license.

Reference protocol
------------------
For each category, train/good paths were sorted deterministically using Python Path ordering;
the first 50 images were retained. All 27 categories supplied at least 50 references, giving
9,800 patches per category. Test images, defect images, and ground-truth masks were excluded
from fitting, reference normalization, parameter selection, and model selection. Masks were
used only for evaluation.

Public benchmark caveat
-----------------------
MVTec AD and VisA are public benchmarks, not a random sample of industrial operating
conditions. The manuscript does not claim factory data, deployment, contamination robustness,
category routing, process drift, or maintenance behavior.
