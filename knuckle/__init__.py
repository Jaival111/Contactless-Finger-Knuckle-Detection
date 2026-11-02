"""Completely contactless finger knuckle identification (Zhou & Kumar, JSTSP 2023).

Modules
-------
model      : Residual Feature Network (RFN) backbone of Fig. 1 (optionally deformable).
alignment  : Rotation/shift alignment of templates and the MRSD distance, Eqs. (3)-(6).
losses     : RSIL (Eq. 2) plus the TL / soft-shifted TL baselines used in Sec. IV-E.
dataset    : Loading of the knuckle ROI images, splits, triplet / PK sampling.
metrics    : ROC, EER, GAR@FAR and CMC computation and plotting.
detection  : Oriented knuckle detection components of Sec. III (CSL, angular loss, OBB head).
"""
