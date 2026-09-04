"""
CholecSeg8k class map (proxy dataset for TIGER SQ-AI pipeline validation).

The CholecSeg8k '*_endo_watershed_mask.png' encodes the class as a single
grayscale code, replicated across R=G=B. The canonical 13-class scheme
(Hong et al., 2020; arXiv:2012.12453) uses these watershed codes:

    code  contiguous_id  name                       rgb_color_mask
    50    0(*bg-merge)   Abdominal Wall             (210,140,140)
    11    1              Liver                      (255,114,114)
    21    2              Grasper / instrument-1     (170,255,0)
    22    3              Connective Tissue          (78,69,124)
    23    4              Blood                      (140,224,222)
    13    5              Fat                        (186,183,75)
    12    6              GI Tract                   (255,160,165)
    31    7              Gallbladder                (231,70,156)
    32    8              Hepatic Vein               (75,183,186)
    33    9              Liver Ligament             (8,72,82)
    24    10             Cystic Duct                (185,255,194) [TINY/rare]
    25    11             L-hook Electrocautery      (45,103,87)  [TINY/rare]
    0/5/255 -> 0        Black Background / watershed-edge artifacts

NOTE on the TIGER mapping (real-data parallel, not used here):
  The challenge masks are *colour* PNGs and the official scorer converts
  RGB -> label_id via an exact-colour lookup (metrics/classes.py
  rgb_mask_to_label_mask). CholecSeg8k's watershed code IS the label,
  so for the proxy we go code -> contiguous id directly. For the REAL
  challenge data the analogous step is RGB-colour -> label_id, and the
  pred->PNG writer must emit EXACT challenge RGB colours (no JPEG).

We merge background (255 black) AND the abdominal-wall-everywhere code 50
into a single id 0 here? -> NO. Abdominal Wall is a real foreground class
in CholecSeg8k and useful; we keep 50 as id 0's *background* only for the
true black/unknown. So we map:
    255 (black bg) -> 0 (background)
    50 (abd wall)  -> 1 ... etc.  (shift everything up by 1; bg=0)

Final 13-class nnU-Net label scheme (id 0 = background):
"""

# watershed_code -> (contiguous_label_id, name)
# id 0 is reserved for background (watershed code 255 black, and stray 0/5).
CHOLEC_MAP = {
    255: (0, "Background"),
    0:   (0, "Background"),      # stray watershed artifact -> bg
    5:   (0, "Background"),      # stray watershed artifact -> bg
    50:  (1, "Abdominal_Wall"),
    11:  (2, "Liver"),
    21:  (3, "Grasper"),
    22:  (4, "Connective_Tissue"),
    23:  (5, "Blood"),
    13:  (6, "Fat"),
    12:  (7, "GI_Tract"),
    31:  (8, "Gallbladder"),
    32:  (9, "Hepatic_Vein"),
    33:  (10, "Liver_Ligament"),
    24:  (11, "Cystic_Duct"),
    25:  (12, "L_hook_Electrocautery"),
}

# id -> name (for nnU-Net dataset.json), ids 0..12.
# NOTE: nnU-Net requires the id-0 key to be the literal lowercase "background".
ID_TO_NAME = {0: "background"}
for code, (lid, name) in CHOLEC_MAP.items():
    if lid != 0:
        ID_TO_NAME[lid] = name
NUM_CLASSES = max(ID_TO_NAME) + 1  # 13


def watershed_to_label(gray):
    """gray: (H,W) uint8 watershed code mask -> (H,W) uint8 contiguous label id.
    Any unknown code maps to background (0)."""
    import numpy as np
    out = np.zeros(gray.shape, dtype=np.uint8)
    for code, (lid, _name) in CHOLEC_MAP.items():
        if lid != 0:
            out[gray == code] = lid
    return out
