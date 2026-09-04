"""
station_boundary_classes — maps each of the 14 thoracic LN stations to the set
of T2 (31-class FINE) boundary structures whose visibility defines whether the
station is "in frame", per ANNOTATION_PROTOCOL.md §"Thoracic lymph-node stations".

Protocol rule (Task 3): a station is visible IFF its boundary structures are
visible. So per-class presence/area of these structures in the T2 seg mask are
the natural features for a seg-conditioned T3 head.

Fine class ids (metrics.classes.CLASSES):
  3 Trachea, 4 Right_Main_Bronchus, 5 Left_Main_Bronchus, 6 Esophagus,
  8 Right_Inf_Pulmonary_Lig, 9 Left_Inf_Pulmonary_Lig, 12 Inferior_Pulmonary_Vein,
  13 Right_Subclavian_Artery, 14 Right_Vagal_Nerve, 15 Aorta, 16 Azygos_Vein,
  17 Superior_Caval_Vein, 21 Left_Subclavian_Artery, 22 Right_Bronchial_Artery,
  23 Pulmonary_Artery, 27 Right_Recurrent_Laryngeal_Nerve,
  28 Left_Recurrent_Laryngeal_Nerve, 19 Lymph_Node.
"""

# Station -> list of fine (31-class) label_ids that bound / define it (protocol).
STATION_BOUNDARY_FINE = {
    # 6R/6L Upper paratracheal: esophagus, R/L vagal & recurrent laryngeal nerves,
    # trachea, R/L subclavian arteries, aortic arch.
    "6R": [6, 3, 14, 27, 13, 15],          # right side: R vagal/RLN, R subclavian
    "6L": [6, 3, 28, 21, 15],              # left side:  LRLN, L subclavian, aorta
    # 7R/7L Lower paratracheal: trachea, R/L main bronchi, right vagal nerve, SCV,
    # azygos vein, right bronchial artery, LRLN, aortic arch.
    "7R": [3, 4, 14, 17, 16, 22],          # right: RMB, R vagal, SCV, azygos, RBA
    "7L": [3, 5, 28, 16, 22, 15],          # left:  LMB, LRLN, azygos arch, RBA, aorta
    # 8 Aortopulmonary window: aortic arch (sup), pulmonary artery (ventral), LMB (distal).
    "8":  [15, 23, 5],
    # 9 Subcarinal: caudal to carina, lateral tracheal margins -> trachea + both bronchi.
    "9":  [3, 4, 5],
    # 10R/10L Upper mediastinal paraesophageal: upper esophagus, aperture->bifurcation.
    "10R": [6, 3, 4, 14],
    "10L": [6, 3, 5, 15],
    # 11R/11L Middle mediastinal paraesophageal: middle esophagus, bifurcation->caudal IPV.
    "11R": [6, 4, 12, 16],
    "11L": [6, 5, 12, 15],
    # 12R/12L Lower mediastinal paraesophageal: lower esophagus, caudal IPV -> EGJ.
    "12R": [6, 12, 8],
    "12L": [6, 12, 9],
    # 13R/13L Pulmonary ligament: LNs within R/L inferior pulmonary ligament.
    "13R": [8, 12],
    "13L": [9, 12],
}

# Official station column order (must match metrics.classes_stations.STATION_NAMES).
STATION_ORDER = ["6L", "6R", "7L", "7R", "8", "9",
                 "10L", "10R", "11L", "11R", "12L", "12R", "13L", "13R"]

# Lymph_Node (19) is informative for EVERY station (an exposed LN is a direct
# positive cue) -> add as a shared feature, handled separately in the trainer.
SHARED_FINE = [19]
