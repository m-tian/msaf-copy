"""Config for the Structural Features algorithm."""

# Serra params
config = {
    "M_gaussian"    : 20,
    "m_embedded"    : 3,
    "k_nearest"     : 0.04,
    "Mp_adaptive"   : 28,
    "offset_thres"  : 0.05
    # "M_gaussian"    : 8,
    # "m_embedded"    : 3,
    # "k_nearest"     : 0.04,
    # "Mp_adaptive"   : 20,
    # "offset_thres"  : 0.03

    # Orig config in msaf
    #"M_gaussian"    : 23,
    #"m_embedded"    : 3,
    #"k_nearest"     : 0.04,
    #"Mp_adaptive"   : 28,
    #"offset_thres"  : 0.05


}

algo_id = "sf"
is_boundary_type = True
is_label_type = False
