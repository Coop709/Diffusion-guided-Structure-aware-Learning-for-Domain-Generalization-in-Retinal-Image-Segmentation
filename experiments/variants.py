"""Table 9 variants; the random-Gaussian row corresponds to BSL+SAM."""

VARIANTS = {
    "gaussian": {"label": "Random Gaussian initialization", "cue_conditioning": False},
    "sam_only": {"label": "SAM-mask-only initialization", "cue_conditioning": False},
    "channel_only": {"label": "Channel-guidance-only initialization", "cue_conditioning": False},
    "additional_condition": {"label": "Both cues as additional conditions + Gaussian initialization",
                             "cue_conditioning": True},
    "joint": {"label": "Joint initialization (Ours)", "cue_conditioning": False},
}
