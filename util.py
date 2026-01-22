import mir_eval.chord as C


# ---------------------------------------
# Utility: extract root, bass, and bitmap
# ---------------------------------------
def parse_chord(label):
    """
    Returns (root_pc, bass_pc, bitmap) for the chord.
    root_pc and bass_pc are integers 0–11.
    bitmap is a tuple of 12 {0,1} intervals relative to root.
    """
    if label == C.NO_CHORD:
        return None, None, None

    # mir_eval.chord.split returns 4 fields:
    root, quality, intervals, bass = C.split(label)

    # Convert root → pitch class (integer 0–11)
    root_pc = C.pitch_class_to_semitone(root)

    # Convert quality → bitmap
    bitmap = C.quality_to_bitmap(quality).tolist()
    bitmap = tuple(bitmap)

    # Bass logic
    if bass is None:
        bass_pc = root_pc
    else:
        try:
            bass_pc = C.pitch_class_to_semitone(bass)
        except Exception:
            # fallback to root if parsing fails
            bass_pc = root_pc

    return root_pc, bass_pc, bitmap


# ---------------------------------------
# Define Major/Minor Vocabulary
# ---------------------------------------
def generate_majmin_vocab():
    """
    Returns list of canonical labels in maj/min vocabulary:
       C:maj, C:min, ..., B:maj, B:min, N
    """
    pcs = C.PITCH_CLASSES  # ['C', 'C#', ..., 'B']
    vocab = ["N"]
    for p in pcs:
        vocab.append(f"{p}:maj")
        vocab.append(f"{p}:min")
    return vocab


MAJMIN_VOCAB = generate_majmin_vocab()


# Precompute vocab interval structures for fast comparison
def build_vocab_interval_sets():
    vocab_map = {}
    for v in MAJMIN_VOCAB:
        if v == "N":
            vocab_map[v] = (None, None, None)
            continue
        root_pc, bass_pc, bitmap = parse_chord(v)
        vocab_map[v] = (root_pc, bass_pc, bitmap)
    return vocab_map


MAJMIN_INTERVAL_MAP = build_vocab_interval_sets()


# ---------------------------------------
# Mapping function (the important part)
# ---------------------------------------
def map_to_majmin(label):
    """
    Map any chord label to the MIREX maj/min vocabulary
    following the Pauwels & Peeters (2013) rule.

    Returns:
        - a maj/min chord label (e.g. "G:maj")
        - or "N"
        - or None if it cannot be mapped
    """

    # no-chord case
    if label == C.NO_CHORD:
        return "N"

    # parse input
    root_pc, bass_pc, bitmap = parse_chord(label)
    if bitmap is None:
        return "N"

    candidates = []

    for vocab_label, (v_root, v_bass, v_bitmap) in MAJMIN_INTERVAL_MAP.items():

        if vocab_label == "N":
            continue

        # 1) root must match
        if v_root != root_pc:
            continue

        # 2) bass must match
        if v_bass != bass_pc:
            continue

        # 3) intervals(vocab) ⊆ intervals(input)
        # bitmap is tuple of 12 ints
        if all((v_bitmap[i] == 0 or bitmap[i] == 1) for i in range(12)):
            # store number of intervals to pick largest later
            interval_count = sum(v_bitmap)
            candidates.append((interval_count, vocab_label))

    if not candidates:
        return None  # cannot be mapped

    # 4) pick the candidate with the largest interval set
    candidates.sort(reverse=True)  # biggest first
    return candidates[0][1]
