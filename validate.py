import mirdata
import soundfile as sf  # audio loading
import numpy as np


def check_alignment(data_home, tolerance=0.1):
    """
    Checks for every track whether audio duration matches
    the annotation end time within tolerance.
    
    Args:
        data_home: path to your Beatles dataset directory
        tolerance: allowed difference in seconds
    """

    dataset = mirdata.initialize("beatles", data_home=data_home)
    #dataset.validate()

    pass_list = []
    fail_list = []

    print("Checking alignment for all tracks...\n")

    for track_id in dataset.track_ids:
        track = dataset.track(track_id)

        # --- Load audio ---
        try:
            audio, sr = track.audio
            audio_duration = len(audio) / sr
        except Exception as e:
            fail_list.append((track_id, f"Audio load error: {e}"))
            continue

        # --- Load chord intervals ---
        try:
            chord_data = track.chords
            intervals = chord_data.intervals
            labels = chord_data.labels
        except Exception as e:
            fail_list.append((track_id, f"Chord annotation error: {e}"))
            continue

        # End time of last annotation
        annotation_end = intervals[-1][1]

        # Difference
        diff = abs(audio_duration - annotation_end)

        if diff <= tolerance:
            pass_list.append(track_id)
            print(f"[PASS] {track_id}  | Δ={diff:.3f}s")
        else:
            fail_list.append((track_id, f"Duration mismatch (Δ={diff:.3f}s)"))
            print(f"[FAIL] {track_id}  | Δ={diff:.3f}s")

    # --- Summary ---
    print("\n--------------------------------------------------")
    print("                Alignment Summary                 ")
    print("--------------------------------------------------")
    print(f"Total tracks: {len(dataset.track_ids)}")
    print(f"Passed: {len(pass_list)}")
    print(f"Failed: {len(fail_list)}")

    if fail_list:
        print("\nMisaligned tracks:")
        for tid, reason in fail_list:
            print(f" - {tid}: {reason}")

    return pass_list, fail_list


if __name__ == "__main__":
    check_alignment("/home/dwawid/projects/acr/mir_datasets2/beatles")
