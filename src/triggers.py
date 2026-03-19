import mne
import numpy as np

from src.utils import write


def set_annotations_from_trigger_file(raw, trigg_df, write_info=True):
    """
    Sets raw.annotations based on:
    - EEG events (mne.find_events)
    - trigger file trigg_file (CSV with columns: trigger_no, trigger_type, acc, n, block_type)

    Annotation descriptions have the format: "<block_type>_<trigger_type>_<n>_<acc>".
    """

    # Create annotations
    events = mne.find_events(raw, min_duration=0.002)
    if len(events) == 0:
        write("Error: No events found!", write_info)
        return None
    event_desc = {e: str(e - 65280) for e in np.unique(events[:, 2])}  # biosemi adds 65280 to triggers values
    raw.set_annotations(mne.annotations_from_events(events, raw.info['sfreq'], event_desc))

    # Check length consistency
    n_events = len(events)
    n_triggs = len(trigg_df)
    if n_events != n_triggs:
        write(f"Error: n_events ({n_events}) != n_trigs ({n_triggs})", write_info)
        return None

    # Check trigger code consistency
    if not np.all(raw.annotations.description.astype(int) == trigg_df['trigger_no'].to_numpy()):
        write("Error: EEG trigger numbers != trigger_no from file; check matching!", write_info)
        return None

    # Build descriptions: "<block_type>_<trigger_type>_<n>_<acc>"
    descriptions = [
        f"{str(row.block_type).strip()}_"
        f"{str(row.trigger_type).strip()}_"
        f"{int(row.n)}_"
        f"{int(row.acc)}"
        for _, row in trigg_df.iterrows()
    ]
    raw.annotations.description = descriptions

    return raw


def drop_training_annotations(raw):
    """
    Remove all annotations whose description starts with 'training'.
    """

    desc = np.asarray(raw.annotations.description, dtype=str)
    keep = ~np.char.startswith(desc, "training")  # Mask: True for annotations to KEEP

    onsets = np.asarray(raw.annotations.onset)[keep]
    durations = np.asarray(raw.annotations.duration)[keep]
    new_desc = desc[keep]

    new_annots = mne.Annotations(
        onset=onsets,
        duration=durations,
        description=new_desc,
        orig_time=raw.annotations.orig_time,
    )

    raw.set_annotations(new_annots)
    return raw
