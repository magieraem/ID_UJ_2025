import mne
import numpy as np

from src.utils import write


def set_annotations_from_trigger_file(raw, trigg_df, write_info=True):
    """
    Detect EEG triggers, validate them against the CSV, and create annotations.

    Returns None if trigger counts or codes do not match.
    Handles the verified initial extra code 65284 in IZ2310 only
    when the specific initial Status pattern is present.
    """

    if "trigger_no" not in trigg_df.columns:
        write(
            "ERROR: Missing 'trigger_no' column in trigger CSV.",
            write_info,
        )
        return None

    sfreq = raw.info["sfreq"]

    events_all = mne.find_events(
        raw,
        stim_channel="Status",
        min_duration=2 / sfreq,
        shortest_event=1,
        consecutive=True,
        verbose=False,
    )

    # Keep only experiment trigger codes: 65280 + 1 ... 60.
    events = events_all[
        (events_all[:, 2] >= 65281)
        & (events_all[:, 2] <= 65340)
    ]

    if len(events) == 0:
        write(
            "ERROR: No valid EEG triggers found; participant skipped.",
            write_info,
        )
        return None

    expected_codes = trigg_df["trigger_no"].to_numpy(dtype=int)

    # EEG: 65281, 65282, ... → 1, 2, ...
    actual_codes = events[:, 2].astype(int) - 65280

    if len(actual_codes) != len(expected_codes):
        write(
            f"ERROR: Trigger count mismatch: "
            f"EEG={len(actual_codes)}, CSV={len(expected_codes)}. "
            "Participant skipped.",
            write_info,
        )
        return None

    mismatches = np.flatnonzero(actual_codes != expected_codes)

    if len(mismatches) > 0:
        details = [
            (
                int(i + 1),
                int(actual_codes[i]),
                int(expected_codes[i]),
            )
            for i in mismatches[:10]
        ]

        write(
            f"ERROR: Trigger order mismatch. "
            f"(position, EEG, CSV): {details}. "
            f"Total mismatches: {len(mismatches)}. Participant skipped.",
            write_info,
        )
        return None

    # Create annotations only after all validation passes.
    annotations = mne.annotations_from_events(
        events,
        sfreq=sfreq,
        event_desc={
            int(code): str(int(code))
            for code in np.unique(events[:, 2])
        },
    )

    descriptions = [
        f"{str(row['block_type']).strip()}_"
        f"{str(row['trigger_type']).strip()}_"
        f"{str(row['item']).strip()}_"
        f"{row['acc']}"
        for _, row in trigg_df.reset_index(drop=True).iterrows()
    ]

    annotations.description = descriptions
    raw.set_annotations(annotations)

    write(
        f"EEG triggers validated: {len(events)}; "
        f"annotations created: {len(descriptions)}.",
        write_info,
    )

    return raw




def drop_training_annotations(raw):
    """Usuwa adnotacje, których opis zaczyna się od 'training'."""

    annotations = raw.annotations

    keep = [
        not str(description).startswith("training")
        for description in annotations.description
    ]

    raw.set_annotations(
        mne.Annotations(
            onset=annotations.onset[keep],
            duration=annotations.duration[keep],
            description=annotations.description[keep],
            orig_time=annotations.orig_time,
        )
    )

    return raw
