import os
import numpy as np
import pandas as pd
import mne

# Import DATA from the module where you define it.
# Adjust this import to match your project structure.
from preprocessing_mod_raven import DATA

OUTPUT_DIR = "trigger_validation"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def check_participant(part):
    participant_id = part["ID"]
    print(f"Checking {participant_id}...")

    result = {
        "ID": participant_id,
        "status": "OK",
        "n_eeg_events": None,
        "n_csv_rows": None,
        "n_bdf_annotations": None,
        "n_unknown_codes": None,
        "count_match": None,
        "details": "",
    }

    try:
        # Read the original EEG without loading all signal data into RAM.
        raw = mne.io.read_raw(
            part["eeg_file"],
            preload=False,
            verbose=False,
        )

        trigg_df = pd.read_csv(
            part["trigg_file"],
            comment="#",
        )

        result["n_csv_rows"] = len(trigg_df)
        result["n_bdf_annotations"] = len(raw.annotations)

        # A. Inspect numerical events detected from the stim channel.
        events = mne.find_events(
            raw,
            min_duration=0,
            verbose=False,
        )

        result["n_eeg_events"] = len(events)
        result["count_match"] = len(events) == len(trigg_df)

        events_df = pd.DataFrame({
            "event_no": np.arange(1, len(events) + 1),
            "sample": events[:, 0],
            "time_sec": events[:, 0] / raw.info["sfreq"],
            "trigger_code": events[:, 2],
        })

        # B. Inspect annotations already stored in the BDF.
        annotations_df = pd.DataFrame({
            "onset_sec": raw.annotations.onset,
            "duration_sec": raw.annotations.duration,
            "description": raw.annotations.description,
        })

        # Save individual reports.
        events_df.to_csv(
            os.path.join(OUTPUT_DIR, f"{participant_id}_events.csv"),
            index=False,
        )

        annotations_df.to_csv(
            os.path.join(
                OUTPUT_DIR,
                f"{participant_id}_bdf_annotations.csv",
            ),
            index=False,
        )

        # C. Record the numerical codes found in this participant.
        codes = events_df["trigger_code"].value_counts().sort_index()

        result["n_unknown_codes"] = None  # Set after loading the trigger map.

        result["details"] = (
            f"EEG codes: {codes.to_dict()}"
        )

        if not result["count_match"]:
            result["status"] = "COUNT_MISMATCH"

        print(
            f"  EEG events: {len(events)} | "
            f"CSV rows: {len(trigg_df)} | "
            f"BDF annotations: {len(raw.annotations)}"
        )

    except Exception as exc:
        result["status"] = "ERROR"
        result["details"] = repr(exc)
        print(f"  ERROR: {exc}")

    return result


def main():
    summary = []

    for part in DATA:
        summary.append(check_participant(part))

    summary_df = pd.DataFrame(summary)

    summary_df.to_csv(
        os.path.join(OUTPUT_DIR, "summary.csv"),
        index=False,
    )

    print("\n--- Validation summary ---")
    print(summary_df.to_string(index=False))
    print(f"\nReports saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
