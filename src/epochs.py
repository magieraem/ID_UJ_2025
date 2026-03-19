import mne
import numpy as np

from src.utils import write


def create_epochs(raw, trigger_type, tmin=0.0, tmax=2.0, baseline=None, write_info=True):
    """
    Create epochs from raw EEG data based on annotations matching a trigger prefix.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data with annotations.
    trigger_type : str
        Prefix of annotation names to use as triggers (e.g., "experiment_matric").
    tmin : float, optional
        Start time of the epoch relative to the trigger, in seconds. Default is 0.0.
    tmax : float, optional
        End time of the epoch relative to the trigger, in seconds. Default is 2.0.
    baseline: None or tuple of length 2
        The time interval to consider as “baseline” when applying baseline correction.
    write_info : bool, optional
        Whether to write information messages. Default is True.

    Returns
    -------
    epochs : mne.Epochs or None
        Epochs object containing the segmented data, or None if no matching triggers found.

    Notes
    -----
    - Only EEG channels are included in the epochs.
    - The function automatically adjusts tmax by one sample to ensure correct epoch length.
    """

    events, event_id = mne.events_from_annotations(raw)

    # Select only triggers starting with the specified prefix
    exp_event_id = {name: code for name, code in event_id.items() if name.startswith(trigger_type)}

    if not exp_event_id:
        write(f"Error: No '{trigger_type}' triggers found", write_info)
        return None

    # Filter events array to selected triggers
    mask = np.isin(events[:, 2], list(exp_event_id.values()))
    events_sel = events[mask]

    # Create epochs; tmax is inclusive, so subtract one sample
    epochs = mne.Epochs(
        raw,
        events_sel,
        event_id=exp_event_id,
        tmin=tmin,
        tmax=tmax - 1.0 / raw.info['sfreq'],
        reject_by_annotation=False,
        baseline=baseline,
        picks="eeg",
        preload=True,
        verbose=False
    )

    write(f"Created {len(epochs)} epochs ({tmax - tmin:.3f} s) for '{trigger_type}'", write_info)
    return epochs
