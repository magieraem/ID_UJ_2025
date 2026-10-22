import mne
from mne.preprocessing import EOGRegression
from tqdm import tqdm
from pyprep.find_noisy_channels import NoisyChannels
import os
import sys
import numpy as np
import pandas as pd
from autoreject import AutoReject

from src.load_data import prepare_files_info
from src.utils import write
from src.triggers import set_annotations_from_trigger_file, drop_training_annotations
from src.epochs import create_epochs

# matplotlib.use('TkAgg')
mne.set_log_level('ERROR')

#JS2812.bdf to small to be analysed

# Paths to data folders
BEH_PATH = "/mnt/data/ociepka_idUJ/BEH/Raven_APM_EEG/results" # Path to the folder with behavioral data files.
EEG_PATH = "/mnt/data/ociepka_idUJ/BDF/2_Raven" # Path to the folder with raw EEG data files.
RESULT_PATH = "/mnt/data/ociepka_idUJ/PREPROCESSING/Raven/PREPROCESSED_1st" # Path to the folder where preprocessed data will be saved.

# Electrode montage settings
MONTAGE = "biosemi64" # Name of the EEG electrode montage used in the recording (e.g., "biosemi64").

# Signal cropping settings
START_CUT_EXTRA_TIME = 2 # Extra time (in seconds) kept before the first event/trigger.
END_CUT_EXTRA_TIME = 2 # Extra time (in seconds) kept after the last event/trigger.

# Filtering settings
LOW_PASS = 0.5 # High‑pass filter cutoff frequency in Hz (removes very slow drifts).
HIGH_PASS = 48 # Low‑pass filter cutoff frequency in Hz (removes high‑frequency noise).

# Bad channel detection (PyPREP)
RANSAC = True # Whether to use the RANSAC algorithm for robust bad channel detection.
SEED = 42 # Random seed to ensure reproducible results.

# Epoching settings
EPOCHS_LEN = [0, 30] # Epoch time window [tmin, tmax] in seconds relative to event onset.
TRIGGER_TYPE = "show_task" # Type/name of triggers used to create epochs.

# AutoReject settings (automatic artifact rejection for epochs)
INTERPOLATION = [1, 2, 3, 4] # Number of channels to interpolate when repairing bad epochs.
CONSENSUS = None # Consensus percentage for rejecting epochs (None = let AutoReject choose).
METHOD = 'bayesian_optimization' # Strategy for optimizing rejection thresholds.
N_JOBS = 1 # Number of parallel jobs used by AutoReject.
CV = 10 # Number of cross‑validation folds used by AutoReject.

# Output and visualization settings
SAVE_DATA = True # If True, save preprocessed epochs to disk as FIF files.
OVERWRITE_DATA = False
WRITE_INFO = True # If True, print textual progress and summary information.
PLOT_INFO = False # If True, show diagnostic plots (e.g., rejected epochs, reject_log).

DATA = prepare_files_info(eeg_path=EEG_PATH, beh_path=BEH_PATH)

EXCLUDED_IDS = {"JS2812"}


for part in tqdm(DATA):
    if part["ID"] in EXCLUDED_IDS:
        msg = f"{part['ID']}: excluded — invalid BDF file"
        print(msg)
        write(msg, WRITE_INFO)
        continue
    
    file_name = f"{part['ID']}_{part['sex']}_{part['age']}_preprocessed-epo.fif"
    output_path = os.path.join(RESULT_PATH, file_name)
    
    if not OVERWRITE_DATA and os.path.isfile(output_path):
        msg = f"{part['ID']}: output already exists — skipped"
        print(msg)
        write(msg, WRITE_INFO)
        continue
    
    write(part["ID"], WRITE_INFO)

    # --- load eeg ---
    try:
        raw = mne.io.read_raw(
            part["eeg_file"],
            preload=True,
            verbose=False
        )
    except Exception as e:
        msg = (
            f"EEG file could not be read; participant skipped. "
            f"ID={part['ID']}; file={part['eeg_file']}; "
            f"error={type(e).__name__}: {e}"
        )
        print(msg)
        write(msg, WRITE_INFO)
        continue

    trigg_df = pd.read_csv(part['trigg_file'], comment='#')

    # --- montage ---
    eog_channels = {f'EXG{i}': 'eog' for i in range(1, 5) if f'EXG{i}' in raw.ch_names}
    if eog_channels:
        raw.set_channel_types(eog_channels)
    drop_channels = [f'EXG{i}' for i in range(5, 9) if f'EXG{i}' in raw.ch_names]
    if drop_channels:
        raw.drop_channels(drop_channels)
    raw.set_montage(MONTAGE, on_missing='warn')

    # --- change triggers to annotation and rename them
    raw = set_annotations_from_trigger_file(raw, trigg_df, write_info=WRITE_INFO)
    
    if raw is None:
        sys.stdout.flush() # for correct order of printed text when using tqdm
        continue

    # --- remove training ---
    raw = drop_training_annotations(raw)

    # --- crop signal ---
    times = raw.annotations.onset
    start_cut = max(0, times[0] - START_CUT_EXTRA_TIME)
    end_cut = min(raw.times[-1], times[-1] + END_CUT_EXTRA_TIME)
    raw.crop(start_cut, end_cut)
    #raw.annotations.onset -= start_cut

    # --- filter ---
    raw.filter(l_freq=LOW_PASS, h_freq=HIGH_PASS, picks="eeg", verbose=False)

    # --- find and mark bad channels ---
    nc = NoisyChannels(raw, ransac=RANSAC, do_detrend=False, random_state=SEED)
    nc.find_all_bads()
    raw.info['bads'] = nc.get_bads()
    write(f"bad channels: {raw.info['bads']}", WRITE_INFO)

    # --- set reference ---
    raw.set_eeg_reference("average", verbose=False)

    # --- remove ocular artifacts ---
    model_plain = EOGRegression(picks="eeg", exclude="bads", picks_artifact="eog").fit(raw)
    raw = model_plain.apply(raw, copy=True)

    # --- interpolate bad channels ---
    if len(raw.info['bads']) > 0:
        raw.interpolate_bads(reset_bads=True, verbose=False)
        write(f"interpolated {len(nc.get_bads())} channels", WRITE_INFO)
    
    # --- create epochs ---
    epochs_all = create_epochs(raw, trigger_type=TRIGGER_TYPE, tmin=EPOCHS_LEN[0],
                               tmax=EPOCHS_LEN[1], write_info=WRITE_INFO)
    
    if epochs_all is None:
        continue
    
    epochs = epochs_all.copy()

    # --- Plot EEG before AutoReject ---
    if PLOT_INFO:
        epochs.plot(
            n_epochs=10,
            n_channels=len(epochs.ch_names),
            scalings=dict(eeg=50e-6),
            title=f"EEG after preprocessing — {part['ID']}",
        )

    epochs_original = epochs.copy()

        # --- remove epochs with artifacts ---
    ar = AutoReject(
            n_interpolate=[2],
            consensus=CONSENSUS,
            thresh_method=METHOD,
            n_jobs=N_JOBS,
            cv = CV,
            random_state=SEED,
            verbose=False)

    epochs, reject_log = ar.fit_transform(epochs, return_log=True)
    
    write(f"Number of epochs: {len(epochs)}", WRITE_INFO)
    write(f"Number of rejected epochs: {reject_log.bad_epochs.sum()}", WRITE_INFO)
    
    if PLOT_INFO:
            epochs.plot(
                n_epochs=10,
                n_channels=len(epochs.ch_names),
                scalings=dict(eeg=50e-6),
                title=f"EEG after AR — {part['ID']}",
            )

    # --- Plot removed epochs ---
    if PLOT_INFO:
        print("bad epochs")
        reject_log.plot('horizontal')
        bad_epochs_idx = np.where(reject_log.bad_epochs)[0] # Bad epoch index
        
        if 0 < len(bad_epochs_idx) < 20:
                epochs_original[bad_epochs_idx].plot(
                    n_epochs=len(bad_epochs_idx),
                    n_channels=64,
                    scalings='auto',
                    title='Removed epochs')

        # --- Save data ---
    if SAVE_DATA:
        result_dir = f"{RESULT_PATH}"
        os.makedirs(result_dir, exist_ok=True)

        output_file = os.path.join(result_dir, file_name)
        epochs.save(output_file, verbose=False, overwrite=False)

        write(f"Data saved to {output_file}", WRITE_INFO)

        sys.stdout.flush() # for correct order of printed text when using tqdm
