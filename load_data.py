import os

def prepare_files_info(eeg_path, beh_path):
    """
        Prepare participant file information by matching EEG files with corresponding behavioral and trigger map files.

        This function scans directories for EEG, behavioral, and trigger map files, matches them by participant ID,
        and creates a structured list of participant information dictionaries.

        Args:
            eeg_path (str): Path to the directory containing EEG files. EEG files should be named with
                           participant ID as the base name (e.g., "participant123.bdf").
            beh_path (str): Path to the directory containing behavioral and trigger map files.
                           Behavioral files should start with 'beh' and trigger map files should start
                           with 'triggermap'. Both should follow the naming pattern:
                           prefix_participantID_sex_age[_other].ext

        Returns:
            list: A list of dictionaries, where each dictionary contains information about one participant:
                - ID (str): Participant identifier
                - sex (str): Participant sex (extracted from behavioral filename)
                - age (str): Participant age (extracted from behavioral filename)
                - eeg_file (str): Full path to the EEG file
                - beh_file (str): Full path to the behavioral file
                - trigg_file (str): Full path to the trigger map file

        Notes:
            - Participants with missing or multiple behavioral/trigger map files are skipped with a warning message
            - File matching is based on participant ID extracted from filenames
            - Behavioral filename format expected: beh_ID_sex_age[_other].ext
            - Trigger map filename format expected: triggermap_ID[_other].ext
            - EEG filename format expected: ID.ext

        Example:
            >>> data = prepare_files_info('/path/to/eeg', '/path/to/behavioral')
            >>> print(data[0])
            {
                'ID': 'sub001',
                'sex': 'M',
                'age': '25',
                'eeg_file': '/path/to/eeg/sub001.eeg',
                'beh_file': '/path/to/behavioral/beh_sub001_M_25.csv',
                'trigg_file': '/path/to/behavioral/triggermap_sub001.txt'
            }
        """
    data = []
    eeg_files = os.listdir(eeg_path)
    beh_files = [f for f in os.listdir(beh_path) if f.startswith('beh')]
    trigg_files = [f for f in os.listdir(beh_path) if f.startswith('triggermap')]
    for eeg_file in eeg_files:
        part_id = eeg_file.split(".")[0]

        beh_file = [f for f in beh_files if f.split('_')[1] == part_id]
        trigg_file = [f for f in trigg_files if f.split('_')[1] == part_id]

        if not beh_file:
            print(f"{part_id} - no beh file")
            continue
        if len(beh_file) > 1:
            print(f"{part_id} - multiple beh files")
            continue
        if not trigg_file:
            print(f"{part_id} - no trigger map file")
            continue
        if len(trigg_file) > 1:
            print(f"{part_id} - multiple trigger map files")
            continue

        part_info = {
            "ID": part_id,
            "sex": beh_file[0].split("_")[2],
            "age": beh_file[0].split("_")[3],
            "eeg_file": os.path.join(eeg_path, eeg_file),
            "beh_file": os.path.join(beh_path, beh_file[0]),
            "trigg_file": os.path.join(beh_path, trigg_file[0])
        }
        data.append(part_info)
    return data