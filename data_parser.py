import os
import numpy as np
import pandas as pd
import wfdb
import h5py


def get_file_list(dir_name='/TRAINING'):
    header_loc, signal_loc, sig_name = [], [], []

    for root, dirs, file_names in sorted(os.walk(dir_name)):
        for name in file_names:
            if '.hea' in name:
                header_loc.append(os.path.join(root, name))

            if '.mat' in name and 'arousal' not in name:
                signal_loc.append(os.path.join(root, name))
                sig_name.append(os.path.splitext(name)[0])

    data_tmp = {'header': header_loc,
                'signal': signal_loc,
                'name': sig_name}

    return pd.DataFrame(data=data_tmp)


sleep_stages = {'W': 1, 'N1': 2, 'N2': 3, 'N3': 4, 'R': 5}
arousal_dict = {'(arousal_bruxism': 1,
                '(arousal_noise': 2,
                '(arousal_plm': 3,
                '(arousal_rera': 4,
                '(arousal_snore': 5,
                '(arousal_spontaneous': 6,
                '(resp_centralapnea': 7,
                '(resp_cheynestokesbreath': 8,
                '(resp_hypopnea': 9,
                '(resp_hypoventilation': 10,
                '(resp_mixedapnea': 11,
                '(resp_obstructiveapnea': 12,
                '(resp_partialobstructive': 13,
                }


def parse_annotations(ann, N):
    sleep_idx = []
    arousal_idx = []

    for (sample, stype) in zip(ann.sample, ann.aux_note):
        if (stype == 'W') | (stype == 'N1') | (stype == 'N2') | (stype == 'N3') | (stype == 'R'):
            sleep_idx.append((sample, stype))
        else:
            arousal_idx.append((sample, stype))

    sleep_ann = np.zeros(N)
    for i in range(0, len(sleep_idx[:-1])):
        sleep_ann[sleep_idx[i][0]:sleep_idx[i+1][0]] = sleep_stages[sleep_idx[i][1]]
    sleep_ann[sleep_idx[-1][0]:] = sleep_stages[sleep_idx[-1][1]]

    arousal_ann = np.zeros(N)
    for start, end in zip(arousal_idx[::2], arousal_idx[1::2]):
        arousal_ann[start[0]:end[0]] = arousal_dict[start[1]]

    return sleep_ann, arousal_ann


def get_sample_data(file, istest=False):

    record_name = os.path.splitext(file['signal'])[0]
    signals, fields = wfdb.rdsamp(record_name)

    if istest is False:
        ann = wfdb.rdann(record_name, 'arousal')
        sleep_ann, arousal_ann = parse_annotations(ann, np.shape(signals)[0])
        fields['sig_name'].append('sleep_label')
        fields['sig_name'].append('arousal_label')

        with h5py.File(record_name + "-arousal.mat", 'r') as f:
            target_ann = np.array(f["data/arousals"]).T[0]
        fields['sig_name'].append('target_label')

        return pd.DataFrame(np.c_[signals, sleep_ann, arousal_ann, target_ann], columns=fields['sig_name'])

    else:
        return pd.DataFrame(signals, columns=fields['sig_name'])
