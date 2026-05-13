from pathlib import Path

import librosa
import numpy as np
import pandas as pd
try:
    from natsort import natsorted
except ImportError:
    def natsorted(items):
        return sorted(items, key=lambda item: str(item))

# Input files
ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = ROOT / "data" / "raw"
MIXED_AUDIO_FOLDER = RAW_DATA_DIR / "mixed"
HEART_AUDIO_FOLDER = RAW_DATA_DIR / "heart"
LUNG_AUDIO_FOLDER = RAW_DATA_DIR / "lung"
MIXED_LABEL_CSV = RAW_DATA_DIR / "Mix.csv"
HEART_LABEL_CSV = RAW_DATA_DIR / "HS.csv"
LUNG_LABEL_CSV = RAW_DATA_DIR / "LS.csv"
PROCESSED_DATA_DIR = ROOT / "data" / "processed"

# Output csv names and folder
OUT_HEART = "heart_only.csv"
OUT_LUNG = "lungs_only.csv"
OUT_MIXED_HEART = "mixed_without_sliding_window.csv"
OUT_MIXED_LUNG = "mixed_with_sliding_window.csv"

# Constants to help ensure our data is collected correctly
N_MFCC = 13
WINDOW_SECONDS = 2.0
SPIKE_PERCENTILE = 75
MIN_SECONDS_BETWEEN_SPIKES = 1.0
RMS_FRAME_SECONDS = 0.05
RMS_HOP_SECONDS = 0.01

HEART_LABEL = "Heart Sound Type"
LUNG_LABEL = "Lung Sound Type"


# Helper function to create the mean and standard deviation columns
def add_mean_std(row, name, values):
    row[name + "_mean"] = float(np.mean(values))
    row[name + "_std"] = float(np.std(values))


# Using the librosa library, we extract features from the audio files
def extract_features(y, sr):
    row = {}

    if len(y) == 0:
        y = np.zeros(int(sr * 0.1))

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)

    # Commonly, the first 13 (Mel-Frequency Cepstral Coefficients) are used in
    # projects like this. This helps summarize the shape and texture of the audio
    for i in range(N_MFCC):
        feature_name = "mfcc_" + str(i + 1)
        add_mean_std(row, feature_name, mfccs[i])

    # Delta MFCCs capture how the spectrum changes over time — heart beats are
    # sharp transients while breath sounds are slower and more sustained, so
    # deltas help distinguish them.
    mfcc_delta = librosa.feature.delta(mfccs)
    for i in range(N_MFCC):
        add_mean_std(row, f"mfcc_{i + 1}_delta", mfcc_delta[i])

    # Gathering other features to ensure our model has as much information as possible
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)

    add_mean_std(row, "centroid", centroid)
    add_mean_std(row, "bandwidth", bandwidth)
    add_mean_std(row, "rolloff", rolloff)
    add_mean_std(row, "zcr", zcr)
    add_mean_std(row, "rms", rms)

    # Spectral contrast measures peak-vs-valley differences per frequency band,
    # which helps distinguish harmonic murmurs from noise-like normal sounds.
    # fmin=100, n_bands=4 keeps all bands within the 2000 Hz Nyquist at sr=4000.
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, fmin=100, n_bands=4)
    for i in range(contrast.shape[0]):
        add_mean_std(row, f"contrast_{i + 1}", contrast[i])

    return row


# Function when using sliding window to ensure the best spikes are chosen
def get_spike_windows(y, sr):
    window_size = int(WINDOW_SECONDS * sr)
    frame_size = int(RMS_FRAME_SECONDS * sr)
    hop_size = int(RMS_HOP_SECONDS * sr)
    min_distance = int(MIN_SECONDS_BETWEEN_SPIKES * sr)

    if frame_size < 1:
        frame_size = 1

    if hop_size < 1:
        hop_size = 1

    # Pad short files so we always have at least one full window
    if len(y) < window_size:
        pad_amount = window_size - len(y)
        y = np.pad(y, (0, pad_amount))

    rms_values = librosa.feature.rms(y=y, frame_length=frame_size, hop_length=hop_size)[
        0
    ]

    nonzero = rms_values[rms_values > 0]

    if len(nonzero) == 0:
        centers = [len(y) // 2]
    else:
        cutoff = np.percentile(nonzero, SPIKE_PERCENTILE)
        active_frames = np.where(rms_values >= cutoff)[0]

        groups = []
        current_group = []

        for frame in active_frames:
            if len(current_group) == 0:
                current_group.append(frame)
            elif frame == current_group[-1] + 1:
                current_group.append(frame)
            else:
                groups.append(current_group)
                current_group = [frame]

        if len(current_group) > 0:
            groups.append(current_group)

        centers = []

        for group in groups:
            loudest_frame = group[0]

            for frame in group:
                if rms_values[frame] > rms_values[loudest_frame]:
                    loudest_frame = frame

            center = loudest_frame * hop_size
            centers.append(center)

    # Remove centers that are too close together
    filtered = []

    for center in centers:
        if len(filtered) == 0:
            filtered.append(center)
        else:
            last_center = filtered[-1]

            if center - last_center >= min_distance:
                filtered.append(center)

    if len(filtered) == 0:
        filtered.append(len(y) // 2)

    windows = []
    half = window_size // 2

    for center in filtered:
        start = center - half

        if start < 0:
            start = 0

        if start + window_size > len(y):
            start = len(y) - window_size

        end = start + window_size
        window = y[start:end]

        windows.append(window)

    return windows


# Combine all data into one row when not using the sliding window
def make_row(path, file_id, label, diagnosis_columns, audio=None, sr=None):
    if audio is None or sr is None:
        audio, sr = librosa.load(path, sr=None, mono=True)

    row = {}
    row["id"] = file_id

    for column in diagnosis_columns:
        row[column] = label[column]

    features = extract_features(audio, sr)

    for key in features:
        row[key] = features[key]

    return row


# Combine all data into one row when using the sliding window
def make_window_rows(path, file_id, label, diagnosis_columns):
    y, sr = librosa.load(path, sr=None, mono=True)
    windows = get_spike_windows(y, sr)

    rows = []

    for window in windows:
        row = make_row(path, file_id, label, diagnosis_columns, window, sr)
        rows.append(row)

    return rows


# Create features folder if needed and save csvs
def save_csv(rows, filename, output_dir=PROCESSED_DATA_DIR):
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)

    print("Saved", len(rows), "rows to", output_path)


def read_label_csv(path):
    df = pd.read_csv(path)

    for column in df.columns:
        if df[column].dtype == "object":
            df[column] = df[column].str.strip()

    return df


def normalize_lung_file_id(file_id):
    return file_id.replace("_C_", "_FC_").replace("_G_", "_CC_")


def list_wav_files(audio_dir):
    if not audio_dir.exists():
        return []

    files = []

    for path in audio_dir.iterdir():
        if path.is_file() and path.suffix.lower() == ".wav":
            files.append(path)

    return natsorted(files)


def build_lookup(label_df, id_column, normalize_id=None):
    lookup = {}

    for _, label in label_df.iterrows():
        file_id = str(label[id_column]).strip()

        if normalize_id is not None:
            file_id = normalize_id(file_id)

        lookup[file_id] = label.to_dict()

    return lookup


def build_feature_tables(
    mixed_audio_dir=MIXED_AUDIO_FOLDER,
    heart_audio_dir=HEART_AUDIO_FOLDER,
    lung_audio_dir=LUNG_AUDIO_FOLDER,
    mixed_labels_csv=MIXED_LABEL_CSV,
    heart_labels_csv=HEART_LABEL_CSV,
    lung_labels_csv=LUNG_LABEL_CSV,
    output_dir=PROCESSED_DATA_DIR,
):
    mixed_label_df = read_label_csv(mixed_labels_csv)

    mixed_labels = {
        "H": build_lookup(mixed_label_df, "Heart Sound ID"),
        "L": build_lookup(mixed_label_df, "Lung Sound ID"),
        "M": build_lookup(mixed_label_df, "Mixed Sound ID"),
    }

    heart_labels = {}
    if heart_labels_csv.exists():
        heart_label_df = read_label_csv(heart_labels_csv)
        heart_labels = build_lookup(heart_label_df, "Heart Sound ID")

    lung_labels = {}
    if lung_labels_csv.exists():
        lung_label_df = read_label_csv(lung_labels_csv)
        lung_labels = build_lookup(
            lung_label_df, "Lung Sound ID", normalize_id=normalize_lung_file_id
        )

    heart_rows = []
    lung_rows = []
    mixed_heart_rows = []
    mixed_lung_rows = []

    for path in list_wav_files(mixed_audio_dir):
        file_id = path.stem.strip()
        prefix = file_id[0].upper()
        label = mixed_labels.get(prefix, {}).get(file_id)

        if label is None:
            print("Skipping file with no label:", path.name)
            continue

        print("Processing", path.name)

        if prefix == "H":
            heart_rows.append(make_row(path, file_id, label, [HEART_LABEL]))

        elif prefix == "L":
            lung_rows.extend(make_window_rows(path, file_id, label, [LUNG_LABEL]))

        elif prefix == "M":
            mixed_heart_rows.append(
                make_row(path, file_id, label, [HEART_LABEL, LUNG_LABEL])
            )
            mixed_lung_rows.extend(
                make_window_rows(path, file_id, label, [HEART_LABEL, LUNG_LABEL])
            )

    for path in list_wav_files(heart_audio_dir):
        file_id = path.stem.strip()
        label = heart_labels.get(file_id)

        if label is None:
            print("Skipping file with no label:", path.name)
            continue

        print("Processing", path.name)
        heart_rows.append(make_row(path, file_id, label, [HEART_LABEL]))

    for path in list_wav_files(lung_audio_dir):
        file_id = path.stem.strip()
        label = lung_labels.get(file_id)

        if label is None:
            print("Skipping file with no label:", path.name)
            continue

        print("Processing", path.name)
        lung_rows.extend(make_window_rows(path, file_id, label, [LUNG_LABEL]))

    save_csv(heart_rows, OUT_HEART, output_dir)
    save_csv(lung_rows, OUT_LUNG, output_dir)
    save_csv(mixed_heart_rows, OUT_MIXED_HEART, output_dir)
    save_csv(mixed_lung_rows, OUT_MIXED_LUNG, output_dir)

def main():
    build_feature_tables()


if __name__ == "__main__":
    main()
