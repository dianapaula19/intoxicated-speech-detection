import os
import json
import librosa
import numpy as np

def process_folder(input_folder, output_folder, fixed_length=100):
    """
    Processes .wav and .json files to extract MFCCs and labels,
    and saves them in .npy format.

    Parameters:
        input_folder (str): Path to the input folder containing .wav and .json files.
        output_folder (str): Path to save the processed .npy files.
        fixed_length (int): Fixed number of frames for padding/trimming.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith("_h_00.wav"):
                identifier = file.split("_")[0]
                wav_file = os.path.join(root, file)
                json_file = os.path.join(root, file.replace("_h_00.wav", "_h_00_annot.json"))
                npy_file = os.path.join(output_folder, f"{identifier}.npy")

                # Process .wav file to extract MFCCs
                y, sr = librosa.load(wav_file, sr=None)
                mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                
                # Normalize MFCCs
                mfccs = (mfccs - np.mean(mfccs)) / np.std(mfccs)

                # Pad or trim MFCCs to fixed length
                if mfccs.shape[1] < fixed_length:
                    pad_width = fixed_length - mfccs.shape[1]
                    mfccs = np.pad(mfccs, ((0, 0), (0, pad_width)), mode='constant')
                else:
                    mfccs = mfccs[:, :fixed_length]

                # Initialize label and metadata
                label = None
                metadata = {}

                if os.path.exists(json_file):
                    with open(json_file, 'r') as f:
                        json_data = json.load(f)
                        # Extract label (alc) from the metadata
                        for label_info in json_data["levels"][0]["items"][0]["labels"]:
                            if label_info["name"] == "alc":
                                label = 1 if label_info["value"] == "a" else 0
                                metadata["label"] = label
                        
                        # Extract all metadata from JSON labels
                        for label_info in json_data["levels"][0]["items"][0]["labels"]:
                            key = label_info["name"]
                            value = label_info["value"]
                            try:
                                # Convert numeric values to float if possible
                                value = float(value)
                            except ValueError:
                                pass  # Keep non-numeric values as strings
                            metadata[key] = value

                if label is None:
                    print(f"Warning: No label found for {identifier}. Skipping.")
                    continue

                # Save MFCCs, label, and metadata in a .npy file
                data = {
                    "mfcc": mfccs,
                    "metadata": metadata  # Include all extracted metadata
                }
                np.save(npy_file, data)
                print(f"Processed and saved: {npy_file}")

# Input and output paths
input_folder = "ALC"
output_folder = "Processed_Stats_ALC"

# Process the folder
process_folder(input_folder, output_folder)
