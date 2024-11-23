import os
import json
import pandas as pd

def process_json_files(input_folder, output_csv):
    """
    Processes annotation JSON files to extract and analyze key information,
    ensuring no duplicate IDs.

    Parameters:
        input_folder (str): Path to the folder containing the .json files.
        output_csv (str): Path to save the summarized data as a CSV file.
    """
    data = []

    # Traverse the folder
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith("_h_00_annot.json"):
                print(file)
                json_file = os.path.join(root, file)

                # Open and parse the JSON file
                with open(json_file, 'r') as f:
                    json_data = json.load(f)
                
                # Extract relevant information from the JSON
                item = json_data["levels"][0]["items"][0]  # "utterance" level
                labels = {label["name"]: label["value"] for label in item["labels"]}

                # Add the data to the list
                data.append({
                    "spn": labels.get("spn"),
                    "alc": labels.get("alc"),
                    "sex": labels.get("sex"),
                    "age": int(labels.get("age", 0)),
                    "acc": labels.get("acc"),
                    "drh": labels.get("drh"),
                    "aak": labels.get("aak"),
                    "bak": float(labels.get("bak")),
                    "ges": labels.get("ges"),
                    "ces": labels.get("ces"),
                    "wea": labels.get("wea"),
                })
    
    # Convert the data into a DataFrame
    df = pd.DataFrame(data)

    # Remove duplicate IDs
    df = df.drop_duplicates(subset=["spn"], keep="first")

    # Perform basic analysis
    print("Summary of Data:")
    print(df.describe(include='all'))
    
    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"Data saved to {output_csv}")


# Define paths
input_folder = "ALC"  # Replace with the actual folder containing JSON files
output_csv = "annotation_analysis.csv"

# Process the files
process_json_files(input_folder, output_csv)
