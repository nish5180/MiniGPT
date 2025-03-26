
'''
import kagglehub
path = kagglehub.dataset_download("deepshah16/song-lyrics-dataset")

print("Path to dataset files:", path)
'''

import os
import pandas as pd

# Set the path to the folder containing the CSV files
folder_path = r"C:\Users\konst\Desktop\TU Delft\Q3\Bayesian ML\ng-video-lecture-master\csv"

# File to save the extracted lyrics
output_file = r"C:\Users\konst\Desktop\TU Delft\Q3\Bayesian ML\ng-video-lecture-master\lyrics.txt"

with open(output_file, 'w', encoding= 'utf-8') as outfile:
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)
            
            if 'Lyric' in df.columns:

                lyrics = df['Lyric']
                

                outfile.write(f"\nLyrics from {filename}:\n")
                for lyric in lyrics:
                    outfile.write(f"{lyric}\n")
            else:
                print(f"No 'Lyrics' column found in {filename}")

print("Extraction completed!")
