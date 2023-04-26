# Basic script for using the OpenAI Whisper model to transcribe a video file. You can uncomment whichever model you want to use.
# Author: ThioJoe ( https://github.com/ThioJoe )

# Required third party packages: whisper
# See instructions for setup here: https://github.com/openai/whisper#setup
#   - You can use the below command to pull the repo and install dependencies, then just put this script in the repo directory:
#     pip install git+https://github.com/openai/whisper.git

import whisper
import io
import time
import os
import json
import pathlib

# Choose model to use by uncommenting
# modelName = "tiny.en"
modelName = "base.en"
# modelName = "small.en"
# modelName = "medium.en"
# modelName = "large-v2"

# Other Variables
# (bool) Whether to export the segment data to a json file. Will include word level timestamps if word_timestamps is True.
exportTimestampData = True
outputFolder = "Output"

#  ----- Select variables for transcribe method  -----
# audio: path to audio file
verbose = True  # (bool): Whether to display the text being decoded to the console. If True, displays all the details, If False, displays minimal details. If None, does not display anything
language = "english"  # Language of audio file
# (bool): Extract word-level timestamps using the cross-attention pattern and dynamic time warping, and include the timestamps for each word in each segment.
word_timestamps = False
# initial_prompt="" # (optional str): Optional text to provide as a prompt for the first window. This can be used to provide, or "prompt-engineer" a context for transcription, e.g. custom vocabularies or proper nouns to make it more likely to predict those word correctly.

#  -------------------------------------------------------------------------
print(f"Using Model: {modelName}")
filePath = input("Path to File Being Transcribed: ")
filePath = filePath.strip("\"")
if not os.path.exists(filePath):
    print("Problem Getting File...")
    input("Press Enter to Exit...")
    exit()

# If output folder does not exist, create it
if not os.path.exists(outputFolder):
    os.makedirs(outputFolder)
    print("Created Output Folder.\n")

# Get filename stem using pathlib (filename without extension)
fileNameStem = pathlib.Path(filePath).stem

resultFileName = f"{fileNameStem}.txt"
jsonFileName = f"{fileNameStem}.json"

model = whisper.load_model(modelName)
start = time.time()

#  ---------------------------------------------------
result = model.transcribe(audio=filePath, language=language,
                          word_timestamps=word_timestamps, verbose=verbose)
#  ---------------------------------------------------

end = time.time()
elapsed = float(end - start)

# Save transcription text to file
print("\nWriting transcription to file...")
with open(os.path.join(outputFolder, resultFileName), "w", encoding="utf-8") as file:
    file.write(result["text"])
print("Finished writing transcription file.")

# Sav
# e the segments data to json file
# if word_timestamps == True:
if exportTimestampData == True:
    print("\nWriting segment data to file...")
    with open(os.path.join(outputFolder, jsonFileName), "w", encoding="utf-8") as file:
        segmentsData = result["segments"]
        json.dump(segmentsData, file, indent=4)
    print("Finished writing segment data file.")

elapsedMinutes = str(round(elapsed/60, 2))
print(f"\nElapsed Time With {modelName} Model: {elapsedMinutes} Minutes")

input("Press Enter to exit...")
exit()
