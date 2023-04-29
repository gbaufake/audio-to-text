import os
import logging
os.system("pip install git+https://github.com/openai/whisper.git")
import gradio as gr
from subprocess import call
import whisper
from datetime import timedelta
from pytube import YouTube
import pandas as pd
import pysrt
# from transformers.pipelines.audio_utils import ffmpeg_read


logger = logging.getLogger("whisper-jax-app")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s;%(levelname)s;%(message)s", "%Y-%m-%d %H:%M:%S")
ch.setFormatter(formatter)
logger.addHandler(ch)


BATCH_SIZE = 16
CHUNK_LENGTH_S = 30
NUM_PROC = 8
FILE_LIMIT_MB = 1000
YT_ATTEMPT_LIMIT = 3


def run_cmd(command):
    try:
        print(command)
        call(command)
    except KeyboardInterrupt:
        print("Process interrupted")
        sys.exit(1)


def inference(text):
    cmd = ['tts', '--text', text]
    run_cmd(cmd)
    return 'tts_output.wav'


baseModel = whisper.load_model("base")
smallModel = whisper.load_model("small")


df_init = pd.DataFrame(columns=['start', 'end', 'text'])
transcription_df = gr.DataFrame(value=df_init, label="Transcription dataframe", row_count=(
    0, "dynamic"), max_rows=30, wrap=True, overflow_row_behaviour='paginate')


inputs = gr.components.Audio(type="filepath", label="Add audio file")
outputs = [gr.components.Textbox(), transcription_df]
title = "Transcribe multi-lingual audio clips"
description = "An example of using TTS to generate speech from text."
article = ""
examples = [
    [""]
]


def transcribe(inputs):
    print('Inputs: ', inputs)
    # print('Text: ', text)
    # progress(0, desc="Loading audio file...")
    if inputs is None:
        logger.warning("No audio file")
        return "No audio file submitted! Please upload an audio file before submitting your request."
    file_size_mb = os.stat(inputs).st_size / (1024 * 1024)
    if file_size_mb > FILE_LIMIT_MB:
        logger.warning("Max file size exceeded")
        return f"File size exceeds file size limit. Got file of size {file_size_mb:.2f}MB for a limit of {FILE_LIMIT_MB}MB."

    # with open(inputs, "rb") as f:
    #     inputs = f.read()

    # load audio and pad/trim it to fit 30 seconds
    result = smallModel.transcribe(audio=inputs, language='english',
                                   verbose=False)
#  ---------------------------------------------------
    segments = result['segments']
    for segment in segments:
        startTime = str(0)+str(timedelta(seconds=int(segment['start'])))+',000'
        endTime = str(0)+str(timedelta(seconds=int(segment['end'])))+',000'
        text = segment['text']
        segmentId = segment['id']+1
        segment = f"{segmentId}\n{startTime} --> {endTime}\n{text[1:] if text[0] is ' ' else text}\n\n"

        srtFilename = os.path.join("output/SrtFiles", inputs.split(
            '/')[-1].split('.')[0]+'.srt')
        with open(srtFilename, 'a', encoding='utf-8') as srtFile:
            srtFile.write(segment)

        rawFilename = os.path.join("output/SrtFiles", inputs.split(
            '/')[-1].split('.')[0]+'.srt')
        with open(rawFilename, 'a', encoding='utf-8') as srtFile:
            srtFile.write(segment)
    try:

        srt_path = srtFilename
        df = pd.DataFrame(columns=['start', 'end', 'text'])
        subs = pysrt.open(srt_path)

        objects = []
        for sub in subs:
            start_hours = str(str(sub.start.hours) + "00")[0:2] if len(
                str(sub.start.hours)) == 2 else str("0" + str(sub.start.hours) + "00")[0:2]
            end_hours = str(str(sub.end.hours) + "00")[0:2] if len(
                str(sub.end.hours)) == 2 else str("0" + str(sub.end.hours) + "00")[0:2]

            start_minutes = str(str(sub.start.minutes) + "00")[0:2] if len(
                str(sub.start.minutes)) == 2 else str("0" + str(sub.start.minutes) + "00")[0:2]
            end_minutes = str(str(sub.end.minutes) + "00")[0:2] if len(
                str(sub.end.minutes)) == 2 else str("0" + str(sub.end.minutes) + "00")[0:2]

            start_seconds = str(str(sub.start.seconds) + "00")[0:2] if len(
                str(sub.start.seconds)) == 2 else str("0" + str(sub.start.seconds) + "00")[0:2]
            end_seconds = str(str(sub.end.seconds) + "00")[0:2] if len(
                str(sub.end.seconds)) == 2 else str("0" + str(sub.end.seconds) + "00")[0:2]

            start = start_hours + ":" + start_minutes + ":" + start_seconds + ",000"
            end = end_hours + ":" + end_minutes + ":" + end_seconds + ",000"
            text = sub.text
            objects.append([start, end, text])

        df = pd.DataFrame(objects, columns=['start', 'end', 'text'])
    except Exception as e:
        print('Error: ', e)
        df = pd.DataFrame(columns=['start', 'end', 'text'])

    return [result["text"], df]


# Transcribe youtube video
# define function for transcription
def youtube_transcript(url):
    try:
        if url:
            yt = YouTube(url, use_oauth=True)
            source = yt.streams.filter(progressive=True, file_extension='mp4').order_by(
                'resolution').desc().first().download('output/youtube')

            transcript = baseModel.transcribe(source)
            return transcript["text"]
    except Exception as e:
        print('Error: ', e)
        return 'Error: ' + str(e)


def displaySrtFile(srtFilename):
    with open(srtFilename, 'r', encoding='utf-8') as srtFile:
        srtContent = srtFile.read()

        try:

            df = pd.DataFrame(columns=['start', 'end', 'text'])
            srt_path = srtFilename
            subs = pysrt.open(srt_path)

            objects = []
            for sub in subs:

                start_hours = str(str(sub.start.hours) + "00")[0:2] if len(
                    str(sub.start.hours)) == 2 else str("0" + str(sub.start.hours) + "00")[0:2]
                end_hours = str(str(sub.end.hours) + "00")[0:2] if len(
                    str(sub.end.hours)) == 2 else str("0" + str(sub.end.hours) + "00")[0:2]

                start_minutes = str(str(sub.start.minutes) + "00")[0:2] if len(
                    str(sub.start.minutes)) == 2 else str("0" + str(sub.start.minutes) + "00")[0:2]
                end_minutes = str(str(sub.end.minutes) + "00")[0:2] if len(
                    str(sub.end.minutes)) == 2 else str("0" + str(sub.end.minutes) + "00")[0:2]

                start_seconds = str(str(sub.start.seconds) + "00")[0:2] if len(
                    str(sub.start.seconds)) == 2 else str("0" + str(sub.start.seconds) + "00")[0:2]
                end_seconds = str(str(sub.end.seconds) + "00")[0:2] if len(
                    str(sub.end.seconds)) == 2 else str("0" + str(sub.end.seconds) + "00")[0:2]

                start_millis = str(str(sub.start.milliseconds) + "000")[0:3]
                end_millis = str(str(sub.end.milliseconds) + "000")[0:3]
                objects.append([sub.text, f'{start_hours}:{start_minutes}:{start_seconds}.{start_millis}',
                               f'{end_hours}:{end_minutes}:{end_seconds}.{end_millis}'])

            for object in objects:
                srt_to_df = {
                    'start': [object[1]],
                    'end': [object[2]],
                    'text': [object[0]]
                }

                df = pd.concat([df, pd.DataFrame(srt_to_df)])
        except Exception as e:
            print("Error creating srt df")

        return srtContent


audio_chunked = gr.Interface(
    fn=transcribe,
    inputs=inputs,
    outputs=outputs,
    allow_flagging="never",
    title=title,
    description=description,
    article=article,
)

microphone_chunked = gr.Interface(
    fn=transcribe,
    inputs=[
        gr.inputs.Audio(source="microphone",
                        optional=True, type="filepath"),
    ],
    outputs=[
        gr.outputs.Textbox(label="Transcription").style(
            show_copy_button=True),
    ],
    allow_flagging="never",
    title=title,
    description=description,
    article=article,
)
youtube_chunked = gr.Interface(
    fn=youtube_transcript,
    inputs=[
        gr.inputs.Textbox(label="Youtube URL", type="text"),
    ],
    outputs=[
        gr.outputs.Textbox(label="Transcription").style(
            show_copy_button=True),
    ],
    allow_flagging="never",
    title=title,

    description=description,
    article=article,
    examples=[
        ["https://www.youtube.com/watch?v=nlMuHtV82q8&ab_channel=NothingforSale24",],
        ["https://www.youtube.com/watch?v=JzPfMbG1vrE&ab_channel=ExplainerVideosByLauren",],
        ["https://www.youtube.com/watch?v=S68vvV0kod8&ab_channel=Pearl-CohnTelevision"]

    ],

)


demo = gr.Blocks()
with demo:
    gr.TabbedInterface([audio_chunked, youtube_chunked, microphone_chunked], [
        "Audio File", "Youtube", "Microphone"])
demo.queue(concurrency_count=1, max_size=5)
demo.launch(show_api=False)


# gr.Interface(
#     inference,
#     inputs,
#     outputs,
#     verbose=True,
#     title=title,
#     description=description,
#     article=article,
#     examples=examples,
#     enable_queue=True,

# ).launch(share=True, debug=True)
