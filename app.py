import os
os.system("pip install git+https://github.com/openai/whisper.git")
from pytube import YouTube
import gradio as gr
from subprocess import call
import whisper
import logging
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


model = whisper.load_model("base")

inputs = gr.components.Audio(type="filepath", label="Add audio file")
outputs = gr.components.Textbox()
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
    result = model.transcribe(audio=inputs, language='english',
                              word_timestamps=False, verbose=True)
#  ---------------------------------------------------

    print(result["text"])
    return result["text"]


# Transcribe youtube video
# define function for transcription
def youtube_transcript(url):
    try:
        if url:
            yt = YouTube(url, use_oauth=True)
            source = yt.streams.filter(progressive=True, file_extension='mp4').order_by(
                'resolution').desc().first().download('output/youtube')

            transcript = model.transcribe(source)
            return transcript["text"]
    except Exception as e:
        print('Error: ', e)
        return 'Error: ' + str(e)


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
        [  "https://www.youtube.com/watch?v=nlMuHtV82q8&ab_channel=NothingforSale24",],
        ["https://www.youtube.com/watch?v=JzPfMbG1vrE&ab_channel=ExplainerVideosByLauren",],
        ["https://www.youtube.com/watch?v=S68vvV0kod8&ab_channel=Pearl-CohnTelevision"]

    ],

)

demo = gr.Blocks()
with demo:
    gr.TabbedInterface([youtube_chunked, audio_chunked, microphone_chunked], [
        "Youtube", "Audio File", "Microphone"])
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
