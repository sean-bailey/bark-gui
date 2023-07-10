"""

We're going to make a FastAPI frontend for this, designed to be used by a Lambda function or similar. Keep it lightweight and barebones.

Let's focus exclusively on TTS functionality here. 


"""

from cProfile import label
import dataclasses
from distutils.command.check import check
from doctest import Example
import os
import sys
import numpy as np
import logging
import torch
import pytorch_seed
import time
import io
from xml.sax import saxutils
from bark.api import generate_with_settings
from bark.api import save_as_prompt
from util.settings import Settings
#import nltk
import urllib

from bark import SAMPLE_RATE
from cloning.clonevoice import clone_voice
from bark.generation import SAMPLE_RATE, preload_models, _load_history_prompt, codec_decode
from scipy.io.wavfile import write as write_wav
from util.parseinput import split_and_recombine_text, build_ssml, is_ssml, create_clips_from_ssml
from datetime import datetime
from tqdm.auto import tqdm
from util.helper import create_filename, add_id3_tag
from swap_voice import swap_voice_from_audio
from training.training_prepare import prepare_semantics_from_text, prepare_wavs_from_semantics
from training.train import training_prepare_files, train
import boto3
import json
import os
from botocore.exceptions import ClientError
import logging
from botocore.config import Config
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from mangum import Mangum

speakerlocation=os.environ.get("SPEAKERLOCATION")
settings = Settings(os.environ.get("SETTINGSDOC"))
bucketname=os.environ.get("BUCKETNAME")
stage = os.environ.get('STAGE', None)
openapi_prefix = f"/{stage}" if stage else "/"
app = FastAPI(title="ttsLambdaApp", openapi_prefix=openapi_prefix) # Here is the magic

@app.get("/tts")
async def generate_text_to_speech(text:str, selected_speaker:str="None", text_temp:float=0.6, waveform_temp:float=0.7, eos_prob:float=0.05, quick_generation:bool=True, save_last_generation: bool=True,use_last_generation_as_history: bool=True, seed: int=-1, batchcount: int=1):
    # Chunk the text into smaller pieces then combine the generated audio

    # generation settings
    if selected_speaker == 'None':
        selected_speaker = None

    voice_name = selected_speaker

    if text == None or len(text) < 1:
       if selected_speaker == None:
            raise gr.Error('No text entered!')

       # Extract audio data from speaker if no text and speaker selected
       voicedata = _load_history_prompt(voice_name)
       audio_arr = codec_decode(voicedata["fine_prompt"])
       result = create_filename(settings.output_folder_path, "None", "extract",".wav")
       save_wav(audio_arr, result)
       return result

    if batchcount < 1:
        batchcount = 1


    silenceshort = np.zeros(int((float(settings.silence_sentence) / 1000.0) * SAMPLE_RATE), dtype=np.int16)  # quarter second of silence
    silencelong = np.zeros(int((float(settings.silence_speakers) / 1000.0) * SAMPLE_RATE), dtype=np.float32)  # half a second of silence
    for l in range(batchcount):
        currentseed = seed
        if seed != None and seed > 2**32 - 1:
            print(f"Seed {seed} > 2**32 - 1 (max), setting to random")
            currentseed = None
        if currentseed == None or currentseed <= 0:
            currentseed = np.random.default_rng().integers(1, 2**32 - 1)
        assert(0 < currentseed and currentseed < 2**32)
        full_generation = None

        all_parts = []
        complete_text = ""
        text = text.lstrip()
        if is_ssml(text):
            list_speak = create_clips_from_ssml(text)
            prev_speaker = None
            for i, clip in tqdm(enumerate(list_speak), total=len(list_speak)):
                selected_speaker = clip[0]
                # Add pause break between speakers
                if i > 0 and selected_speaker != prev_speaker:
                    all_parts += [silencelong.copy()]
                prev_speaker = selected_speaker
                text = clip[1]
                text = saxutils.unescape(text)
                if selected_speaker == "None":
                    selected_speaker = None

                print(f"\nGenerating Text ({i+1}/{len(list_speak)}) -> {selected_speaker} (Seed {currentseed}):`{text}`")
                complete_text += text
                with pytorch_seed.SavedRNG(currentseed):
                    audio_array = generate_with_settings(text_prompt=text, voice_name=selected_speaker, semantic_temp=text_temp, coarse_temp=waveform_temp, eos_p=eos_prob)
                    currentseed = torch.random.initial_seed()
                if len(list_speak) > 1:
                    filename = create_filename(settings.output_folder_path, currentseed, "audioclip",".wav")
                    save_wav(audio_array, filename)
                    add_id3_tag(filename, text, selected_speaker, currentseed)

                all_parts += [audio_array]
        else:
            texts = split_and_recombine_text(text, settings.input_text_desired_length, settings.input_text_max_length)
            for i, text in tqdm(enumerate(texts), total=len(texts)):
                print(f"\nGenerating Text ({i+1}/{len(texts)}) -> {selected_speaker} (Seed {currentseed}):`{text}`")
                complete_text += text
                if quick_generation == True:
                    with pytorch_seed.SavedRNG(currentseed):
                        audio_array = generate_with_settings(text_prompt=text, voice_name=selected_speaker, semantic_temp=text_temp, coarse_temp=waveform_temp, eos_p=eos_prob)
                        currentseed = torch.random.initial_seed()
                else:
                    full_output = use_last_generation_as_history or save_last_generation
                    if full_output:
                        full_generation, audio_array = generate_with_settings(text_prompt=text, voice_name=voice_name, semantic_temp=text_temp, coarse_temp=waveform_temp, eos_p=eos_prob, output_full=True)
                    else:
                        audio_array = generate_with_settings(text_prompt=text, voice_name=voice_name, semantic_temp=text_temp, coarse_temp=waveform_temp, eos_p=eos_prob)

                # Noticed this in the HF Demo - convert to 16bit int -32767/32767 - most used audio format  
                # audio_array = (audio_array * 32767).astype(np.int16)

                if len(texts) > 1:
                    filename = create_filename(settings.output_folder_path, currentseed, "audioclip",".wav")
                    save_wav(audio_array, filename)
                    add_id3_tag(filename, text, selected_speaker, currentseed)

                if quick_generation == False and (save_last_generation == True or use_last_generation_as_history == True):
                    # save to npz
                    voice_name = create_filename(settings.output_folder_path, seed, "audioclip", ".npz")
                    save_as_prompt(voice_name, full_generation)
                    if use_last_generation_as_history:
                        selected_speaker = voice_name

                all_parts += [audio_array]
                # Add short pause between sentences
                if text[-1] in "!?.\n" and i > 1:
                    all_parts += [silenceshort.copy()]

        # save & play audio
        result = create_filename(settings.output_folder_path, currentseed, "final",".wav")
        save_wav(np.concatenate(all_parts), result)
        # write id3 tag with text truncated to 60 chars, as a precaution...
        #add_id3_tag(result, complete_text, selected_speaker, currentseed)
    returndict={}
    returndict['audiolink']=create_presigned_url(bucketname,result,expiration=3600*24)
    return returndict


def save_wav(audio_array, filename):
    #hypothetically, we should be able to use a bytesio buffer, keeping everything in memory. This means we can use this method to upload things to S3.
    f=io.BytesIO()
    write_wav(f, SAMPLE_RATE, audio_array)
    upload_to_s3(f,bucketname,filename)


def create_presigned_url(bucket_name, object_name, expiration=60):
    """Generate a presigned URL to share an S3 object

    :param bucket_name: string
    :param object_name: string
    :param expiration: Time in seconds for the presigned URL to remain valid
    :return: Presigned URL as string. If error, returns None.
    """

    my_config = Config(
    signature_version = 's3v4',
)

    # Generate a presigned URL for the S3 object
    s3_client = boto3.client('s3',config=my_config)
    try:
        response = s3_client.generate_presigned_url('get_object',
                                                    Params={'Bucket': bucket_name,
                                                            'Key': object_name},
                                                    ExpiresIn=expiration,
                                                    )
    except ClientError as e:
        logging.error(e)
        return None

    # The response contains the presigned URL
    return response


def getJsonFromS3(bucket,key):
    filestream=io.BytesIO()
    s3=boto3.resource('s3')
    bucket=s3.Bucket(bucket)
    objectname=key
    bucket.download_fileobj(objectname,filestream)
    filecontent_bytes=filestream.getvalue()
    data=json.loads(filecontent_bytes)
    return data


#take in the bytesio buffer, bucket and filename, upload to bucket.
def upload_to_s3(buffer, bucket, filename):
    extension="."+filename.split('.')[-1]
    #tempfilename=str(uuid.uuid4()).split('-')[0]+str(time.time()).split('.')[0]
    filename=filename#regex.sub('_',str(filename))+extension
    buffer.seek(0)
    s3=boto3.resource('s3')
    boto_bucket=s3.Bucket(bucket)
    boto_bucket.upload_fileobj(buffer,filename)
    print("wrote "+filename+" to s3 bucket "+bucket)


def get_bucket_key(event):
    print(event)
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = urllib.parse.unquote_plus(
        event['Records'][0]['s3']['object']['key'], encoding='utf-8')
    url=create_presigned_url(bucket,key)
    #it's not this simple. We'll need to be able to generate a presigned key.
    #url="https://"+bucket+".s3.amazonaws.com/"+key
    return bucket, key, url


@app.get("/listspeakers")
def returnspeakers():
    speakers_list = []

    for root, dirs, files in os.walk(speakerlocation):
        for file in files:
            if file.endswith(".npz"):
                pathpart = root.replace(speakerlocation, "")
                name = os.path.join(pathpart, file[:-4])
                if name.startswith("/") or name.startswith("\\"):
                     name = name[1:]
                speakers_list.append(name)

    speakers_list = sorted(speakers_list, key=lambda x: x.lower())
    speakers_list.insert(0, 'None')
    returndict={}
    returndict['speakers']=speakers_list
    return returndict

handler=Mangum(app)
