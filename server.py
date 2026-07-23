import librosa
import argparse
import torch
import numpy as np
from model import Speech2BlendshapeNetwork
from transformers import Wav2Vec2FeatureExtractor

import uvicorn
from fastapi import FastAPI, Response, Request
import io
import soundfile as sf


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="ANONYMIZED-FOR-REVIEW/speech2blendshape")
    parser.add_argument("--condition_subjects", type=str, default="3")
    parser.add_argument("--emotion", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--host", type=str, default='0.0.0.0', help='host to expose rest api')
    parser.add_argument("--port", type=int, default=7222, help='port to expose rest api')

    args = parser.parse_args()
    return args

def load_model(model_path, device):
    with torch.no_grad():
        model = Speech2BlendshapeNetwork.from_pretrained(model_path)
        model.eval()
        model.to(torch.device(device))
    print(model)
    return model

WAV_PATH = "temp.wav"
args = parse_args()
model = load_model(args.model_path, args.device)
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-xlarge-ls960-ft")
app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "Lipsync"}


@app.post("/lipsync/", response_class=Response)
async def lipsync(data: Request, emotion: int = args.emotion, condition_subjects: str = args.condition_subjects):
    data_b = await data.body()
    bio = io.BytesIO(data_b)
    speech_array, sampling_rate = sf.read(bio, dtype='float32')
    if speech_array.ndim > 1:
        speech_array = np.mean(speech_array, axis=1)
    if sampling_rate != 16000:
        speech_array = librosa.resample(speech_array, orig_sr=sampling_rate, target_sr=16000)
        sampling_rate = 16000
    
    subj_one_hot = np.zeros((1,model.subj_emb_len))
    for s in condition_subjects.split():
        if s in model.train_subjects.split():
            subj_one_hot[:,model.train_subjects.split().index(s)] = 1
        else:
            print(f'Warning: subject {s} was not in train_subjects')
    
    emo_one_hot = np.zeros((1,model.emo_emb_len))
    if emotion > 0:
        emo_one_hot[:,emotion-1] = 1    

    with torch.no_grad():
        subj_one_hot = torch.FloatTensor(subj_one_hot).to(device=torch.device(args.device))
        emo_one_hot = torch.FloatTensor(emo_one_hot).to(device=torch.device(args.device))
        
        audio = feature_extractor(speech_array, return_tensors="pt", padding="longest",
                                    sampling_rate=sampling_rate).input_values
        audio = audio.to(device=torch.device(args.device))
        
        output = model(audio, subj_one_hot, emo_one_hot).detach().cpu().numpy()[0]

        torch.cuda.empty_cache()
        return Response(content=output.tobytes(), media_type='application/octet-stream')
    

if __name__ == '__main__':
    uvicorn.run(app, port=args.port, host=args.host)