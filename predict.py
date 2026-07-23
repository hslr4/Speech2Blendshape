import argparse
from model import Speech2BlendshapeNetwork
import torch
import os
import librosa
from transformers import Wav2Vec2FeatureExtractor
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="ANONYMIZED-FOR-REVIEW/speech2blendshape")
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--input_file", type=str, default="example/example.wav")
    parser.add_argument("--result_path", type=str, default="example")
    parser.add_argument("--condition_subjects", type=str, default="3")
    parser.add_argument("--emotion", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    return args

def load_model(model_path, device):
    model = Speech2BlendshapeNetwork.from_pretrained(model_path)
    model.eval()
    model.to(torch.device(device))
    print(model)
    return model

def load_input(audio_path, subjects, emotion, device, model):
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-xlarge-ls960-ft")
    speech_array, sampling_rate = librosa.load(os.path.join(audio_path), sr=16000)

    audio = feature_extractor(speech_array, return_tensors="pt", padding="longest",
                                sampling_rate=sampling_rate).input_values

    audio = audio.to(device=torch.device(device))

    if model.subj_emb_len > 0:
        subj_one_hot = np.zeros((1,model.subj_emb_len))
        for s in subjects:
            if s in model.train_subjects.split():
                subj_one_hot[:,model.train_subjects.split().index(s)] = 1
            else:
                print(f'Warning: subject {s} was not in train_subjects')

    if model.emo_emb_len > 0:
        emo_one_hot = np.zeros((1,model.emo_emb_len))
        if emotion > 0:
            emo_one_hot[:,emotion-1] = 1

    subj_one_hot = torch.FloatTensor(subj_one_hot).to(device=torch.device(device))
    emo_one_hot = torch.FloatTensor(emo_one_hot).to(device=torch.device(device))

    return audio, subj_one_hot, emo_one_hot
    


if __name__ == "__main__":
    args = parse_args()

    model = load_model(args.model_path, args.device)
    if os.path.isdir(args.input_file):
        for file_name in os.listdir(args.input_file):
            file_path = os.path.join(args.input_file, file_name)
            if file_name.endswith('.wav'):
                inp_path = os.path.join(args.input_file, file_name)
                audio, subj, emo = load_input(inp_path, args.condition_subjects.split(), args.emotion, args.device, model)
                output = model(audio, subj, emo)
                fn = inp_path.split('/')[-1].split('.')[0]
                np.save(os.path.join(args.result_path, f'{fn}.npy'), output.detach().cpu().numpy()[0])
                print('output saved to', os.path.join(args.result_path, f'{fn}.npy'))
    elif os.path.isfile(args.input_file):
        audio, subj, emo = load_input(args.input_file, args.condition_subjects.split(), args.emotion, args.device, model)
        output = model(audio, subj, emo)
        fn = args.input_file.split('/')[-1].split('.')[0]
        np.save(os.path.join(args.result_path, f'{fn}.npy'), output.detach().cpu().numpy()[0])
        print('output saved to', os.path.join(args.result_path, f'{fn}.npy'))
    else:
        print(f'invalid input file {args.input_file}, needs to be a valid path to a directory or audio file')