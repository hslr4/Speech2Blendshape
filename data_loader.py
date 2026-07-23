import os
import torch
from torch.utils import data
import numpy as np
import random
from transformers import Wav2Vec2FeatureExtractor
from sklearn.model_selection import train_test_split
import librosa


class AudioBlendshapeDataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""

    def __init__(self, data_path, filenames, audio_processor, subjects_dict, embed_eval_subj='zeros', emotions=7, device='cuda'):
        self.filenames = filenames
        self.data_path = data_path
        self.audio_path = os.path.join(data_path, 'wav')
        self.vertice_path = os.path.join(data_path, 'vertices_npy')
        self.audio_processor = audio_processor
        self.device = device
        self.subjects_dict = subjects_dict
        self.subj_one_hots = np.eye(len(subjects_dict["train"]))
        self.embed_eval_subj = embed_eval_subj
        self.emo_one_hots = np.eye(emotions+1,emotions,-1) #first row all zeros for neutral

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        file_name = self.filenames[index]

        speech_array, sampling_rate = librosa.load(os.path.join(self.audio_path, file_name+'.wav'), sr=16000)
        
        audio = self.audio_processor(speech_array, return_tensors="pt", padding="max_length",
                                    max_length=sampling_rate*10, # data preprocessing made 10s clips
                                    sampling_rate=sampling_rate).input_values
        audio = np.squeeze(audio)
        
        vertice = np.load(os.path.join(self.vertice_path, file_name+'.npy'), allow_pickle=True)
        vertice = vertice.astype(np.float32)
        vertice = torch.from_numpy(vertice)
        
        audio, vertice = audio.to(device=torch.device(self.device)), vertice.to(device=torch.device(self.device))

        subject = file_name.split('_')[0]
        subj_one_hot = np.zeros_like(self.subj_one_hots[0])
        if subject in self.subjects_dict['train']:
            subj_one_hot = self.subj_one_hots[self.subjects_dict['train'].index(subject)]
        elif self.embed_eval_subj == 'random':
            subj_one_hot = self.subj_one_hots[self.subjects_dict['train'].index(random.choice(self.subjects_dict['train']))]
        elif self.embed_eval_subj == 'ones':
            subj_one_hot = np.ones_like(self.subj_one_hots[0])
        elif self.embed_eval_subj in self.subjects_dict['train']:
            subj_one_hot = self.subj_one_hots[self.subjects_dict['train'].index(self.embed_eval_subj)]
        elif isinstance(self.embed_eval_subj, list):
            for i in self.embed_eval_subj:
                if i in self.subjects_dict['train']:
                    subj_one_hot[self.subjects_dict['train'].index(i)] = 1
        # else:
        #     if self.embed_eval_subj != 'zeros':
        #         print(f'subject embedding strategy {self.embed_eval_subj} ended up using zeros. Check if subject {subject} is in training subjects {self.subjects_dict["train"]}')

        emo = file_name.split('_')[1]
        emo_one_hot = self.emo_one_hots[int(emo)]

        subj_one_hot = torch.FloatTensor(subj_one_hot).to(device=torch.device(self.device))
        emo_one_hot = torch.FloatTensor(emo_one_hot).to(device=torch.device(self.device))

        return audio, vertice, subj_one_hot, emo_one_hot

    def __len__(self):
        return len(self.filenames)



def get_filenames(data_path, subject_lists):
    audio_path = os.path.join(data_path, 'wav')
    vertices_path = os.path.join(data_path, 'vertices_npy')
    all_subjects = []
    for l in subject_lists.values():
        all_subjects += l
    print(f'all_subjects {all_subjects}')
    filenames = []
    for f in os.listdir(audio_path):
        if f.endswith('.wav'):
            if os.path.isfile(os.path.join(vertices_path, f.replace('.wav', '.npy'))):
                subject_id = f.split("_")[0]
                if subject_id in all_subjects:
                    filenames.append(f.removesuffix('.wav'))
            else:
                print(f'missing npy for {f}')
    return filenames

def split(filenames, subject_lists):
    subj_counts = {}
    for l in subject_lists.values():
        for s in l:
            if s in subj_counts:
                subj_counts[s] += 1
            else:
                subj_counts[s] = 1
    split_arr = []
    splitted_names = {k: [] for k in subject_lists.keys()}
    for fn in sorted(filenames):
        subject_id, emo_id, sentence_id = fn.split('_')
        if subj_counts[subject_id] > 1:
            split_arr.append([int(subject_id), int(emo_id), int(sentence_id)])
        else: # add all sentences of subjects exclusive to a list
            for k, v in subject_lists.items():
                if subject_id in v:
                    splitted_names[k].append(fn)
    print('exclusive subject sentences: {}'.format({k: len(v) for k,v in splitted_names.items()}))
    split_arr = np.array(split_arr)
    train_indices, test_indices = train_test_split(
        split_arr, test_size=0.1, stratify=split_arr[:, 0:2], random_state=42
    )
    train_indices, val_indices = train_test_split(
        train_indices, test_size=0.1, stratify=train_indices[:, 0:2], random_state=42
    )
    print(f'splitted sentences: train: {train_indices.shape[0]}, validation: {val_indices.shape[0]}, test: {test_indices.shape[0]}')
    dif_emos, counts = np.unique(train_indices[:,1], return_counts=True)
    print('different emotions in train data (emo: n_times): {}'.format([(e,c) for e,c in zip(dif_emos, counts)]))
    n_emos = len(dif_emos)
    for n, i in zip(['train', 'validation', 'test'], [train_indices, val_indices, test_indices]):
        for row in i:
            if str(row[0]) in subject_lists[n]:
                splitted_names[n].append(str(row[0])+'_'+str(row[1])+'_'+str(row[2]).zfill(2))
            # else:
            #     print('not adding {} to {}'.format('_'.join([str(e) for e in row]), n))
    print('all sentences: {}'.format({k: len(v) for k,v in splitted_names.items()}))
    return splitted_names, n_emos

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_dataloaders(data_path, train_subjects, validation_subjects, test_subjects, batch_size, embed_eval_subj, device):
    g = torch.Generator()
    g.manual_seed(0)

    subject_lists = {'train': train_subjects.split(), 'validation': validation_subjects.split(), 'test': test_subjects.split()}

    fns = get_filenames(data_path, subject_lists)

    splitted, n_emos = split(fns, subject_lists)

    processor = Wav2Vec2FeatureExtractor.from_pretrained(
        "facebook/hubert-xlarge-ls960-ft")  # HuBERT uses the processor of Wav2Vec 2.0

    dataloaders = {}
    for k in splitted.keys():
        if len(splitted[k]) > 0:
            dataset = AudioBlendshapeDataset(data_path, splitted[k], processor, subject_lists, embed_eval_subj=embed_eval_subj, emotions=n_emos, device=device)
            dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker, generator=g)
            dataloaders[k] = dataloader

    return dataloaders, n_emos
