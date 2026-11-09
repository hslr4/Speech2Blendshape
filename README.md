# Speech2Blendshape

Predict 51 AR Kit Blendshapes for speech input. (Tongue is missing in the BEAT dataset.)

Code for speech-to-blendshape model as used in the paper: *"Multimodal Emotion Expression for Android Robot Heads*


## Setup
`pip install -r requirements.txt`

## Prediction
To predict blendshapes for an audio file following example command can be used:
`python predict.py --input_file="example/example.wav" --condition_subjects="3"`
This will download the [pretrained model from huggingface](https://huggingface.co/ANONYMIZED-FOR-REVIEW/speech2blendshape).

## Training
### Data preparation
[BEAT](https://huggingface.co/datasets/H-Liu1997/BEAT) was used to train models in this repository. Data was preprocessed as in [FaceDiffuser](https://github.com/uuembodiedsocialai/FaceDiffuser). See [data/beat/](./data/beat/) for further information.

### Training
To train custom models `main.py` can be used. In [slurm/](./slurm/) the config used to train the model reported in the paper can be found besides an `sbatch` script to train models in a slurm cluster.

## Acknowledgements
We use and adapt code from [FaceDiffuser](https://github.com/uuembodiedsocialai/FaceDiffuser) and [FaceXHubert](https://github.com/galib360/FaceXHuBERT).