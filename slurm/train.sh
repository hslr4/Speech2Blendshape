#!/bin/bash
#SBATCH --job-name train_speech2blendshape
#SBATCH --partition gpu
#SBATCH --gpus 1
#SBATCH --output trainlogs/%j.out
#SBATCH --time 48:00:00

# Print some node information
echo "$(date)"
echo "On partition: GPU"
echo "Running on: $(hostname)"
echo "Available CPUs: $(taskset -c -p $$) (logical CPU ids)"
echo "Available GPUs: $(nvidia-smi)"

echo "SLURM_JOB_ID $SLURM_JOB_ID"


while getopts ":i:t:e:s:m:d:l:r:b:z:n:f:u:o:w:p:a:c:h:q:v:" opt; do
    case $opt in
        i) i=$OPTARG ;;
        t) train_subjects=$OPTARG ;;
        e) test_subjects=$OPTARG ;;
        s) embed_subjects=$OPTARG ;;
        m) embed_emotions=$OPTARG ;;
        d) gru_latent_dim=$OPTARG ;;
        l) gru_layers=$OPTARG ;;
        r) gru_dropout=$OPTARG ;;
        b) batch_size=$OPTARG ;;
        z) learning_rate=$OPTARG ;;
        n) epochs=$OPTARG ;;
        f) forzen_encoder_layers=$OPTARG ;;
        u) loss_fn=$OPTARG ;;
        o) optimizer=$OPTARG ;;
        w) weight_decay=$OPTARG ;;
        p) layer_dropout=$OPTARG ;;
        a) subject_dropout=$OPTARG ;;
        c) model_norm=$OPTARG ;;
        h) huber_delta=$OPTARG ;;
        q) scale_loss=$OPTARG ;;
        v) embedding_integration=$OPTARG ;;
        \?) echo "Invalid option: $OPTARG" ;;
    esac
done

# Print all parameters
echo "i: $i"
echo "train_subjects: $train_subjects"
echo "test_subjects: $test_subjects"
echo "embed_subjects: $embed_subjects"
echo "embed_emotions: $embed_emotions"
echo "gru_latent_dim: $gru_latent_dim"
echo "gru_layers: $gru_layers"
echo "gru_dropout: $gru_dropout"
echo "batch_size: $batch_size"
echo "learning_rate: $learning_rate"
echo "epochs: $epochs"
echo "forzen_encoder_layers: $forzen_encoder_layers"
echo "loss_fn: $loss_fn"
echo "optimizer: $optimizer"
echo "weight_decay: $weight_decay"
echo "layer_dropout: $layer_dropout"
echo "subject_dropout: $subject_dropout"
echo "model_norm: $model_norm"
echo "huber_delta: $huber_delta"
echo "scale_loss: $scale_loss"
echo "embedding_integration: $embedding_integration"

# Create the enroot container if necessary
if enroot list | grep -q -w "^lipsynccontainer$"; then 
	echo "Found container: lipsynccontainer";
else 
	echo "Can't find lipsynccontainer container"
	exit
fi

enroot start -w -m $HOME/speech2blendshape:$HOME/speech2blendshape lipsynccontainer bash -c "echo $pwd && cd speech2blendshape && /opt/miniconda3/envs/customlipsync/bin/python main.py --train_subjects '$train_subjects' --test_subjects '$test_subjects' --embed_subjects $embed_subjects --embed_emotions $embed_emotions --gru_latent_dim $gru_latent_dim --gru_layers $gru_layers --gru_dropout $gru_dropout --batch_size $batch_size --learning_rate $learning_rate --epochs $epochs --forzen_encoder_layers '$forzen_encoder_layers' --loss_fn $loss_fn --optimizer $optimizer --weight_decay $weight_decay --layer_dropout $layer_dropout --subject_dropout $subject_dropout --model_norm $model_norm --huber_delta $huber_delta --scale_loss $scale_loss --embedding_integration $embedding_integration --save_path models/$SLURM_JOB_ID"
