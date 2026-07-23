import subprocess

config = {
    "train_subjects": "1 2 3 4 6 7 8 9 11 12 13 14 15 16 17 18 19 21 22 23 24 25 26 27 28 29",
    "test_subjects": "1 2 3 4 6 7 8 9 11 12 13 14 15 16 17 18 19 21 22 23 24 25 26 27 28 29",
    "validation_subjects": "1 2 3 4 6 7 8 9 11 12 13 14 15 16 17 18 19 21 22 23 24 25 26 27 28 29",
    # "train_subjects": "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24",
    # "test_subjects": "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24",
    # "validation_subjects": "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24",
    "embed_subjects": "true",
    "embed_emotions": "true",
    "gru_latent_dim": 256,
    "gru_layers": 1,
    "gru_dropout": 0.3,
    "batch_size": 16,
    "learning_rate": 0.00002,
    "epochs": 50,
    "forzen_encoder_layers": "0 1",
    "loss_fn": "huber",
    "optimizer": "adamw",
    "weight_decay": 0.01,
    "layer_dropout": 0,
    "subject_dropout": 0,
    "model_norm": "layer",
    "huber_delta": 0.7,
    "scale_loss": "true",
    "embedding_integration": "cat"
}

cmd = [
    "sbatch", "train.sh",
    "-t", config["train_subjects"],
    "-e", config["test_subjects"],
    "-s", config["embed_subjects"],
    "-m", config["embed_emotions"],
    "-d", str(config["gru_latent_dim"]),
    "-l", str(config["gru_layers"]),
    "-r", str(config["gru_dropout"]),
    "-b", str(config["batch_size"]),
    "-z", str(config["learning_rate"]),
    "-n", str(config["epochs"]),
    "-f", config["forzen_encoder_layers"],
    "-u", config["loss_fn"],
    "-o", config["optimizer"],
    "-w", str(config["weight_decay"]),
    "-p", str(config["layer_dropout"]),
    "-a", str(config["subject_dropout"]),
    "-c", config["model_norm"],
    "-h", str(config["huber_delta"]),
    "-q", config["scale_loss"],
    "-v", config["embedding_integration"],
    "-i", str(0)
]

subprocess.run(cmd)