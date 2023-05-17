import yaml

hyperparameters = {
    "model_name": "distilbert-base-uncased",
    "max_sent_size": 512,
    "mlm_probability": 0.15,
    "pad_to_multiple_of_8": True,
    "learning_rate": 5e-5,
    "batch_size": 16,
    "steps": 5000,
    "warmup_ratio": 0.1,
    "eval_steps": 500,
    "logging_steps": 100
}

with open('/Users/benjaminhaussmann/Documents/GitHub/10k/pre_training_DistillBERT/distilbert_params.yaml', 'w') as outfile:
    yaml.dump(hyperparameters, outfile, default_flow_style=False)
