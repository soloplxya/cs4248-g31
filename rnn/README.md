## About this sub-folder 
This sub-folder contains all the relevant code for our RNN models. 

To run the code, first install the requirements in the requirements.txt file by running the following command in the terminal:
```
pip install -r requirements.txt
```

Then, you can run `rnn-experiments.py` to train and evaluate the RNN models. The script will train the models and save the checkpoints in the `checkpoints` folder. The evaluation results will be saved in the `results` folder.

For instance,
```
python rnn-experiments.py -h # View valid arguments

# Example command
python rnn-experiments.py --encoder_type=rnn --decoder_type=rnn --hidden_dim=2 --num_layers=1 --batch_size=2 --max_epochs=1 --learning_rate=0.001 --bidirectional=False --max_epochs=1
```
## Instructions
- The `rnn-experiments.py` script will train the RNN models with the specified hyperparameters and save the results in the `results` folder.

- The script uses huggingface datasets to handle data loading and preprocessing.
- The sentencepiece tokenizer will be trained and saved locally on the first run.

## Supported Arguments
- `--encoder_type`: Type of encoder (rnn, lstm, gru)
- `--decoder_type`: Type of decoder (rnn, lstm, gru)
- `--hidden_dim`: Hidden dimension of the RNN
- `--num_layers`: Number of layers in the RNN
- `--batch_size`: Batch size for training
- `--max_epochs`: Maximum number of epochs to train
- `--learning_rate`: Learning rate for training
- `--bidirectional`: Whether to use bidirectional RNN
- `--teacher_forcing_prob`: Probability of using teacher forcing
- `--teacher_forcing_schedule`: Use Schedule for teacher forcing (True or Force)
- `--loss`: Loss function to use (cross_entropy, intrust)
- `--vocab_size`: Vocabulary size
- `--train_batch_size`: Batch size for training
- `--checkpoint_prefix`: Prefix for checkpoint files
- `--seq_length`: Sequence length
