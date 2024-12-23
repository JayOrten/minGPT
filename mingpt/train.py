import torch
from mingpt.utils import set_seed
from mingpt.dataset import PileDataset
from mingpt.model import GPT
from mingpt.trainer import Trainer

from pathlib import Path

set_seed(3407)

def main():
    sequence_length = 512
    model_type = 'gpt2-medium'
    # output_folder = Path('/home/jo288/nobackup/autodelete/cs674project2/runs/run4_iter50000_ul2')
    output_folder = Path('/home/jo288/nobackup/autodelete/cs674project2/runs/run5')
    output_folder.mkdir(parents=True, exist_ok=True)

    print('Loading dataset...', flush=True)
    # dataset = PileDataset('/home/jo288/nobackup/autodelete/cs674project2/pile_data_10.jsonl', sequence_length, use_UL2=True, ul2_percentage=0.4)
    # dataset = PileDataset('/home/jo288/nobackup/autodelete/cs674project2/pile_data_short.jsonl', sequence_length, use_UL2=True, ul2_percentage=0.2)
    dataset = PileDataset('/home/jo288/nobackup/autodelete/cs674project2/pile_data_10.jsonl', sequence_length, use_UL2=False)

    print('Loading model...', flush=True)
    # checkpoint_path = "/home/jo288/nobackup/autodelete/cs674project2/runs/run4/model_50000_loss_0.24781.pt"
    checkpoint_path = None

    model_config = GPT.get_default_config()
    model_config.model_type = model_type
    model_config.vocab_size = dataset.get_vocab_size()
    model_config.block_size = dataset.get_block_size()
    model_config.padding_token_id = dataset.pad_tok

    if checkpoint_path is not None:
        model = GPT.from_pretrained(model_type, 
            checkpoint_file=checkpoint_path,
            vocab_size=dataset.get_vocab_size(),
            block_size=1023,
            padding_token_id=dataset.pad_tok)
        print(f"loaded model from {checkpoint_path}", flush=True)
    else:
        model = GPT(model_config)

    print('Training model...', flush=True)
    train_config = Trainer.get_default_config()
    train_config.learning_rate = 5e-6
    train_config.max_iters = 20000
    train_config.num_workers = 0
    train_config.checkpoint_every = 500
    train_config.log_every = 1
    trainer = Trainer(train_config, model, dataset)

    # Save config parameters to output_folder
    with open(output_folder / 'config.txt', 'a+') as f:
        f.write(f"model_config: {model_config}\n")
        f.write(f"train_config: {train_config}\n")

    def log_callback(trainer):
        if trainer.iter_num % train_config.log_every == 0:
            print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}", flush=True)
            # Save to output log file
            with open(output_folder / 'log.txt', 'a') as f:
                f.write(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}\n")

            # Log loss to csv file with iter_num and loss
            with open(output_folder / 'loss.csv', 'a') as f:
                f.write(f"{trainer.iter_num},{trainer.loss.item()}\n")
    
    def checkpoint_callback(trainer):
        if trainer.iter_num % train_config.checkpoint_every == 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': trainer.optimizer.state_dict(),
                'iter_num': trainer.iter_num,
            }
            torch.save(checkpoint, output_folder / f'model_{trainer.iter_num}_loss_{trainer.loss.item():.5f}.pt')
            print(f"saved checkpoint to model_{trainer.iter_num}_loss_{trainer.loss.item():.5f}.pt", flush=True)
            # Save to output log file
            with open(output_folder / 'log.txt', 'a+') as f:
                f.write(f"saved checkpoint to model_{trainer.iter_num}.pt\n")

    trainer.set_callback('log', log_callback)
    trainer.set_callback('checkpoint', checkpoint_callback)

    trainer.run()

if __name__ == '__main__':
    main()