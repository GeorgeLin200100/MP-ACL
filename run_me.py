resume_run = False

import sys

sys.path.append("..")

from pathlib import Path
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from model.llm import LLM
from model.tokenizer import Tokenizer, train_tokenizer

from tools.dataset import NextTokenPredictionDataset
from tools.trainer import train
from tools.config import LLMConfig, TrainingConfig


def generate_and_print(prompt_text, tokenizer, model, device, stop_tokens, max_seq_len):
    # Encode the prompt
    prompt = tokenizer.encode(
        prompt_text,
        beg_of_string=True,
        pad_seq=True,
        seq_len=max_seq_len,
    )
    
    # Prepare the inputs for the model
    inputs = torch.tensor(prompt, dtype=torch.int32).unsqueeze(0).to(device)
    model.to(device)

    # Generate output
    out = model.module.generate(inputs, max_seq_len=1024, stop_tokens=stop_tokens, temperature = 0.8)
    
    # Decode and print the output
    decoded_output = tokenizer.decode(out.tolist())
    truncated_output = decoded_output.split("<|endoftext|>")[0]
    print(truncated_output)
    print()

def save_checkpoint(model, filename="checkpoint.pt.tar"):
    state = {'state_dict': model.state_dict()}
    torch.save(state, filename)

def load_checkpoint(model, filename="checkpoint.pt.tar", optim=None):
    # checkpoint = torch.load(filename, weights_only=True)
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['state_dict'])
    if optim is not None:
        optim.load_state_dict(checkpoint['optimizer'])

llm_config = LLMConfig(
    vocab_size=4096,
    seq_len=128,
    dim_emb=256,
    # num_layers=4,
    num_layers=8,
    num_heads=8,
    emb_dropout=0.1,
    ffn_dim_hidden=4 * 256,
    ffn_bias=False,
)

train_config = TrainingConfig(
    retrain_tokenizer=True,
    #device=get_device(),
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    #batch_size=64,
    batch_size=512,
    learning_rate=3e-5,
    weight_decay=1e-5,
    # max_epochs=2,
    max_epochs=1,
    log_frequency=10,
)

input_file = "./TinyStories-valid.txt"
output_file = Path(input_file).with_suffix(".model")

if not output_file.exists() or train_config.retrain_tokenizer:
    train_tokenizer(input_file, llm_config.vocab_size)

tokenizer = Tokenizer(path = str(output_file), eos_token = str("<|endoftext|>"))

sentence = "<|endoftext|>"
print (tokenizer.sp.EncodeAsPieces(sentence))
print(tokenizer.encode(sentence))

sentence = "Project for the artificial intelligence class in Fudan."
print(tokenizer.sp.EncodeAsPieces(sentence))

assert tokenizer.decode(tokenizer.encode(sentence)) == sentence

ds_train = NextTokenPredictionDataset(input_file, llm_config.seq_len, tokenizer)
dl_train = DataLoader(ds_train, batch_size=train_config.batch_size, shuffle=True)

for inputs, labels in dl_train:
    print(inputs.shape, labels.shape)
    break

stop_tokens = set()

print(f"The <|endoftext|> id is : {tokenizer.sp.piece_to_id("<|endoftext|>")}")

model = LLM(
    vocab_size=tokenizer.vocab_size,
    seq_len=llm_config.seq_len,
    dim_emb=llm_config.dim_emb,
    num_layers=llm_config.num_layers,
    attn_num_heads=llm_config.num_heads,
    emb_dropout=llm_config.emb_dropout,
    ffn_hidden_dim=llm_config.ffn_dim_hidden,
    ffn_bias=llm_config.ffn_bias,
)

model = nn.DataParallel(model)

params_size = sum(p.nelement() * p.element_size() for p in model.parameters())
buffer_size = sum(p.nelement() * p.element_size() for p in model.buffers())
size = (params_size + buffer_size) / 1024**2

print(f"total params: {sum(p.numel() for p in model.parameters()):,d}")
print(f"model size: {size:.3f}MB")

# print(model)
if resume_run:
    load_checkpoint(model)
else: 
    loss_history = train(
        model,
        dl_train,
        train_config.device,
        lr=train_config.learning_rate,
        max_epochs=train_config.max_epochs,
        weight_decay=train_config.weight_decay,
        log_every=train_config.log_frequency,
    )
    print(f"exit train!")

# List of prompts
prompts = [
    "Once upon a time, Tom is very hungry, but he doesn't",
    "I like apple, but Lily loves",
    "Once upon a time, there is a boy named Tom.",
    "Once upon a time, there is a girl named Lily. One day, she",
    "I love the monkey, but",
    "Once upon a time, there is a monkey",
    "Once upon a time, the sun is dimmed.",
    "Once upon a time, the water is dirty.",
    "Once upon a time, sophia won the first prize in the competition.",
    "Once upon a time, there was a little girl named Lucy. She had a pet cat named Tom.",
    "Once upon a time, there was a little brown dog named Spot.",
    "Once upon a time, there was a little boy named Tom.",
    "Once upon a time, there was a big whale. ",
    "Once upon a time, there was a ",
    "Once upon a time, ",
    "Once",
    "Tim and Lily were playing in the park.",
    "Tom had a coin that he liked very much.",
    "Tim and Mia like to play in the backyard.",
    "Tom and Mia went to the zoo with Mom and Dad.",
    "Anna liked to speak to her toys.",
    "Lily was playing with her doll in the garden.",
    "Tim likes to talk about politics.",
    "Sophia never eats breakfast.",
    "Lucy tell a weird story.",
    "Lucy and Lily are playing computer games."
]

# Generate and print outputs for each prompt
for prompt in prompts:
    generate_and_print(prompt, tokenizer, model, train_config.device, stop_tokens, llm_config.seq_len)

if resume_run:
    print("no checkpoint save!")
else:
    save_checkpoint(model)
    print("checkpoint saved!")
