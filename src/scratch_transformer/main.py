import math
import os
import time
from collections import Counter
from datetime import datetime

import torch
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import login
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from scratch_transformer.transformer import TransformerModel

load_dotenv()
token = os.getenv("TOKEN")
login(token)

dataset = load_dataset(
    "Salesforce/wikitext", "wikitext-2-v1"
)  # You can pick other configs such as "wikitext-103-v1" too

print(dataset)

train_texts = dataset["train"]["text"]
val_texts = dataset["validation"]["text"]
test_texts = dataset["test"]["text"]


tokenizer = get_tokenizer("basic_english")
counter = Counter()

for line in train_texts:
    counter.update(tokenizer(line))


def yield_tokens(texts):
    for line in texts:
        yield tokenizer(line)


vocab = build_vocab_from_iterator(yield_tokens(train_texts), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])


def data_process(raw_texts):
    data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in raw_texts]
    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))


ntokens = len(vocab)

train_data = data_process(train_texts)
val_data = data_process(val_texts)
test_data = data_process(test_texts)

# 4️⃣ Batchify (same as before)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def batchify(data, bsz):
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)


batch_size = 20
eval_batch_size = 10
train_data = batchify(train_data, batch_size)
val_data = batchify(val_data, eval_batch_size)
test_data = batchify(test_data, eval_batch_size)

# 5️⃣ Same model setup
bptt = 35


def get_batch(source, i):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i : i + seq_len]
    target = source[i + 1 : i + 1 + seq_len].reshape(-1)
    return data, target


ntokens = len(vocab)

emsize = 200  # embedding dimension
nhid = 200  # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 2  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 2  # the number of heads in the multiheadattention models
dropout = 0.2  # the dropout value
model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)

criterion = torch.nn.CrossEntropyLoss()
lr = 5.0  # learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)


def train():
    model.train()  # Turn on the train mode
    total_loss = 0.0
    start_time = time.time()
    src_mask = model.generate_square_subsequent_mask(bptt).to(device)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, targets = get_batch(train_data, i)
        optimizer.zero_grad()
        if data.size(0) != bptt:
            src_mask = model.generate_square_subsequent_mask(data.size(0)).to(device)
        output = model(data, src_mask)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        log_interval = 200
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print(
                f"| epoch {epoch:3d} | {batch:5d}/{len(train_data) // bptt:5d} batches | "
                f"lr {scheduler.get_last_lr()[0]:02.2f} | ms/batch {elapsed * 1000 / log_interval:5.2f} | "
                f"loss {cur_loss:5.2f} | ppl {math.exp(cur_loss):8.2f}"
            )
            total_loss = 0
            start_time = time.time()


def evaluate(eval_model, data_source):
    eval_model.eval()  # Turn on the evaluation mode
    total_loss = 0.0
    src_mask = model.generate_square_subsequent_mask(bptt).to(device)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, bptt):
            data, targets = get_batch(data_source, i)
            if data.size(0) != bptt:
                src_mask = model.generate_square_subsequent_mask(data.size(0)).to(device)
            output = eval_model(data, src_mask)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
    return total_loss / (len(data_source) - 1)


best_val_loss = float("inf")
epochs = 3  # The number of epochs
best_model = None

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train()
    val_loss = evaluate(model, val_data)
    print("-" * 89)
    print(
        f"| end of epoch {epoch:3d} | time: {time.time() - epoch_start_time:5.2f}s | valid loss {val_loss:5.2f} | valid ppl {math.exp(val_loss):8.2f}"
    )
    print("-" * 89)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model

    scheduler.step()
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

torch.save(best_model.state_dict(), f"transformer_best_{timestamp}.pt")
