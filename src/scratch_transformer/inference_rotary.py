import torch
import torch.nn.functional as F
from datasets import load_dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from scratch_transformer.transformer import TransformerModelRotary

# -------------------------------
# 1. Load dataset and build vocab
# -------------------------------
print("Loading dataset and building vocab...")
dataset = load_dataset("Salesforce/wikitext", "wikitext-2-v1")
train_texts = dataset["train"]["text"]

tokenizer = get_tokenizer("basic_english")


def yield_tokens(texts):
    for line in texts:
        yield tokenizer(line)


vocab = build_vocab_from_iterator(yield_tokens(train_texts), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

ntokens = len(vocab)
print(f"Vocab size: {ntokens}")

# -------------------------------
# 2. Load trained model
# -------------------------------
emsize = 200
nhid = 200
nlayers = 2
nhead = 2
dropout = 0.2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading trained model weights...")
model = TransformerModelRotary(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)
model.load_state_dict(torch.load("transformer_best_20251106_000420.pt", map_location=device))
model.eval()


# -------------------------------
# 3. Text generation function
# -------------------------------
def generate_text(
    model,
    prompt,
    vocab,
    tokenizer,
    max_len=50,
    temperature=0.7,
    top_p=0.8,
    repeat_penalty=1.2,
):
    model.eval()
    tokens = tokenizer(prompt)
    input_ids = torch.tensor([vocab[token] for token in tokens], dtype=torch.long).unsqueeze(1)
    input_ids = input_ids.to(next(model.parameters()).device)
    generated = input_ids

    with torch.no_grad():
        for _ in range(max_len):
            src_mask = model.generate_square_subsequent_mask(generated.size(0)).to(input_ids.device)
            output = model(generated, src_mask)
            logits = output[-1, 0, :] / temperature

            # Apply repeat penalty only for last 10 tokens
            recent_tokens = generated[-10:].squeeze()
            for token_id in recent_tokens:
                logits[token_id] /= repeat_penalty

            # Top-p (nucleus) sampling
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            probs = F.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(probs, dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[0] = False  # always keep at least 1 token
            probs[sorted_indices_to_remove] = 0.0
            probs = probs / probs.sum()

            next_token = sorted_indices[torch.multinomial(probs, 1)]
            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=0)

    generated_tokens = [vocab.lookup_token(token_id.item()) for token_id in generated.squeeze()]
    return " ".join(generated_tokens)


# -------------------------------
# 4. Run inference
# -------------------------------
if __name__ == "__main__":
    prompt = "What is "
    print(f"\nPrompt: {prompt}")
    generated_text = generate_text(model, prompt, vocab, tokenizer, max_len=100, temperature=0.8)
    print("\nGenerated text:")
    print(generated_text)
