"""
Train a tokenizer using our own BPE Tokenizer library.
In the style of GPT-4 tokenizer.
"""

import argparse
import os
import time

import torch

from nanochat.common import get_base_dir
from nanochat.dataset import parquets_iter_batched
from nanochat.tokenizer import RustBPETokenizer

# -----------------------------------------------------------------------------
# Parse command line arguments

parser = argparse.ArgumentParser(description="Train a BPE tokenizer")
parser.add_argument(
    "--max-chars",
    type=int,
    default=2_000_000_000,
    help="Maximum characters to train on (default: 2B)",
)
parser.add_argument(
    "--doc-cap",
    type=int,
    default=10_000,
    help="Maximum characters per document (default: 10,000)",
)
parser.add_argument(
    "--vocab-size",
    type=int,
    default=32768,
    help="Vocabulary size (default: 32768 = 2^15)",
)
args = parser.parse_args()
print(f"max_chars: {args.max_chars:,}")
print(f"doc_cap: {args.doc_cap:,}")
print(f"vocab_size: {args.vocab_size:,}")

# -----------------------------------------------------------------------------
# Text iterator


def text_iterator():
    """
    1) Flatten the batches into a single iterator
    2) Crop every document to args.doc_cap characters
    3) Break when we've seen args.max_chars characters
    """
    nchars = 0
    for batch in parquets_iter_batched(split="train"):
        for doc in batch:
            doc_text = doc
            if len(doc_text) > args.doc_cap:
                doc_text = doc_text[: args.doc_cap]
            nchars += len(doc_text)
            yield doc_text
            if nchars > args.max_chars:
                return


text_iter = text_iterator()

# -----------------------------------------------------------------------------
# Train the tokenizer
t0 = time.time()
tokenizer = RustBPETokenizer.train_from_iterator(text_iter, args.vocab_size)
t1 = time.time()
train_time = t1 - t0
print(f"Training time: {train_time:.2f}s")

# -----------------------------------------------------------------------------
# Save the tokenizer to disk
base_dir = get_base_dir()
tokenizer_dir = os.path.join(base_dir, "tokenizer")
tokenizer.save(tokenizer_dir)

# -----------------------------------------------------------------------------
# Quick inline sanity check
test_text = """Hello world! This is a test.
Numbers: 123, 4567, 89
Contractions: I'm, you're, it's
Special chars: @#$%^&*()
Unicode: 你好世界 🌍"""
encoded = tokenizer.encode(test_text)
decoded = tokenizer.decode(encoded)
assert decoded == test_text

# -----------------------------------------------------------------------------
# Cache a lookup table of raw byte lengths for bits-per-byte (BPB) evaluation.
#
# Why not just compare the usual mean cross-entropy loss?
# For target tokens y_1, ..., y_N, let p_i = P(y_i | context_i). With natural
# logarithms, the loss at position i is l_i = -ln(p_i), measured in nats.
# The usual mean loss is L_token = sum_i(l_i) / N, measured in nats per token.
# However, tokenizers with different vocabularies can split the same text into
# different numbers of tokens, so the denominator N depends on the tokenizer.
# A longer token can also contain more information than a shorter token.
# Therefore, a lower loss per token alone does not imply better text modeling
# when comparing models that use different tokenizers.
#
# BPB instead normalizes the total negative log-likelihood by the number of raw
# bytes represented by the evaluated targets. Define:
#   b(v) = number of raw bytes represented by ordinary token ID v;
#          0 for special tokens, which are excluded from this metric.
#   S    = evaluated target positions, excluding special tokens and ignored
#          targets (e.g. labels set to -1 by a training/evaluation mask).
#   B    = sum_{i in S} b(y_i), the total number of evaluated bytes.
# Then, for B > 0:
#   total_nats = sum_{i in S} -ln(p_i)
#   total_bits = total_nats / ln(2), since log2(p) = ln(p) / ln(2)
#   BPB        = total_bits / B
#              = sum_{i in S} -ln(p_i) / (ln(2) * sum_{i in S} b(y_i))
# This is a ratio of totals, not an unweighted mean of per-token l_i / b(y_i).
# Lower BPB means fewer idealized coding bits per byte of evaluated text.
# For example, if 12 bytes have a total NLL of 24 * ln(2) nats, their BPB is
# 24 / 12 = 2, whether those bytes were split into 3 tokens or 6 tokens,
# PROVIDED the total NLL is the same. Changing tokenizers can change the total
# NLL too: BPB gives a common unit, not a guarantee of identical scores.
# Comparisons should use the same text and compatible evaluation conditions.
#
# Why save this table here?
# Byte length is fixed once the tokenizer is trained. Save b(v) once as an
# int32 CPU tensor of shape (vocab_size,), rather than repeatedly decoding
# tokens during model evaluation. nanochat.loss_eval.evaluate_bpb later uses
# token_bytes[y] to obtain target byte lengths by tensor indexing, sums the
# lengths, and excludes zero-length special tokens from BOTH loss and bytes.
# Ignored labels must be masked before indexing; they are not vocabulary IDs.
# Special-token lengths are zero by evaluation convention, not because their
# printable names contain no bytes. This convention does not itself remove
# special tokens from the model's training objective.
#
# Count raw bytes, not Python characters: UTF-8 characters can occupy multiple
# bytes, and a BPE token may contain only part of a character's byte sequence.
# For example, "\u4f60" encodes to three bytes (E4 BD A0), but a token could contain
# just E4. Decoding that token to text may insert a replacement character;
# re-encoding that string would no longer recover its original byte length.
# decode_single_token_bytes() preserves the exact bytes needed by the formula.
vocab_size = tokenizer.get_vocab_size()
special_ids = {tokenizer.encode_special(s) for s in tokenizer.get_special_tokens()}
token_bytes = []
for token_id in range(vocab_size):
    if token_id in special_ids:
        token_bytes.append(0)  # special tokens are not counted
    else:
        # use the raw bytes of the token: decoding to a string first corrupts
        # tokens that are not valid standalone UTF-8 (e.g. the raw bytes >= 0x80)
        num_bytes = len(tokenizer.decode_single_token_bytes(token_id))
        token_bytes.append(num_bytes)
token_bytes = torch.tensor(token_bytes, dtype=torch.int32, device="cpu")
token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.pt")
with open(token_bytes_path, "wb") as f:
    torch.save(token_bytes, f)
print(f"Saved token_bytes to {token_bytes_path}")
