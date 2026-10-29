"""
BPE Tokenizer in the style of GPT-4: train with rustbpe, inference with tiktoken.
"""

import copy
import os
from functools import lru_cache

# Structural markers: each spelling below corresponds to one dedicated token ID.
# train_from_iterator() reserves space for these tokens and assigns their IDs
# after the ordinary BPE vocabulary, in the order listed here. They are not
# learned by BPE merges; the language model learns their use during training.
# Use encode_special() to obtain a marker's ID. encode() uses ordinary encoding,
# so a literal marker spelling inside user text does not become a special ID.
#
# Example conversation layout (line breaks below are only for readability):
#   <|bos|>
#   <|user_start|>What is 2 + 3?<|user_end|>
#   <|assistant_start|>Let me calculate.
#   <|python_start|>2 + 3<|python_end|>
#   <|output_start|>5<|output_end|>
#   The answer is 5.<|assistant_end|>
# The model generates the tool call; the inference engine inserts the result.
# In the current engine, tool calls are handled by use_calculator().
#
# render_conversation() also returns an SFT supervision mask:
#   mask=1: assistant text, Python call markers/content, and assistant_end.
#   mask=0: BOS, user markers/content, assistant_start, and tool output
#           markers/content. These still serve as context for later predictions.
# A completion prompt ends with assistant_start; assistant_end is learned as
# the end of the assistant's reply. No separate system marker is defined here:
# render_conversation() merges an initial system message into the first user message.
SPECIAL_TOKENS = [
    # Beginning of Sequence: prepended to each pretraining document and each
    # rendered conversation. Marks a boundary; it does not itself reset attention
    # or the KV cache. The inference engine also stops a row if BOS is generated.
    "<|bos|>",

    # User message boundaries, inserted by the conversation renderer.
    "<|user_start|>",
    "<|user_end|>",

    # Assistant message boundaries. The start marker is supplied in the prompt;
    # the model learns to generate the end marker, which stops generation.
    "<|assistant_start|>",
    "<|assistant_end|>",

    # Tool call boundaries within an assistant message. The engine collects the
    # expression after python_start and executes it when python_end is generated.
    # Ending a tool call does not end the assistant message.
    "<|python_start|>",
    "<|python_end|>",

    # Tool result boundaries. The engine inserts these markers and the returned
    # result, then the model continues its reply using that result as context.
    # SFT does not train the model to predict externally supplied tool outputs.
    "<|output_start|>",
    "<|output_end|>",
]

# Regex pre-tokenization: split ordinary text into pieces BEFORE byte-level BPE.
# Each piece is encoded as UTF-8 bytes and processed independently by BPE;
# merges cannot cross piece boundaries. A piece is not necessarily one final
# token: it may become several tokens, depending on the learned merge ranks.
# These rules impose boundaries, not linguistic word segmentation.
#
# train_from_iterator() passes this pattern to rustbpe, then retrieves it and
# supplies it as tiktoken.Encoding(pat_str=...). Training and encoding therefore
# use the same boundaries. The pattern belongs to the trained encoding; changing
# this constant does not rewrite an already saved tokenizer.
#
# Syntax used below:
#   |          alternatives, tried left to right at a given position
#   \p{L}      Unicode letters (including Chinese characters, not just A-Z)
#   \p{N}      Unicode numbers (broader than ASCII digits 0-9)
#   \s / \S    whitespace / non-whitespace; \r / \n are CR / LF
#   [^...]     a character outside the listed characters/categories
#   ? / + / *  zero or one / one or more / zero or more repetitions
#   ?+ / ++    possessive repetitions: do not give matched characters back
#              through backtracking
# This is a Unicode-aware regex dialect; Python's standard re does not support
# the \p{...} properties used here. The raw string preserves regex backslashes.
#
# The seven alternatives, in matching order:
#
# 1. '(?i:[sdmt]|ll|ve|re)
#    Match common English apostrophe suffixes: 's, 'd, 'm, 't, 'll, 've, 're.
#    (?i:...) makes only this group case-insensitive, so 'RE also matches.
#    Example: "I'm" splits into "I" and "'m" before BPE.
#
# 2. [^\r\n\p{L}\p{N}]?+\p{L}+
#    Match a run of letters, optionally prefixed by ONE character that is not
#    CR, LF, a letter, or a number. That prefix can be a space or punctuation;
#    it is not restricted to spaces. For example, " world" can stay together.
#    A continuous Chinese letter sequence can also be one piece; this rule
#    does not perform dictionary-based Chinese word segmentation.
#
# 3. \p{N}{1,2}
#    Match numbers in groups of at most two characters:
#    "12345" -> ["12", "34", "5"]. BPE cannot rejoin these separate pieces.
#    This differs from the GPT-4-style {1,3} grouping. The original author's
#    note reports that two-character groups worked best for a 32K vocabulary,
#    avoiding too much vocabulary capacity being spent on number combinations.
#    This limits numeric PIECE length, not the number of digits in the input.
#
# 4.  ?[^\s\p{L}\p{N}]++[\r\n]*
#    Match a run of non-whitespace, non-letter, non-number characters (such as
#    punctuation or symbols), with an optional literal ASCII space before it
#    and any immediately following CR/LF characters. Example: " !!!\n".
#
# 5. \s*[\r\n]
#    Match whitespace ending in CR or LF, grouping line breaks with preceding
#    whitespace. Because \s includes line breaks, a piece may span several.
#
# 6. \s+(?!\S)
#    Match whitespace only if the next position is not a non-whitespace
#    character: either another whitespace character or the end of the text.
#    This captures trailing whitespace. Before a word, it can also backtrack
#    to leave the final space for rule 2: "  hello" -> [" ", " hello"].
#    Unlike ?+ and ++ above, this ordinary + allows backtracking.
#
# 7. \s+
#    Fallback for remaining whitespace, such as a space before a number.
#
# Example pre-tokenization (not final token IDs):
#   "Hello world! 12345" -> ["Hello", " world", "!", " ", "12", "34", "5"]
# No text is discarded: spaces, punctuation, and line breaks remain in pieces.
# Dedicated special-token IDs are inserted separately; this pattern handles
# ordinary text, including literal marker spellings passed through encode().
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

# -----------------------------------------------------------------------------
# Tokenizer based on rustbpe + tiktoken combo
import pickle

import rustbpe
import tiktoken


class RustBPETokenizer:
    """Light wrapper around tiktoken (for efficient inference) but train with rustbpe"""

    def __init__(self, enc, bos_token):
        self.enc = enc
        self.bos_token_id = self.encode_special(bos_token)

    @classmethod
    def train_from_iterator(cls, text_iterator, vocab_size):
        # 1) train using rustbpe
        tokenizer = rustbpe.Tokenizer()  # ty: ignore[unresolved-attribute] - native extension without type stubs
        # the special tokens are inserted later in __init__, we don't train them here
        vocab_size_no_special = vocab_size - len(SPECIAL_TOKENS)
        assert vocab_size_no_special >= 256, (
            f"vocab_size_no_special must be at least 256, got {vocab_size_no_special}"
        )
        tokenizer.train_from_iterator(
            text_iterator, vocab_size_no_special, pattern=SPLIT_PATTERN
        )
        # 2) construct the associated tiktoken encoding for inference
        pattern = tokenizer.get_pattern()
        mergeable_ranks_list = tokenizer.get_mergeable_ranks()
        mergeable_ranks = {bytes(k): v for k, v in mergeable_ranks_list}
        tokens_offset = len(mergeable_ranks)
        special_tokens = {
            name: tokens_offset + i for i, name in enumerate(SPECIAL_TOKENS)
        }
        enc = tiktoken.Encoding(
            name="rustbpe",
            pat_str=pattern,
            mergeable_ranks=mergeable_ranks,  # dict[bytes, int] (token bytes -> merge priority rank)
            special_tokens=special_tokens,  # dict[str, int] (special token name -> token id)
        )
        return cls(enc, "<|bos|>")

    @classmethod
    def from_directory(cls, tokenizer_dir):
        pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
        with open(pickle_path, "rb") as f:
            enc = pickle.load(f)
        return cls(enc, "<|bos|>")

    @classmethod
    def from_pretrained(cls, tiktoken_name):
        # https://github.com/openai/tiktoken/blob/eedc8563/tiktoken_ext/openai_public.py
        enc = tiktoken.get_encoding(tiktoken_name)
        # tiktoken calls the special document delimiter token "<|endoftext|>"
        # yes this is confusing because this token is almost always PREPENDED to the beginning of the document
        # it most often is used to signal the start of a new sequence to the LLM during inference etc.
        # so in nanoChat we always use "<|bos|>" short for "beginning of sequence", but historically it is often called "<|endoftext|>".
        return cls(enc, "<|endoftext|>")

    def get_vocab_size(self):
        return self.enc.n_vocab

    def get_special_tokens(self):
        return self.enc.special_tokens_set

    def id_to_token(self, id):
        return self.enc.decode([id])

    @lru_cache(maxsize=32)  # noqa: B019 - tokenizer instances are long-lived
    def encode_special(self, text):
        return self.enc.encode_single_token(text)

    def get_bos_token_id(self):
        return self.bos_token_id

    def encode(self, text, prepend=None, append=None, num_threads=8):
        # Encode str -> list[int], or list[str] -> list[list[int]].
        # Results are Python lists, not tensors; sequences may have different
        # lengths. This method does not pad, truncate, or concatenate documents.
        # prepend/append each add ONE token per sequence: pass a Python int ID
        # directly, or a special-token name such as "<|bos|>" to look up its ID.
        # num_threads controls batch encoding only; it is unused for a single str.

        # Resolve optional boundary tokens once, before encoding any text.
        # Test against None rather than truthiness: token ID 0 is valid too.
        if prepend is not None:
            prepend_id = (
                prepend if isinstance(prepend, int) else self.encode_special(prepend)
            )
        if append is not None:
            append_id = (
                append if isinstance(append, int) else self.encode_special(append)
            )

        if isinstance(text, str):
            # tiktoken performs regex pre-tokenization and byte-level BPE.
            # "ordinary" treats literal special-token spellings in the text as
            # ordinary content, rather than inserting their dedicated IDs.
            ids = self.enc.encode_ordinary(text)
            if prepend is not None:
                # Insert at the front; existing elements shift right (O(n)).
                ids.insert(0, prepend_id)  # TODO: slightly inefficient here? :( hmm
            if append is not None:
                ids.append(append_id)
        elif isinstance(text, list):
            # Encode each string independently using the batch thread pool.
            # The outer list preserves input order; each inner list contains
            # the token IDs for one document, without padding.
            ids = self.enc.encode_ordinary_batch(text, num_threads=num_threads)
            # Add the same boundary token to EVERY document, not just once
            # around the batch. Each ids_row is a mutable list[int].
            if prepend is not None:
                for ids_row in ids:
                    ids_row.insert(0, prepend_id)  # TODO: same
            if append is not None:
                for ids_row in ids:
                    ids_row.append(append_id)
        else:
            # This wrapper accepts str or list[str], not tuples or tensors.
            raise ValueError(f"Invalid input type: {type(text)}")

        # With hypothetical IDs: body [10, 20], prepend=1, append=2 ->
        # [1, 10, 20, 2]. Batch input returns one such list per input string.
        return ids

    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)

    def decode(self, ids):
        return self.enc.decode(ids)

    def decode_single_token_bytes(self, token_id):
        return self.enc.decode_single_token_bytes(token_id)

    def save(self, tokenizer_dir):
        # save the encoding object to disk
        os.makedirs(tokenizer_dir, exist_ok=True)
        pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
        with open(pickle_path, "wb") as f:
            pickle.dump(self.enc, f)
        print(f"Saved tokenizer encoding to {pickle_path}")

    def render_conversation(self, conversation, max_tokens=2048):
        """
        Tokenize a single Chat conversation (which we call a "doc" or "document" here).
        Input: {"messages": [{"role": "user", "content": "Hi"},
                             {"role": "assistant", "content": "Hello"}, ...]}.
        After an optional initial system message is merged into the first user
        message, roles must alternate user/assistant, starting with user.
        User content is a string; assistant content is a string or a list of
        {"type": "text" | "python" | "python_output", "text": str} parts.

        Returns:
        - ids: list[int] is a list of token ids of this rendered conversation
        - mask: list[int] of same length, mask = 1 for tokens that the Assistant is expected to train on.

        This is a LOSS mask, not an attention mask. A zero masks supervision
        for that target token; the token remains available as input context.
        For next-token training, use inputs=ids[:-1], targets=ids[1:], and
        target_mask=mask[1:]. The caller applies this shift, not this method.
        Thus, assistant_start has mask 0, but predicting the first answer token
        from assistant_start is supervised because the ANSWER token has mask 1.

        Example, assuming "Hi" and "Hello" each encode to one token:
          tokens: BOS user_start Hi user_end assistant_start Hello assistant_end
          mask:    0      0      0     0            0          1         1
        Outputs are Python lists, with no padding or tensor/device conversion.
        """
        # Build tokens and supervision flags in parallel: len(ids) == len(mask).
        ids, mask = [], []

        def add_tokens(token_ids, mask_val):
            # Accept either one special-token ID or a list of ordinary IDs.
            # extend() appends individual elements, keeping both lists flat.
            if isinstance(token_ids, int):
                token_ids = [token_ids]
            ids.extend(token_ids)
            mask.extend([mask_val] * len(token_ids))

        # There is no dedicated system marker in this format. If present,
        # prepend the system text to the first user's text, separated by two
        # newlines, then omit the separate system message from rendering.
        # The input is expected to contain at least one message; the system
        # case additionally requires a following user with string content.
        if conversation["messages"][0]["role"] == "system":
            # Copy nested messages before editing, preserving the caller's data.
            conversation = copy.deepcopy(conversation)  # avoid mutating the original
            messages = conversation["messages"]
            assert messages[1]["role"] == "user", (
                "System message must be followed by a user message"
            )
            messages[1]["content"] = (
                messages[0]["content"] + "\n\n" + messages[1]["content"]
            )
            messages = messages[1:]
        else:
            messages = conversation["messages"]
        assert len(messages) >= 1, f"Conversation has less than 1 message: {messages}"

        # Look up dedicated structural IDs rather than encoding their spellings
        # as ordinary text. The tokenizer must have these chat/tool markers.
        bos = self.get_bos_token_id()
        user_start, user_end = (
            self.encode_special("<|user_start|>"),
            self.encode_special("<|user_end|>"),
        )
        assistant_start, assistant_end = (
            self.encode_special("<|assistant_start|>"),
            self.encode_special("<|assistant_end|>"),
        )
        python_start, python_end = (
            self.encode_special("<|python_start|>"),
            self.encode_special("<|python_end|>"),
        )
        output_start, output_end = (
            self.encode_special("<|output_start|>"),
            self.encode_special("<|output_end|>"),
        )

        # One BOS for the whole conversation, not one BOS per message.
        add_tokens(bos, 0)
        for i, message in enumerate(messages):
            # Enforce user -> assistant -> user -> ...; ending on user is allowed.
            must_be_from = "user" if i % 2 == 0 else "assistant"
            assert message["role"] == must_be_from, (
                f"Message {i} is from {message['role']} but should be from {must_be_from}"
            )

            # content can be either a simple string or a list of parts (e.g. containing tool calls)
            content = message["content"]

            if message["role"] == "user":
                # User text and both boundaries supply context, not SFT targets.
                assert isinstance(content, str), (
                    "User messages are simply expected to be strings"
                )
                value_ids = self.encode(content)
                add_tokens(user_start, 0)
                add_tokens(value_ids, 0)
                add_tokens(user_end, 0)
            elif message["role"] == "assistant":
                # The prompt supplies this start marker; supervise the reply
                # that follows it, rather than predicting the start marker.
                add_tokens(assistant_start, 0)
                if isinstance(content, str):
                    # A plain assistant reply: supervise every content token.
                    value_ids = self.encode(content)
                    add_tokens(value_ids, 1)
                elif isinstance(content, list):
                    # Preserve part order within ONE assistant message. Tool
                    # outputs are embedded parts, not separate role messages.
                    # This renderer serializes existing data; it executes no tools.
                    for part in content:
                        value_ids = self.encode(part["text"])
                        if part["type"] == "text":
                            # string part => simply add the tokens
                            add_tokens(value_ids, 1)
                        elif part["type"] == "python":
                            # Learn both the expression and the markers that
                            # request tool execution during inference.
                            add_tokens(python_start, 1)
                            add_tokens(value_ids, 1)
                            add_tokens(python_end, 1)
                        elif part["type"] == "python_output":
                            # At inference the engine supplies the result and
                            # its boundaries. Exclude all of them from SFT loss,
                            # but keep them as context for the rest of the reply.
                            add_tokens(output_start, 0)
                            add_tokens(value_ids, 0)
                            add_tokens(output_end, 0)
                        else:
                            raise ValueError(f"Unknown part type: {part['type']}")
                else:
                    raise ValueError(f"Unknown content type: {type(content)}")
                # Learn when to finish the reply, including after tool use.
                add_tokens(assistant_end, 1)

        # Keep the same prefix of both lists. This is a hard token limit, not
        # message-aware truncation: it can cut a reply/tool block or remove its
        # closing marker. Full encoding happens before this final truncation.
        ids = ids[:max_tokens]
        mask = mask[:max_tokens]
        return ids, mask

    def visualize_tokenization(self, ids, mask, with_token_id=False):
        """Small helper function useful in debugging: visualize the tokenization of render_conversation"""
        RED = "\033[91m"
        GREEN = "\033[92m"
        RESET = "\033[0m"
        GRAY = "\033[90m"
        tokens = []
        for i, (token_id, mask_val) in enumerate(zip(ids, mask)):
            token_str = self.decode([token_id])
            color = GREEN if mask_val == 1 else RED
            tokens.append(f"{color}{token_str}{RESET}")
            if with_token_id:
                tokens.append(f"{GRAY}({token_id}){RESET}")
        return "|".join(tokens)

    def render_for_completion(self, conversation):
        """
        Used during Reinforcement Learning. In that setting, we want to
        render the conversation priming the Assistant for a completion.
        Unlike the Chat SFT case, we don't need to return the mask.
        """
        # We have some surgery to do: we need to pop the last message (of the Assistant)
        conversation = copy.deepcopy(conversation)  # avoid mutating the original
        messages = conversation["messages"]
        assert messages[-1]["role"] == "assistant", (
            "Last message must be from the Assistant"
        )
        messages.pop()  # remove the last message (of the Assistant) inplace

        # Now tokenize the conversation
        ids, mask = self.render_conversation(conversation)

        # Finally, to prime the Assistant for a completion, append the Assistant start token
        assistant_start = self.encode_special("<|assistant_start|>")
        ids.append(assistant_start)
        return ids


# -----------------------------------------------------------------------------
# nanochat-specific convenience functions


def get_tokenizer():
    from nanochat.common import get_base_dir

    base_dir = get_base_dir()
    tokenizer_dir = os.path.join(base_dir, "tokenizer")
    return RustBPETokenizer.from_directory(tokenizer_dir)


def get_token_bytes(device="cpu"):
    import torch

    from nanochat.common import get_base_dir

    base_dir = get_base_dir()
    tokenizer_dir = os.path.join(base_dir, "tokenizer")
    token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.pt")
    assert os.path.exists(token_bytes_path), (
        f"Token bytes not found at {token_bytes_path}? It gets written by tok_train.py"
    )
    with open(token_bytes_path, "rb") as f:
        token_bytes = torch.load(f, map_location=device)
    return token_bytes
