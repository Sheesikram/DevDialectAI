import os
import shutil

hub_cache = os.path.expanduser("~/.cache/torch/hub")
if os.path.exists(hub_cache):
    shutil.rmtree(hub_cache)
import torch
torch.classes.__path__ = []

import streamlit as st
import pickle
import re
import math
from my_transformer import TransformerSeq2Seq, generate_square_subsequent_mask

# ----------------------------
# Special Tokens and Indices
# ----------------------------
PAD_TOKEN = "<PAD>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"
UNK_TOKEN = "<UNK>"

PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3

# ----------------------------
# Tokenization and Detokenization
# ----------------------------
def tokenize(text):
    # Updated regex to capture multi-character operators and words/punctuation.
    pattern = r'==|!=|<=|>=|[A-Za-z0-9_]+|[^\sA-Za-z0-9_]'
    return re.findall(pattern, text)

def detokenize(token_ids, idx2word):
    tokens = []
    for token in token_ids:
        if token in [PAD_IDX, SOS_IDX, EOS_IDX]:
            continue
        tokens.append(idx2word.get(token, UNK_TOKEN))
    return " ".join(tokens)

def format_output(text, language="cpp"):
    # For C++: add newline after semicolons and around braces.
    # For pseudocode, add newline after semicolons.
    if language == "cpp":
        text = text.replace(";", ";\n")
        text = text.replace("{", "{\n")
        text = text.replace("}", "\n}\n")
    else:
        text = text.replace(";", ";\n")
    text = re.sub(r'\n+', '\n', text)
    return text.strip()

# ----------------------------
# Helper Functions for Embeddings
# ----------------------------
def get_src_embedding(model, src_tensor):
    if hasattr(model, 'src_embedding'):
        return model.src_embedding(src_tensor)
    elif hasattr(model, 'src_tok_emb'):
        return model.src_tok_emb(src_tensor)
    elif hasattr(model, 'embedding'):
        return model.embedding(src_tensor)
    else:
        raise AttributeError("Model does not have a known source embedding attribute.")

def get_tgt_embedding(model, tgt_tensor):
    if hasattr(model, 'tgt_embedding'):
        return model.tgt_embedding(tgt_tensor)
    elif hasattr(model, 'tgt_tok_emb'):
        return model.tgt_tok_emb(tgt_tensor)
    elif hasattr(model, 'embedding'):
        return model.embedding(tgt_tensor)
    else:
        raise AttributeError("Model does not have a known target embedding attribute.")

# ----------------------------
# Load Vocabularies and Models for Pseudocode → C++ (Forward)
# ----------------------------
with open("src_vocab1.pkl", "rb") as f:
    src_word2idx_fwd = pickle.load(f)
with open("tgt_vocab1.pkl", "rb") as f:
    tgt_word2idx_fwd = pickle.load(f)
tgt_idx2word_fwd = {idx: word for word, idx in tgt_word2idx_fwd.items()}

SRC_VOCAB_SIZE_FWD = len(src_word2idx_fwd)
TGT_VOCAB_SIZE_FWD = len(tgt_word2idx_fwd)

# ----------------------------
# Hyperparameters and Device
# ----------------------------
d_model = 256
nhead = 8
num_encoder_layers = 6
num_decoder_layers = 6
dim_feedforward = 512
dropout = 0.1
max_seq_length = 256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate and load the forward model.
model_fwd = TransformerSeq2Seq(SRC_VOCAB_SIZE_FWD, TGT_VOCAB_SIZE_FWD, d_model, nhead,
                                num_encoder_layers, num_decoder_layers, dim_feedforward,
                                dropout, max_seq_length)
state_dict_fwd = torch.load("transformer_Trained.pth", map_location=device)
new_state_dict_fwd = {}
for key, value in state_dict_fwd.items():
    new_key = key
    if key.startswith("src_embedding"):
        new_key = key.replace("src_embedding", "src_tok_emb")
    elif key.startswith("tgt_embedding"):
        new_key = key.replace("tgt_embedding", "tgt_tok_emb")
    elif key.startswith("fc_out"):
        new_key = key.replace("fc_out", "generator")
    new_state_dict_fwd[new_key] = value
model_fwd.load_state_dict(new_state_dict_fwd)
model_fwd = model_fwd.to(device)
model_fwd.eval()

# ----------------------------
# Load Vocabularies and Model for C++ → Pseudocode (Reverse)
# ----------------------------
with open("cpp2pseudo_vocab.pkl", "rb") as f:
    src_word2idx_rev = pickle.load(f)
with open("pseudo_vocab.pkl", "rb") as f:
    tgt_word2idx_rev = pickle.load(f)
tgt_idx2word_rev = {idx: word for word, idx in tgt_word2idx_rev.items()}

SRC_VOCAB_SIZE_REV = len(src_word2idx_rev)
TGT_VOCAB_SIZE_REV = len(tgt_word2idx_rev)

# Instantiate and load the reverse model.
model_rev = TransformerSeq2Seq(SRC_VOCAB_SIZE_REV, TGT_VOCAB_SIZE_REV, d_model, nhead,
                                num_encoder_layers, num_decoder_layers, dim_feedforward,
                                dropout, max_seq_length)
state_dict_rev = torch.load("transformer_cpp2pseudo.pth", map_location=device)
new_state_dict_rev = {}
for key, value in state_dict_rev.items():
    new_key = key
    if key.startswith("src_embedding"):
        new_key = key.replace("src_embedding", "src_tok_emb")
    elif key.startswith("tgt_embedding"):
        new_key = key.replace("tgt_embedding", "tgt_tok_emb")
    elif key.startswith("fc_out"):
        new_key = key.replace("fc_out", "generator")
    new_state_dict_rev[new_key] = value
model_rev.load_state_dict(new_state_dict_rev)
model_rev = model_rev.to(device)
model_rev.eval()

# ----------------------------
# Greedy Decoding Function (Shared for Both Directions)
# ----------------------------
def greedy_decode(model, src_tensor, max_len=256):
    tgt_tokens = torch.tensor([SOS_IDX], device=device).unsqueeze(1)  # (1, 1)
    for _ in range(max_len):
        tgt_mask = generate_square_subsequent_mask(tgt_tokens.size(0)).to(device)
        out = model(src_tensor, tgt_tokens, tgt_mask=tgt_mask,
                    src_padding_mask=(src_tensor == PAD_IDX).transpose(0, 1),
                    tgt_padding_mask=(tgt_tokens == PAD_IDX).transpose(0, 1))
        prob = out[-1, 0]
        next_token = torch.argmax(prob).unsqueeze(0).unsqueeze(1)
        tgt_tokens = torch.cat([tgt_tokens, next_token], dim=0)
        if next_token.item() == EOS_IDX:
            break
    return tgt_tokens

# ----------------------------
# Streamlit Interface with Two Tabs
# ----------------------------
st.set_page_config(page_title="DevDialect 🚀", page_icon="💡", layout="wide")
st.title("🚀 DevDialect")
st.markdown("Welcome to **DevDialect** – your interactive translator for code and pseudocode! Choose a tab below to convert between pseudocode and C++ code. 😎")

tab1, tab2 = st.tabs(["📝 Pseudocode → C++", "💻 C++ → Pseudocode"])

# Tab 1: Pseudocode to C++ Translator (Forward)
with tab1:
    st.header("📝 Translate Pseudocode to C++ Code")
    pseudocode_input = st.text_area("Enter pseudocode:", height=200, key="fwd_input")
    if st.button("Translate to C++", key="fwd_button"):
        if pseudocode_input.strip():
            tokens = tokenize(pseudocode_input)
            src_indices = [src_word2idx_fwd.get(token, UNK_IDX) for token in tokens]
            if len(src_indices) < max_seq_length:
                src_indices += [PAD_IDX] * (max_seq_length - len(src_indices))
            else:
                src_indices = src_indices[:max_seq_length]
            src_tensor = torch.tensor(src_indices, device=device).unsqueeze(1)
            with torch.no_grad():
                tgt_tokens = greedy_decode(model_fwd, src_tensor, max_len=max_seq_length)
            tgt_tokens = tgt_tokens.squeeze().tolist()
            code_output = detokenize(tgt_tokens, tgt_idx2word_fwd)
            formatted_output = format_output(code_output, language="cpp")
            st.subheader("Generated C++ Code:")
            st.code(formatted_output, language="cpp")
        else:
            st.error("🚫 Please enter pseudocode!")
            
# Tab 2: C++ to Pseudocode Translator (Reverse)
with tab2:
    st.header("💻 Translate C++ Code to Pseudocode")
    cpp_input = st.text_area("Enter C++ code:", height=200, key="rev_input")
    if st.button("Translate to Pseudocode", key="rev_button"):
        if cpp_input.strip():
            tokens = tokenize(cpp_input)
            src_indices = [src_word2idx_rev.get(token, UNK_IDX) for token in tokens]
            if len(src_indices) < max_seq_length:
                src_indices += [PAD_IDX] * (max_seq_length - len(src_indices))
            else:
                src_indices = src_indices[:max_seq_length]
            src_tensor = torch.tensor(src_indices, device=device).unsqueeze(1)
            with torch.no_grad():
                tgt_tokens = greedy_decode(model_rev, src_tensor, max_len=max_seq_length)
            tgt_tokens = tgt_tokens.squeeze().tolist()
            pseudo_output = detokenize(tgt_tokens, tgt_idx2word_rev)
            formatted_output = format_output(pseudo_output, language="pseudo")
            st.subheader("Generated Pseudocode:")
            st.code(formatted_output, language="text")
        else:
            st.error("🚫 Please enter C++ code!")
