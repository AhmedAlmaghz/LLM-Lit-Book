# -*- coding: utf-8 -*-
"""
الفصل 6: تضمينات الجُمل — RNN وLSTM وGRU: كيف تتذكر الشبكة ما نسيته؟

الكتاب: المحوّلات والنماذج اللغوية الكبيرة — من النظرية إلى بناء أنظمة ذكية
الجزء الثاني · الفصل 6 · عدد كتل بايثون في هذا الملف: 9

طريقة التشغيل:
  • Colab:  ارفع الملف إلى /content ثم:  !python ch06_sentence-embeddings-rnn-lstm.py
  • Kaggle: أضفه كـ Script أو شغّله خليةً خلية عبر دفتر notebooks/ المقابل.
  • محليًا: pip install -r requirements.txt  ثم  python ch06_sentence-embeddings-rnn-lstm.py

ملاحظة العتاد (CPU/GPU):
  الأكواد تعمل على CPU كما هي، لكن فصول التدريب والاستدلال الثقيل أسرع كثيرًا على GPU.
  في Colab: Runtime ▸ Change runtime type ▸ T4/A100. في Kaggle: Settings ▸ Accelerator ▸ GPU.
  أي كتلة تفترض CUDA صراحةً تجد بجانبها بديل device = "cuda" if torch.cuda.is_available() else "cpu".

الحزم الشائعة (ثبّتها عند الحاجة فقط):
  pip install -q torch transformers datasets accelerate peft trl bitsandbytes \
              sentence-transformers faiss-cpu evaluate scikit-learn pandas numpy matplotlib
"""


# ════════ [1/9] class SimpleRNN(nn.Module): ════════


import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.W_xh = nn.Linear(input_dim, hidden_dim, bias=False)
        self.W_hh = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.W_hy = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, h0=None):
        # x: (batch, seq_len, input_dim)
        batch_size, seq_len, _ = x.size()
        if h0 is None:
            h0 = torch.zeros(batch_size, self.hidden_dim, device=x.device)

        h = h0
        outputs = []
        for t in range(seq_len):
            h = torch.tanh(self.W_xh(x[:, t, :]) + self.W_hh(h))
            outputs.append(h)

        outputs = torch.stack(outputs, dim=1)  # (batch, seq_len, hidden)
        logits = self.W_hy(outputs)             # (batch, seq_len, output_dim)
        return logits, h


# ════════ [2/9] def train_rnn(model, data, epochs=10, seq_len=35, lr=1e-3): ════════


def train_rnn(model, data, epochs=10, seq_len=35, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        h = None
        total_loss = 0
        for batch in data:
            optimizer.zero_grad()
            logits, h = model(batch["input"], h0=h.detach() if h is not None else None)
            loss = criterion(logits.view(-1, logits.size(-1)), batch["target"].view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} | Loss: {total_loss / len(data):.4f}")


# ════════ [3/9] class LSTMCell(nn.Module): ════════


class LSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        # نجمع كل الأوزان في مصفوفة واحدة للكفاءة
        self.W = nn.Linear(input_dim + hidden_dim, 4 * hidden_dim)

    def forward(self, x, state):
        h_prev, c_prev = state
        combined = torch.cat([x, h_prev], dim=-1)
        gates = self.W(combined)
        i, f, g, o = gates.chunk(4, dim=-1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)

        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, (h, c)

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1, dropout=0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.cells = nn.ModuleList([
            LSTMCell(input_dim if l == 0 else hidden_dim, hidden_dim)
            for l in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, state=None):
        # x: (batch, seq_len, input_dim)
        batch_size, seq_len, _ = x.size()
        if state is None:
            h = torch.zeros(batch_size, self.hidden_dim, device=x.device)
            c = torch.zeros(batch_size, self.hidden_dim, device=x.device)
            state = [(h, c) for _ in range(self.num_layers)]

        outputs = []
        for t in range(seq_len):
            inp = x[:, t, :]
            new_state = []
            for l, cell in enumerate(self.cells):
                h, c = cell(inp, state[l])
                inp = self.dropout(h) if l < self.num_layers - 1 else h
                new_state.append((h, c))
            state = new_state
            outputs.append(inp)

        outputs = torch.stack(outputs, dim=1)
        return outputs, state


# ════════ [4/9] packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enfo ════════


from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# lengths: (batch,) طول كل تسلسل قبل الحشو
packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
output_packed, (h_n, c_n) = lstm(packed)
output, _ = pad_packed_sequence(output_packed, batch_first=True)


# ════════ [5/9] class GRUCell(nn.Module): ════════


class GRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.W_z = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.W_r = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.W_h = nn.Linear(input_dim + hidden_dim, hidden_dim)

    def forward(self, x, h):
        combined = torch.cat([x, h], dim=-1)
        z = torch.sigmoid(self.W_z(combined))
        r = torch.sigmoid(self.W_r(combined))
        combined_r = torch.cat([x, r * h], dim=-1)
        h_tilde = torch.tanh(self.W_h(combined_r))
        h_new = (1 - z) * h + z * h_tilde
        return h_new


# ════════ [6/9] class BiLSTM(nn.Module): ════════


class BiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, x):
        out, _ = self.lstm(x)  # (batch, seq, hidden*2)
        return self.fc(out)


# ════════ [7/9] class Encoder(nn.Module): ════════


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        emb = self.embed(x)
        outputs, (h, c) = self.lstm(emb)
        return outputs, (h, c)

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, state):
        emb = self.embed(x)
        outputs, state = self.lstm(emb, state)
        logits = self.fc(outputs)
        return logits, state


# ════════ [8/9] def mean_pooling(hidden_states, attention_mask): ════════


def mean_pooling(hidden_states, attention_mask):
    """
    hidden_states: (batch, seq, hidden)
    attention_mask: (batch, seq) - 1 للكلمات الحقيقية، 0 للحشو
    """
    mask = attention_mask.unsqueeze(-1).float()
    summed = (hidden_states * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


# ════════ [9/9] y_true = [["B-PER", "I-PER", "O", "B-LOC"]] ════════


from seqeval.metrics import classification_report, f1_score

y_true = [["B-PER", "I-PER", "O", "B-LOC"]]
y_pred = [["B-PER", "I-PER", "O", "B-LOC"]]
print(classification_report(y_true, y_pred))
