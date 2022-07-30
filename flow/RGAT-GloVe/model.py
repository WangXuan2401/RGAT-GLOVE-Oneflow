import oneflow as flow
import oneflow.nn as nn
import numpy as np

from common.tree import head_to_adj
from common.transformer_encoder import TransformerEncoder
from common.RGAT import RGATEncoder


class RGATABSA(nn.Module):
    def __init__(self, args, emb_matrix=None):
        super().__init__()
        in_dim = args.hidden_dim
        self.args = args
        self.enc = ABSAEncoder(args, emb_matrix=emb_matrix)
        self.classifier = nn.Linear(in_dim, args.num_class)

    def forward(self, inputs):
        hiddens = self.enc(inputs)
        logits = self.classifier(hiddens)
        return logits, hiddens


class ABSAEncoder(nn.Module):
    def __init__(self, args, emb_matrix=None):
        super().__init__()
        self.args = args
        self.emb_matrix = emb_matrix

        # #################### Embeddings ###################
        self.emb = nn.Embedding(args.tok_size, args.emb_dim, padding_idx=0)
        if emb_matrix is not None:
            self.emb.weight = nn.Parameter(emb_matrix.cuda(), requires_grad=False)
        self.pos_emb = (
            nn.Embedding(args.pos_size, args.pos_dim, padding_idx=0) if args.pos_dim > 0 else None
        )  # POS emb
        self.post_emb = (
            nn.Embedding(args.post_size, args.post_dim, padding_idx=0)
            if args.post_dim > 0
            else None
        )  # position emb
        
        # #################### Encoder ###################
        # 初始化self.encoder = DoubleEncoder
        if self.args.model.lower() in ["std", "gat"]:
            embeddings = (self.emb, self.pos_emb, self.post_emb)
            self.encoder = DoubleEncoder(
                args, embeddings, args.hidden_dim, args.num_layers
            )
        elif self.args.model.lower() == "rgat":
            self.dep_emb = (
                nn.Embedding(args.dep_size, args.dep_dim, padding_idx=0)
                if args.dep_dim > 0
                else None
            )  # position emb
            embeddings = (self.emb, self.pos_emb, self.post_emb, self.dep_emb)
            self.encoder = DoubleEncoder(
                args, embeddings, args.hidden_dim, args.num_layers, use_dep=True
            )
        else:
            print(
                "Invalid model name {}, it should be (std, GAT, RGAT)".format(
                    self.args.model.lower()
                )
            )
            exit(0)

        # #################### pooling and fusion modules ###################
        # 初始化聚合模型，基于参数args.output_merge.lower的选定，实现不同形式的fushion
        if self.args.pooling.lower() == "attn":
            self.attn = flow.nn.Linear(args.hidden_dim, 1)

        if self.args.output_merge.lower() != "none":
            self.inp_map = flow.nn.Linear(args.hidden_dim * 2, args.hidden_dim)
        if self.args.output_merge.lower() == "none":
            pass
        elif self.args.output_merge.lower() == "attn":
            self.out_attn_map = flow.nn.Linear(args.hidden_dim * 2, 1)
        elif self.args.output_merge.lower() == "gate":
            self.out_gate_map = flow.nn.Linear(args.hidden_dim * 2, args.hidden_dim)
        elif self.args.output_merge.lower() == "gatenorm" or self.args.output_merge.lower() == "gatenorm2":
            self.out_gate_map = flow.nn.Linear(args.hidden_dim * 2, args.hidden_dim)
            self.out_norm = nn.LayerNorm(args.hidden_dim)
        elif self.args.output_merge.lower() == "addnorm":
            self.out_norm = nn.LayerNorm(args.hidden_dim)
        else:
            print("Invalid output_merge type: ", self.args.output_merge)
            exit()


    def forward(self, inputs):
        tok, asp, pos, head, deprel, post, mask_ori, lengths = inputs  # unpack inputs
        maxlen = max(lengths.data)


        # #################### Encoder ###################
        # DoubleEncoder获得左右结构处理后的值:sent_output, graph_output
        adj_lst, label_lst = [], []
        for idx in range(len(lengths)):
            adj_i, label_i = head_to_adj(
                maxlen,
                head[idx],
                tok[idx],
                deprel[idx],
                lengths[idx],
                mask_ori[idx],
                directed=self.args.direct,
                self_loop=self.args.loop,
            )
            adj_lst.append(adj_i.reshape(1, maxlen, maxlen))
            label_lst.append(label_i.reshape(1, maxlen, maxlen))

        # [B, maxlen, maxlen]
        adj = np.concatenate(adj_lst, axis=0)
        adj = flow.from_numpy(adj).cuda()

        # [B, maxlen, maxlen]
        labels = np.concatenate(label_lst, axis=0)
        label_all = flow.from_numpy(labels).cuda()

        if self.args.model.lower() == "std":
            sent_out, graph_out = self.encoder(adj=None, inputs=inputs, lengths=lengths)
        elif self.args.model.lower() == "gat":
            sent_out, graph_out = self.encoder(adj=adj, inputs=inputs, lengths=lengths)
        elif self.args.model.lower() == "rgat":
            sent_out, graph_out = self.encoder(
                adj=adj, relation_matrix=label_all, inputs=inputs, lengths=lengths
            )
        elif self.args.model.lower() == "rgat-noadj":
            sent_out, graph_out = self.encoder(
                adj=None, relation_matrix=label_all, inputs=inputs, lengths=lengths
            )
        else:
            print(
                "Invalid model name {}, it should be (std, GAT, RGAT)".format(
                    self.args.model.lower()
                )
            )
            exit(0)

        # ###########pooling and fusion #################
        # 对上述Encoder部分获得的sent_output, graph_output进行聚合fushion操作
        # fushion操作的选择由parameter决定
        asp_wn = mask_ori.sum(dim=1).unsqueeze(-1)
        mask = mask_ori.unsqueeze(-1).repeat(1, 1, self.args.hidden_dim)  # mask for h

        
        if self.args.pooling.lower() == "avg":      # avg pooling
            graph_out = (graph_out * mask).sum(dim=1) / asp_wn  # masking
        elif self.args.pooling.lower() == "max":    # max pooling
            graph_out = flow.max(graph_out * mask, dim=1).values
        elif self.args.pooling.lower() == "attn":
            # [B, seq_len, 1]
            attns = flow.tanh(self.attn(graph_out))

            for i in range(mask_ori.size(0)):
                for j in range(mask_ori.size(1)):
                    if mask_ori[i, j] == 0:
                        mask_ori[i, j] = -1e10
            masked_attns = F.softmax(mask_ori * attns.squeeze(-1), dim=1)
            graph_out = flow.matmul(masked_attns.unsqueeze(1), graph_out).squeeze(1)

        if self.args.output_merge.lower() == "none":
            return graph_out

        sent_out = self.inp_map(sent_out)      # avg pooling
        if self.args.pooling.lower() == "avg":
            sent_out = (sent_out * mask).sum(dim=1) / asp_wn
        elif self.args.pooling.lower() == "max":    # max pooling
            sent_out = flow.max(sent_out * mask, dim=1).values

        if self.args.output_merge.lower() == "gate":    # gate feature fusion
            gate = flow.sigmoid(
                self.out_gate_map(flow.cat([graph_out, sent_out], dim=-1))
            )  
            outputs = graph_out * gate + (1 - gate) * sent_out
        elif self.args.output_merge.lower() == "gatenorm":
            gate = flow.sigmoid(
                self.out_gate_map(flow.cat([graph_out, sent_out], dim=-1))
            )  # gatenorm merge
            outputs = self.out_norm(graph_out * gate + (1 - gate) * sent_out)
        elif self.args.output_merge.lower() == "gatenorm2":
            gate = self.out_norm(flow.sigmoid(
                self.out_gate_map(flow.cat([graph_out, sent_out], dim=-1))
            ))  # gatenorm2 merge
            outputs = graph_out * gate + (1 - gate) * sent_out
        elif self.args.output_merge.lower() == "addnorm":
            outputs = self.out_norm(graph_out + sent_out)
        elif self.args.output_merge.lower() == "add":
            outputs = graph_out + sent_out
        elif self.args.output_merge.lower() == "attn":
            att = flow.sigmoid(
                self.out_attn_map(flow.cat([graph_out, sent_out], dim=-1))
            )  # attn merge
            outputs = graph_out * att + (1 - att) * sent_out
        return outputs


class DoubleEncoder(nn.Module):
    def __init__(self, args, embeddings, mem_dim, num_layers, use_dep=False):
        super(DoubleEncoder, self).__init__()
        self.args = args
        self.layers = num_layers
        self.mem_dim = mem_dim
        self.in_dim = args.emb_dim + args.post_dim + args.pos_dim
        if use_dep:
            self.emb, self.pos_emb, self.post_emb, self.dep_emb = embeddings
        else:
            self.emb, self.pos_emb, self.post_emb = embeddings

         # Sentence Encoder:BiLstm
        input_size = self.in_dim
        self.Sent_encoder = nn.LSTM(
            input_size,
            args.rnn_hidden,
            args.rnn_layers,
            batch_first=True,
            dropout=args.rnn_dropout,
            bidirectional=args.bidirect,        # 参数设置为True，使用双向LSTM
        )
        if args.bidirect:
            self.in_dim = args.rnn_hidden * 2
        else:
            self.in_dim = args.rnn_hidden

        # dropout
        self.rnn_drop = nn.Dropout(args.rnn_dropout)
        self.in_drop = nn.Dropout(args.input_dropout)

        # Graph Encoder
        if use_dep:
            self.graph_encoder = RGATEncoder(
                num_layers=num_layers,
                d_model=args.rnn_hidden * 2,
                heads=args.attn_heads,
                d_ff=args.rnn_hidden * 2,
                dropout=args.layer_dropout,
                att_drop=args.att_dropout,
                use_structure=True,
                alpha=args.alpha,
                beta=args.beta,
            )
        else:
            self.graph_encoder = TransformerEncoder(
                num_layers=num_layers,
                d_model=args.rnn_hidden * 2,
                heads=args.attn_heads,
                d_ff=args.rnn_hidden * 2,
                dropout=args.layer_dropout,
            )

        self.out_map = nn.Linear(args.rnn_hidden * 2, args.rnn_hidden)

    def encode_with_rnn(self, rnn_inputs, seq_lens, batch_size):
        h0, c0 = rnn_zero_state(
            batch_size, self.args.rnn_hidden, self.args.rnn_layers, self.args.bidirect
        )

        # rnn_inputs = nn.utils.rnn.pack_padded_sequence(rnn_inputs, seq_lens, batch_first=True)
        rnn_outputs, (ht, ct) = self.Sent_encoder(rnn_inputs, (h0, c0))
        # rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(rnn_outputs, batch_first=True)
        return rnn_outputs

    def forward(self, adj, inputs, lengths, relation_matrix=None):
        tok, asp, pos, head, deprel, post, a_mask, seq_len = inputs  # unpack inputs
        # embedding
        word_embs = self.emb(tok)
        embs = [word_embs]
        if self.args.pos_dim > 0:
            embs += [self.pos_emb(pos)]
        if self.args.post_dim > 0:
            embs += [self.post_emb(post)]
        embs = flow.cat(embs, dim=2)
        embs = self.in_drop(embs)

        # Sentence encoding，右侧结构
        # sent_encoder(BiLSTM)->droput
        sent_output = self.rnn_drop(
            self.encode_with_rnn(embs, seq_len, tok.size()[0])
        )  # [B, seq_len, H]

        mask = adj.eq(0) if adj is not None else None
        key_padding_mask = sequence_mask(lengths) if lengths is not None else None  # [B, seq_len]
        dep_relation_embs = self.dep_emb(relation_matrix) if relation_matrix is not None else None

        # Graph encoding，左侧结构(使用到右侧输出的sent_output)
        inp = sent_output
        graph_output = self.graph_encoder(
            inp, mask=mask, src_key_padding_mask=key_padding_mask, structure=dep_relation_embs,
        )  # [bsz, seq_len, H]
        graph_output = self.out_map(graph_output)
        return sent_output, graph_output


def rnn_zero_state(batch_size, hidden_dim, num_layers, bidirectional=True):
    total_layers = num_layers * 2 if bidirectional else num_layers
    state_shape = (total_layers, batch_size, hidden_dim)
    h0 = c0 = flow.zeros(*state_shape)
    return h0.cuda(), c0.cuda()


def sequence_mask(lengths, max_len=None):
    """
    create a boolean mask from sequence length `[batch_size, 1, seq_len]`
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    max_len = max_len.item()
    return flow.arange(0, max_len, device=lengths.device).type_as(lengths).unsqueeze(0).expand(
        batch_size, max_len
    ) >= (lengths.unsqueeze(1))
