import torch
import torch.nn as nn
import torch.nn.functional as F


def dot_prod_attention(h_t, src_encoding, src_encoding_att_linear, mask=None):
    """
    :param h_t: (batch_size, hidden_size)
    :param src_encoding: (batch_size, src_sent_len, hidden_size * 2)
    :param src_encoding_att_linear: (batch_size, src_sent_len, hidden_size)
    :param mask: (batch_size, src_sent_len)
    """
    # (batch_size, src_sent_len)
    att_weight = torch.bmm(src_encoding_att_linear, h_t.unsqueeze(2)).squeeze(2)
    if mask is not None:
        att_weight.data.masked_fill_(mask, -float('inf'))
    att_weight = F.softmax(att_weight, dim=-1)

    att_view = (att_weight.size(0), 1, att_weight.size(1))
    # (batch_size, hidden_size)
    ctx_vec = torch.bmm(att_weight.view(*att_view), src_encoding).squeeze(1)

    return ctx_vec, att_weight


class Attention(nn.Module):
    """
    Applies an attention mechanism on the output features from the decoder.

    .. math::
            \begin{array}{ll}
            x = context*output \\
            attn = exp(x_i) / sum_j exp(x_j) \\
            output = \tanh(w * (attn * context) + b * output)
            \end{array}

    Args:
        dim(int): The number of expected features in the output

    Inputs: output, context
        - **output** (batch, output_len, dimensions): tensor containing the output features from the decoder.
        - **context** (batch, input_len, dimensions): tensor containing features of the encoded input sequence.

    Outputs: output, attn
        - **output** (batch, output_len, dimensions): tensor containing the attended output features from the decoder.
        - **attn** (batch, output_len, input_len): tensor containing attention weights.

    Attributes:
        linear_out (torch.nn.Linear): applies a linear transformation to the incoming data: :math:`y = Ax + b`.
        mask (torch.Tensor, optional): applies a :math:`-inf` to the indices specified in the `Tensor`.

    Examples::

         >>> attention = seq2seq.models.Attention(256)
         >>> context = Variable(torch.randn(5, 3, 256))
         >>> output = Variable(torch.randn(5, 5, 256))
         >>> output, attn = attention(output, context)

    """

    def __init__(self, dim):
        super(Attention, self).__init__()
        self.linear_out = nn.Linear(dim * 2, dim)
        self.mask = None

    def set_mask(self, mask):
        """
        Sets indices to be masked

        Args:
            mask (torch.Tensor): tensor containing indices to be masked
        """
        self.mask = mask

    def forward(self, output, context):
        batch_size = output.size(0)
        hidden_size = output.size(2)
        input_size = context.size(1)
        # (batch, out_len, dim) * (batch, in_len, dim) -> (batch, out_len, in_len)
        attn = torch.bmm(output, context.transpose(1, 2))
        if self.mask is not None:
            self.mask = self.mask.unsqueeze(1)
            attn.data.masked_fill_(self.mask, -float('inf'))
        attn = F.softmax(attn.view(-1, input_size), dim=1).view(batch_size, -1, input_size)

        # (batch, out_len, in_len) * (batch, in_len, dim) -> (batch, out_len, dim)
        mix = torch.bmm(attn, context)

        # concat -> (batch, out_len, 2*dim)
        combined = torch.cat((mix, output), dim=2)
        # output -> (batch, out_len, dim)
        output = torch.tanh(self.linear_out(combined.view(-1, 2 * hidden_size))).view(batch_size, -1, hidden_size)

        return output, attn


class LuongAttention(nn.Module):
    def __init__(self, hidden_size, emb_size, pool_size=0, **kwargs):
        super(LuongAttention, self).__init__()
        self.hidden_size, self.emb_size, self.pool_size = hidden_size, emb_size, pool_size
        self.linear_in = nn.Linear(hidden_size, hidden_size)
        if pool_size > 0:
            self.linear_out = MaxOut(2 * hidden_size + emb_size, hidden_size, pool_size)
        else:
            self.linear_out = nn.Sequential(nn.Linear(2 * hidden_size + emb_size, hidden_size), nn.Tanh())
        self.softmax = nn.Softmax(dim=1)

    def init_context(self, context):
        self.context = context.transpose(0, 1)

    def forward(self, h, x):
        gamma_h = self.linear_in(h).unsqueeze(2)  # batch * size * 1
        weights = torch.bmm(self.context, gamma_h).squeeze(2)  # batch * time
        weights = self.softmax(weights)  # batch * time
        c_t = torch.bmm(weights.unsqueeze(1), self.context).squeeze(1)  # batch * size
        output = self.linear_out(torch.cat([c_t, h, x], 1))

        return output, weights


class LuongGateAttention(nn.Module):

    def __init__(self, hidden_size, emb_size, prob=0.22, **kwargs):
        super(LuongGateAttention, self).__init__()
        self.hidden_size, self.emb_size = hidden_size, emb_size
        self.linear_in = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Dropout(p=prob))
        self.feed = nn.Sequential(nn.Linear(2 * hidden_size, hidden_size), nn.SELU(), nn.Dropout(p=prob),
                                  nn.Linear(hidden_size, hidden_size), nn.Sigmoid(), nn.Dropout(p=prob))
        self.remove = nn.Sequential(nn.Linear(2 * hidden_size, hidden_size), nn.SELU(), nn.Dropout(p=prob),
                                    nn.Linear(hidden_size, hidden_size), nn.Sigmoid(), nn.Dropout(p=prob))
        self.linear_out = nn.Sequential(nn.Linear(2 * hidden_size, hidden_size), nn.SELU(), nn.Dropout(p=prob),
                                        nn.Linear(hidden_size, hidden_size), nn.SELU(), nn.Dropout(p=prob))
        self.mem_gate = nn.Sequential(nn.Linear(2 * hidden_size, hidden_size), nn.SELU(), nn.Dropout(p=prob),
                                      nn.Linear(hidden_size, hidden_size), nn.Sigmoid(), nn.Dropout(p=prob))
        self.feed_vec = nn.Sequential(nn.Linear(2 * hidden_size, hidden_size), nn.SELU(), nn.Dropout(p=prob))
        self.softmax = nn.Softmax(dim=1)

    def init_context(self, context):
        self.context = context.transpose(0, 1)

    def forward(self, h, embs, m, hops=1):
        x = h
        for i in range(hops):
            gamma_h = self.linear_in(x).unsqueeze(2)
            weights = torch.bmm(self.context, gamma_h).squeeze(2)
            weights = self.softmax(weights)
            c_t = torch.bmm(weights.unsqueeze(1), self.context).squeeze(1)
            x = c_t + x
        feed_gate = self.feed(torch.cat([x, h], 1))
        remove_gate = self.remove(torch.cat([x, h], 1))
        fv = self.feed_vec(torch.cat([x, h], 1))
        # mem_gate = self.mem_gate(torch.cat([m, h], 1))
        # m_x = mem_gate * x
        # output = self.linear_out(torch.cat([m_x, h], 1))
        memory = (remove_gate * m) + (feed_gate * fv)
        mem_gate = self.mem_gate(torch.cat([memory, h], 1))
        m_x = mem_gate * x
        output = self.linear_out(torch.cat([m_x, h], 1))

        return output, weights, memory


class BahdanauAttention(nn.Module):

    def __init__(self, hidden_size, emb_size, **kwargs):
        super(BahdanauAttention, self).__init__()
        self.linear_encoder = nn.Linear(hidden_size, hidden_size)
        self.linear_decoder = nn.Linear(hidden_size, hidden_size)
        self.linear_v = nn.Linear(hidden_size, 1)
        self.linear_r = nn.Linear(hidden_size * 2 + emb_size, hidden_size * 2)
        self.hidden_size = hidden_size
        self.emb_size = emb_size
        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()

    def init_context(self, context):
        self.context = context.transpose(0, 1)

    def forward(self, h, x):
        gamma_encoder = self.linear_encoder(self.context)  # batch * time * size
        gamma_decoder = self.linear_decoder(h).unsqueeze(1)  # batch * 1 * size
        weights = self.linear_v(self.tanh(gamma_encoder + gamma_decoder)).squeeze(2)  # batch * time
        weights = self.softmax(weights)  # batch * time
        c_t = torch.bmm(weights.unsqueeze(1), self.context).squeeze(1)  # batch * size
        r_t = self.linear_r(torch.cat([c_t, h, x], dim=1))
        output = r_t.view(-1, self.hidden_size, 2).max(2)[0]

        return output, weights


class MaxOut(nn.Module):

    def __init__(self, in_feature, out_feature, pool_size):
        super(MaxOut, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.pool_size = pool_size
        self.linear = nn.Linear(in_feature, out_feature * pool_size)

    def forward(self, x):
        output = self.linear(x)
        output = output.view(-1, self.out_feature, self.pool_size)
        output = output.max(2)[0]

        return output


# author yang .
def yang_dot_prod_attention(h_t, src_encoding, src_encoding_att_linear, mask=None):
    """
    :param h_t: (batch_size, tgt_sent_len,dec_hidden_size)
    :param src_encoding: (batch_size, src_sent_len, hidden_size * 2)
    :param src_encoding_att_linear: (batch_size, src_sent_len, hidden_size)
    :param mask: (batch_size, src_sent_len) , 1：no pad .
    :return (batch_size, tgt_sent_len, hidden_size)
    """
    # (batch_size, src_sent_len)
    # att_weight = torch.bmm(src_encoding_att_linear, h_t.unsqueeze(2)).squeeze(2)
    att_weight = torch.bmm(h_t, src_encoding_att_linear.transpose(1, 2))
    if mask is not None:
        mask = mask.unsqueeze(1)
        # pad --> 1
        mask = 1 - mask
        att_weight.data.masked_fill_(mask, -float('inf'))
    att_weight = F.softmax(att_weight, dim=-1)

    # att_view = (att_weight.size(0), 1, att_weight.size(1))
    # (batch_size, hidden_size)
    ctx_vec = torch.bmm(att_weight, src_encoding)

    return ctx_vec, att_weight


class Attention_two_encoder(nn.Module):
    def __init__(self, hidden_size):
        super(Attention_two_encoder, self).__init__()
        self.dim = hidden_size
        self.gate_linear = nn.Linear(3 * hidden_size, hidden_size, bias=False)
        self.w_sent = nn.Linear(self.dim, self.dim, bias=False)
        self.w_template = nn.Linear(self.dim, self.dim, bias=False)
        self.w_output = nn.Linear(self.dim, self.dim, bias=False)
        self.sent_mask = None
        self.template_mask = None

    def set_mask(self, sent, template):
        self.sent_mask = sent
        self.template_mask = template

    def forward(self, output, sent, template):
        batch_size = output.size(0)
        tgt_sent_length = output.size(1)
        sent_ctx, sent_weight = yang_dot_prod_attention(output, sent, sent, mask=self.sent_mask)
        template_ctx, template_weight = yang_dot_prod_attention(output, template, template, mask=self.template_mask)
        gate_start = self.gate_linear(
            torch.cat([sent_ctx, template_ctx, output], dim=-1).view(batch_size * tgt_sent_length, -1))
        gate = torch.sigmoid(gate_start)
        fusion = (1 - gate) * self.w_sent(
            sent_ctx.contiguous().view(batch_size * tgt_sent_length, -1)) + gate * self.w_template(
            template_ctx.contiguous().view(batch_size * tgt_sent_length, -1)) + self.w_output(
            output.contiguous().view(batch_size * tgt_sent_length, -1))
        fusion = torch.tanh(fusion).view(batch_size, tgt_sent_length, -1)
        return fusion, sent_weight, template_weight


class bilinear_attention(nn.Module):
    def __init__(self, size):
        super(bilinear_attention, self).__init__()
        self.w = nn.Parameter(torch.Tensor(size, size))
        nn.init.xavier_uniform(self.w.data)
        self.mask = None

    def set_mask(self, mask):
        self.mask = mask

    def forward(self, dec_hidden, encoder_outputs):
        """"
        dec_hidden : (b, d)
        encoder_outputs: (b, l, d)
        """
        b_hn = dec_hidden.mm(self.w)
        scores = b_hn[:, None, :] * encoder_outputs
        scores = torch.sum(scores, 2)
        if self.mask is not None:
            self.mask = self.mask
            # pad --> 1
            mask = 1 - self.mask
            scores.data.masked_fill_(mask, -float('inf'))
        scores = F.softmax(scores)
        ctx = torch.sum(scores[:, :, None] * encoder_outputs, 1)
        return ctx, scores


class GlobalAttention(nn.Module):
    def __init__(self, dim, alignment_function='general'):
        super(GlobalAttention, self).__init__()
        self.alignment_function = alignment_function
        if self.alignment_function == 'general':
            self.linear_align = nn.Linear(dim, dim, bias=False)
        elif self.alignment_function != 'dot':
            raise ValueError('Invalid alignment function: {}'.format(self.alignment_function))
        self.softmax = nn.Softmax(dim = -1)
        self.linear_context = nn.Linear(dim, dim, bias=False)
        self.linear_query = nn.Linear(dim, dim, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, query, context, context_mask):
        q = query if self.alignment_function=='dot' else self.linear_align(query)
        align = context.bmm(q.unsqueeze(2)).squeeze(2)

        if context_mask is not None:
            align.data.masked_fill_(context_mask==0, -float('inf'))
        # compute attention
        attention = self.softmax(align)
        # compute weight context
        weight_context = attention.unsqueeze(1).bmm(context).squeeze(1)
        # combine weight context and query

        return self.tanh(self.linear_context(weight_context)+self.linear_query(query)), attention, weight_context
















