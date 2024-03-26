# models.py
import copy
from typing import Optional, Union, Callable
from torch import Tensor
from torch.nn import MultiheadAttention
from torch.nn import Dropout
from torch.nn import functional as F
from torch.nn import Module
from torch.nn import ModuleList
from torch.nn import Linear
from torch.nn import LayerNorm
from transformers import AutoModel, AutoConfig
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.models.bert.modeling_bert import TokenClassifierOutput
class SpanBERTForRE(torch.nn.Module):
    def __init__(self, num_labels):
        super(SpanBERTForRE, self).__init__()
        self.num_labels = num_labels
        config = AutoConfig.from_pretrained('spanbert_hf_base/', hidden_dropout_prob = 0.3, attention_probs_dropout_prob = 0.3)
        self.spanbert = AutoModel.from_pretrained('spanbert_hf_base/')
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(768, self.num_labels)
    def forward(
        self,
        input_ids = None,
        attention_mask = None,
        head_mask = None,
        inputs_embeds = None,
        labels = None,
        output_attentions = None,
        output_hidden_states = None,
        return_dict = None,
    ):
        outputs = self.spanbert(
            input_ids,
            attention_mask = attention_mask,
            head_mask = head_mask,
            inputs_embeds = inputs_embeds,
            position_ids = None,
            output_attentions = output_attentions,
            output_hidden_states = output_hidden_states,
            return_dict = return_dict,
        )
        x = outputs[0]
        y = self.dropout(x)
        logits = self.classifier(y)
        outputs = (logits,)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
            return TokenClassifierOutput(
                loss = loss,
                logits = logits,
                hidden_states = x,
            )
        else:
          return TokenClassifierOutput(
            logits = logits,
            hidden_states = x,
            )

def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))
class TransformerEncoder(Module):
    __constants__ = ['norm']
    def __init__(self, encoder_layer, num_layers, num_entities, device, batch_num, max_len, norm=None, enable_nested_tensor=True, mask_check=True):
        super().__init__()
        super().__init__()
        self.num_entities = num_entities
        self.device = device
        self.batch_num = batch_num
        self.max_len = max_len
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")
        self.layers = _get_clones(encoder_layer, num_layers)
        self.word_embeddings = nn.Embedding(30619, 768)
        self.num_layers = num_layers
        self.norm = norm
        layer_norm_eps = 1e-5
        self.norm = LayerNorm(768, eps=layer_norm_eps)
        self.enable_nested_tensor = enable_nested_tensor
        self.mask_check = mask_check
        self.rnn = nn.LSTM(768,
                           768,
                           num_layers = 1,
                           batch_first = True,
                           bidirectional = True,
                           dropout = 0.1)
        self.h0 = torch.randn(2, self.batch_num, 768).to(self.device)
        self.c0 = torch.randn(2, self.batch_num, 768).to(self.device)
        self.classifier2=Linear(768, num_entities)
        self.token_type_embeddings = nn.Embedding(2, 768)
        self.LayerNorm = nn.LayerNorm(768, eps=1e-12)
        self.dropout = nn.Dropout(0.3)
    def forward(
            self,
            b_input_ids: Tensor,
            mask_combined: Tensor,
            b_input_mask: Tensor,
            token_type_ids: Tensor,
            entities: Optional[Tensor] = None,
            mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            is_causal: Optional[bool] = None) -> Tensor:

        src_key_padding_mask = F._canonical_mask(
            mask = src_key_padding_mask,
            mask_name = "src_key_padding_mask",
            other_type = F._none_or_dtype(mask),
            other_name = "mask",
            target_type = b_input_ids.dtype
        )

        mask = F._canonical_mask(
            mask = mask,
            mask_name = "mask",
            other_type = None,
            other_name = "",
            target_type = b_input_ids.dtype,
            check_other = False,
        )

        output = b_input_ids
        convert_to_nested = False
        first_layer = self.layers[0]
        src_key_padding_mask_for_layers = src_key_padding_mask
        why_not_sparsity_fast_path = ''
        str_first_layer = "self.layers[0]"
        if not isinstance(first_layer, torch.nn.TransformerEncoderLayer):
            why_not_sparsity_fast_path = f"{str_first_layer} was not TransformerEncoderLayer"
        elif first_layer.norm_first :
            why_not_sparsity_fast_path = f"{str_first_layer}.norm_first was True"
        elif first_layer.training:
            why_not_sparsity_fast_path = f"{str_first_layer} was in training mode"
        elif not first_layer.self_attn.batch_first:
            why_not_sparsity_fast_path = f" {str_first_layer}.self_attn.batch_first was not True"
        elif not first_layer.self_attn._qkv_same_embed_dim:
            why_not_sparsity_fast_path = f"{str_first_layer}.self_attn._qkv_same_embed_dim was not True"
        elif not first_layer.activation_relu_or_gelu:
            why_not_sparsity_fast_path = f" {str_first_layer}.activation_relu_or_gelu was not True"
        elif not (first_layer.norm1.eps == first_layer.norm2.eps) :
            why_not_sparsity_fast_path = f"{str_first_layer}.norm1.eps was not equal to {str_first_layer}.norm2.eps"
        elif not b_input_ids.dim() == 3:
            why_not_sparsity_fast_path = f"input not batched; expected value.dim() of 3 but got {b_input_ids.dim()}"
        elif not self.enable_nested_tensor:
            why_not_sparsity_fast_path = "enable_nested_tensor was not True"
        elif src_key_padding_mask is None:
            why_not_sparsity_fast_path = "src_key_padding_mask was None"
        elif (((not hasattr(self, "mask_check")) or self.mask_check)
                and not torch._nested_tensor_from_mask_left_aligned(b_input_ids, src_key_padding_mask.logical_not())):
            why_not_sparsity_fast_path = "mask_check enabled, and src and src_key_padding_mask was not left aligned"
        elif output.is_nested:
            why_not_sparsity_fast_path = "NestedTensor input is not supported"
        elif mask is not None:
            why_not_sparsity_fast_path = "src_key_padding_mask and mask were both supplied"
        elif first_layer.self_attn.num_heads % 2 == 1:
            why_not_sparsity_fast_path = "num_head is odd"
        elif torch.is_autocast_enabled():
            why_not_sparsity_fast_path = "autocast is enabled"
        if not why_not_sparsity_fast_path:
            tensor_args = (
                b_input_ids,
                first_layer.self_attn.in_proj_weight,
                first_layer.self_attn.in_proj_bias,
                first_layer.self_attn.out_proj.weight,
                first_layer.self_attn.out_proj.bias,
                first_layer.norm1.weight,
                first_layer.norm1.bias,
                first_layer.norm2.weight,
                first_layer.norm2.bias,
                first_layer.linear1.weight,
                first_layer.linear1.bias,
                first_layer.linear2.weight,
                first_layer.linear2.bias,
            )
            if torch.overrides.has_torch_function(tensor_args):
                why_not_sparsity_fast_path = "some Tensor argument has_torch_function"
            elif not (b_input_ids.is_cuda or 'cpu' in str(b_input_ids.device)):
                why_not_sparsity_fast_path = "src is neither CUDA nor CPU"
            elif torch.is_grad_enabled() and any(x.requires_grad for x in tensor_args):
                why_not_sparsity_fast_path = ("grad is enabled and at least one of value or the "
                                              "input/output projection weights or biases requires_grad")
            if (not why_not_sparsity_fast_path) and (src_key_padding_mask is not None):
                convert_to_nested = True
                output = torch._nested_tensor_from_mask(output, src_key_padding_mask.logical_not(), mask_check=False)
                src_key_padding_mask_for_layers = None
        make_causal = (is_causal is True)
        if is_causal is None:
            if mask is not None:
                sz = mask.size(0)
                causal_comparison = torch.triu(
                    torch.ones(sz, sz, device=mask.device) * float('-inf'), diagonal=1
                ).to(mask.dtype)
                if torch.equal(mask, causal_comparison):
                    make_causal = True
        is_causal = make_causal
        for mod in self.layers:
            input_shape = b_input_ids.size()
            seq_length = input_shape[1]
            buffered_token_type_ids = token_type_ids[:, :seq_length]
            buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
            token_type_ids = buffered_token_type_ids_expanded
            query = torch.zeros(self.batch_num, self.max_len, 768).to(self.device)
            for i in range(mask_combined.shape[0]):
              indices = (mask_combined[i] == True).nonzero(as_tuple = True)[0]
              if indices.nelement() != 0 :
                  query[i] = self.word_embeddings(b_input_ids[i, indices[0]]).clone().detach().long() + self.token_type_embeddings(token_type_ids[i, indices[0]]).clone().detach().long()
            kkk = self.word_embeddings(b_input_ids).clone().detach().long() + self.token_type_embeddings(token_type_ids).clone().detach().long()
            output1 = mod(query, kkk, kkk, entities, src_mask = mask, is_causal = is_causal, src_key_padding_mask = src_key_padding_mask_for_layers)
        output2, (hidden, cell) = self.rnn(output1,(self.h0, self.c0))
        out=torch.add(output2[:,:,:768], output2[:,:,768:])
        O = self.dropout(out)
        logits = self.classifier2(O)
        outputs = (logits,)
        if entities is not None:
            loss_fct = CrossEntropyLoss()
            if src_key_padding_mask is not None:
                active_loss = src_key_padding_mask.view(-1) == 0
                active_logits = logits.view(-1, self.num_entities)
                active_entities = torch.where(
                    active_loss, entities.view(-1), torch.tensor(loss_fct.ignore_index).type_as(entities)
                )
                loss = loss_fct(active_logits, active_entities)
            else:
                loss = loss_fct(logits.view(-1, self.num_entities), entities.view(-1))
            outputs = (loss,) + outputs
        return outputs


class TransformerEncoderLayer(Module):
    __constants__ = ['batch_first', 'norm_first']
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = True, norm_first: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs)
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)
        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        if isinstance(activation, str):
            activation = _get_activation_fn(activation)
        if activation is F.relu or isinstance(activation, torch.nn.ReLU):
            self.activation_relu_or_gelu = 1
        elif activation is F.gelu or isinstance(activation, torch.nn.GELU):
            self.activation_relu_or_gelu = 2
        else:
            self.activation_relu_or_gelu = 0
        self.activation = activation
    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, 'activation'):
            self.activation = F.relu
    def forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            entities: Tensor,
            src_mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            is_causal: bool = False) -> Tensor:
        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(src_mask),
            other_name="src_mask",
            target_type=value.dtype
        )
        src_mask = F._canonical_mask(
            mask = src_mask,
            mask_name = "src_mask",
            other_type = None,
            other_name = "",
            target_type = value.dtype,
            check_other = False,
        )
        why_not_sparsity_fast_path = ''
        if not value.dim() == 3:
            why_not_sparsity_fast_path = f"input not batched; expected src.dim() of 3 but got {value.dim()}"
        elif self.training:
            why_not_sparsity_fast_path = "training is enabled"
        elif not self.self_attn.batch_first :
            why_not_sparsity_fast_path = "self_attn.batch_first was not True"
        elif not self.self_attn._qkv_same_embed_dim :
            why_not_sparsity_fast_path = "self_attn._qkv_same_embed_dim was not True"
        elif not self.activation_relu_or_gelu:
            why_not_sparsity_fast_path = "activation_relu_or_gelu was not True"
        elif not (self.norm1.eps == self.norm2.eps):
            why_not_sparsity_fast_path = "norm1.eps is not equal to norm2.eps"
        elif value.is_nested and (src_key_padding_mask is not None or src_mask is not None):
            why_not_sparsity_fast_path = "neither src_key_padding_mask nor src_mask are not supported with NestedTensor input"
        elif self.self_attn.num_heads % 2 == 1:
            why_not_sparsity_fast_path = "num_head is odd"
        elif torch.is_autocast_enabled():
            why_not_sparsity_fast_path = "autocast is enabled"
        if not why_not_sparsity_fast_path:
            tensor_args = (
                value,
                self.self_attn.in_proj_weight,
                self.self_attn.in_proj_bias,
                self.self_attn.out_proj.weight,
                self.self_attn.out_proj.bias,
                self.norm1.weight,
                self.norm1.bias,
                self.norm2.weight,
                self.norm2.bias,
                self.linear1.weight,
                self.linear1.bias,
                self.linear2.weight,
                self.linear2.bias,
            )
            if torch.overrides.has_torch_function(tensor_args):
                why_not_sparsity_fast_path = "some Tensor argument has_torch_function"
            elif not all((x.is_cuda or 'cpu' in str(x.device)) for x in tensor_args):
                why_not_sparsity_fast_path = "some Tensor argument is neither CUDA nor CPU"
            elif torch.is_grad_enabled() and any(x.requires_grad for x in tensor_args):
                why_not_sparsity_fast_path = ("grad is enabled and at least one of value or the "
                                              "input/output projection weights or biases requires_grad")

            if not why_not_sparsity_fast_path:
                merged_mask, mask_type = self.self_attn.merge_masks(src_mask, src_key_padding_mask, value)
                return torch._transformer_encoder_layer_fwd(
                    value.float(),
                    self.self_attn.embed_dim,
                    self.self_attn.num_heads,
                    self.self_attn.in_proj_weight,
                    self.self_attn.in_proj_bias,
                    self.self_attn.out_proj.weight,
                    self.self_attn.out_proj.bias,
                    self.activation_relu_or_gelu == 2,
                    self.norm_first,
                    self.norm1.eps,
                    self.norm1.weight,
                    self.norm1.bias,
                    self.norm2.weight,
                    self.norm2.bias,
                    self.linear1.weight,
                    self.linear1.bias,
                    self.linear2.weight,
                    self.linear2.bias,
                    merged_mask.bool(),
                    mask_type,
                )
        if self.norm_first:
            x = value.float() + self._sa_block(self.norm1(query.float()), self.norm1(key.float()), self.norm1(value.float()), src_mask, src_key_padding_mask, is_causal = is_causal)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(value.float() + self._sa_block(self.norm1(query.float()), self.norm1(key.float()), self.norm1(value.float()), src_mask, src_key_padding_mask, is_causal = is_causal))
            x = self.norm2(x + self._ff_block(x))
        return x
    def _sa_block(self, query: Tensor, key: Tensor, value: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False) -> Tensor:
        x = self.self_attn(query, key, value,
                           attn_mask = attn_mask,
                           key_padding_mask = key_padding_mask,
                           need_weights = False, is_causal = is_causal)[0]
        return self.dropout1(x)
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)
class ONIEXNetwork(torch.nn.Module):
    def __init__(self, modelR, modelE, device, tag2idx):
        super(ONIEXNetwork, self).__init__()
        self.NetA = modelR
        self.NetB = modelE
        self.device = device
        self.tag2idx = tag2idx
    def forward(
        self,
        input_ids = None,
        attention_mask = None,
        token_type_ids = None,
        head_mask = None,
        inputs_embeds = None,
        labels = None,
        output_attentions = None,
        output_hidden_states = None,
        return_dict = None,
        b_ppos = None,
        entities = None
    ):
      output1 = self.NetA(
            input_ids,
            attention_mask = attention_mask,
            head_mask = head_mask,
            inputs_embeds = inputs_embeds,
            output_attentions = output_attentions,
            output_hidden_states = output_hidden_states,
            return_dict = return_dict,
            labels = labels
        )
      loss_R, output_R, x = output1[:3]
      logits = torch.argmax(F.log_softmax(output_R,dim = 2),dim = 2)
      mask_a = torch.logical_or(torch.eq(logits, self.tag2idx['R-B']), torch.eq(logits, self.tag2idx['R-I']))
      mask_c = torch.eq(b_ppos, 1)
      mask_combined = torch.logical_and(mask_a, mask_c)
      mask = (torch.logical_or(torch.logical_not(attention_mask), mask_combined)).int()
      output2 = self.NetB( b_input_ids = input_ids.to(self.device), mask_combined = mask_combined.to(self.device),b_input_mask = attention_mask.to(self.device),
                         entities = entities.to(self.device),token_type_ids = b_ppos.to(self.device), src_key_padding_mask = mask.to(self.device).float())
      loss_E, output_E = output2[:3]
      return loss_R, loss_E, output_R, output_E