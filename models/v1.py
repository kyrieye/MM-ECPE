import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    Blip2Config,
    Blip2ForConditionalGeneration,
    Blip2QFormerModel,
    Blip2VisionModel,
    Blip2PreTrainedModel
)
from transformers.modeling_outputs import BaseModelOutputWithPooling
from torch.nn import CrossEntropyLoss
from typing import Any, Optional, Tuple, Union
from models.transformer.Models import Transformer_emo
from torch.autograd import Variable
import torch.nn.functional as F
from transformers.generation.logits_process import LogitsProcessorList,TemperatureLogitsWarper,TopKLogitsWarper,TopPLogitsWarper
from transformers.generation.stopping_criteria import StoppingCriteriaList
import numpy as np
class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):

        energies = torch.bmm(q, k.transpose(1, 2))
        energies = energies / self.temperature

        if mask is not None:
            energies = energies.masked_fill(mask, -np.inf)
        
        attn = self.softmax(energies)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, energies, attn

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class VideoBlipVisionModel(Blip2VisionModel):
    """A simple, augmented version of Blip2VisionModel to handle videos."""

    def forward(
        self,
        pixel_values,
        output_attentions,
        output_hidden_states,
        return_dict,
    ):
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        batch, _, time, _, _ = pixel_values.size()

        # flatten along the batch and time dimension to create a tensor of shape
        # (batch * time, channel, height, width)
        flat_pixel_values = pixel_values.permute(0, 2, 1, 3, 4).flatten(end_dim=1)

        (global_embeddings, global_pooled_output, local_embeddings, local_pooled_output) = super().forward(
            pixel_values=flat_pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        seq_len = global_embeddings.size(1)
        global_embeddings = global_embeddings.view(
            batch, time * seq_len, -1
        )
        global_pooler_output = global_pooled_output.view(batch, time, -1)

        seq_len = local_embeddings.size(1)
        local_embeddings = local_embeddings.view(
            batch, time * seq_len, -1
        )
        local_pooler_output = local_pooled_output.view(batch, time, -1)

        return (global_embeddings, global_pooler_output, local_embeddings, local_pooler_output)

    def feature_aggregation(self,phr_feat,video_feat):
        N = video_feat.shape[1]
        phr_feat = self.agg_pool(N)(phr_feat.transpose(1,2)).transpose(1,2)
        combine_feat = torch.concat([self.w_v(phr_feat),video_feat],dim=1)
        A_agg = self.agg_w_a(combine_feat)
        feat_a = self.agg_w_c(combine_feat)
        feat = torch.bmm((self.agg_softmax(A_agg)).transpose(1,2),feat_a)
        return feat
    
    def element_routing(self,agg_feat,emo_embedding):
        A_cross = torch.bmm(emo_embedding,agg_feat.transpose(1,2))
        A_cross_pool = torch.mean(A_cross,dim=1).reshape(-1,1,self.aggregation_k)
        route_feat = torch.bmm(A_cross_pool,agg_feat).repeat(1,emo_embedding.shape[1],1)
        concat_feat = torch.concat([emo_embedding,route_feat],dim=-1)
        R_e = self.alpha * self.ele_route_tanh(self.ele_route_w(concat_feat))
        route_emo_feat = R_e * emo_embedding
        return route_emo_feat

class Blip2ForCauseEmotion(Blip2PreTrainedModel):
    config_class = Blip2Config
    main_input_name = "pixel_values"

    def __init__(self, config: Blip2Config):
        super().__init__(config)
        # self.vision_model = Blip2VisionModel(config.vision_config)

        self.language_projection = nn.Linear(config.qformer_config.hidden_size, config.text_config.hidden_size)
        if config.use_decoder_only_language_model:
            language_model = AutoModelForCausalLM.from_config(config.text_config)
        else:
            language_model = AutoModelForSeq2SeqLM.from_config(config.text_config)

        # Update _tied_weights_keys using the base model used.
        if language_model._tied_weights_keys is not None:
            self._tied_weights_keys = [f"language_model.{k}" for k in language_model._tied_weights_keys]

        self.language_model = language_model

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def get_output_embeddings(self) -> nn.Module:
        return self.language_model.get_output_embeddings()

    def get_encoder(self):
        return self.language_model.get_encoder()

    def get_decoder(self):
        return self.language_model.get_decoder()

    def _tie_weights(self):
        if not self.config.use_decoder_only_language_model:
            self.language_model.encoder.embed_tokens = self.language_model.shared
            self.language_model.decoder.embed_tokens = self.language_model.shared

    def _preprocess_accelerate(self):
        hf_device_map = self.hf_device_map

        if hasattr(self.language_model, "_hf_hook"):
            self.language_model._hf_hook.io_same_device = True 

    def language_model_forward(self,query_output,input_ids,attention_mask,label,return_dict):
        language_model_inputs = self.language_projection(query_output)
        language_model_attention_mask = torch.ones(
            language_model_inputs.size()[:-1], dtype=torch.long, device=language_model_inputs.device
        )
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([language_model_inputs, inputs_embeds.to(language_model_inputs.device)], dim=1)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        expected_device = language_model_attention_mask.device
        attention_mask = torch.cat([language_model_attention_mask, attention_mask.to(expected_device)], dim=1)

        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=return_dict,
            labels=label,
        )
        loss = outputs.loss if return_dict else outputs[0]
        logits = outputs.logits if return_dict else outputs[1]

        if not return_dict:
            return loss, logits
    
    @torch.no_grad()
    def generate_sentence(self,query_output,input_ids,attention_mask):
        batch_size = query_output.shape[0]
        generate_kwargs = {'num_beams': 4, 'max_new_tokens': 128, 'temperature': 0.7, 'top_p': 0.9, 'repetition_penalty': 1.5, 'do_sample': True}
        language_model_inputs = self.language_projection(query_output)
        language_attention_mask = torch.ones(
            language_model_inputs.size()[:-1], dtype=torch.long, device=language_model_inputs.device
        )
        attention_mask = torch.cat([language_attention_mask, attention_mask.to(language_attention_mask.device)], dim=1)

        # concatenate query embeddings with prompt embeddings
        inputs_embeds = self.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([language_model_inputs, inputs_embeds.to(language_model_inputs.device)], dim=1)

        outputs = self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **generate_kwargs,
        )
        return outputs

    def forward(
        self,
        vocab,
        vis_embed: torch.FloatTensor,
        cause_input_ids: torch.FloatTensor,
        caption_input_ids: torch.FloatTensor,
        cause_attention_mask: Optional[torch.LongTensor] = None,
        caption_attention_mask: Optional[torch.LongTensor] = None,
        caption_labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size = vis_embed.shape[0]

        # step0: emotional mask attention
        embedding_emo = nn.Embedding.from_pretrained(torch.FloatTensor(vocab.embedding_weights_em), freeze=False).cuda()

        emotion_vocab_feats = []
        for idx,word in enumerate(vocab.em_vocabs):
            word = Variable(torch.cuda.LongTensor(1).fill_(vocab.em_word2idx[word])) 
            weights = embedding_emo(word)
            emotion_vocab_feats.append(weights)
        emotion_vocabs = torch.stack(emotion_vocab_feats).transpose(0,1)
        emotion_vocab_feats = emotion_vocabs.repeat(batch_size,1,1) 

        emotion_cate_feats = []
        for idx,word in enumerate(vocab.em_cates):
            word = Variable(torch.cuda.LongTensor(1).fill_(vocab.em_word2idx[word])) 
            weights = embedding_emo(word)
            emotion_cate_feats.append(weights)
        emotion_cates = torch.stack(emotion_cate_feats).transpose(0,1)
        emotion_cate_feats = emotion_cates.repeat(batch_size,1,1) 

        emotion_vocab_feats = self.vocab_linear(emotion_vocab_feats)
        emotion_cate_feats = self.cate_linear(emotion_cate_feats)

        emotion_tilde, _, _ = self.emo_attention(src_seq=emotion_vocab_feats, trg_seq=vis_embed,src_seq_34=emotion_cate_feats)
        cause_feature,_,_ = self.attention_cau(emotion_tilde,vis_embed,vis_embed)
        vis_embed_tilde ,_,_  = self.reweight_att(emotion_vocab_feats,vis_embed,vis_embed)

        emotion_feature,em_logits1 ,em_logits2 = self.emo_attention(src_seq=emotion_vocab_feats,trg_seq=vis_embed_tilde,src_seq_34=emotion_cate_feats)
        zero_clo = Variable(torch.cuda.FloatTensor(batch_size,1).fill_(0))
        em_logits1 = torch.log_softmax(em_logits1, dim=1)
        em_logits2 = torch.log_softmax(em_logits2, dim=1) 
        cls_attention1 = torch.cat((zero_clo,em_logits1),1).view(batch_size,-1,1).repeat(1,1,3)
        cls_attention2 = torch.cat((zero_clo,em_logits2),1).view(batch_size,-1,1).repeat(1,1,3)

        # combine global video feature, cause feature(local video feature), emotion_feature
        cause_input_embeds = torch.concat([vis_embed,cause_feature,emotion_feature],dim=1)

        # forward the query tokens through the QFormer, using the image embeddings for cross-attention
        cause_input_attention_mask = torch.ones(cause_input_embeds.size()[:-1], dtype=torch.long, device=vis_embed.device)

        query_tokens = self.query_tokens.expand(cause_input_embeds.shape[0], -1, -1)
        cause_query_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=cause_input_embeds,
            encoder_attention_mask=cause_input_attention_mask,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=return_dict,
        )
        cause_query_output = cause_query_outputs[0]
        
        fc_sentence = self.generate_sentence(cause_query_output,cause_input_ids,cause_attention_mask) # [B, N] step1 sentence about factual and emotional respectively
        
        # step2 generate emotional captions based on factual and emotional semantics obtained in step1
        caption_input_ids = torch.concat([caption_input_ids,fc_sentence],dim=1)
        caption_attention_mask = torch.ones(caption_input_ids.shape, dtype=torch.long, device=caption_input_ids.device)

        caption_input_embeds = vis_embed

        # forward the query tokens through the QFormer, using the image embeddings for cross-attention
        caption_input_attention_mask = torch.ones(caption_input_embeds.size()[:-1], dtype=torch.long, device=caption_input_embeds.device)

        query_tokens = self.query_tokens.expand(caption_input_embeds.shape[0], -1, -1)
        caption_query_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=caption_input_embeds,
            encoder_attention_mask=caption_input_attention_mask,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=return_dict,
        )
        caption_query_output = caption_query_outputs[0]
        caption_loss, _ = self.language_model_forward(caption_query_output,caption_input_ids,caption_attention_mask,caption_labels,return_dict)

        return caption_loss, cls_attention1, cls_attention2
    
    @torch.no_grad()
    def generate(
        self,
        vocab,
        vis_embed,
        cause_input_ids: torch.FloatTensor,
        caption_input_ids: torch.FloatTensor,
        cause_attention_mask: Optional[torch.LongTensor] = None,
        caption_attention_mask: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.LongTensor:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size = vis_embed.shape[0]

        # step0: emotional mask attention
        embedding_emo = nn.Embedding.from_pretrained(torch.FloatTensor(vocab.embedding_weights_em), freeze=False).cuda()

        emotion_vocab_feats = []
        for idx,word in enumerate(vocab.em_vocabs):
            word = Variable(torch.cuda.LongTensor(1).fill_(vocab.em_word2idx[word])) 
            weights = embedding_emo(word)
            emotion_vocab_feats.append(weights)
        emotion_vocabs = torch.stack(emotion_vocab_feats).transpose(0,1)
        emotion_vocab_feats = emotion_vocabs.repeat(batch_size,1,1) 

        emotion_cate_feats = []
        for idx,word in enumerate(vocab.em_cates):
            word = Variable(torch.cuda.LongTensor(1).fill_(vocab.em_word2idx[word])) 
            weights = embedding_emo(word)
            emotion_cate_feats.append(weights)
        emotion_cates = torch.stack(emotion_cate_feats).transpose(0,1)
        emotion_cate_feats = emotion_cates.repeat(batch_size,1,1) 

        emotion_vocab_feats = self.vocab_linear(emotion_vocab_feats)
        emotion_cate_feats = self.cate_linear(emotion_cate_feats)

        emotion_tilde, _, _ = self.emo_attention(src_seq=emotion_vocab_feats, trg_seq=vis_embed,src_seq_34=emotion_cate_feats)
        cause_feature,_,_ = self.attention_cau(emotion_tilde,vis_embed,vis_embed)
        vis_embed_tilde ,_,_  = self.reweight_att(emotion_vocab_feats,vis_embed,vis_embed)

        emotion_feature,_ ,_ = self.attention_emo(src_seq=emotion_vocab_feats,trg_seq=vis_embed_tilde,src_seq_34=emotion_cate_feats)

        # combine global video feature, cause feature(local video feature), emotion_feature
        cause_input_embeds = torch.concat([vis_embed,cause_feature,emotion_feature],dim=1)

        # forward the query tokens through the QFormer, using the image embeddings for cross-attention
        cause_input_attention_mask = torch.ones(cause_input_embeds.size()[:-1], dtype=torch.long, device=vis_embed.device)

        query_tokens = self.query_tokens.expand(cause_input_embeds.shape[0], -1, -1)
        cause_query_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=cause_input_embeds,
            encoder_attention_mask=cause_input_attention_mask,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=return_dict,
        )
        cause_query_output = cause_query_outputs[0]
        
        fc_sentence = self.generate_sentence(cause_query_output,cause_input_ids,cause_attention_mask) # [B, N] step1 sentence about factual and emotional respectively
        
        # step2 generate emotional captions based on factual and emotional semantics obtained in step1
        caption_input_ids = torch.concat([caption_input_ids,fc_sentence],dim=1)
        caption_attention_mask = torch.ones(caption_input_ids.shape, dtype=torch.long, device=caption_input_ids.device)

        caption_input_embeds = vis_embed

        # forward the query tokens through the QFormer, using the image embeddings for cross-attention
        caption_input_attention_mask = torch.ones(caption_input_embeds.size()[:-1], dtype=torch.long, device=caption_input_embeds.device)

        query_tokens = self.query_tokens.expand(caption_input_embeds.shape[0], -1, -1)
        caption_query_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=caption_input_embeds,
            encoder_attention_mask=caption_input_attention_mask,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=return_dict,
        )
        caption_query_output = caption_query_outputs[0]
        caption_sentence = self.generate_sentence(caption_query_output,caption_input_ids,caption_attention_mask)

        return caption_sentence

class VideoBlipForConditionalGeneration(Blip2ForCauseEmotion):
    def __init__(self, config: Blip2Config) -> None:
        # HACK: we call the grandparent super().__init__() to bypass
        # Blip2ForConditionalGeneration.__init__() so we can replace
        # self.vision_model
        super(Blip2ForCauseEmotion, self).__init__(config)
        
        # self.vision_model = VideoBlipVisionModel(config.vision_config)
        # print(sum(p.numel() for p in self.vision_model.parameters() if p.requires_grad))
        self.vocab = vocab
        self.emo_attention = Transformer_emo(d_model=1408,d_inner=512,d_outer = 1408,emo_num = 179)

        self.cate_linear = nn.Linear(300,1408)
        self.vocab_linear = nn.Linear(300,1408)
        
        self.reweight_att = ScaledDotProductAttention(temperature=np.power(1408, 0.5))
        self.attention_cau = ScaledDotProductAttention(temperature=np.power(1408, 0.5))

        self.query_tokens = nn.Parameter(torch.zeros(1, config.num_query_tokens, config.qformer_config.hidden_size))
        self.qformer = Blip2QFormerModel(config.qformer_config)

        self.language_projection = nn.Linear(
            config.qformer_config.hidden_size, config.text_config.hidden_size
        )
        if config.use_decoder_only_language_model:
            language_model = AutoModelForCausalLM.from_config(config.text_config)
        else:
            language_model = AutoModelForSeq2SeqLM.from_config(config.text_config)
        self.language_model = language_model
        print(sum(p.numel() for p in self.language_model.parameters() if p.requires_grad))
        # Initialize weights and apply final processing
        self.post_init()