# -*- coding: utf-8 -*-

import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration, Blip2Config
import torch.nn as nn
import torch.nn.functional as F
from models.VAT_Former import BertConfig, BertLMHeadModel


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class VATFormer(nn.Module):
    def __init__(self, num_query_token, vision_graph_width, model_config, cross_attention_freq=2):
        super().__init__()
        encoder_config = BertConfig.from_pretrained("allenai/scibert_scivocab_uncased",
                                                    cache_dir='../preTrain_model/allenai/scibert_scivocab_uncased/')
        encoder_config.encoder_width = vision_graph_width
        encoder_config.add_cross_attention = True
        encoder_config.add_pooling_layer = False
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        self.Qformer = BertLMHeadModel.from_pretrained(
            "allenai/scibert_scivocab_uncased", config=encoder_config,
            cache_dir='../preTrain_model/allenai/scibert_scivocab_uncased/'
        )
        self.query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size) # num_query_token [768], hidden_size [768]
        )
        self.query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)


class VATModel(nn.Module):
    def __init__(self, args_config, fp=False, modal=None, device=None):
        super().__init__()

        self.args_config = args_config
        self.blip2conf = Blip2Config()
        self.model = Blip2ForConditionalGeneration(self.blip2conf)
        self.model.vision_model = ImageTransform()
        self.model.video_model = VideoTransform()
        self.model.audio_model = AudioTransform(self.args_config)

        self.model.ln_vision = LayerNorm(768)
        self.model.ln_audio = LayerNorm(768)
        self.model.ln_video = LayerNorm(768)

        vatformer = VATFormer(384, 768, model_config=self.args_config)
        self.model.vat_former = vatformer.Qformer
        self.model.query_tokens = vatformer.query_tokens #

        self.isml_head = nn.Linear(self.model.vat_former.config.hidden_size, 2)
        self.vsml_head = nn.Linear(self.model.vat_former.config.hidden_size, 2)
        self.asml_head = nn.Linear(self.model.vat_former.config.hidden_size, 2)
        self.tsml_head = nn.Linear(self.model.vat_former.config.hidden_size, 2)

        embed_dim = 256
        self.vision_proj = nn.Linear(self.model.vat_former.config.hidden_size, embed_dim)
        self.video_proj = nn.Linear(self.model.vat_former.config.hidden_size, embed_dim)
        self.audio_proj = nn.Linear(self.model.vat_former.config.hidden_size, embed_dim)
        self.cs_text_proj = nn.Linear(self.model.vat_former.config.hidden_size, embed_dim)
        self.text_proj = nn.Linear(self.model.vat_former.config.hidden_size, embed_dim)

        self.model_freeze()
        self.device = device
        self.temp = nn.Parameter(0.07 * torch.ones([]))
        self.task = []
        self.modal = modal  # task
        self.input_modal = [m['inputs_modal'][0] for m in self.modal]
        self.output_modal = [m['outputs_modal'][0] for m in self.modal]
        if 'summary' in self.output_modal:
            if 'image' in self.input_modal:
                self.task.append('isml')
                self.task.append('iscl')
            if 'text' in self.input_modal:
                self.task.append('tsml')
                self.task.append('tscl')
            if 'video' in self.input_modal:
                self.task.append('vsml')
                self.task.append('vscl')
            if 'audio' in self.input_modal:
                self.task.append('asml')
                self.task.append('ascl')

    def model_freeze(self):

        for param in self.model.audio_model.parameters():
            param.requires_grad = False
        for param in self.model.video_model.parameters():
            param.requires_grad = False
        for param in self.model.vision_model.parameters():
            param.requires_grad = False

    def forward(self, sci):
        loss = 0
        output_modality = self.output_modal[0]
        if output_modality == 'summary':
            text = sci['Summary']
        else:
            text = sci[output_modality]
        batch_size = text['input_ids'].size(0)

        if 'image' in self.input_modal:
            image_embeds = self.model.ln_vision(self.model.vision_model(sci))
            image_embeds = image_embeds.float()
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
                image_embeds.device
            )
            image_targets = torch.arange(batch_size).to(image_embeds.device)

        if 'video' in self.input_modal:
            video_embeds = self.model.ln_video(self.model.video_model(sci))
            video_embeds = video_embeds.float()
            video_atts = torch.ones(video_embeds.size()[:-1], dtype=torch.long).to(
                video_embeds.device
            )
            video_targets = torch.arange(batch_size).to(video_embeds.device)

        if 'audio' in self.input_modal:
            audio_embeds = self.model.ln_audio(self.model.audio_model(sci))
            audio_embeds = audio_embeds.float()
            audio_atts = torch.ones(audio_embeds.size()[:-1], dtype=torch.long).to(
                audio_embeds.device
            )
            audio_targets = torch.arange(batch_size).to(audio_embeds.device)

        if 'text' in self.input_modal:
            cs_text = sci['Text']
            cs_text_embeds = self.model.vat_former.bert(
                cs_text['input_ids'], # The input tensor containing sequences of token IDs to be fed into the model.
                attention_mask=cs_text['attention_mask'], # The attention mask tensor that specifies which tokens should be attended to (typically 1 for tokens to be considered and 0 for padding tokens).
                return_dict=True,
            ).last_hidden_state # [3, 445, 768] batch_size, seq_len, feat_len
            cs_text_atts = torch.ones(cs_text_embeds.size()[:-1], dtype=torch.long).to(
                cs_text_embeds.device
            )
            cs_text_targets = torch.arange(batch_size).to(cs_text_embeds.device)
        text_output = self.model.vat_former.bert(
            text['input_ids'],
            attention_mask=text['attention_mask'],
            return_dict=True,
        )
        text_feat = F.normalize(
            self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
        )


        if "isml" in self.task:

            image_embeds_list = []
            text_input_ids_list = []
            text_attention_mask_list = []

            for i in range(image_embeds.shape[0]):
                # Original samples
                image_embeds_list.append(image_embeds[i])
                text_input_ids_list.append(text['input_ids'][i])
                text_attention_mask_list.append(text['attention_mask'][i])

                # Negative samples (neg_text_input_ids corresponds to image_embeds)
                neg_text_input_ids = text['input_ids'][i - 1] if i == image_embeds.shape[0] - 1 else text['input_ids'][
                    i + 1]
                neg_text_attention_mask = text['attention_mask'][i - 1] if i == image_embeds.shape[0] - 1 else \
                text['attention_mask'][i + 1]
                text_input_ids_list.append(neg_text_input_ids)
                text_attention_mask_list.append(neg_text_attention_mask)
                image_embeds_list.append(image_embeds[i])

                # Negative samples (text_input_ids corresponds to neg_image_embeds)
                neg_image_embeds = image_embeds[i - 1] if i == image_embeds.shape[0] - 1 else image_embeds[i + 1]
                image_embeds_list.append(neg_image_embeds)
                text_input_ids_list.append(text['input_ids'][i])
                text_attention_mask_list.append(text['attention_mask'][i])

            # Stack all samples into two large tensors (list -> tensor)
            image_embeds_all = torch.stack(image_embeds_list, dim=1).reshape(-1, image_embeds.size(1),
                                                                             image_embeds.size(2))
            text_input_ids_all = torch.stack(text_input_ids_list, dim=1).reshape(-1, text['input_ids'].size(1))
            text_attenetion_mask_all = torch.stack(text_attention_mask_list, dim=1).reshape(-1,
                                                                                            text['attention_mask'].size(
                                                                                                1))

            # Create image attention masks for the concatenated tensor
            image_atts_all = torch.ones(image_embeds_all.size()[:-1], dtype=torch.long).to(
                image_embeds_all.device
            )
            query_tokens_isml = self.model.query_tokens.expand(text_input_ids_all.shape[0], -1, -1)
            query_atts_isml = torch.ones(query_tokens_isml.size()[:-1], dtype=torch.long).to(
                image_embeds_all.device
            )
            attention_mask_all = torch.cat([query_atts_isml, text_attenetion_mask_all], dim=1)

            output_isml = self.model.vat_former.bert(
                text_input_ids_all,
                query_embeds=query_tokens_isml,
                attention_mask=attention_mask_all,
                encoder_hidden_states=image_embeds_all,
                encoder_attention_mask=image_atts_all,
                modal='image',
                return_dict=True,
            )
            isml_embeddings = output_isml.last_hidden_state[:, : query_tokens_isml.size(1), :]

            isml_logit = self.isml_head(isml_embeddings)
            isml_logit = isml_logit.mean(dim=1)

            # # Create labels: 1 for the original samples, 0 for the negative samples
            labels = torch.cat([torch.ones(batch_size), torch.zeros(batch_size * 2)], dim=0).long().to(isml_logit.device)

            # Calculate cross entropy loss
            loss_isml = F.cross_entropy(isml_logit, labels)

            loss = loss + loss_isml

        if "vsml" in self.task:
            video_embeds_list = []
            text_input_ids_list = []
            text_attention_mask_list = []

            for i in range(video_embeds.shape[0]):
                # Original samples
                video_embeds_list.append(video_embeds[i])
                text_input_ids_list.append(text['input_ids'][i])
                text_attention_mask_list.append(text['attention_mask'][i])

                # Negative samples (neg_text_input_ids corresponds to image_embeds)
                neg_text_input_ids = text['input_ids'][i - 1] if i == video_embeds.shape[0] - 1 else text['input_ids'][
                    i + 1]
                neg_text_attention_mask = text['attention_mask'][i - 1] if i == video_embeds.shape[0] - 1 else \
                text['attention_mask'][i + 1]
                text_input_ids_list.append(neg_text_input_ids)
                text_attention_mask_list.append(neg_text_attention_mask)
                video_embeds_list.append(video_embeds[i])

                # Negative samples (text_input_ids corresponds to neg_image_embeds)
                neg_video_embeds = video_embeds[i - 1] if i == video_embeds.shape[0] - 1 else video_embeds[i + 1]
                video_embeds_list.append(neg_video_embeds)
                text_input_ids_list.append(text['input_ids'][i])
                text_attention_mask_list.append(text['attention_mask'][i])

            # Stack all samples into two large tensors
            video_embeds_all = torch.stack(video_embeds_list, dim=1).reshape(-1, video_embeds.size(1),
                                                                             video_embeds.size(2))
            text_input_ids_all = torch.stack(text_input_ids_list, dim=1).reshape(-1, text['input_ids'].size(1))
            text_attenetion_mask_all = torch.stack(text_attention_mask_list, dim=1).reshape(-1,
                                                                                            text['attention_mask'].size(
                                                                                                1))
            # Create image attention masks for the concatenated tensor
            video_atts_all = torch.ones(video_embeds_all.size()[:-1], dtype=torch.long).to(
                video_embeds_all.device
            )
            query_tokens_vsml = self.model.query_tokens.expand(text_input_ids_all.shape[0], -1, -1)
            query_atts_vsml = torch.ones(query_tokens_vsml.size()[:-1], dtype=torch.long).to(
                video_embeds_all.device
            )
            attention_mask_all = torch.cat([query_atts_vsml, text_attenetion_mask_all], dim=1)

            output_vsml = self.model.vat_former.bert(
                text_input_ids_all,
                query_embeds=query_tokens_vsml,
                attention_mask=attention_mask_all,
                encoder_hidden_states=video_embeds_all,
                encoder_attention_mask=video_atts_all,
                modal='video',
                return_dict=True,
            )
            vsml_embeddings = output_vsml.last_hidden_state[:, : query_tokens_vsml.size(1), :]

            vsml_logit = self.vsml_head(vsml_embeddings)
            vsml_logit = vsml_logit.mean(dim=1)
            # isml_logit = self.isml_head(isml_embeddings)
            # Create labels: 1 for the original samples, 0 for the negative samples
            labels = torch.cat([torch.ones(batch_size), torch.zeros(batch_size * 2)], dim=0).long().to(vsml_logit.device)

            # Calculate cross entropy loss
            loss_vsml = F.cross_entropy(vsml_logit, labels)

            loss = loss + loss_vsml

        if "asml" in self.task:
            audio_embeds_list = []
            text_input_ids_list = []
            text_attention_mask_list = []

            for i in range(audio_embeds.shape[0]):
                # Original samples
                audio_embeds_list.append(audio_embeds[i])
                text_input_ids_list.append(text['input_ids'][i])
                text_attention_mask_list.append(text['attention_mask'][i])

                # Negative samples (neg_text_input_ids corresponds to image_embeds)
                neg_text_input_ids = text['input_ids'][i - 1] if i == audio_embeds.shape[0] - 1 else text['input_ids'][
                    i + 1]
                neg_text_attention_mask = text['attention_mask'][i - 1] if i == audio_embeds.shape[0] - 1 else \
                text['attention_mask'][i + 1]
                text_input_ids_list.append(neg_text_input_ids)
                text_attention_mask_list.append(neg_text_attention_mask)
                audio_embeds_list.append(audio_embeds[i])

                # Negative samples (text_input_ids corresponds to neg_image_embeds)
                neg_audio_embeds = audio_embeds[i - 1] if i == audio_embeds.shape[0] - 1 else audio_embeds[i + 1]
                audio_embeds_list.append(neg_audio_embeds)
                text_input_ids_list.append(text['input_ids'][i])
                text_attention_mask_list.append(text['attention_mask'][i])

            # Stack all samples into two large tensors
            audio_embeds_all = torch.stack(audio_embeds_list, dim=1).reshape(-1, audio_embeds.size(1),
                                                                             audio_embeds.size(2))
            text_input_ids_all = torch.stack(text_input_ids_list, dim=1).reshape(-1, text['input_ids'].size(1))
            text_attenetion_mask_all = torch.stack(text_attention_mask_list, dim=1).reshape(-1,
                                                                                            text['attention_mask'].size(
                                                                                                1))
            # Create image attention masks for the concatenated tensor
            audio_atts_all = torch.ones(audio_embeds_all.size()[:-1], dtype=torch.long).to(
                audio_embeds_all.device
            )
            query_tokens_asml = self.model.query_tokens.expand(text_input_ids_all.shape[0], -1, -1)
            query_atts_asml = torch.ones(query_tokens_asml.size()[:-1], dtype=torch.long).to(
                audio_embeds_all.device
            )
            attention_mask_all = torch.cat([query_atts_asml, text_attenetion_mask_all], dim=1)

            output_asml = self.model.vat_former.bert(
                text_input_ids_all,
                query_embeds=query_tokens_asml,
                attention_mask=attention_mask_all,
                encoder_hidden_states=audio_embeds_all,
                encoder_attention_mask=audio_atts_all,
                modal='audio',
                return_dict=True,
            )
            asml_embeddings = output_asml.last_hidden_state[:, : query_tokens_asml.size(1), :]

            asml_logit = self.asml_head(asml_embeddings)
            asml_logit = asml_logit.mean(dim=1)

            # Create labels: 1 for the original samples, 0 for the negative samples
            labels = torch.cat([torch.ones(batch_size), torch.zeros(batch_size * 2)], dim=0).long().to(asml_logit.device)

            # Calculate cross entropy loss
            loss_asml = F.cross_entropy(asml_logit, labels)

            loss = loss + loss_asml


        if "tsml" in self.task:
            cs_text_embeds_list = []
            text_input_ids_list = []
            text_attention_mask_list = []

            for i in range(cs_text_embeds.shape[0]):
                # Original samples
                cs_text_embeds_list.append(cs_text_embeds[i])
                text_input_ids_list.append(text['input_ids'][i])
                text_attention_mask_list.append(text['attention_mask'][i])

                # Negative samples (neg_text_input_ids corresponds to image_embeds)
                neg_text_input_ids = text['input_ids'][i - 1] if i == cs_text_embeds.shape[0] - 1 else \
                text['input_ids'][i + 1]
                neg_text_attention_mask = text['attention_mask'][i - 1] if i == cs_text_embeds.shape[0] - 1 else \
                text['attention_mask'][i + 1]
                text_input_ids_list.append(neg_text_input_ids)
                text_attention_mask_list.append(neg_text_attention_mask)
                cs_text_embeds_list.append(cs_text_embeds[i])

                # Negative samples (text_input_ids corresponds to neg_image_embeds)
                neg_cs_text_embeds = cs_text_embeds[i - 1] if i == cs_text_embeds.shape[0] - 1 else cs_text_embeds[
                    i + 1]
                cs_text_embeds_list.append(neg_cs_text_embeds)
                text_input_ids_list.append(text['input_ids'][i])
                text_attention_mask_list.append(text['attention_mask'][i])

            # Stack all samples into two large tensors
            cs_text_embeds_all = torch.stack(cs_text_embeds_list, dim=1).reshape(-1, cs_text_embeds.size(1),
                                                                                 cs_text_embeds.size(2))
            text_input_ids_all = torch.stack(text_input_ids_list, dim=1).reshape(-1, text['input_ids'].size(1))
            text_attenetion_mask_all = torch.stack(text_attention_mask_list, dim=1).reshape(-1,
                                                                                            text['attention_mask'].size(
                                                                                                1))
            # Create image attention masks for the concatenated tensor
            cs_text_atts_all = torch.ones(cs_text_embeds_all.size()[:-1], dtype=torch.long).to(
                cs_text_embeds_all.device
            )
            query_tokens_tsml = self.model.query_tokens.expand(text_input_ids_all.shape[0], -1, -1)
            query_atts_tsml = torch.ones(query_tokens_tsml.size()[:-1], dtype=torch.long).to(
                cs_text_embeds_all.device
            )
            attention_mask_all = torch.cat([query_atts_tsml, text_attenetion_mask_all], dim=1)

            output_tsml = self.model.vat_former.bert(
                text_input_ids_all,
                query_embeds=query_tokens_tsml,
                attention_mask=attention_mask_all,
                encoder_hidden_states=cs_text_embeds_all,
                encoder_attention_mask=cs_text_atts_all,
                modal='cs_text',
                return_dict=True,
            )
            tsml_embeddings = output_tsml.last_hidden_state[:, : query_tokens_tsml.size(1), :]

            tsml_logit = self.tsml_head(tsml_embeddings)
            tsml_logit = tsml_logit.mean(dim=1)
            # isml_logit = self.isml_head(isml_embeddings)

            # Create labels: 1 for the original samples, 0 for the negative samples
            labels = torch.cat([torch.ones(batch_size), torch.zeros(batch_size * 2)], dim=0).long().to(tsml_logit.device)

            # Calculate cross entropy loss
            loss_tsml = F.cross_entropy(tsml_logit, labels)

            loss = loss + loss_tsml

        if "iscl" in self.task:
            query_tokens = self.model.query_tokens.expand(image_embeds.shape[0], -1, -1)

            query_output = self.model.vat_former.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                modal='image',
                return_dict=True,
            )

            image_feats = F.normalize(
                self.vision_proj(query_output.last_hidden_state), dim=-1
            )
            sim_q2t = torch.matmul(
                image_feats.unsqueeze(1), text_feat.unsqueeze(-1)
            ).squeeze()
            # [batch_size, batch_size*num_gpu, num_query_tokens]

            # image-text similarity: aggregate across all query tokens
            sim_i2t, _ = sim_q2t.max(-1)
            sim_i2t = sim_i2t / self.temp

            # text-query similarity: [batch_size, batch_size*num_gpu, num_query_tokens]
            sim_t2q = torch.matmul(
                text_feat.unsqueeze(1).unsqueeze(1), image_feats.permute(0, 2, 1)
            ).squeeze()

            # text-image similarity: aggregate across all query tokens
            sim_t2i, _ = sim_t2q.max(-1)
            sim_t2i = sim_t2i / self.temp  # [batch_size, batch_size*num_gpu]
            loss_iscl = (
                               F.cross_entropy(sim_i2t, image_targets, label_smoothing=0.1)
                               + F.cross_entropy(sim_t2i, image_targets, label_smoothing=0.1)
                       ) / 2

            loss = loss + loss_iscl

        if "vscl" in self.task:
            query_tokens = self.model.query_tokens.expand(video_embeds.shape[0], -1, -1)

            query_output = self.model.vat_former.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=video_embeds,
                encoder_attention_mask=video_atts,
                modal='video',
                return_dict=True,
            )

            video_feats = F.normalize(
                self.video_proj(query_output.last_hidden_state), dim=-1
            )

            sim_q2t = torch.matmul(
                video_feats.unsqueeze(1), text_feat.unsqueeze(-1)
            ).squeeze()
            # [batch_size, batch_size*num_gpu, num_query_tokens]

            # image-text similarity: aggregate across all query tokens
            sim_g2t, _ = sim_q2t.max(-1)
            sim_g2t = sim_g2t / self.temp

            # text-query similarity: [batch_size, batch_size*num_gpu, num_query_tokens]
            sim_t2q = torch.matmul(
                text_feat.unsqueeze(1).unsqueeze(1), video_feats.permute(0, 2, 1)
            ).squeeze()

            # text-image similarity: aggregate across all query tokens
            sim_t2g, _ = sim_t2q.max(-1)
            sim_t2g = sim_t2g / self.temp  # [batch_size, batch_size*num_gpu]
            loss_vscl = (
                               F.cross_entropy(sim_g2t, video_targets, label_smoothing=0.1)
                               + F.cross_entropy(sim_t2g, video_targets, label_smoothing=0.1)
                       ) / 2

            loss = loss + loss_vscl

        if "ascl" in self.task:
            query_tokens = self.model.query_tokens.expand(audio_embeds.shape[0], -1, -1)

            query_output = self.model.vat_former.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=audio_embeds,
                encoder_attention_mask=audio_atts,
                modal='audio',
                return_dict=True,
            )

            audio_feats = F.normalize(
                self.audio_proj(query_output.last_hidden_state), dim=-1
            )

            sim_q2t = torch.matmul(
                audio_feats.unsqueeze(1), text_feat.unsqueeze(-1)
            ).squeeze()
            # [batch_size, batch_size*num_gpu, num_query_tokens]

            # image-text similarity: aggregate across all query tokens
            sim_a2t, _ = sim_q2t.max(-1)
            sim_a2t = sim_a2t / self.temp

            # text-query similarity: [batch_size, batch_size*num_gpu, num_query_tokens]
            sim_t2q = torch.matmul(
                text_feat.unsqueeze(1).unsqueeze(1), audio_feats.permute(0, 2, 1)
            ).squeeze()

            # text-image similarity: aggregate across all query tokens
            sim_t2a, _ = sim_t2q.max(-1)
            sim_t2a = sim_t2a / self.temp  # [batch_size, batch_size*num_gpu]
            loss_ascl = (
                               F.cross_entropy(sim_a2t, audio_targets, label_smoothing=0.1)
                               + F.cross_entropy(sim_t2a, audio_targets, label_smoothing=0.1)
                       ) / 2

            loss = loss + loss_ascl

        if "tscl" in self.task:
            query_tokens = self.model.query_tokens.expand(cs_text_embeds.shape[0], -1, -1)
            query_output = self.model.vat_former.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=cs_text_embeds,
                encoder_attention_mask=cs_text_atts,
                modal='cs_text',
                return_dict=True,
            )

            cs_text_feats = F.normalize(
                self.cs_text_proj(query_output.last_hidden_state), dim=-1
            )
            sim_q2t = torch.matmul(
                cs_text_feats.unsqueeze(1), text_feat.unsqueeze(-1)
            ).squeeze()
            # [batch_size, batch_size*num_gpu, num_query_tokens]

            # image-text similarity: aggregate across all query tokens
            sim_c2t, _ = sim_q2t.max(-1)
            sim_c2t = sim_c2t / self.temp

            # text-query similarity: [batch_size, batch_size*num_gpu, num_query_tokens]
            # reversed order
            sim_t2q = torch.matmul(
                text_feat.unsqueeze(1).unsqueeze(1), cs_text_feats.permute(0, 2, 1)
            ).squeeze()

            # text-image similarity: aggregate across all query tokens
            sim_t2c, _ = sim_t2q.max(-1)
            sim_t2c = sim_t2c / self.temp  # [batch_size, batch_size*num_gpu]


            loss_tscl = (
                               F.cross_entropy(sim_c2t, cs_text_targets, label_smoothing=0.1)
                               + F.cross_entropy(sim_t2c, cs_text_targets, label_smoothing=0.1)
                       ) / 2

            loss = loss + loss_tscl

        loss = loss / len(self.task)
        return loss


class ImageTransform(nn.Module):
    def __init__(self):
        super(ImageTransform, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=49, kernel_size=1).cuda()

    def forward(self, sci):
        image_feat = sci['image'].float()
        features_transposed = image_feat.unsqueeze(1)
        transformed_features_transposed = self.conv1d(features_transposed)
        image_embeds = transformed_features_transposed

        return image_embeds

class VideoTransform(nn.Module):
    def __init__(self):
        super(VideoTransform, self).__init__()
        # Define a 1D convolutional layer to transform from 2048 to 768 features
        self.conv1d = nn.Conv1d(in_channels=2048, out_channels=768, kernel_size=1).cuda()

    def forward(self, sci):
        video_feat = sci['video'].float()
        features_transposed = video_feat.transpose(1, 2)
        # Apply the convolution
        transformed_features_transposed = self.conv1d(features_transposed)
        video_embeds = transformed_features_transposed.transpose(1, 2)
        return video_embeds

class AudioTransform(nn.Module):
    def __init__(self, audio_config):
        super(AudioTransform, self).__init__()
        if audio_config.data_name == 'cellpress':
            self.conv1d = nn.Conv1d(in_channels=43, out_channels=768, kernel_size=1).cuda()
        elif audio_config.data_name == 'aviate':
            self.conv1d = nn.Conv1d(in_channels=20, out_channels=768, kernel_size=1).cuda()

    def forward(self, sci):
        audio_feat = sci['audio'].float()
        features_transposed = audio_feat.transpose(1, 2)
        # Apply the convolution
        transformed_features_transposed = self.conv1d(features_transposed)
        # Transpose the features back to the original shape
        audio_embeds = transformed_features_transposed.transpose(1, 2)
        return audio_embeds