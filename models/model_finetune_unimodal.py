# -*- coding: utf-8 -*-
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration, Blip2Config
import torch.nn as nn
import torch.nn.functional as F
from transformers import (T5Tokenizer, T5ForConditionalGeneration, BertTokenizer,
                          LongT5ForConditionalGeneration, AutoTokenizer, LEDForConditionalGeneration,
                          PegasusForConditionalGeneration as PegasusGen, Blip2ForConditionalGeneration,
                          BertGenerationEncoder, BertGenerationDecoder, EncoderDecoderModel,
                          EncoderDecoderConfig, BertConfig, PegasusConfig)

from models.VAT_Former import BertConfig, BertLMHeadModel
from transformers.modeling_outputs import BaseModelOutput
import pickle
import rdkit.Chem as Chem
import numpy as np
from peft import LoraConfig
from peft import get_peft_model

modalities = {
    'image': 'image representation',
    'audio': 'audio representation',
    'src_text': 'text representation',
    'image_ind': 'textual img_ind',
    'summary': 'textual summary',
    'video': 'video representation'
}


def generate_prompt(inputs, outputs):
    inputs_desc = [modalities[i] for i in inputs]
    outputs_desc = [modalities[o] for o in outputs]

    inputs_str = ' and '.join(inputs_desc)
    outputs_str = ', '.join(outputs_desc)

    prompt = f"Given the input {inputs_str}, please generate the {outputs_str}."
    return prompt


class LayerNorm(nn.LayerNorm):

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class VATFormer(nn.Module):
    def __init__(self, num_query_token, vision_graph_width, cross_attention_freq=2):
        super().__init__()

        encoder_config = BertConfig.from_pretrained("allenai/scibert_scivocab_uncased")

        encoder_config.encoder_width = vision_graph_width
        encoder_config.add_cross_attention = True
        encoder_config.add_pooling_layer = False
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token

        self.Qformer = BertLMHeadModel.from_pretrained("allenai/scibert_scivocab_uncased", config=encoder_config)

        self.query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        self.query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)


class SciSumModel(nn.Module):
    def __init__(self, args_config, fp=False, task=None, device=None):
        super().__init__()

        self.args_config = args_config
        self.blip2conf = Blip2Config()
        self.model = Blip2ForConditionalGeneration(self.blip2conf)
        # self.model.language_model = T5ForConditionalGeneration.from_pretrained("../../ckpts/text_ckpts/molt5-base")
        #
        # self.processor = Blip2Processor.from_pretrained("../../ckpts/fusion_ckpts/blip2")
        # self.tokenizer = T5Tokenizer.from_pretrained("../../ckpts/text_ckpts/molt5-base",  model_max_length=512)
        # self.model.language_model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-base", cache_dir='/home/allen/Documents/research/summarization/code/GIT-Mol/preTrain_model/t5-base/')

        # self.model.language_model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-pubmed",
        #                                                                             cache_dir='preTrain_model/google/pegasus-pubmed/')

        self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b",
                                                        cache_dir='../../preTrain_model/blip2-opt-2.7b/')
        # self.tokenizer = AutoTokenizer.from_pretrained("google/pegasus-pubmed",
        #                                                cache_dir='preTrain_model/google/pegasus-pubmed/')
        # self.processor = Blip2Processor(self.processor.current_processor, self.tokenizer)

        # TODO: New added #############################################################################################################
        # For direct test on Pegasus Model
        Pegasus_Model_Name = 'google/pegasus-pubmed'

        self.pegasus_tokenizer = AutoTokenizer.from_pretrained(Pegasus_Model_Name,
                                                               cache_dir=f'../../preTrain_model/{Pegasus_Model_Name}/')
        new_tokens = [f"img_ind_{_ind}" for _ind in range(20)]
        self.pegasus_tokenizer.add_tokens(new_tokens)
        
        self.pegasus_model = PegasusGen.from_pretrained(Pegasus_Model_Name,
                                                        cache_dir=f'../../preTrain_model/{Pegasus_Model_Name}/')
        self.pegasus_model.resize_token_embeddings(len(self.pegasus_tokenizer))
        
        self.processor = Blip2Processor(self.processor.current_processor, self.pegasus_tokenizer)

        # self.pegasus_tokenizer = AutoTokenizer.from_pretrained("allenai/led-large-16384-arxiv",
        #                                                 cache_dir='preTrain_model/allenai/led-large-16384-arxiv/')
        # self.pegasus_model = LEDForConditionalGeneration.from_pretrained("allenai/led-large-16384-arxiv",
        #                                                 cache_dir='preTrain_model/allenai/led-large-16384-arxiv/')

        # self.pegasus_tokenizer = AutoTokenizer.from_pretrained("google/long-t5-local-base",
        #                                                 cache_dir='preTrain_model/google/long-t5-local-base/')
        # self.pegasus_model = LongT5ForConditionalGeneration.from_pretrained("google/long-t5-local-base",
        #                                                 cache_dir='preTrain_model/google/long-t5-local-base/')

        ## EncoderDecoder (Bert and Pegasus)
        # self.bert_tokenizer = BertTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
        # self.pegasus_tokenizer = AutoTokenizer.from_pretrained("google/pegasus-pubmed")
        #
        # self.bert_encoder_config = BertConfig.from_pretrained("allenai/scibert_scivocab_uncased")
        # self.pegasus_decoder_config = PegasusConfig.from_pretrained("google/pegasus-pubmed")
        #
        # self.encoder_decoder_config = EncoderDecoderConfig.from_encoder_decoder_configs(self.bert_encoder_config,
        #                                                                                 self.pegasus_decoder_config)
        # self.encoder_decoder_model = EncoderDecoderModel(config=self.encoder_decoder_config)
        # self.encoder_decoder_model.decoder.resize_token_embeddings(len(self.pegasus_tokenizer))
        #################################################################################################################################

        ##############################################
        # TODO: Use when working with Pegasus and Led
        self.model.feat_1024_transform = FeatureTransform1()
        self.model.feat_768_transform = FeatureTransform7()
        ##############################################
        if args_config.data_name == "pubmed":
            self.model.vision_model = PubmedImageTransform()
        else:
            self.model.vision_model = ImageTransform()
        self.model.video_model = VideoTransform()
        self.model.audio_model = AudioTransform(args_config)

        self.model.ln_vision = LayerNorm(768)
        self.model.ln_audio = LayerNorm(768)
        self.model.ln_video = LayerNorm(768)

        # self.model.ln_text = LayerNorm(1024)
        # self.model.vision_model = VisonEncoder()
        # self.model.ln_vision = LayerNorm(self.model.vision_model.hidden_size)
        # self.model.graph_encoder = GraphEncoder(config)
        # self.model.ln_graph = LayerNorm(self.model.graph_encoder.hidden_size)

        # language_model = T5ForConditionalGeneration.from_pretrained("google/pegasus-pubmed",
        #                                                             cache_dir='preTrain_model/google/pegasus-pubmed/')

        language_lora_config = LoraConfig(
            peft_type="LORA",
            r=16,  # rank of the update matrices
            lora_alpha=16,  # scaling factor for LoRA updates
            target_modules=["q", "v", "lm_head", "shared"],  # apply LoRA to query and value matrices
            lora_dropout=0.1,  # dropout rate for LoRA updates
            bias="none",  # do not train bias parameters
                #modules_to_save=["lm_head"]  # also train the classifier parameters
        )
        self.model.language_model = get_peft_model(self.pegasus_model, language_lora_config)

        gitformer = VATFormer(384, 768)
        self.model.git_former = gitformer.Qformer
        self.model.query_tokens = gitformer.query_tokens

        for param in self.model.feat_1024_transform.parameters():
            param.requires_grad = False
        for param in self.model.feat_768_transform.parameters():
            param.requires_grad = False
        # for param in self.pegasus_model.parameters():
        #     param.requires_grad = False

        embed_dim = 256
        self.device = device

        self.task_list = task
        self.inputs_modal = []
        self.outputs_modal = []

        for task in self.task_list:
            inputs_modal = task['inputs_modal']
            self.inputs_modal = self.inputs_modal + inputs_modal
            outputs_modal = task['outputs_modal']
            self.outputs_modal = self.outputs_modal + outputs_modal

        self.inputs_modal = list(set(self.inputs_modal))
        self.outputs_modal = list(set(self.outputs_modal))

    def get_git_former_outputs(self, mol, inputs_modal, outputs_modal):
        language_model_inputs_image = None
        language_model_inputs_text = None
        language_model_inputs_graph = None
        input_tensors = []
        batch_size = len(mol['src_text'])

        prompt = generate_prompt(inputs_modal, outputs_modal)
        # print(f"prompt : {prompt}")
        prompt = self.processor(text=prompt, return_tensors="pt")
        input_ids = prompt['input_ids']
        attention_mask = prompt['attention_mask']
        mol['input_ids'] = input_ids.repeat(batch_size, 1).to(self.device)
        mol['attention_mask'] = attention_mask.repeat(batch_size, 1).to(self.device)
        
        if 'src_text' in inputs_modal:
            language_model_inputs_text = self.get_text_git_former_features(mol, inputs_modal)
            input_tensors.append(language_model_inputs_text) # (batch_size, 896, 1024)
            
        if 'image' in inputs_modal:
            language_model_inputs_image = self.get_image_git_former_features(mol)
            input_tensors.append(language_model_inputs_image) # (batch_size, 384, 1024)
            
        if 'video' in inputs_modal:
            language_model_inputs_video = self.get_video_git_former_features(mol)
            input_tensors.append(language_model_inputs_video) # (batch_size, 384, 1024)

        if 'audio' in inputs_modal:
            language_model_inputs_audio = self.get_audio_git_former_features(mol)
            input_tensors.append(language_model_inputs_audio) # (batch_size, 384, 1024)
            
        qformer_outputs = torch.cat(input_tensors, dim=1) # (batch_size, 2048, 1024)

        return qformer_outputs, mol

    def forward(self, mol):
        loss_list = []
        for i, task in enumerate(self.task_list):
            inputs_modal = task['inputs_modal']
            outputs_modal = task['outputs_modal']
            qformer_outputs, mol = self.get_git_former_outputs(mol, inputs_modal, outputs_modal)

            # if 'summary' in outputs_modal:
            loss_text = self.loss_text(mol, qformer_outputs, outputs_modal)
            loss_list.append(loss_text)

        loss_tensor = torch.stack(loss_list)
        loss = torch.mean(loss_tensor)
        # loss = torch.sum(torch.stack(loss_list) * self.loss_weights_base_task())
        return loss

    def loss_text(self, mol, qformer_outputs, outputs_modal):
        '''Q-Former + pegasus -> Pegasus'''
        language_model_inputs = qformer_outputs

        if 'summary' in outputs_modal:
            tgt = mol['tgt_Caption']

        if self.args_config.prompt:
            # print('>>> Using Prompt >>>')
            prompt_embeddings = self.model.language_model.model.model.encoder(input_ids=mol['input_ids']).last_hidden_state
            language_model_inputs = torch.cat([prompt_embeddings, language_model_inputs], dim=1)

        # bert_embeddings = self.bert_model(input_ids=src['input_ids'],
        #                                   attention_mask=src['attention_mask'])
        #
        # bert_masks = torch.ones(
        #     bert_embeddings.size()[:-1], dtype=torch.long, device=bert_embeddings.device)
        #
        # inputs_embeds = torch.cat([bert_embeddings, language_model_inputs], dim=1)
        # attention_mask = torch.cat([bert_masks, language_model_attention_mask], dim=1)

        ## EncoderDecoderModel
        # self.encoder_decoder_model.config.decoder_start_token_id = self.bert_tokenizer.pad_token_id
        # self.encoder_decoder_model.config.pad_token_id = 0
        # outputs = self.encoder_decoder_model(input_ids=src['input_ids'],
        #                                      attention_mask=src['attention_mask'],
        #                                      labels=tgt['input_ids'],
        #                                      decoder_attention_mask=tgt['attention_mask'])

        h = BaseModelOutput(
            last_hidden_state=language_model_inputs,
            hidden_states=None,
            attentions=None
        )
        # original ids
        outputs = self.model.language_model.model(encoder_outputs=h,
                                     labels=tgt['input_ids'],
                                     decoder_attention_mask=tgt['attention_mask'])

        # h = BaseModelOutput(
        #             last_hidden_state=language_model_inputs,
        #             hidden_states=None,
        #             attentions=None
        #         )
        # outputs = self.encoder_decoder_model(encoder_outputs=h,
        #                                      labels=tgt['input_ids'])

        loss_text = outputs['loss']
        return loss_text


    def get_text_git_former_features(self, mol, inputs_modal):
        if 'src_text' in inputs_modal:
            src = mol['src_Text']


        text_embeds_1 = self.model.language_model.model.model.encoder(input_ids=src['input_ids']).last_hidden_state

        text_embeds = self.model.feat_768_transform(
            self.model.language_model.model.model.encoder(input_ids=src['input_ids']).last_hidden_state)
        text_attention_mask = torch.ones(text_embeds.size()[:-1], dtype=torch.long, device=text_embeds.device)
        query_tokens = self.model.query_tokens.expand(text_embeds.shape[0], -1, -1)
        query_outputs = self.model.git_former.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=text_embeds,
            encoder_attention_mask=text_attention_mask,
            modal='cs_text',
            is_decoder=False
        )
        query_output = self.model.feat_1024_transform(query_outputs.last_hidden_state)

        # Text + Query
        language_model_inputs_text = torch.cat([text_embeds_1, query_output], dim=1)

        # # Query + Text
        # language_model_inputs_text = torch.cat([query_output, text_embeds_1], dim=1)

        return language_model_inputs_text

    def get_image_git_former_features(self, mol):
        image_embeds = self.model.ln_vision(self.model.vision_model(mol))

        image_embeds = image_embeds.float()
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image_embeds.device
        )
        image_embeds = self.model.vision_model(mol)

        query_tokens = self.model.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_outputs = self.model.git_former.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            modal='image',
            is_decoder=False
        )
        query_output = query_outputs.last_hidden_state

        language_model_inputs_image = self.model.feat_1024_transform(query_output)

        return language_model_inputs_image

    def get_video_git_former_features(self, mol):
        '''Get the video and output the laten vector for decoder'''
        video_embeds = self.model.ln_video(self.model.video_model(mol))
        video_embeds = video_embeds.float()
        video_attention_mask = torch.ones(video_embeds.size()[:-1], dtype=torch.long).to(
            video_embeds.device
        )
        # video_embeds = self.model.video_model(mol)
        query_tokens = self.model.query_tokens.expand(video_embeds.shape[0], -1, -1)
        query_outputs = self.model.git_former.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=video_embeds,
            encoder_attention_mask=video_attention_mask,
            modal='video',
            is_decoder=False
        )
        query_output = query_outputs.last_hidden_state
        language_model_inputs_video = self.model.feat_1024_transform(query_output)
        return language_model_inputs_video

    def get_audio_git_former_features(self, mol):
        '''Get the video and output the laten vector for decoder'''
        audio_embeds = self.model.ln_audio(self.model.audio_model(mol))
        audio_embeds = audio_embeds.float()
        audio_attention_mask = torch.ones(audio_embeds.size()[:-1], dtype=torch.long).to(
            audio_embeds.device
        )
        # audio_embeds = self.model.audio_model(mol)
        query_tokens = self.model.query_tokens.expand(audio_embeds.shape[0], -1, -1)
        query_outputs = self.model.git_former.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=audio_embeds,
            encoder_attention_mask=audio_attention_mask,
            modal='audio',
            is_decoder=False
        )
        query_output = query_outputs.last_hidden_state
        language_model_inputs_audio = self.model.feat_1024_transform(query_output)
        return language_model_inputs_audio

    def generate_text(self, mol, inputs_modal, outputs_modal):
        generated_ids = self.generate_language(mol, inputs_modal, outputs_modal)
        # print(f"Generated_ids are: {generated_ids, generated_ids.shape}")
        ori_texts = self.processor.batch_decode(mol['src_Text']['input_ids'], skip_special_tokens=True)

        # generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        # print(f"generated_texts are: {generated_texts}")

        return ori_texts, generated_texts

    def generate_language(self, mol, inputs_modal, outputs_modal):
        '''Q-Former + pegasus -> Pegasus'''
        language_model_inputs, mol = self.get_git_former_outputs(mol, inputs_modal, outputs_modal)
        # language_model_attention_mask = torch.ones(
        #     language_model_inputs.size()[:-1], dtype=torch.long, device=language_model_inputs.device)

        # if ('caption' in outputs_modal):
        #     tgt = mol['tgt_Caption']
        #     src = mol['src_isoSMILES']
        #     bert_src = mol['isoSMILES']

        # pegasus_embeddings = self.pegasus_model(input_ids=src['input_ids'],
        #                                         attention_mask=src['attention_mask'],
        #                                         decoder_input_ids=src['input_ids']).encoder_last_hidden_state
        # pegasus_masks = torch.ones(
        #     pegasus_embeddings.size()[:-1], dtype=torch.long, device=pegasus_embeddings.device)
        #
        # inputs_embeds = torch.cat([pegasus_embeddings, language_model_inputs], dim=1)
        # attention_mask = torch.cat([pegasus_masks, language_model_attention_mask], dim=1)

        # EncoderDecoderModel
        # output_ids = self.encoder_decoder_model.generate(
        #     input_ids=src['input_ids'],
        #     attention_mask=src['attention_mask'],
        #     num_beams=5,
        #     no_repeat_ngram_size=3)

        h = BaseModelOutput(
            last_hidden_state=language_model_inputs,
            hidden_states=None,
            attentions=None
        )
        ## original ids
        output_ids = self.model.language_model.model.generate(
            encoder_outputs=h,
            num_beams=5,
            no_repeat_ngram_size=3)

        # h = BaseModelOutput(
        #     last_hidden_state=language_model_inputs,
        #     hidden_states=None,
        #     attentions=None
        # )
        # output_ids = self.encoder_decoder_model.generate(encoder_outputs=h,
        #                                                  num_beams=5,
        #                                                  max_length=256,
        #                                                  no_repeat_ngram_size=3)

        return output_ids


class FeatureTransform1(nn.Module):
    '''
    Use when handling Pegasus and Led, their d_model is 1024, required to transform back to 768
    '''

    def __init__(self):
        super(FeatureTransform1, self).__init__()
        # Define a 1D convolutional layer to transform from 2048 to 768 features
        self.conv1d = nn.Conv1d(in_channels=768, out_channels=1024, kernel_size=1).cuda()

    def forward(self, x):
        # Apply the 1D convolutional layer to the entire sequence
        # First, transpose the features to put the sequence in the correct dimension
        features_transposed = x.transpose(1, 2)  # Now shape is [batch_size, 2048, seq_length]

        # Apply the convolution
        transformed_features_transposed = self.conv1d(features_transposed)  # Now shape is [batch_size, 768, seq_length]

        # Transpose the features back to the original shape
        transformed_embeds = transformed_features_transposed.transpose(1,
                                                                       2)  # Now shape is [batch_size, seq_length, 768]

        return transformed_embeds


class FeatureTransform7(nn.Module):
    '''
    Use when handling Pegasus and Led, their d_model is 1024, required to transform back to 768
    '''

    def __init__(self):
        super(FeatureTransform7, self).__init__()
        # Define a 1D convolutional layer to transform from 2048 to 768 features
        self.conv1d = nn.Conv1d(in_channels=1024, out_channels=768, kernel_size=1).cuda()

    def forward(self, x):
        # Apply the 1D convolutional layer to the entire sequence
        # First, transpose the features to put the sequence in the correct dimension
        features_transposed = x.transpose(1, 2)  # Now shape is [batch_size, 2048, seq_length]

        # Apply the convolution
        transformed_features_transposed = self.conv1d(features_transposed)  # Now shape is [batch_size, 768, seq_length]

        # Transpose the features back to the original shape
        transformed_embeds = transformed_features_transposed.transpose(1,
                                                                       2)  # Now shape is [batch_size, seq_length, 768]

        return transformed_embeds

class VideoTransform(nn.Module):
    def __init__(self):
        super(VideoTransform, self).__init__()
        # Define a 1D convolutional layer to transform from 2048 to 768 features
        self.conv1d = nn.Conv1d(in_channels=2048, out_channels=768, kernel_size=1).cuda()

    def forward(self, mol):
        # Apply the 1D convolutional layer to the entire sequence
        # First, transpose the features to put the sequence in the correct dimension
        features_transposed = mol['video'].transpose(1, 2)  # Now shape is [batch_size, 2048, seq_length]

        # Apply the convolution
        transformed_features_transposed = self.conv1d(features_transposed)  # Now shape is [batch_size, 768, seq_length]

        # Transpose the features back to the original shape
        video_embeds = transformed_features_transposed.transpose(1, 2)  # Now shape is [batch_size, seq_length, 768]

        return video_embeds

class AudioTransform(nn.Module):
    def __init__(self, audio_config):
        super(AudioTransform, self).__init__()
        # Define a 1D convolutional layer to transform from 2048 to 768 features
        if audio_config.data_name == 'aviate':
            self.conv1d = nn.Conv1d(in_channels=20, out_channels=768, kernel_size=1).cuda()
        elif audio_config.data_name == 'cellpress':
            self.conv1d = nn.Conv1d(in_channels=43, out_channels=768, kernel_size=1).cuda()

    def forward(self, mol):
        # Apply the 1D convolutional layer to the entire sequence
        # First, transpose the features to put the sequence in the correct dimension
        features_transposed = mol['audio'].transpose(1, 2)

        # Apply the convolution (ava)
        transformed_features_transposed = self.conv1d(features_transposed.float())

        # Transpose the features back to the original shape
        audio_embeds = transformed_features_transposed.transpose(1, 2)  # Now shape is [batch_size, seq_length, 768]

        return audio_embeds

class ImageTransform(nn.Module):
    def __init__(self):
        super(ImageTransform, self).__init__()
        # Define a 1D convolutional layer to transform from 2048 to 768 features
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=49, kernel_size=1).cuda()

    def forward(self, mol):
        # Apply the 1D convolutional layer to the entire sequence
        # First, transpose the features to put the sequence in the correct dimension
        image_feat = mol['image'].float()
        features_transposed = image_feat.unsqueeze(1)  # Now shape is [batch_size, 2048, seq_length]

        # Apply the convolution
        transformed_features_transposed = self.conv1d(features_transposed)  # Now shape is [batch_size, 768, seq_length]
        # Transpose the features back to the original shape
        image_embeds = transformed_features_transposed  # Now shape is [batch_size, seq_length, 768]

        return image_embeds
    
class PubmedImageTransform(nn.Module):
    def __init__(self):
        super(PubmedImageTransform, self).__init__()
        # Define a 1D convolutional layer to transform from 2048 to 768 features
        self.conv1d = nn.Conv1d(in_channels=2048, out_channels=768, kernel_size=1).cuda()

    def forward(self, mol):
        # Apply the 1D convolutional layer to the entire sequence
        # First, transpose the features to put the sequence in the correct dimension
        features_transposed = mol['image'].unsqueeze(1)
        features_transposed = features_transposed.transpose(1, 2)  # Now shape is [batch_size, 2048, seq_length]

        # Apply the convolution
        transformed_features_transposed = self.conv1d(features_transposed)  # Now shape is [batch_size, 768, seq_length]

        # Transpose the features back to the original shape
        video_embeds = transformed_features_transposed.transpose(1, 2)  # Now shape is [batch_size, seq_length, 768]

        return video_embeds