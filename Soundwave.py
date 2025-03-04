from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss


from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM
from transformers.trainer_pt_utils import LabelSmoother
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.models.whisper.modeling_whisper import WhisperEncoder, WhisperConfig


IGNORE_TOKEN_ID = LabelSmoother.ignore_index


class SoundwaveConfig(LlamaConfig):
    model_type = "Soundwave"
    
class LookBackModule(nn.Module):
    def __init__(self, cfg: LlamaConfig):
        super().__init__()
        self.encoder_attn = nn.MultiheadAttention(
            cfg.hidden_size,
            cfg.num_attention_heads,
            dropout=0.1,
            batch_first=True
        )
        self.atten_layer_norm = nn.LayerNorm(cfg.hidden_size)


    def forward(self, x, wav_feature, bf_shrink_padding_mask):

        residual = x
        x, _ = self.encoder_attn(
            query=x,
            key=wav_feature,
            value=wav_feature,
            key_padding_mask=bf_shrink_padding_mask,
        )
        x += residual
        x = self.atten_layer_norm(x)
        return x

class SoundwaveModel(LlamaModel):
    config_class = SoundwaveConfig

    def __init__(self, config: LlamaConfig):
        super(SoundwaveModel, self).__init__(config)

        if hasattr(config, "adapter_size"):
            self.mm_projector1 = nn.Linear(config.adapter_size*2 , config.hidden_size)
            self.lbm =  LookBackModule(config)
            self.out_norm = nn.LayerNorm(config.hidden_size)
            self.audio_feature_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            
            asr_encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.hidden_size,
                nhead=config.num_attention_heads,
                dim_feedforward=config.hidden_size*2,
                dropout=0.1,
                norm_first=True
            )
            self.asr_transformer_encoder = nn.TransformerEncoder(asr_encoder_layer, num_layers=1)
        
        if hasattr(config, "audio_tower"):
            self.audio_tower = WhisperEncoder(WhisperConfig.from_pretrained(config.audio_tower))
        self.mask_tensor=(torch.ones([1,1024])>0)
        self.length=-1

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        audios: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if (input_ids.shape[1] != 1 or self.training) and audios is not None:
            audio_list=[]
            
            for audio in audios:
                with torch.no_grad():
                    audio=audio.unsqueeze(0)
                    audio_feature = self.audio_tower(audio).last_hidden_state
           
                audio_feature = audio_feature.view(audio_feature.shape[0], audio_feature.shape[1]//2, 2 * audio_feature.shape[2])
                audio_feature = self.mm_projector1(audio_feature)
                audio_feature = self.asr_transformer_encoder(audio_feature)
                audio_feature = self.out_norm(audio_feature)
                audio_list.append(audio_feature[0])

            audio_features = torch.stack(audio_list, dim=0)

            predict_logits = self.audio_feature_head(audio_features)

            new_input_embeds = []
            label_shift = []
            label_extend = -1
            new_input_ids = []
            tokens = predict_logits.argmax(dim=-1)
            shrink_mask = tokens.roll(1) != tokens
            shrink_mask[:,0] = True
                
            lengths = shrink_mask.long().sum(-1)
            shrink_2d = audio_features[shrink_mask]
            num_patches = self.config.audio_patch_size
            l_index=0
            shrink_features = []
            for v, audio_feature, mask in zip(lengths, audio_features, ~shrink_mask):
                shrink_feature = shrink_2d[l_index:l_index+v]
                shrink_feature = self.lbm(shrink_feature, audio_feature, bf_shrink_padding_mask=mask)
                shrink_features.append(shrink_feature)
                l_index += v
            
            if self.training: 
                maxn_length = lengths.max()
                label_extend = maxn_length - num_patches
                for cur_input_ids, cur_input_embeds, shrink_feature in zip(input_ids, inputs_embeds, shrink_features):
                    pad_ids = torch.full(size=(maxn_length,), fill_value=self.config.llm_pad_token_id, dtype=torch.long).to(attention_mask.device)
                    pad_embeds = self.embed_tokens(pad_ids)
                    v = shrink_feature.shape[0]
                    audio_start_token_pos = torch.where(cur_input_ids == self.config.audio_patch_token)[0][:1]
                    cur_new_input_id = torch.cat((cur_input_ids[:audio_start_token_pos], cur_input_ids[audio_start_token_pos: audio_start_token_pos+1].repeat(v), cur_input_ids[audio_start_token_pos + num_patches:], pad_ids[:maxn_length - v]), dim=0)
                    cur_new_input_embeds = torch.cat((
                    cur_input_embeds[:audio_start_token_pos],
                    shrink_feature,
                    cur_input_embeds[audio_start_token_pos + num_patches:],pad_embeds[:maxn_length-v]), dim=0)
                    new_input_embeds.append(cur_new_input_embeds)
                    new_input_ids.append(cur_new_input_id)
                    label_shift.append(v - num_patches)
                    
                input_ids = torch.stack(new_input_ids, dim=0)
                attention_mask=input_ids.ne(self.config.llm_pad_token_id)
                inputs_embeds = torch.stack(new_input_embeds, dim=0) 
            else:
                for cur_input_ids, cur_input_embeds, shrink_feature in zip(input_ids, inputs_embeds, shrink_features):
                    v = shrink_feature.shape[0]

                    audio_start_token_pos = torch.where(cur_input_ids == self.config.audio_patch_token)[0][:1]
                    cur_new_input_id = torch.cat((cur_input_ids[:audio_start_token_pos],cur_input_ids[audio_start_token_pos: audio_start_token_pos+1].repeat(v), cur_input_ids[audio_start_token_pos + num_patches:]),dim=0)
                    cur_new_input_embeds = torch.cat((
                    cur_input_embeds[:audio_start_token_pos],
                    shrink_feature,
                    cur_input_embeds[audio_start_token_pos + num_patches:]), dim=0)
                    new_input_embeds.append(cur_new_input_embeds)
                    new_input_ids.append(cur_new_input_id)
                input_ids = torch.stack(new_input_ids, dim=0)
                attention_mask=input_ids.ne(self.config.llm_pad_token_id)
                inputs_embeds = torch.stack(new_input_embeds, dim=0)
                self.mask_tensor.to(input_ids.device)[0][:attention_mask.shape[1]]=attention_mask[0]
                self.length=attention_mask.shape[1]

        if not self.training:
            attention_mask=self.mask_tensor.to(input_ids.device)[:,:self.length]
            self.length+=1

        return_state=super(SoundwaveModel, self).forward(
            input_ids=None, attention_mask=attention_mask, past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, use_cache=use_cache,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        if self.training:
            return_state["audio_features"] = predict_logits
            return_state["label_shift"] = label_shift
            return_state["label_extend"] = label_extend
        
        return return_state 


class SoundwaveForCausalLM(LlamaForCausalLM):
    config_class = SoundwaveConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = SoundwaveModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        asr_targets: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        audios: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            audios=audios
        )
        

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            if asr_targets is not None:
                mask_asr_targets = (asr_targets != IGNORE_TOKEN_ID)
                target_lengths = mask_asr_targets.sum(1)
                input_lengths = torch.full(size=(outputs["audio_features"].shape[0],), fill_value=outputs["audio_features"].shape[1], dtype=torch.long)
                asr_logits = outputs["audio_features"]

                log_probs = F.log_softmax(asr_logits, dim=-1).transpose(0, 1)

                with torch.backends.cudnn.flags(enabled=False):
                    loss_asr = F.ctc_loss(
                        log_probs,
                        asr_targets,
                        input_lengths,
                        target_lengths,
                        blank=self.model.config.audio_patch_token,
                        reduction='mean',
                        zero_infinity=True,
                    )
            else:
                loss_asr=0

            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            if len(outputs["label_shift"]) >0:
                if outputs["label_extend"] != -1:
                    new_shift_labels = torch.full(size=(shift_labels.shape[0], outputs["label_extend"]+shift_labels.shape[1]), fill_value=IGNORE_TOKEN_ID, dtype=torch.long).to(shift_labels.device)
                    for i in range(len(outputs["label_shift"])):
                        new_shift_labels[i][outputs["label_shift"][i]:outputs["label_shift"][i] + len(shift_labels[i])]= shift_labels[i]
                    shift_labels = new_shift_labels
                else:
                    for i in range(len(outputs["label_shift"])):
                        shift_labels[i]= shift_labels[i].roll(-outputs["label_shift"][i])

            loss_fct = CrossEntropyLoss()
            # Flatten the tokens
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
                    
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            loss = loss + 0.3*loss_asr 

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        **kwargs,
    ):
        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        if past_key_values is not None:
            if inputs_embeds is not None:  # Exception 1
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids.contiguous()}  # `contiguous()` needed for compilation use cases

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )
        model_inputs.update({"audios": kwargs["audios"]} if "audios" in kwargs.keys() else {})
        return model_inputs


AutoConfig.register("Soundwave", SoundwaveConfig)
AutoModelForCausalLM.register(SoundwaveConfig, SoundwaveForCausalLM)