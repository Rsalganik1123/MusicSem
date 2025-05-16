'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
'''

from transformers import BertTokenizer
import transformers
transformers.logging.set_verbosity_error()
from typing import List
import os 
os.environ['TORCH_HOME'] = '/model/tteng/MusicEnocdeFactory/models/LARP/models'
import torch
from torch import nn
import torch.nn.functional as F
import ipdb 


from models.LARP.models.bert import BertConfig, BertModel, BertLMHeadModel, init_tokenizer
from models.LARP.models.vit import create_vit


class LARP(nn.Module):
    def __init__(self,                 
                config,  
                audio_cfg,                 
                text_cfg,               
                embed_dim = 256,     
                queue_size = 57600,
                momentum = 0.995):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            audio_size (int): input audio size
            vit (str): model size of vision transformer
        """               
        super().__init__()
        
        self.config = config

        audio_module = __import__('models.LARP.models.' + audio_cfg['model_type'].lower(), fromlist=['']) # audio_cfg['name']
        self.audio_encoder = getattr(audio_module, f"create_{audio_cfg['model_type'].lower()}_model")(audio_cfg)
        audio_width = 768 
        self.tokenizer = init_tokenizer()  
        encoder_config = BertConfig.from_dict(text_cfg) 
        encoder_config.encoder_width = audio_width
        self.text_encoder = BertModel.from_pretrained('bert-base-uncased',config=encoder_config, add_pooling_layer=False)
        self.text_encoder.resize_token_embeddings(len(self.tokenizer)) 

        text_width = self.text_encoder.config.hidden_size
        
        self.audio_proj = nn.Linear(audio_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)

        self.itm_head = nn.Linear(text_width, 2) 
        
        # create momentum encoders  
        
        self.audio_encoder_m = getattr(audio_module, f"create_{audio_cfg['model_type'].lower()}_model")(audio_cfg)  # audio_cfg['name']       
        self.audio_proj_m = nn.Linear(audio_width, embed_dim)
        self.text_encoder_m = BertModel(config=encoder_config, add_pooling_layer=False)  
        self.text_encoder_m.resize_token_embeddings(len(self.tokenizer))   #la add  
         
        self.text_proj_m = nn.Linear(text_width, embed_dim)
        
        self.model_pairs = [[self.audio_encoder,self.audio_encoder_m],
                            [self.audio_proj,self.audio_proj_m],
                            [self.text_encoder,self.text_encoder_m],
                            [self.text_proj,self.text_proj_m],
                           ]       
        self.copy_params()

        # create the queue
        self.register_buffer("audio_queue", torch.randn(embed_dim, queue_size))
        self.register_buffer("text_queue", torch.randn(embed_dim, queue_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))  

        self.audio_queue = nn.functional.normalize(self.audio_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)
        
        self.queue_size = queue_size
        self.momentum = momentum
        self.temp = nn.Parameter(0.07*torch.ones([]))   
        
        # create the decoder
        decoder_config = BertConfig.from_dict(text_cfg) 
        decoder_config.encoder_width = audio_width        
        self.text_decoder = BertLMHeadModel.from_pretrained('bert-base-uncased',config=decoder_config)    
        self.text_decoder.resize_token_embeddings(len(self.tokenizer)) 
        tie_encoder_decoder_weights(self.text_encoder,self.text_decoder.bert,'','/attention')


        # CIP Loss Part >>>
        self.num_tracks = self.config["num_tracks"]
        self.register_buffer("audio_features_all", torch.zeros(self.num_tracks, embed_dim))
        self.register_buffer("text_features_all", torch.zeros(self.num_tracks, embed_dim))

        self.playlist_constructor = Playlist_Constructor(conf={
            "num_tracks": self.num_tracks,
            "method": self.config["fusion_method"], # {average, soft_weight, self_attn}
            "embedding_size": embed_dim,
            "device": self.config["device"],
        })
        
        # CIP Loss Part <<<

    # cross_mask: mask for cross-item audio-text pairs, used for avioding misleading negative pairs 
    def forward(self, audio, caption, sequences, alpha, loss_type, update_momentum, cross_mask=None, update_features=False, epoch=None):

        idx, seq, mask, length = sequences

        with torch.no_grad():
            self.temp.clamp_(0.001,0.5)
        
        audio_embeds = self.audio_encoder(audio) 
        audio_atts = torch.ones(audio_embeds.size()[:-1],dtype=torch.long).to(audio.device)        
        audio_feat = F.normalize(self.audio_proj(audio_embeds[:,0,:]),dim=-1)          
        
        text = self.tokenizer(caption, padding='max_length', truncation=True, max_length=30, return_tensors="pt").to(audio.device)  
        text_output = self.text_encoder(text.input_ids, attention_mask = text.attention_mask,return_dict = True, mode = 'text')            
        text_feat = F.normalize(self.text_proj(text_output.last_hidden_state[:,0,:]),dim=-1)   

        # update item features (CIP) >>>
        if update_features:
            self.audio_features_all[idx] = audio_feat[:int(audio_feat.shape[0]/2)]
            self.text_features_all[idx] = text_feat[:int(text_feat.shape[0]/2)]
        # update item features (CIP) <<<
             
        # get momentum features
        with torch.no_grad():
            self._momentum_update()
            audio_embeds_m = self.audio_encoder_m(audio) 
            audio_feat_m = F.normalize(self.audio_proj_m(audio_embeds_m[:,0,:]),dim=-1)  
            audio_feat_all = torch.cat([audio_feat_m.t(),self.audio_queue.clone().detach()],dim=1)                   
            
            text_output_m = self.text_encoder_m(text.input_ids, attention_mask = text.attention_mask, return_dict = True, mode = 'text')    
            text_feat_m = F.normalize(self.text_proj_m(text_output_m.last_hidden_state[:,0,:]),dim=-1) 
            text_feat_all = torch.cat([text_feat_m.t(),self.text_queue.clone().detach()],dim=1)

            sim_i2t_m = audio_feat_m @ text_feat_all / self.temp  
            sim_t2i_m = text_feat_m @ audio_feat_all / self.temp 

            sim_targets = torch.zeros(sim_i2t_m.size()).to(audio.device)
            sim_targets.fill_diagonal_(1)          

            sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
            sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets        

        sim_i2t = audio_feat @ text_feat_all / self.temp
        sim_t2i = text_feat @ audio_feat_all / self.temp

        losses = {}

        ###============== WTC ===================### 
        loss_wtc = torch.zeros(1).to(audio.device)
        if "wtc" in loss_type:                   
            loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1)*sim_i2t_targets,dim=1).mean()
            loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1)*sim_t2i_targets,dim=1).mean() 

            loss_wtc = (loss_i2t+loss_t2i)/2
            losses["wtc"] = loss_wtc

        ###============== TTC ===================### 
        loss_ttc = torch.zeros(1).to(audio.device)
        if "ttc" in loss_type:                   
            loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1)*sim_i2t_targets,dim=1).mean()
            loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1)*sim_t2i_targets,dim=1).mean() 

            loss_ttc = (loss_i2t+loss_t2i)/2
            losses["ttc"] = loss_ttc
            # return loss_itc
        if update_momentum: 
            self._dequeue_and_enqueue(audio_feat_m, text_feat_m)       


        ###============== TPC ===================### 
        
        loss_tpc = torch.zeros(1).to(audio.device)
        if "tpc" in loss_type:
            if True: #epoch != 0:
                # audio_p = self.p_audio_features[pidx] 
                # caption_p = self.p_text_features[pidx]   
                audio_p = self.playlist_constructor(seq, self.audio_features_all[seq].detach(), mask, length)  
                text_p = self.playlist_constructor(seq, self.text_features_all[seq].detach(), mask, length)       

                sim_cip_a = audio_feat @ text_p.t() / self.temp
                sim_cip_t = text_feat @ audio_p.t() / self.temp
                trargets = torch.zeros(sim_cip_a.size()).to(audio.device).fill_diagonal_(1) 
                loss_cip_a = -torch.sum(F.log_softmax(sim_cip_a, dim=1)*trargets,dim=1).mean()
                loss_cip_t = -torch.sum(F.log_softmax(sim_cip_t, dim=1)*trargets,dim=1).mean() 
                
                loss_tpc = (loss_cip_a+loss_cip_t)/2

            losses["tpc"] = loss_tpc
        
        losses["loss"] = loss_wtc + loss_ttc + loss_tpc 
        # print(losses)
        return losses
    
    def calc_losses(self, ablation_loss):
        return 0

    def set_queue_ptr(self, val):
        self.queue_ptr[0] = val 

    
    # Fix the parameters in the first half of the text/visual encoders
    def fix_first_half_encoders(self):
        for params in self.text_encoder.parameters():
            params.requires_grad = False
        for params in self.visual_encoder.parameters():
            params.requires_grad = False

        div_layer_idx = int(len(self.text_encoder.encoder.layer) / 2)
        for params in self.text_encoder.encoder.layer[div_layer_idx:].parameters():
            params.requires_grad = True

        div_layer_idx = int(len(self.visual_encoder.blocks) / 2)
        for params in self.visual_encoder.blocks[div_layer_idx:].parameters():
            params.requires_grad = True
 
    @torch.no_grad() 
    def return_features(self, audio, caption): 
        self.temp.clamp_(0.001,0.5)
        audio_feat = None
        text_feat = None
        if audio is not None:
            audio_embeds = self.audio_encoder(audio) 
            audio_atts = torch.ones(audio_embeds.size()[:-1],dtype=torch.long).to(audio.device)        
            audio_feat = F.normalize(self.audio_proj(audio_embeds[:,0,:]),dim=-1)          
        
        if caption is not None:
            device =  audio.device if audio is not None else self.config["device"]
            text = self.tokenizer(caption, padding='max_length', truncation=True, max_length=30,
                                    return_tensors="pt").to(device)  

            text_output = self.text_encoder(text.input_ids, attention_mask = text.attention_mask,
                                    return_dict = True, mode = 'text')            
            text_feat = F.normalize(self.text_proj(text_output.last_hidden_state[:,0,:]),dim=-1)                 
        
        return audio_feat, text_feat 
    
    @torch.no_grad() 
    def return_playlists(self, sequences, audio_features, text_features):
        idx, seq, mask, length = sequences
        # print(idx.device, seq.device, mask.device, length.device)
        audio_p = self.playlist_constructor(seq, audio_features[seq], mask, length)  
        text_p = self.playlist_constructor(seq, text_features[seq], mask, length)       
        return audio_p, text_p
         
    @torch.no_grad()    
    def copy_params(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient    

            
    @torch.no_grad()        
    def _momentum_update(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)

                        
    @torch.no_grad()
    def _dequeue_and_enqueue(self, audio_feat, text_feat):
        # gather keys before updating queue
        audio_feats = concat_all_gather(audio_feat)
        text_feats = concat_all_gather(text_feat)
        batch_size = audio_feats.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity
        
        end_ptr = ptr + batch_size
        if end_ptr > self.queue_size:
            self.audio_queue[:, ptr:] = audio_feats[:self.queue_size-ptr].T
            self.text_queue[:, ptr:] = text_feats[:self.queue_size-ptr].T
            audio_feats = audio_feats[self.queue_size-ptr:]
            text_feats = text_feats[self.queue_size-ptr:]

            ptr = 0
            end_ptr = end_ptr - self.queue_size
        
        # replace the keys at ptr (dequeue and enqueue)
        self.audio_queue[:, ptr:end_ptr] = audio_feats.T
        self.text_queue[:, ptr:end_ptr] = text_feats.T
        ptr = (end_ptr) % self.queue_size  # move pointer

        # import pdb
        # pdb.set_trace()

        self.queue_ptr[0] = ptr 

def larp(**kwargs):
    model = LARP(**kwargs)
    return model 


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output     


from typing import List
def tie_encoder_decoder_weights(encoder: nn.Module, decoder: nn.Module, base_model_prefix: str, skip_key:str, verbose=False):
    uninitialized_encoder_weights: List[str] = []
    if decoder.__class__ != encoder.__class__:
        logger.info(
            f"{decoder.__class__} and {encoder.__class__} are not equal. In this case make sure that all encoder weights are correctly initialized."
        )

    def tie_encoder_to_decoder_recursively(
        decoder_pointer: nn.Module,
        encoder_pointer: nn.Module,
        module_name: str,
        uninitialized_encoder_weights: List[str],
        skip_key: str,
        depth=0,
    ):
        assert isinstance(decoder_pointer, nn.Module) and isinstance(
            encoder_pointer, nn.Module
        ), f"{decoder_pointer} and {encoder_pointer} have to be of type torch.nn.Module"
        if hasattr(decoder_pointer, "weight") and skip_key not in module_name:
            assert hasattr(encoder_pointer, "weight")
            encoder_pointer.weight = decoder_pointer.weight
            if hasattr(decoder_pointer, "bias"):
                assert hasattr(encoder_pointer, "bias")
                encoder_pointer.bias = decoder_pointer.bias                
            if verbose: 
                print(module_name+' is tied')    
            return

        encoder_modules = encoder_pointer._modules
        decoder_modules = decoder_pointer._modules
        if len(decoder_modules) > 0:
            assert (
                len(encoder_modules) > 0
            ), f"Encoder module {encoder_pointer} does not match decoder module {decoder_pointer}"

            all_encoder_weights = set([module_name + "/" + sub_name for sub_name in encoder_modules.keys()])
            encoder_layer_pos = 0
            for name, module in decoder_modules.items():
                if name.isdigit():
                    encoder_name = str(int(name) + encoder_layer_pos)
                    decoder_name = name
                    if not isinstance(decoder_modules[decoder_name], type(encoder_modules[encoder_name])) and len(
                        encoder_modules
                    ) != len(decoder_modules):
                        # this can happen if the name corresponds to the position in a list module list of layers
                        # in this case the decoder has added a cross-attention that the encoder does not have
                        # thus skip this step and subtract one layer pos from encoder
                        encoder_layer_pos -= 1
                        continue
                elif name not in encoder_modules:
                    continue
                elif depth > 500:
                    raise ValueError(
                        "Max depth of recursive function `tie_encoder_to_decoder` reached. It seems that there is a circular dependency between two or more `nn.Modules` of your model."
                    )
                else:
                    decoder_name = encoder_name = name
                tie_encoder_to_decoder_recursively(
                    decoder_modules[decoder_name],
                    encoder_modules[encoder_name],
                    module_name + "/" + name,
                    uninitialized_encoder_weights,
                    skip_key,
                    depth=depth + 1,
                )
                all_encoder_weights.remove(module_name + "/" + encoder_name)

            uninitialized_encoder_weights += list(all_encoder_weights)

    # tie weights recursively
    tie_encoder_to_decoder_recursively(decoder, encoder, base_model_prefix, uninitialized_encoder_weights, skip_key) 

def forward_and_backward(model, config, ego_pairs, cross_pairs, sequences, cross_mask, alpha, epoch):
    losses = {}
    
    ego_audio, ego_caption = ego_pairs

    ego_itc = model(ego_audio, ego_caption, sequences, alpha, {"wtc"}, update_momentum=True, update_features=True if "tpc" in config['ablation_loss'] else False)["loss"]
    ego_itc.backward()
    losses["ego_itc"] = ego_itc

    if config['ablation_loss'] != "":
        cross_audio, cross_caption = cross_pairs
        losses2 = model(cross_audio, cross_caption, sequences, alpha, config['ablation_loss'], update_momentum=False, cross_mask=cross_mask, epoch=epoch)
        if losses2["loss"] != 0: losses2["loss"].backward()
        for l in losses2:
            losses[l] = losses2[l]

    return losses 


class TransformerEncoderLayer(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        self.conf["component"] = ["w_v"] #["ln", "w_v"]
        self.embedding_size = conf["embedding_size"]
        self.w_q = nn.Linear(self.embedding_size,
                             self.embedding_size, bias=False)
        nn.init.xavier_normal_(self.w_q.weight)
        self.w_k = nn.Linear(self.embedding_size,
                            self.embedding_size, bias=False)
        nn.init.xavier_normal_(self.w_k.weight)
        self.w_v = nn.Linear(self.embedding_size,
                            self.embedding_size, bias=False)
        nn.init.xavier_normal_(self.w_v.weight)
        self.ln = nn.LayerNorm(self.embedding_size, elementwise_affine=False)
        
    def forward(self, features, mask):
        # features: [bs, n_seq, d]
        if "ln" in self.conf["component"]:
            features = self.ln(features)
        q = self.w_q(features)
        k = self.w_k(features)
        v = self.w_v(features) if "w_v" in self.conf["component"] else features
        # [bs, n_seq, n_seq]
        attn = q.mul(self.embedding_size ** -0.5) @ k.transpose(-1, -2)
        attn = attn.masked_fill(mask.unsqueeze(-2), -1e9) # mask: [bs, 1, N_max]
        attn = attn.softmax(dim=-1)

        features = attn @ v  # [bs, n_seq, d]
        return features

class Playlist_Constructor(nn.Module):
    def __init__(self, conf):
        super().__init__()

        self.method = conf["method"]
        self.device = conf["device"]
        if 'num_layers' not in conf: 
            self.num_layers = 1
        else:
            self.num_layers = conf["num_layers"]
        if self.method == "soft_weight":
            self.num_tracks = conf["num_tracks"]
            self.soft_weights = nn.Embedding(self.num_tracks, 1) 
            torch.nn.init.ones_(self.soft_weights.weight)
        elif self.method == "self_attn":
            self.transformer_encoder = nn.ModuleList([TransformerEncoderLayer(conf).to(self.device) for _ in range(self.num_layers)])
        
    def forward(self, track_idx_seq, track_feat_seq, mask, length):
        # track_idx_seq: [bs, N_max]
        # track_feat_seq: [bs, N_max, dim]
        # mask: [bs, N_max]
        # length: [bs]
        if self.method == "average":
            track_feat_seq = track_feat_seq.masked_fill(mask.unsqueeze(-1), 0) # [bs, N_max, dim]
            feat = track_feat_seq.sum(-2) / length.unsqueeze(-1)

        elif self.method == "soft_weight":
            track_feat_seq = self.soft_weights(track_idx_seq) * track_feat_seq
            track_feat_seq = track_feat_seq.masked_fill(mask.unsqueeze(-1), 0) # [bs, N_max, dim]
            feat = track_feat_seq.sum(-2) / length.unsqueeze(-1)

        elif self.method == "self_attn":
            feat = track_feat_seq
            for encoder_layer in self.transformer_encoder:
                feat = encoder_layer(feat, mask)
            feat = feat.sum(-2) / length.unsqueeze(-1)
        return feat
        