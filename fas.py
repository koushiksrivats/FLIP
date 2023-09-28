import os
import sys
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import third_party

sys.path.append('../../')
import timm
import clip
from prompt_templates import spoof_templates, real_templates
from collections import OrderedDict


def l2_norm(input, axis=1):
  norm = torch.norm(input, 2, axis, True)
  output = torch.div(input, norm)
  return output


# feature generators
class feature_generator_adapt(nn.Module):

  def __init__(self, gamma, beta):
    super(feature_generator_adapt, self).__init__()
    self.vit = third_party.create_model(
        'vit_base_patch16_224', pretrained=True, gamma=gamma, beta=beta)

  def forward(self, input):
    feat, total_loss = self.vit.forward_features(input)
    return feat, total_loss


class feature_generator(nn.Module):

  def __init__(self):
    super(feature_generator, self).__init__()
    self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)
    # self.vit.head = nn.Identity() # remove the classification head for timm version 0.6.12

  def forward(self, input):
    # feat = self.vit.forward_features(input).detach()
    feat = self.vit.forward_features(input) # for timm version 0.4.9
    # feat = self.vit.forward_features(input) # for timm version 0.4.9
    # feat = self.vit.forward(input) # for timm version 0.6.12
    print(f'feat : {feat}')
    return feat


class feature_generator_beit(nn.Module):

  def __init__(self):
    super(feature_generator_beit, self).__init__()
    self.vit = timm.create_model('beitv2_base_patch16_224', pretrained=True) # This will load the v2 of BEiT and the weights will be the pretrained on in1k and intermediate finetuned on 21kto1k
    self.vit.head = nn.Identity()

  def forward(self, input):
    feat = self.vit(input)
    return feat


class feature_generator_clip(nn.Module):

  def __init__(self):
    super(feature_generator_clip, self).__init__()
    self.vit, _ = clip.load("ViT-B/16", device='cuda')
    self.vit = self.vit.visual

  def forward(self, input):
    feat = self.vit.forward_full(input)
    # feat, _ = self.vit.forward_full(input)
    return feat


class feature_generator_fix_vitaf(nn.Module):

  def __init__(self):
    super(feature_generator_fix_vitaf, self).__init__()
    self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)
    self.vit.head = nn.Identity() # remove the classification head for timm version 0.6.12

  def forward(self, input):
    # feat = self.vit.forward_features(input).detach() # for timm version 0.4.9
    feat = self.vit.forward(input).detach() # for timm version 0.6.12
    # feat = self.vit.forward_features(input)
    return feat


# feature embedders
class feature_embedder(nn.Module):

  def __init__(self):
    super(feature_embedder, self).__init__()
    self.bottleneck_layer_fc = nn.Linear(768, 512)
    self.bottleneck_layer_fc.weight.data.normal_(0, 0.005)
    self.bottleneck_layer_fc.bias.data.fill_(0.1)
    self.bottleneck_layer = nn.Sequential(self.bottleneck_layer_fc, nn.ReLU(),
                                          nn.Dropout(0.5))

  def forward(self, input, norm_flag=True):
    feature = self.bottleneck_layer(input)
    if (norm_flag):
      feature_norm = torch.norm(
          feature, p=2, dim=1, keepdim=True).clamp(min=1e-12)**0.5 * (2)**0.5
      feature = torch.div(feature, feature_norm)
    return feature

# classifier
class classifier(nn.Module):

  def __init__(self):
    super(classifier, self).__init__()
    self.classifier_layer = nn.Linear(512, 2)
    self.classifier_layer.weight.data.normal_(0, 0.01)
    self.classifier_layer.bias.data.fill_(0.0)

  def forward(self, input, norm_flag=True):
    if (norm_flag):
      self.classifier_layer.weight.data = l2_norm(
          self.classifier_layer.weight, axis=0)
      classifier_out = self.classifier_layer(input)
    else:
      classifier_out = self.classifier_layer(input)
    return classifier_out


# Models

# adaptive transfomer
class fas_model_adapt(nn.Module):

  def __init__(self, gamma, beta):
    super(fas_model_adapt, self).__init__()
    self.backbone = feature_generator_adapt(gamma, beta)
    self.embedder = feature_embedder()
    self.classifier = classifier()

  def forward(self, input, norm_flag=True):
    feature, total_loss = self.backbone(input)
    feature = self.embedder(feature, norm_flag)
    classifier_out = self.classifier(feature, norm_flag)
    return classifier_out, feature, total_loss

# base vit
class fas_model_fix(nn.Module):

  def __init__(self):
    super(fas_model_fix, self).__init__()
    self.backbone = feature_generator()
    self.embedder = feature_embedder()
    self.classifier = classifier()

  def forward(self, input, norm_flag=True):
    feature = self.backbone(input)
    feature = self.embedder(feature, norm_flag)
    classifier_out = self.classifier(feature, norm_flag)
    return classifier_out, feature

# base vit - stage 1 for vitaf*
class fas_model_fix_vitaf(nn.Module):

  def __init__(self):
    super(fas_model_fix_vitaf, self).__init__()
    self.backbone = feature_generator_fix_vitaf()
    self.embedder = feature_embedder()
    self.classifier = classifier()

  def forward(self, input, norm_flag=True):
    feature = self.backbone(input)
    feature = self.embedder(feature, norm_flag)
    classifier_out = self.classifier(feature, norm_flag)
    return classifier_out, feature

# beit
class fas_model_beit(nn.Module):

  def __init__(self):
    super(fas_model_beit, self).__init__()
    self.backbone = feature_generator_beit()
    self.embedder = feature_embedder()
    self.classifier = classifier()

  def forward(self, input, norm_flag=True):
    feature = self.backbone(input)
    feature = self.embedder(feature, norm_flag)
    classifier_out = self.classifier(feature, norm_flag)
    return classifier_out, feature

# FLIP-V
class flip_v(nn.Module):

  def __init__(self):
    super(flip_v, self).__init__()
    self.backbone = feature_generator_clip()
    self.embedder = feature_embedder()
    self.classifier = classifier()

  def forward(self, input, norm_flag=True):
    feature,_ = self.backbone(input)
    feature = self.embedder(feature, norm_flag)
    classifier_out = self.classifier(feature, norm_flag)
    return classifier_out, feature

# FLIP-IT
class flip_it(nn.Module):

  def __init__(self):
    super(flip_it, self).__init__()
    self.model, _ = clip.load("ViT-B/16", 'cuda:0')
    # self.text_inputs = torch.cat([clip.tokenize("a photo of a spoof face") , clip.tokenize("a photo of a real face")]).cuda()
    
  def forward(self, input, norm_flag=True):
    # single text prompt per class
    # logits_per_image, logits_per_text = self.model(input, self.text_inputs)

    # Ensemble of text features
    # tokenize the spoof and real templates
    spoof_texts = clip.tokenize(spoof_templates).cuda(non_blocking=True) #tokenize
    real_texts = clip.tokenize(real_templates).cuda(non_blocking=True) #tokenize
    #embed with text encoder
    spoof_class_embeddings = self.model.encode_text(spoof_texts) 
    spoof_class_embeddings = spoof_class_embeddings.mean(dim=0)
    real_class_embeddings = self.model.encode_text(real_texts)
    real_class_embeddings = real_class_embeddings.mean(dim=0)

    # stack the embeddings for image-text similarity
    ensemble_weights = [spoof_class_embeddings, real_class_embeddings ] 
    text_features = torch.stack(ensemble_weights, dim=0).cuda()
    # get the image features
    image_features = self.model.encode_image(input)
    # normalized features
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # cosine similarity as logits
    logit_scale = self.model.logit_scale.exp()
    logits_per_image = logit_scale * image_features @ text_features.t()

    similarity = logits_per_image
    return similarity, None


  def forward_vis(self, input, norm_flag=True):
    # image_features, image_features_proj = self.model.encode_image(input)
    _, image_features_proj = self.model.visual.forward_full(input)
    feature = image_features_proj

    return None, feature

# FLIP-MCL
class flip_mcl(nn.Module):

  def __init__(self, in_dim, ssl_mlp_dim, ssl_emb_dim):
    super(flip_mcl, self).__init__()
    # load the model
    self.model, _ = clip.load("ViT-B/16", 'cuda:0')

    # define the SSL parameters
    self.image_mlp = self._build_mlp(in_dim=in_dim, mlp_dim=ssl_mlp_dim, out_dim=ssl_emb_dim)
    self.n_views = 2
    self.temperature = 0.1

    # dot product similarity
    self.cosine_similarity = nn.CosineSimilarity()
    # mse loss
    self.mse_loss = nn.MSELoss()

  def _build_mlp(self, in_dim, mlp_dim, out_dim):
    return nn.Sequential(OrderedDict([
        ("layer1", nn.Linear(in_dim, mlp_dim)),
        ("bn1", nn.BatchNorm1d(mlp_dim)),
        ("relu1", nn.ReLU(inplace=True)),
        ("layer2", nn.Linear(mlp_dim, mlp_dim)),
        ("bn2", nn.BatchNorm1d(mlp_dim)),
        ("relu2", nn.ReLU(inplace=True)),
        ("layer3", nn.Linear(mlp_dim, out_dim)),
    ]))

  def info_nce_loss(self, feature_view_1, feature_view_2):
        assert feature_view_1.shape == feature_view_2.shape
        features = torch.cat([feature_view_1, feature_view_2], dim=0)

        labels = torch.cat([torch.arange(feature_view_1.shape[0]) for i in range(self.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.cuda()

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).cuda()
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        logits = logits / self.temperature
        return logits, labels

  def forward(self, input, input_view_1, input_view_2, source_labels, norm_flag=True):
    # tokenize the spoof and real templates
    spoof_texts = clip.tokenize(spoof_templates).cuda(non_blocking=True) #tokenize
    real_texts = clip.tokenize(real_templates).cuda(non_blocking=True) #tokenize
    # encode the spoof and real templates with the text encoder
    all_spoof_class_embeddings = self.model.encode_text(spoof_texts)
    all_real_class_embeddings = self.model.encode_text(real_texts)
    
    # ------------------- Image-Text similarity branch -------------------
    # Ensemble of text features
    # embed with text encoder
    spoof_class_embeddings = all_spoof_class_embeddings.mean(dim=0)
    real_class_embeddings = all_real_class_embeddings.mean(dim=0)

    # stack the embeddings for image-text similarity
    ensemble_weights = [spoof_class_embeddings, real_class_embeddings ] 
    text_features = torch.stack(ensemble_weights, dim=0).cuda()
    # get the image features
    image_features = self.model.encode_image(input)
    # normalized features
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # cosine similarity as logits
    logit_scale = self.model.logit_scale.exp()
    logits_per_image = logit_scale * image_features @ text_features.t()

    similarity = logits_per_image
    # ------------------- Image-Text similarity branch -------------------


    # ------------------------------ Image SSL branch -------------------------------- # 
    # Get the image embeddings for the ssl views
    aug1 = self.model.encode_image(input_view_1) # Bx512
    aug2 = self.model.encode_image(input_view_2) # Bx512

    # Project the image embeddings to the SSL embedding space
    aug1_embed = self.image_mlp(aug1) # Bx256
    aug2_embed = self.image_mlp(aug2) # Bx256
    
    # Get the logits for the SSL loss
    logits_ssl, labels_ssl = self.info_nce_loss(aug1_embed, aug2_embed)
    # ------------------------------ Image SSL branch --------------------------------

    # ------------------------------ Image-Text dot product branch --------------------------------
    # Split the prompts into 2 views
    text_embedding_v1 = []
    text_embedding_v2 = []
    for label in source_labels:
        label = int(label.item())

        if label == 0: # spoof
          # Randomly choose indices for the 2 views
          available_indices = np.arange(0, len(spoof_templates))
          pair_1 = np.random.choice(available_indices, len(spoof_templates)//2)
          pair_2 = np.setdiff1d(available_indices, pair_1)
          # slice embedding based on the indices
          spoof_texts_v1 = [ all_spoof_class_embeddings[i] for i in pair_1] # slice from the embedded fake templates
          spoof_texts_v2 = [ all_spoof_class_embeddings[i] for i in pair_2] # slice from the embedded fake templates
          # stack the embeddings
          spoof_texts_v1 = torch.stack(spoof_texts_v1, dim=0).cuda() # 3x512
          spoof_texts_v2 = torch.stack(spoof_texts_v2, dim=0).cuda() # 3x512
          assert int(spoof_texts_v1.shape[1]) == 512 and int(spoof_texts_v2.shape[1]) == 512 , "text embedding shape is not 512"
          # append the embeddings
          text_embedding_v1.append(spoof_texts_v1.mean(dim=0))
          text_embedding_v2.append(spoof_texts_v2.mean(dim=0))

        elif label == 1: # real
          # Randomly choose indices for the 2 views
          available_indices = np.arange(0, len(real_templates))
          pair_1 = np.random.choice(available_indices, len(real_templates)//2)
          pair_2 = np.setdiff1d(available_indices, pair_1)
          # slice embedding based on the indices
          real_texts_v1 = [ all_real_class_embeddings[i] for i in pair_1] # slice from the tokenized templates
          real_texts_v2 = [ all_real_class_embeddings[i] for i in pair_2] # slice from the tokenized templates
          # stack the embeddings
          real_texts_v1 = torch.stack(real_texts_v1, dim=0).cuda() # 3x512
          real_texts_v2 = torch.stack(real_texts_v2, dim=0).cuda() # 3x512
          assert int(real_texts_v1.shape[1]) == 512 and int(real_texts_v2.shape[1]) == 512 , "text embedding shape is not 512"
          # append the embeddings
          text_embedding_v1.append(real_texts_v1.mean(dim=0))
          text_embedding_v2.append(real_texts_v2.mean(dim=0))
    
    text_embed_v1 = torch.stack(text_embedding_v1, dim=0).cuda() # Bx512
    text_embed_v2 = torch.stack(text_embedding_v2, dim=0).cuda() # Bx512
    assert int(text_embed_v1.shape[1]) == 512 and int(text_embed_v2.shape[1]) == 512 , "text embedding shape is not 512"

    # dot product of image and text embeddings
    aug1_norm = aug1 / aug1.norm(dim=-1, keepdim=True)
    aug2_norm = aug2 / aug2.norm(dim=-1, keepdim=True)

    text_embed_v1_norm = text_embed_v1 / text_embed_v1.norm(dim=-1, keepdim=True)
    text_embed_v2_norm = text_embed_v2 / text_embed_v2.norm(dim=-1, keepdim=True)

    aug1_text_dot_product = self.cosine_similarity(aug1_norm, text_embed_v1_norm)
    aug2_text_dot_product = self.cosine_similarity(aug2_norm, text_embed_v2_norm)

    # mse loss between the dot product of aug1 and aug2
    dot_product_loss = self.mse_loss(aug1_text_dot_product, aug2_text_dot_product)
    # ------------------------------ Image-Text dot product branch --------------------------------

    
    return similarity, logits_ssl, labels_ssl, dot_product_loss

  def forward_eval(self, input, norm_flag=True):
    # single text prompt per class
    # logits_per_image, logits_per_text = self.model(input, self.text_inputs)

    # Ensemble of text features
    ensemble_weights = []
    for classname in (['spoof', 'real']):
        if classname == 'spoof':
          texts = spoof_templates #format with spoof class
        elif classname == 'real':
          texts = real_templates #format with real class
        
        texts = clip.tokenize(texts).cuda() #tokenize
        class_embeddings = self.model.encode_text(texts) #embed with text encoder
        class_embedding = class_embeddings.mean(dim=0)
        ensemble_weights.append(class_embedding)
    text_features = torch.stack(ensemble_weights, dim=0).cuda()

    # get the image features
    image_features = self.model.encode_image(input)
    # normalized features
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # cosine similarity as logits
    logit_scale = self.model.logit_scale.exp()
    logits_per_image = logit_scale * image_features @ text_features.t()

    similarity = logits_per_image
    return similarity, None


  def forward_vis(self, input, norm_flag=True):
    # image_features, image_features_proj = self.model.encode_image(input)
    # _, image_features_proj = self.model.visual.forward_full(input)
    image_features, image_features_proj = self.model.visual.forward_full(input)
    feature = image_features_proj

    # return None, feature
    return image_features, feature
