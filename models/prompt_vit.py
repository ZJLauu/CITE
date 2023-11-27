from typing import List, Literal
import torch
import torch.nn as nn
import torch.nn.modules.activation as act
from mmcls.models.backbones import VisionTransformer
from mmcls.models import BACKBONES, HEADS, NECKS
from mmcls.models.builder import MODELS
from mmcls.models.heads import ClsHead
from mmcls.models.utils import resize_pos_embed
from transformers import AutoTokenizer, AutoModel

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


@BACKBONES.register_module()
class PromptedVisionTransformer(VisionTransformer):
    """Vision Transformer with visual prompts. Based on mmcls implementation."""

    def __init__(self,
                 prompt_length: int = 1,
                 prompt_layers: List[int] = None,
                 prompt_pos: Literal['prepend'] = 'prepend',
                 prompt_init: Literal['normal', 'uniform', 'zero', 'kaiming', 'token'] = 'normal',
                 fix: bool = True,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        for param in self.parameters():
            param.requires_grad = not fix

        self.prompt_layers = [0] if prompt_layers is None else prompt_layers
        prompt = torch.empty(
            len(self.prompt_layers),
            prompt_length,
            self.embed_dims
        )
        if prompt_init == 'uniform':
            nn.init.uniform_(prompt, -0.08, 0.08)
        elif prompt_init == 'zero':
            nn.init.zeros_(prompt)
        elif prompt_init == 'kaiming':
            nn.init.kaiming_normal_(prompt)
        elif prompt_init == 'token':
            nn.init.zeros_(prompt)
            self.prompt_initialized = False
        else:
            nn.init.normal_(prompt, std=0.02)
        self.prompt = nn.Parameter(prompt, requires_grad=True)
        self.prompt_length = prompt_length
        self.prompt_pos = prompt_pos

    def forward(self, x):
        """Following mmcls implementation."""
        B = x.shape[0]
        x, patch_resolution = self.patch_embed(x)

        # stole cls_tokens impl from Phil Wang, thanks
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + resize_pos_embed(
            self.pos_embed,
            self.patch_resolution,
            patch_resolution,
            mode=self.interpolate_mode,
            num_extra_tokens=self.num_extra_tokens)
        x = self.drop_after_pos(x)

        # Add prompt
        if hasattr(self, 'prompt_initialized') and not self.prompt_initialized:
            with torch.no_grad():
                self.prompt.data += x.mean([0, 1]).detach().clone()
            self.prompt_initialized = True
        prompt = self.prompt.unsqueeze(1).expand(-1, B, -1, -1)
        # prompt: [layer, batch, length, dim]
        if self.prompt_pos == 'prepend':
            x = torch.cat([x[:, :1, :], prompt[0, :, :, :], x[:, 1:, :]], dim=1)

        outs = []
        for i, layer in enumerate(self.layers):
            if i in self.prompt_layers:
                if self.prompt_pos == 'prepend':
                    x = torch.cat([
                        x[:, :1, :],
                        prompt[i, :, :, :],
                        x[:, 1 + self.prompt_length:, :]
                    ], dim=1)
            x = layer(x)

            if i == len(self.layers) - 1 and self.final_norm:
                x = self.norm1(x)

            if i in self.out_indices:
                outs.append(x[:, 0])

        return tuple(outs)


@NECKS.register_module()
class ProjectionNeck(nn.Linear):
    """Linear projection layer."""

    def __init__(self,
                 *args,
                 init='kaiming_uniform',
                 act_before=None,
                 act_after=None,
                 fix=False,
                 float16=False,
                 **kwargs):
        kwargs['dtype'] = torch.float16 if float16 else torch.float32
        super().__init__(*args, **kwargs)
        if init == 'identity':
            nn.init.eye_(self.weight)
        elif init == 'normal':
            nn.init.normal_(self.weight, std=0.02)
        elif init == 'normal_identity':
            nn.init.normal_(self.weight, std=0.02)
            dim = min(self.weight.shape)
            with torch.no_grad():
                self.weight[:dim, :dim] += torch.eye(dim)
        elif init == 'kaiming_uniform':
            pass
        else:
            raise NotImplementedError
        if act_before is not None:
            self.act_before = getattr(act, act_before)()
        if act_after is not None:
            self.act_after = getattr(act, act_after)()
        for param in self.parameters():
            param.requires_grad = not fix

    def forward(self, x):
        if isinstance(x, tuple):
            x = x[0]
        if isinstance(x, list):
            x = x[1]  # cls token
        if hasattr(self, 'act_before'):
            x = self.act_before(x)
        x = super().forward(x)
        if hasattr(self, 'act_after'):
            x = self.act_after(x)
        return x


@HEADS.register_module()
class TextEmbeddingHead(ClsHead):
    """Text embedding head."""

    def __init__(self,
                 texts: List[str],
                 text_encoder: dict,
                 temperature: float = 1.0,
                 learnable_t: bool = False,
                 float16: bool = False,
                 **kwargs):
        super().__init__(**kwargs)

        dtype = torch.float16 if float16 else torch.float32
        text_encoder = MODELS.build(text_encoder)
        self.weights = text_encoder(texts).type(dtype)  # [n_cls, dim]
        self.temperature = torch.tensor(temperature, dtype=dtype).to(DEVICE)
        if learnable_t:
            self.temperature = nn.Parameter(self.temperature)

    def pre_logits(self, x):
        if isinstance(x, tuple):
            x = x[-1]
        if isinstance(x, list):
            x = x[-1]  # cls token
        return x

    def forward(self, x):
        dtype = x.dtype
        x = self.pre_logits(x)
        x = x / x.norm(dim=-1, keepdim=True)
        weights = self.weights / self.weights.norm(dim=-1, keepdim=True)
        t = self.temperature.exp().type(dtype)
        cls_score = t * x @ weights.type(dtype).t()
        return cls_score

    def forward_train(self, x, gt_label, **kwargs):
        cls_score = self.forward(x)
        losses = self.loss(cls_score, gt_label, **kwargs)
        return losses

    def simple_test(self, x, **kwargs):
        cls_score = self.forward(x)
        return super().simple_test(cls_score, **kwargs)


@MODELS.register_module()
class BERT(nn.Module):
    """A wrapper of BERT model for text embedding."""

    def __init__(self,
                 model: str = 'michiyasunaga/BioLinkBERT-large',
                 key: Literal['pooler_output', 'last_hidden_state'] = 'pooler_output',
                 text_feature_file: str = None):
        super().__init__()
        self.key = key
        if text_feature_file is not None:
            self.text_embeddings = torch.load(text_feature_file, map_location='cpu')
            return
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModel.from_pretrained(model)

    def forward(self, texts):
        if hasattr(self, 'text_embeddings'):
            return self.text_embeddings.to(DEVICE)
        texts = [self.tokenizer(t, return_tensors='pt') for t in texts]

        with torch.no_grad():
            if self.key == 'pooler_output':
                text_embeddings = torch.cat([self.model(**inputs)[self.key] for inputs in texts], dim=0)
            elif self.key == 'last_hidden_state':
                # use [CLS] token
                text_embeddings = torch.cat([self.model(**inputs)[self.key][:, 0] for inputs in texts], dim=0)
            else:
                raise NotImplementedError

        return text_embeddings.to(DEVICE)


@HEADS.register_module()
class MyTextEmbeddingHead(ClsHead):
    """Text embedding head."""

    def __init__(self,
                 text_encoder: dict,
                 temperature: float = 1.0,
                 learnable_t: bool = False,
                 float16: bool = False,
                 **kwargs):
        super().__init__(**kwargs)

        dtype = torch.float16 if float16 else torch.float32
        self.text_encoder = MODELS.build(text_encoder)
        self.temperature = torch.tensor(temperature, dtype=dtype).to(DEVICE)
        if learnable_t:
            self.temperature = nn.Parameter(self.temperature)

    def pre_logits(self, x):
        if isinstance(x, tuple):
            x = x[-1]
        if isinstance(x, list):
            x = x[-1]  # cls token
        return x

    def forward(self, x):
        dtype = x.dtype
        x = self.pre_logits(x)
        x = x / x.norm(dim=-1, keepdim=True)
        weights = self.text_encoder().type(dtype)
        weights = weights / weights.norm(dim=-1, keepdim=True)
        t = self.temperature.exp().type(dtype)
        cls_score = t * x @ weights.type(dtype).t()
        return cls_score

    def forward_train(self, x, gt_label, **kwargs):
        cls_score = self.forward(x)
        losses = self.loss(cls_score, gt_label, **kwargs)
        return losses

    def simple_test(self, x, **kwargs):
        cls_score = self.forward(x)
        return super().simple_test(cls_score, **kwargs)


@MODELS.register_module()
class TextEncoderWithPrompt(nn.Module):

    def __init__(self,
                 arch: str,
                 n_classes: int,
                 n_attr_per_class: int,
                 n_prompts_per_attr: int,
                 attributes: list,
                 text_feature_file: str = None):
        super().__init__()
        self.n_classes = n_classes
        self.n_attr_per_class = n_attr_per_class
        self.has_attributes = (attributes != [])
        if self.has_attributes:
            assert len(attributes) == n_classes
            for cls in attributes:
                assert len(cls) == n_attr_per_class
        if text_feature_file is not None:
            self.text_embeddings = torch.load(text_feature_file, map_location='cpu')
            return

        model, _ = clip.load(arch, device=DEVICE)
        for name, param in self.model.named_parameters():
            param.requires_grad_(False)

        self.transformer = model.transformer
        self.positional_embedding = model.positional_embedding
        self.ln_final = model.ln_final
        self.text_projection = model.text_projection
        self.dtype = model.dtype

        self.prompt_learner = PromptLearner(n_classes, n_attr_per_class, n_prompts_per_attr, attributes, model)

    def forward(self):
        if hasattr(self, 'text_embeddings'):
            return self.text_embeddings.to(DEVICE)
        x = self.prompt_learner() + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [n_cls * n_attr, n_prompts, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]),
              self.prompt_learner.tokenized_prompts.argmax(dim=-1)] @ self.text_projection  # [n_cls * n_attr, dim]
        # x = x[torch.arange(x.shape[0]),
        #       self.prompt_learner.tokenized_prompts.argmax(dim=-1) if not self.has_attributes else
        #       self.prompt_learner.tokenized_prompts[cls_i].argmax(dim=-1)] @ self.text_projection  # [n_attr, dim]

        return x.continguous().view(self.n_classes, self.n_attr_per_class, -1).mean(dim=1)  # [n_cls, dim]


@MODELS.register_module()
class PromptLearner(nn.Module):
    def __init__(self, n_classes, n_attr_per_class, n_prompts_per_attr, attributes, clip_model):
        super().__init__()

        self.has_attributes = (attributes != [])
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]

        # random initializing class-specific contexts
        ctx_vectors = torch.empty(n_classes * n_attr_per_class, n_prompts_per_attr, ctx_dim, dtype=dtype)

        nn.init.normal_(ctx_vectors, std=0.02)
        prompt_prefix = " ".join(["X"] * n_prompts_per_attr)

        self.text_prompts = nn.Parameter(ctx_vectors).to(DEVICE)  # to be optimized
        self._tokenizer = _Tokenizer()
        if not self.has_attributes:
            prompts = [prompt_prefix for i in range(n_classes * n_attr_per_class)]
            # tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(DEVICE)
            # with torch.no_grad():
            #     embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
            #
            # # These token vectors will be saved when in save_model(),
            # # but they should be ignored in load_model() as we want to use
            # # those computed using the current class names
            # self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
            # self.register_buffer("token_suffix", embedding[:, 1 + n_prompts_per_attr:, :])  # CLS, EOS
            #
            # self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        else:
            prompts = []
            for cls_attr in attributes:
                prompts.append(prompt_prefix + " " + attr for attr in cls_attr)
            # tokenized_prompts = [torch.cat([clip.tokenize(p) for p in cls_prompt]).to(DEVICE)
            #                      for cls_prompt in prompts]
            # with torch.no_grad():
            #     embedding = torch.stack([clip_model.token_embedding(tokenized_prompt).type(dtype)
            #                              for tokenized_prompt in tokenized_prompts]).to(DEVICE)
            # self.register_buffer("token_prefix", embedding[:, :, :1, :])  # SOS
            # self.register_buffer("token_suffix", embedding[:, :, 1 + n_prompts_per_attr:, :])  # CLS, EOS
            # self.tokenized_prompts = tokenized_prompts  # torch.Tensor

        self.tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(DEVICE)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)  # [n_clss * n_attr, len, dim]
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_prompts_per_attr:, :])  # CLS, EOS

    def forward(self):

        prefix = self.token_prefix
        suffix = self.token_suffix

        prompts = torch.cat(
            [
                prefix,  # (n_clss * n_attr, 1, dim)
                self.text_prompts,  # (n_clss * n_attr, n_prompts_per_attr, dim)
                suffix,  # (n_clss * n_attr, *, dim)
            ],
            dim=1,
        )
        return prompts
