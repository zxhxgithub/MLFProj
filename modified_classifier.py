# final

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from segment_anything.modeling.common import LayerNorm2d

class MaskDecoderClassifier(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        cla_head_depth: int = 3,
        cla_head_hidden_dim: int = 256,
        class_num: int = 13,
        den_emb_in_channel: int = 256,
        use_den_emb: bool = False,
    ) -> None:
        super().__init__()
        self.class_num = class_num
        self.transformer_dim = transformer_dim
        self.transformer = transformer
        self.cla_token = nn.Embedding(1, transformer_dim)
        self.mask_tokens = nn.Embedding(1, transformer_dim)
        self.cla_prediction_head = MLP(
            transformer_dim, cla_head_hidden_dim, class_num, cla_head_depth
        )
        self.use_den_emb = use_den_emb
        self.den_emb_downscaling = nn.Sequential(
            nn.Conv2d(den_emb_in_channel, den_emb_in_channel, kernel_size=2, stride=2),
            LayerNorm2d(den_emb_in_channel),
            nn.GELU(),
            nn.Conv2d(den_emb_in_channel, den_emb_in_channel, kernel_size=2, stride=2),
            LayerNorm2d(den_emb_in_channel),
            nn.GELU(),
            nn.Conv2d(den_emb_in_channel, den_emb_in_channel, kernel_size=1),
        )
        self.den_emb_encoder = nn.Sequential(
            nn.Conv2d(den_emb_in_channel, den_emb_in_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(den_emb_in_channel),
            nn.GELU(),
            nn.Conv2d(den_emb_in_channel, den_emb_in_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(den_emb_in_channel),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(den_emb_in_channel, den_emb_in_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(den_emb_in_channel),
            nn.GELU(),
            nn.Conv2d(den_emb_in_channel, den_emb_in_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(den_emb_in_channel),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(den_emb_in_channel, den_emb_in_channel, kernel_size=1),
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        den_emb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict classes given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs

        Returns:
          torch.Tensor: batched predicted classes of the corresponding masks
        """
        with torch.no_grad():
            den_emb = self.den_emb_downscaling(den_emb)
        pred_classes = self.predict_classes(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            den_emb = den_emb,
        )

        # Prepare output
        return pred_classes

    def predict_classes(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        den_emb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict classes. See 'forward' for more details."""
    # with torch.no_grad():
        # Concatenate output tokens
        output_tokens = torch.cat([self.cla_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        # print(output_tokens.shape[0], image_embeddings.shape, src.shape, sparse_prompt_embeddings.shape, dense_prompt_embeddings.shape)

        if self.use_den_emb:
            src = src + den_emb
        else:
            src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        cla_token_out = hs[:, 0, :]
        # print(cla_token_out.shape, cla_token_out.shape)

        # Generate class predictions
        cla_pred = self.cla_prediction_head(cla_token_out)

        return cla_pred
    

# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x


# class den_emb_downscaling(nn.Module):
#     def __init__(self, den_emb_in_channel: int = 256) -> None:
#         super().__init__()
#         self.den_emb_downscaling = nn.Sequential(
#             nn.Conv2d(den_emb_in_channel, den_emb_in_channel, kernel_size=2, stride=2),
#             LayerNorm2d(den_emb_in_channel),
#             nn.GELU(),
#             nn.Conv2d(den_emb_in_channel, den_emb_in_channel, kernel_size=2, stride=2),
#             LayerNorm2d(den_emb_in_channel),
#             nn.GELU(),
#             nn.Conv2d(den_emb_in_channel, den_emb_in_channel, kernel_size=1),
#         )
#         self.pred = nn.Sequential(
#             nn.Conv2d(256, 128, kernel_size=3, padding=1, stride=2),
#             LayerNorm2d(128),
#             nn.GELU(),
#             nn.Conv2d(128, 64, kernel_size=3, padding=1, stride=2),
#             LayerNorm2d(64),
#             nn.GELU(),
#             nn.Conv2d(den_emb_in_channel, den_emb_in_channel, kernel_size=1),
#         )
#
#     def forward(self, den_emb):
#         return self.den_emb_downscaling(den_emb)