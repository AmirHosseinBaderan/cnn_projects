import torch.nn as nn


from models.blocks.conv_block import ConvBlock
from models.blocks.down_block import DownBlock
from models.blocks.up_block import UpBlock
from models.blocks.bottleneck import Bottleneck


from models.embeddings.timestep_embedding import TimeEmbedding
from models.embeddings.label_embedding import LabelEmbedding



class UNet(nn.Module):

    def __init__(
        self,
        time_dim=64,
        label_dim=64,
    ):
        super().__init__()


        self.time_embedding = TimeEmbedding(
            time_dim
        )


        self.label_embedding = LabelEmbedding(
            10,
            label_dim
        )


        self.encoder = ConvBlock(
            1,
            32
        )


        self.down = DownBlock(
            32,
            64
        )


        self.bottleneck = Bottleneck(
            64
        )


        self.up = UpBlock(
            128,
            64,
            32
        )


        self.output = nn.Conv2d(
            32,
            1,
            kernel_size=1
        )



    def forward(
        self,
        x,
        timestep,
        label,
    ):

        # embeddings

        time_emb = self.time_embedding(
            timestep
        )


        label_emb = self.label_embedding(
            label
        )


        condition = (
            time_emb
            +
            label_emb
        )


        # encoder

        x = self.encoder(x)


        x, skip = self.down(x)


        # bottleneck

        x = self.bottleneck(
            x,
            condition
        )


        # decoder

        x = self.up(
            x,
            skip
        )


        # output

        return self.output(x)