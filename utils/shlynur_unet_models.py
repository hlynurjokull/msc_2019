import torch
import torch.nn as nn
import torch.nn.functional as functional
import bruegger_code.revtorch.revtorch as rv


def initialize_weights(*models):
    """
    Initialize the weights via the Kaiming initialization.
    :param models: model.
    """
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


# class _EncoderBlock2D(nn.Module):
#     def __init__(self, in_channels, out_channels, dropout=False):
#         super(_EncoderBlock2D, self).__init__()
#         layers = [
#             nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3),
#             nn.BatchNorm2d(num_features=out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3),
#             nn.BatchNorm2d(num_features=out_channels),
#             nn.ReLU(inplace=True)
#         ]
#         if dropout:
#             layers.append(nn.Dropout())
#         layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
#         self.encode = nn.Sequential(*layers)
#
#     def forward(self, x):
#         return self.encode(x)
#
#
# class _DecoderBlock2D(nn.Module):
#     def __init__(self, in_channels, middle_channels, out_channels):
#         super(_DecoderBlock2D, self).__init__()
#         layers = [
#             nn.Conv2d(in_channels=in_channels, out_channels=middle_channels, kernel_size=3),
#             nn.BatchNorm2d(num_features=middle_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels=middle_channels, out_channels=middle_channels, kernel_size=3),
#             nn.BatchNorm2d(num_features=middle_channels),
#             nn.ReLU(inplace=True),
#             nn.ConvTranspose2d(in_channels=middle_channels, out_channels=out_channels, kernel_size=2, stride=2)
#         ]
#         self.decode = nn.Sequential(*layers)
#
#     def forward(self, x):
#         return self.decode(x)
#
#
# class _ConvBlock3D(nn.Module):
#     def __init__(self, in_channels, middle_channels, out_channels, kernel_size, padding, max_pooling_d,
#                  downsample=False, upsample=False):
#         super(_ConvBlock3D, self).__init__()
#         layers = [
#             nn.Conv3d(in_channels=in_channels, out_channels=middle_channels,
#                       kernel_size=kernel_size, padding=padding, bias=False),
#             nn.BatchNorm3d(num_features=middle_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv3d(in_channels=middle_channels, out_channels=out_channels,
#                       kernel_size=kernel_size, padding=padding, bias=False),
#             nn.BatchNorm3d(num_features=out_channels),
#             nn.ReLU(inplace=True)
#         ]
#         if downsample:
#             layers.insert(0, nn.MaxPool3d(kernel_size=(max_pooling_d, 2, 2), stride=(max_pooling_d, 2, 2)))
#         elif upsample:
#             layers.insert(0, nn.ConvTranspose3d(in_channels=))
#         self.encode = nn.Sequential(*layers)
#
#     def forward(self, x):
#         return self.encode(x)


class _EncoderBlock3D(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, kernel_size, padding, max_pooling_d, dropout=False,
                 down_sample=False):
        super(_EncoderBlock3D, self).__init__()
        layers = [
            nn.Conv3d(in_channels=in_channels, out_channels=middle_channels,
                      kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm3d(num_features=middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=middle_channels, out_channels=out_channels,
                      kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm3d(num_features=out_channels),
            nn.ReLU(inplace=True)
        ]
        if dropout:
            layers.append(nn.Dropout3d(p=0.2))
        if down_sample:
            layers.insert(0, nn.MaxPool3d(kernel_size=(max_pooling_d, 2, 2), stride=(max_pooling_d, 2, 2)))
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class _DecoderBlock3D(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, kernel_size, padding, conv_transposing_d,
                 down_sample=False):
        super(_DecoderBlock3D, self).__init__()
        layers = [
            nn.Conv3d(in_channels=in_channels, out_channels=middle_channels,
                      kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm3d(num_features=middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=middle_channels, out_channels=middle_channels,
                      kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm3d(num_features=middle_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(in_channels=middle_channels, out_channels=out_channels,
                               kernel_size=(conv_transposing_d, 2, 2), stride=(conv_transposing_d, 2, 2))
        ]
        if down_sample:
            layers.insert(0, nn.MaxPool3d(kernel_size=(conv_transposing_d, 2, 2), stride=(conv_transposing_d, 2, 2)))
        self.decode = nn.Sequential(*layers)

    def forward(self, x):
        return self.decode(x)


# class UNet2D(nn.Module):
#     def __init__(self, number_of_classes, number_of_filters):
#         super(UNet2D, self).__init__()
#         self.enc1 = _EncoderBlock2D(in_channels=1, out_channels=number_of_filters)
#         self.enc2 = _EncoderBlock2D(in_channels=number_of_filters, out_channels=number_of_filters * 2)
#         self.enc3 = _EncoderBlock2D(in_channels=number_of_filters * 2, out_channels=number_of_filters * 4)
#         self.enc4 = _EncoderBlock2D(in_channels=number_of_filters * 4, out_channels=number_of_filters * 8,
#                                     dropout=True)
#         self.center = _DecoderBlock2D(in_channels=number_of_filters * 8, middle_channels=number_of_filters * 16,
#                                       out_channels=number_of_filters * 8)
#         self.dec4 = _DecoderBlock2D(in_channels=number_of_filters * 16, middle_channels=number_of_filters * 8,
#                                     out_channels=number_of_filters * 4)
#         self.dec3 = _DecoderBlock2D(in_channels=number_of_filters * 8, middle_channels=number_of_filters * 4,
#                                     out_channels=number_of_filters * 2)
#         self.dec2 = _DecoderBlock2D(in_channels=number_of_filters * 4, middle_channels=number_of_filters * 2,
#                                     out_channels=number_of_filters)
#         self.dec1 = nn.Sequential(
#             nn.Conv2d(in_channels=number_of_filters * 2, out_channels=number_of_filters, kernel_size=3),
#             nn.BatchNorm2d(num_features=number_of_filters),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels=number_of_filters, out_channels=number_of_filters, kernel_size=3),
#             nn.BatchNorm2d(num_features=number_of_filters),
#             nn.ReLU(inplace=True)
#         )
#         self.final = nn.Conv2d(in_channels=number_of_filters, out_channels=number_of_classes, kernel_size=1)
#         initialize_weights(self)
#
#     def forward(self, x):
#         enc1 = self.enc1(x)
#         enc2 = self.enc2(enc1)
#         enc3 = self.enc3(enc2)
#         enc4 = self.enc4(enc3)
#         center = self.center(enc4)
#         dec4 = self.dec4(torch.cat((center,
#                                     functional.interpolate(enc4, center.size()[2:], mode='bilinear',
#                                                            align_corners=True)), 1))
#         dec3 = self.dec3(torch.cat((dec4,
#                                     functional.interpolate(enc3, dec4.size()[2:], mode='bilinear',
#                                                            align_corners=True)), 1))
#         dec2 = self.dec2(torch.cat((dec3,
#                                     functional.interpolate(enc2, dec3.size()[2:], mode='bilinear',
#                                                            align_corners=True)), 1))
#         dec1 = self.dec1(torch.cat((dec2,
#                                     functional.interpolate(enc1, dec2.size()[2:], mode='bilinear',
#                                                            align_corners=True)), 1))
#         final = self.final(dec1)
#         # print("x: {} \tfinal: {}".format(x.shape, final.shape))
#         return functional.interpolate(final, x.size()[2:], mode='bilinear', align_corners=True)


class UNet3D(nn.Module):
    def __init__(self, bool_prostate_data, in_channels, number_of_classes, number_of_filters, kernel_size=(3, 3, 3),
                 padding=(1, 1, 1), max_pooling_d=1):
        super(UNet3D, self).__init__()
        self.bool_prostate_data = bool_prostate_data

        self.enc1 = _EncoderBlock3D(
            in_channels=in_channels, middle_channels=(number_of_filters // 2), out_channels=number_of_filters,
            kernel_size=kernel_size, padding=padding, max_pooling_d=max_pooling_d, dropout=False, down_sample=False)
        self.enc2 = _EncoderBlock3D(
            in_channels=number_of_filters, middle_channels=number_of_filters, out_channels=number_of_filters*2,
            kernel_size=kernel_size, padding=padding, max_pooling_d=max_pooling_d, dropout=False, down_sample=True)
        self.enc3 = _EncoderBlock3D(
            in_channels=number_of_filters*2, middle_channels=number_of_filters*2, out_channels=number_of_filters*4,
            kernel_size=kernel_size, padding=padding, max_pooling_d=max_pooling_d, dropout=True, down_sample=True)
        self.center = _DecoderBlock3D(
            in_channels=number_of_filters*4, middle_channels=number_of_filters*4, out_channels=number_of_filters*8,
            kernel_size=kernel_size, padding=padding, conv_transposing_d=max_pooling_d, down_sample=True)
        self.dec3 = _DecoderBlock3D(
            in_channels=number_of_filters*12, middle_channels=number_of_filters*4, out_channels=number_of_filters*4,
            kernel_size=kernel_size, padding=padding, conv_transposing_d=max_pooling_d)
        self.dec2 = _DecoderBlock3D(
            in_channels=number_of_filters*6, middle_channels=number_of_filters*2, out_channels=number_of_filters*2,
            kernel_size=kernel_size, padding=padding, conv_transposing_d=max_pooling_d)
        self.dec1 = nn.Sequential(
            nn.Conv3d(in_channels=number_of_filters*3, out_channels=number_of_filters, kernel_size=kernel_size,
                      padding=padding, bias=False),
            nn.BatchNorm3d(num_features=number_of_filters),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=number_of_filters, out_channels=number_of_filters, kernel_size=kernel_size,
                      padding=padding, bias=False),
            nn.BatchNorm3d(num_features=number_of_filters),
            nn.ReLU(inplace=True)
        )
        self.final = nn.Conv3d(in_channels=number_of_filters, out_channels=number_of_classes, kernel_size=(1, 1, 1))
        initialize_weights(self)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        center = self.center(enc3)

        # if self.bool_prostate_data:
        dec3 = self.dec3(torch.cat((center, enc3), 1))
        dec2 = self.dec2(torch.cat((dec3, enc2), 1))
        dec1 = self.dec1(torch.cat((dec2, enc1), 1))
        final = self.final(dec1)
        return final
        # else:
        #     dec3 = self.dec3(torch.cat((center, functional.interpolate(enc3, center.size()[2:], mode='trilinear',
        #                                                                align_corners=True)), 1))
        #     dec2 = self.dec2(torch.cat((dec3, functional.interpolate(enc2, dec3.size()[2:], mode='trilinear',
        #                                                              align_corners=True)), 1))
        #     dec1 = self.dec1(torch.cat((dec2, functional.interpolate(enc1, dec2.size()[2:], mode='trilinear',
        #                                                              align_corners=True)), 1))
        #     final = self.final(dec1)
        #     return functional.interpolate(final, x.size()[2:], mode='trilinear', align_corners=True)

        # return final
        # return functional.interpolate(final, x.size()[2:], mode='trilinear', align_corners=True)


# class UNetCascade2D(nn.Module):
#     def __init__(self, number_of_classes, number_of_filters):
#         super(UNetCascade2D, self).__init__()
#         self.enc1 = _EncoderBlock2D(in_channels=5, out_channels=number_of_filters)
#         self.enc2 = _EncoderBlock2D(in_channels=number_of_filters, out_channels=number_of_filters * 2)
#         self.enc3 = _EncoderBlock2D(in_channels=number_of_filters * 2, out_channels=number_of_filters * 4)
#         self.enc4 = _EncoderBlock2D(in_channels=number_of_filters * 4, out_channels=number_of_filters * 8,
#                                     dropout=True)
#         self.center = _DecoderBlock2D(in_channels=number_of_filters * 8, middle_channels=number_of_filters * 16,
#                                       out_channels=number_of_filters * 8)
#         self.dec4 = _DecoderBlock2D(in_channels=number_of_filters * 16, middle_channels=number_of_filters * 8,
#                                     out_channels=number_of_filters * 4)
#         self.dec3 = _DecoderBlock2D(in_channels=number_of_filters * 8, middle_channels=number_of_filters * 4,
#                                     out_channels=number_of_filters * 2)
#         self.dec2 = _DecoderBlock2D(in_channels=number_of_filters * 4, middle_channels=number_of_filters * 2,
#                                     out_channels=number_of_filters)
#         self.dec1 = nn.Sequential(
#             nn.Conv2d(in_channels=number_of_filters * 2, out_channels=number_of_filters, kernel_size=3),
#             nn.BatchNorm2d(num_features=number_of_filters),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels=number_of_filters, out_channels=number_of_filters, kernel_size=3),
#             nn.BatchNorm2d(num_features=number_of_filters),
#             nn.ReLU(inplace=True)
#         )
#         self.final = nn.Conv2d(in_channels=number_of_filters, out_channels=number_of_classes, kernel_size=1)
#         initialize_weights(self)
#
#     def forward(self, input_1, input_2, input_3, input_4, input_5):
#         x = torch.cat((input_1, input_2, input_3, input_4, input_5), 1)
#         enc1 = self.enc1(x)
#         enc2 = self.enc2(enc1)
#         enc3 = self.enc3(enc2)
#         enc4 = self.enc4(enc3)
#         center = self.center(enc4)
#         dec4 = self.dec4(torch.cat((center,
#                                     functional.interpolate(enc4, center.size()[2:], mode='bilinear',
#                                                            align_corners=True)), 1))
#         dec3 = self.dec3(torch.cat((dec4,
#                                     functional.interpolate(enc3, dec4.size()[2:], mode='bilinear',
#                                                            align_corners=True)), 1))
#         dec2 = self.dec2(torch.cat((dec3,
#                                     functional.interpolate(enc2, dec3.size()[2:], mode='bilinear',
#                                                            align_corners=True)), 1))
#         dec1 = self.dec1(torch.cat((dec2,
#                                     functional.interpolate(enc1, dec2.size()[2:], mode='bilinear',
#                                                            align_corners=True)), 1))
#         final = self.final(dec1)
#         return functional.interpolate(final, x.size()[2:], mode='bilinear', align_corners=True)


# class UNetCascade3D(nn.Module):
#     def __init__(self, number_of_classes, number_of_filters, kernel_size=(3, 3, 3), padding=(1, 1, 1),
#                  max_pooling_d=1):
#         super(UNetCascade3D, self).__init__()
#         self.enc1 = _EncoderBlock3D(in_channels=5, middle_channels=int(number_of_filters / 2),
#                                     out_channels=number_of_filters, kernel_size=kernel_size, padding=padding,
#                                     max_pooling_d=max_pooling_d)
#         self.enc2 = _EncoderBlock3D(in_channels=number_of_filters, middle_channels=number_of_filters,
#                                     out_channels=number_of_filters * 2, kernel_size=kernel_size, padding=padding,
#                                     max_pooling_d=max_pooling_d)
#         self.enc3 = _EncoderBlock3D(in_channels=number_of_filters * 2, middle_channels=number_of_filters * 2,
#                                     out_channels=number_of_filters * 4, kernel_size=kernel_size, padding=padding,
#                                     dropout=True, max_pooling_d=max_pooling_d)
#         self.center = _DecoderBlock3D(in_channels=number_of_filters * 4, middle_channels=number_of_filters * 4,
#                                       out_channels=number_of_filters * 8, kernel_size=kernel_size, padding=padding,
#                                       conv_transposing_d=max_pooling_d)
#         self.dec3 = _DecoderBlock3D(in_channels=number_of_filters * 12, middle_channels=number_of_filters * 4,
#                                     out_channels=number_of_filters * 4, kernel_size=kernel_size, padding=padding,
#                                     conv_transposing_d=max_pooling_d)
#         self.dec2 = _DecoderBlock3D(in_channels=number_of_filters * 6, middle_channels=number_of_filters * 2,
#                                     out_channels=number_of_filters * 2, kernel_size=kernel_size, padding=padding,
#                                     conv_transposing_d=max_pooling_d)
#         self.dec1 = nn.Sequential(
#             nn.Conv3d(in_channels=number_of_filters * 3, out_channels=number_of_filters,
#                       kernel_size=kernel_size, padding=padding),
#             nn.BatchNorm3d(num_features=number_of_filters),
#             nn.ReLU(inplace=True),
#             nn.Conv3d(in_channels=number_of_filters, out_channels=number_of_filters,
#                       kernel_size=kernel_size, padding=padding),
#             nn.BatchNorm3d(num_features=number_of_filters),
#             nn.ReLU(inplace=True)
#         )
#         self.final = nn.Conv3d(in_channels=number_of_filters, out_channels=number_of_classes, kernel_size=1)
#         initialize_weights(self)
#         # self.enc1 = _EncoderBlock3D(in_channels=5, out_channels=64)
#         # self.enc2 = _EncoderBlock3D(in_channels=64, out_channels=128)
#         # self.enc3 = _EncoderBlock3D(in_channels=128, out_channels=256)
#         # self.enc4 = _EncoderBlock3D(in_channels=256, out_channels=512, dropout=True)
#         # self.center = _DecoderBlock3D(in_channels=512, middle_channels=1024, out_channels=512)
#         # self.dec4 = _DecoderBlock3D(in_channels=1024, middle_channels=512, out_channels=256)
#         # self.dec3 = _DecoderBlock3D(in_channels=512, middle_channels=256, out_channels=128)
#         # self.dec2 = _DecoderBlock3D(in_channels=256, middle_channels=128, out_channels=64)
#         # self.dec1 = nn.Sequential(
#         #     nn.Conv3d(in_channels=128, out_channels=64, kernel_size=3),
#         #     nn.BatchNorm3d(num_features=64),
#         #     nn.ReLU(inplace=True),
#         #     nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3),
#         #     nn.BatchNorm3d(num_features=64),
#         #     nn.ReLU(inplace=True)
#         # )
#         # self.final = nn.Conv3d(in_channels=64, out_channels=number_of_classes, kernel_size=1)
#         # initialize_weights(self)
#
#     def forward(self, input_1, input_2, input_3, input_4, input_5):
#         x = torch.cat((input_1, input_2, input_3, input_4, input_5), 1)
#         enc1 = self.enc1(x)
#         enc2 = self.enc2(enc1)
#         enc3 = self.enc3(enc2)
#         center = self.center(enc3)
#         dec3 = self.dec3(torch.cat((center,
#                                     functional.interpolate(enc3, center.size()[2:], mode='trilinear',
#                                                            align_corners=True)), 1))
#         dec2 = self.dec2(torch.cat((dec3,
#                                     functional.interpolate(enc2, dec3.size()[2:], mode='trilinear',
#                                                            align_corners=True)), 1))
#         dec1 = self.dec1(torch.cat((dec2,
#                                     functional.interpolate(enc1, dec2.size()[2:], mode='trilinear',
#                                                            align_corners=True)), 1))
#         final = self.final(dec1)
#         return functional.interpolate(final, x.size()[2:], mode='trilinear', align_corners=True)
#         # enc1 = self.enc1(x)
#         # enc2 = self.enc2(enc1)
#         # enc3 = self.enc3(enc2)
#         # enc4 = self.enc4(enc3)
#         # center = self.center(enc4)
#         # dec4 = self.dec4(torch.cat((center,
#         #                             functional.interpolate(enc4, center.size()[2:], mode='trilinear',
#         #                                                    align_corners=True)), 1))
#         # dec3 = self.dec3(torch.cat((dec4,
#         #                             functional.interpolate(enc3, dec4.size()[2:], mode='trilinear',
#         #                                                    align_corners=True)), 1))
#         # dec2 = self.dec2(torch.cat((dec3,
#         #                             functional.interpolate(enc2, dec3.size()[2:], mode='trilinear',
#         #                                                    align_corners=True)), 1))
#         # dec1 = self.dec1(torch.cat((dec2,
#         #                             functional.interpolate(enc1, dec2.size()[2:], mode='trilinear',
#         #                                                    align_corners=True)), 1))
#         # final = self.final(dec1)
#         # return functional.interpolate(final, x.size()[2:], mode='trilinear', align_corners=True)


########################################################################################################################
# # # # # # # # # # # # # # # # # # # # # # # # # # Bruegger's code # # # # # # # # # # # # # # # # # # # # # # # # # #
########################################################################################################################
# Not thoroughly enough tested.

# total_channels = [100, 200, 300, 400, 500]
total_channels = [4, 8, 16]


class ResidualInner(nn.Module):
    def __init__(self, channels, groups):
        super(ResidualInner, self).__init__()
        self.gn = nn.GroupNorm(groups, channels)
        self.conv = nn.Conv3d(channels, channels, 3, padding=1, bias=False)

    def forward(self, x):
        x = self.conv(functional.leaky_relu(self.gn(x), inplace=True))
        return x


def make_reversible_sequence(channels):
    inner_channels = channels // 2
    groups = total_channels[0] // 4
    f_block = ResidualInner(inner_channels, groups)
    g_block = ResidualInner(inner_channels, groups)
    return rv.ReversibleBlock(f_block=f_block, g_block=g_block)


def make_reversible_component(channels, block_count):
    modules = []
    for i in range(block_count):
        modules.append(make_reversible_sequence(channels))
    return rv.ReversibleSequence(nn.ModuleList(modules))


def get_channels_at_index(index):
    if index < 0:
        index = 0
    elif index >= len(total_channels):
        index = len(total_channels) - 1
    return total_channels[index]


class BrueggerEncoderModule(nn.Module):
    def __init__(self, in_channels, out_channels, depth, downsample=True):
        super(BrueggerEncoderModule, self).__init__()
        self.downsample = downsample
        if self.downsample:
            self.conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.reversible_blocks = make_reversible_component(channels=out_channels, block_count=depth)

    def forward(self, x):
        if self.downsample:
            x = functional.max_pool3d(x, 2)
            x = self.conv(x)  # increase the number of channels
        x = self.reversible_blocks(x)
        return x


class BrueggerDecoderModule(nn.Module):
    def __init__(self, in_channels, out_channels, depth, upsample=True):
        super(BrueggerDecoderModule, self).__init__()
        self.upsample = upsample
        if self.upsample:
            self.conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.reversible_blocks = make_reversible_component(channels=in_channels, block_count=depth)

    def forward(self, x):
        x = self.reversible_blocks(x)
        if self.upsample:
            x = self.conv(x)
            x = functional.interpolate(input=x, scale_factor=2, mode='trilinear', align_corners=False)
        return x


class Reversible3D(nn.Module):
    def __init__(self, encoder_depth):
        super(Reversible3D, self).__init__()
        self.encoder_depth = encoder_depth
        self.decoder_depth = 1
        self.levels = 5

        self.first_conv = nn.Conv3d(in_channels=4, out_channels=total_channels[0], kernel_size=3, padding=1, bias=False)
        self.dropout = nn.Dropout3d(p=0.2, inplace=True)
        self.last_conv = nn.Conv3d(in_channels=total_channels[0], out_channels=3, kernel_size=1, bias=True)

        # create encoder levels
        encoder_modules = []
        for i in range(self.levels):
            encoder_modules.append(BrueggerEncoderModule(in_channels=get_channels_at_index(i - 1),
                                                         out_channels=get_channels_at_index(i),
                                                         depth=self.encoder_depth, downsample=(i != 0)))
        self.encoders = nn.ModuleList(encoder_modules)

        # create decoder levels
        decoder_modules = []
        for i in range(self.levels):
            decoder_modules.append(BrueggerDecoderModule(in_channels=get_channels_at_index(self.levels - i - 1),
                                                         out_channels=get_channels_at_index(self.levels - i - 2),
                                                         depth=self.decoder_depth, upsample=(i != (self.levels - 1))))
        self.decoders = nn.ModuleList(decoder_modules)

    def forward(self, x):
        x = self.first_conv(x)
        x = self.dropout(x)

        input_stack = []
        for i in range(self.levels):
            x = self.encoders[i](x)
            if i < self.levels - 1:
                input_stack.append(x)
        for i in range(self.levels):
            x = self.decoders[i](x)
            if i < self.levels - 1:
                x = x + input_stack.pop()

        x = self.last_conv(x)
        # x = torch.tanh(x)
        return x
