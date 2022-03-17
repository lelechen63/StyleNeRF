import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

class LinearWN( torch.nn.Linear ):
    def __init__( self, in_features, out_features, bias = True ):
        super().__init__( in_features, out_features, bias )
        self.g = torch.nn.Parameter( torch.ones( out_features ) )
        self.is_fused = False

    def forward( self, input ):
        if self.is_fused:
            return F.linear( input, self.weight, self.bias )
        else:
            wnorm = torch.sqrt( torch.sum( self.weight ** 2 ) )
            return F.linear( input, self.weight * self.g[ :, None ] / wnorm, self.bias )

    def fuse( self ):
        wnorm = torch.sqrt( (self.weight ** 2).sum() )
        self.weight.data = self.weight.data * self.g.data[ :, None ] / wnorm
        del self._parameters[ "g" ]
        self.is_fused = True

    def unfuse( self ):
        self.g = torch.nn.Parameter( torch.ones( out_features ) )
        self.is_fused = False

class Conv2dWNUB( torch.nn.Conv2d ):
    def __init__( self, in_channels, out_channels, height, width, kernel_size,
                  stride = 1, padding = 0, dilation = 1, groups = 1, bias = False ):
        super().__init__( in_channels, out_channels, kernel_size, stride,
                          padding, dilation, groups, False )
        self.g = torch.nn.Parameter( torch.ones( out_channels // groups ) )
        self.bias = torch.nn.Parameter( torch.zeros( out_channels, height, width ) )
        self.is_fused = False

    def forward( self, x ):
        if self.is_fused:
            return F.conv2d( x, self.weight,
                               bias = None, stride = self.stride, padding = self.padding,
                               dilation = self.dilation, groups = self.groups ) + self.bias[ None, ... ]
        else:
            wnorm = torch.sqrt( torch.sum( self.weight ** 2 ) )
            return F.conv2d( x, self.weight * self.g[ :, None, None, None ] / wnorm,
                               bias = None, stride = self.stride, padding = self.padding,
                               dilation = self.dilation, groups = self.groups ) + self.bias[ None, ... ]

    def fuse( self ):
        wnorm = torch.sqrt( (self.weight ** 2).sum() )
        self.weight.data = self.weight.data * self.g.data[ :, None, None, None ] / wnorm
        del self._parameters[ "g" ]
        self.is_fused = True

    def unfuse( self ):
        self.g = torch.nn.Parameter( torch.ones( self.out_channels // self.groups ) )
        self.is_fused = False


class Conv2dWN( torch.nn.Conv2d ):
    def __init__( self, in_channels, out_channels, kernel_size,
                  stride = 1, padding = 0, dilation = 1, groups = 1, bias = True ):
        super().__init__( in_channels, out_channels, kernel_size, stride,
                          padding, dilation, groups, True )
        self.g = torch.nn.Parameter( torch.ones( out_channels // groups ) )
        self.is_fused = False

    def forward( self, x ):
        if self.is_fused:
            return F.conv2d( x, self.weight, bias = self.bias,
                               stride = self.stride, padding = self.padding,
                               dilation = self.dilation, groups = self.groups )
        else:
            wnorm = torch.sqrt( torch.sum( self.weight ** 2 ) )
            return F.conv2d( x, self.weight * self.g[ :, None, None, None ] / wnorm,
                               bias = self.bias, stride = self.stride, padding = self.padding,
                               dilation = self.dilation, groups = self.groups )

    def fuse( self ):
        wnorm = torch.sqrt( (self.weight ** 2).sum() )
        self.weight.data = self.weight.data * self.g.data[ :, None, None, None ] / wnorm
        del self._parameters[ "g" ]
        self.is_fused = True

    def unfuse( self ):
        self.g = torch.nn.Parameter( torch.ones( self.out_channels // self.groups ) )
        self.is_fused = False

class ConvTranspose2dWNUB( torch.nn.ConvTranspose2d ):
    def __init__( self, in_channels, out_channels, height, width, kernel_size,
                  stride = 1, padding = 0, dilation = 1, groups = 1, bias = False ):
        super().__init__( in_channels, out_channels, kernel_size, stride,
                          padding, dilation, groups, False )
        self.g = torch.nn.Parameter( torch.ones( out_channels // groups ) )
        self.bias = torch.nn.Parameter( torch.zeros( out_channels, height, width ) )
        self.is_fused = False

    def forward( self, x ):
        if self.is_fused:
            return F.conv_transpose2d( x, self.weight,
                                         bias = None, stride = self.stride, padding = self.padding,
                                         dilation = self.dilation, groups = self.groups ) + self.bias[ None, ... ]
        else:
            wnorm = torch.sqrt( torch.sum( self.weight ** 2 ) )
            return F.conv_transpose2d( x, self.weight * self.g[ None, :, None, None ] / wnorm,
                                         bias = None, stride = self.stride, padding = self.padding,
                                         dilation = self.dilation, groups = self.groups ) + self.bias[ None, ... ]

    def fuse( self ):
        wnorm = torch.sqrt( (self.weight ** 2).sum() )
        self.weight.data = self.weight.data * self.g.data[ None, :, None, None ] / wnorm
        del self._parameters[ "g" ]
        self.is_fused = True

    def unfuse( self ):
        self.g = torch.nn.Parameter( torch.ones( self.out_channels // self.groups ) )
        self.is_fused = False


class ConvTranspose2dWN( torch.nn.ConvTranspose2d ):
    def __init__( self, in_channels, out_channels, kernel_size,
                  stride = 1, padding = 0, dilation = 1, groups = 1, bias = True ):
        super().__init__( in_channels, out_channels, kernel_size, stride,
                          padding, dilation, groups, True )
        self.g = torch.nn.Parameter( torch.ones( out_channels // groups ) )
        self.is_fused = False

    def forward( self, x ):
        if self.is_fused:
            return F.conv_transpose2d( x, self.weight, bias = self.bias,
                                         stride = self.stride, padding = self.padding,
                                         dilation = self.dilation, groups = self.groups )
        else:
            wnorm = torch.sqrt( torch.sum( self.weight ** 2 ) )
            return F.conv_transpose2d( x, self.weight * self.g[ None, :, None, None ] / wnorm,
                                         bias = self.bias, stride = self.stride, padding = self.padding,
                                         dilation = self.dilation, groups = self.groups )

    def fuse( self ):
        wnorm = torch.sqrt( (self.weight ** 2).sum() )
        self.weight.data = self.weight.data * self.g.data[ None, :, None, None ] / wnorm
        del self._parameters[ "g" ]
        self.is_fused = True

    def unfuse( self ):
        self.g = torch.nn.Parameter( torch.ones( self.out_channels // self.groups ) )
        self.is_fused = False


def glorot( m, alpha ):
    gain = np.sqrt( 2. / (1. + alpha ** 2) )

    if isinstance( m, torch.nn.Conv2d ):
        ksize = m.kernel_size[ 0 ] * m.kernel_size[ 1 ]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * np.sqrt( 2.0 / ((n1 + n2) * ksize) )
    elif isinstance( m, torch.nn.ConvTranspose2d ):
        ksize = m.kernel_size[ 0 ] * m.kernel_size[ 1 ] // 4
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * np.sqrt( 2.0 / ((n1 + n2) * ksize) )
    elif isinstance( m, torch.nn.ConvTranspose3d ):
        ksize = m.kernel_size[ 0 ] * m.kernel_size[ 1 ] * m.kernel_size[ 2 ] // 8
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * matorch.sqrt( 2.0 / ((n1 + n2) * ksize) )
    elif isinstance( m, torch.nn.Linear ):
        n1 = m.in_features
        n2 = m.out_features

        std = gain * np.sqrt( 2.0 / (n1 + n2) )
    else:
        return

    # m.weight.data.normal_(0, std)
    m.weight.data.uniform_( -std * np.sqrt( 3.0 ), std * np.sqrt( 3.0 ) )
    m.bias.data.zero_()

    if isinstance( m, torch.nn.ConvTranspose2d ):
        # hardcoded for stride=2 for now
        m.weight.data[ :, :, 0::2, 1::2 ] = m.weight.data[ :, :, 0::2, 0::2 ]
        m.weight.data[ :, :, 1::2, 0::2 ] = m.weight.data[ :, :, 0::2, 0::2 ]
        m.weight.data[ :, :, 1::2, 1::2 ] = m.weight.data[ :, :, 0::2, 0::2 ]

    if isinstance( m, Conv2dWNUB ) or isinstance( m, Conv2dWN ) or isinstance( m, ConvTranspose2dWN ) or \
            isinstance( m, ConvTranspose2dWNUB ) or isinstance( m, LinearWN ):
        norm = np.sqrt( torch.sum( m.weight.data[ : ] ** 2 ) )
        m.g.data[ : ] = norm


def fuse( m ):
    if hasattr( m, "fuse" ) and isinstance( m, torch.nn.Module ):
        m.fuse()


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out
        
class ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm, activation, pad_type):
        super(ResBlocks, self).__init__()
        self.model = []
        for i in range(num_blocks):
            self.model += [ResBlock(dim,
                                    norm=norm,
                                    activation=activation,
                                    pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


class ResBlock(nn.Module):
    def __init__(self, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlock, self).__init__()
        model = []
        model += [Conv2dBlock(dim, dim, 3, 1, 1,
                              norm=norm,
                              activation=activation,
                              pad_type=pad_type)]
        model += [Conv2dBlock(dim, dim, 3, 1, 1,
                              norm=norm,
                              activation='none',
                              pad_type=pad_type)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out


class ActFirstResBlock(nn.Module):
    def __init__(self, fin, fout, fhid=None,
                 activation='lrelu', norm='none'):
        super().__init__()
        self.learned_shortcut = (fin != fout)
        self.fin = fin
        self.fout = fout
        self.fhid = min(fin, fout) if fhid is None else fhid
        self.conv_0 = Conv2dBlock(self.fin, self.fhid, 3, 1,
                                  padding=1, pad_type='reflect', norm=norm,
                                  activation=activation, activation_first=True)
        self.conv_1 = Conv2dBlock(self.fhid, self.fout, 3, 1,
                                  padding=1, pad_type='reflect', norm=norm,
                                  activation=activation, activation_first=True)
        if self.learned_shortcut:
            self.conv_s = Conv2dBlock(self.fin, self.fout, 1, 1,
                                      activation='none', use_bias=False)

    def forward(self, x):
        x_s = self.conv_s(x) if self.learned_shortcut else x
        dx = self.conv_0(x)
        dx = self.conv_1(dx)
        out = x_s + dx
        return out


class LinearBlock(nn.Module):
    def __init__(self, in_dim, out_dim, norm='none', activation='relu'):
        super(LinearBlock, self).__init__()
        use_bias = True
        self.fc = nn.Linear(in_dim, out_dim, bias=use_bias)

        # initialize normalization
        norm_dim = out_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out


class Conv2dBlock(nn.Module):
    def __init__(self, in_dim, out_dim, ks, st, padding=0,
                 norm='none', activation='relu', pad_type='zero',
                 use_bias=True, activation_first=False):
        super(Conv2dBlock, self).__init__()
        self.use_bias = use_bias
        self.activation_first = activation_first
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = out_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        self.conv = nn.Conv2d(in_dim, out_dim, ks, st, bias=self.use_bias)

    def forward(self, x):
        if self.activation_first:
            if self.activation:
                x = self.activation(x)
            x = self.conv(self.pad(x))
            if self.norm:
                x = self.norm(x)
        else:
            x = self.conv(self.pad(x))
            if self.norm:
                x = self.norm(x)
            if self.activation:
                x = self.activation(x)
        return x


class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight = None
        self.bias = None
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and \
               self.bias is not None, "Please assign AdaIN weight first"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])
        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)
        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'


class DownSampleConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel=4, strides=2, padding=1, activation=True, batchnorm=True):
        """
        Paper details:
        - C64-C128-C256-C512-C512-C512-C512-C512
        - All convolutions are 4×4 spatial filters applied with stride 2
        - Convolutions in the encoder downsample by a factor of 2
        """
        super().__init__()
        self.activation = activation
        self.batchnorm = batchnorm

        self.conv = nn.Conv2d(in_channels, out_channels, kernel, strides, padding)

        if batchnorm:
            self.bn = nn.BatchNorm2d(out_channels)

        if activation:
            self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv(x)
        if self.batchnorm:
            x = self.bn(x)
        if self.activation:
            x = self.act(x)
        return x

class UpSampleConv(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel=4,
        strides=2,
        padding=1,
        activation=True,
        batchnorm=True,
        dropout=False
    ):
        super().__init__()
        self.activation = activation
        self.batchnorm = batchnorm
        self.dropout = dropout

        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel, strides, padding)

        if batchnorm:
            self.bn = nn.BatchNorm2d(out_channels)

        if activation:
            self.act = nn.ReLU(True)

        if dropout:
            self.drop = nn.Dropout2d(0.5)

    def forward(self, x):
        x = self.deconv(x)
        if self.batchnorm:
            x = self.bn(x)

        if self.dropout:
            x = self.drop(x)
        return x