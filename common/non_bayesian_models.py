import torch
from torch import nn 
    
class LeNet(nn.Module):

    def __init__(self, outputs, inputs, layer_type='lrt', activation_type='softplus'):
        '''
        Base LeNet model that matches the architecture of BayesianLeNet with randomly 
        initialized weights
        '''
        super(LeNet, self).__init__()
        
        # initialization follows the BBBLeNet initialization, changing
        # BBBLinear and BBBConv2D layers to nn.Linear and nn.Conv2D
        
        if activation_type == 'softplus':
            self.act = nn.Softplus
        elif activation_type == 'relu':
            self.act = nn.ReLU

        self.conv1 = nn.Conv2d(inputs, 6, 5, padding=0, bias=True)
        self.act1 = self.act()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, 5, padding=0, bias=True)
        self.act2 = self.act()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(5 * 5 * 16, 120, bias=True)
        self.act3 = self.act()
        self.fc2 = nn.Linear(120, 84, bias=True)
        self.act4 = self.act()
        self.fc3 = nn.Linear(84, outputs, bias=True)


    def sample(self, bbbnet):
        '''
        Takes in a BBBLeNet instance and copies the structure into a LeNet model.
        Replaces the BBBLinear and BBBConv2D that uses sampling in their forward steps
        with regular nn.Linear and nn.Conv2d layers whose weights are initialized by 
        sampling the BBBLeNet model.
        '''    
        ### store activation function used by BNN, only relu and softplus  currently supported
        self.act1 = bbbnet.act()
        self.act2 = bbbnet.act()
        self.act3 = bbbnet.act()
        self.act4 = bbbnet.act()

        ### maxpool
        self.pool1 = nn.MaxPool2d(kernel_size=bbbnet.pool1.kernel_size, stride=bbbnet.pool1.stride)
        self.pool2 = nn.MaxPool2d(kernel_size=bbbnet.pool2.kernel_size, stride=bbbnet.pool2.stride)
        
        ### Create Convolution layers
        self.conv1 = nn.Conv2d(bbbnet.conv1.in_channels, bbbnet.conv1.out_channels, bbbnet.conv1.kernel_size,
                                stride=bbbnet.conv1.stride, padding=bbbnet.conv1.padding, dilation=bbbnet.conv1.dilation,
                                groups=bbbnet.conv1.groups)
        
        self.conv2 = nn.Conv2d(bbbnet.conv2.in_channels, bbbnet.conv2.out_channels, bbbnet.conv2.kernel_size,
                        stride=bbbnet.conv2.stride, padding=bbbnet.conv2.padding, dilation=bbbnet.conv2.dilation,
                        groups=bbbnet.conv2.groups)
        

        # Sample convolutional layers
        self.conv1.weight.data, self.conv1.bias.data = sample_conv2d(bbbnet.conv1)
        self.conv2.weight.data, self.conv2.bias.data = sample_conv2d(bbbnet.conv2)
        
        ### Create Linear Layers
        self.fc1 = nn.Linear(bbbnet.fc1.in_features, bbbnet.fc1.out_features, bbbnet.fc1.use_bias)
        self.fc2 = nn.Linear(bbbnet.fc2.in_features, bbbnet.fc2.out_features, bbbnet.fc2.use_bias)
        self.fc3 = nn.Linear(bbbnet.fc3.in_features, bbbnet.fc3.out_features, bbbnet.fc3.use_bias)

        # Sample linear layers
        self.fc1.weight.data, self.fc1.bias.data = sample_linear(bbbnet.fc1)            
        self.fc2.weight.data, self.fc2.bias.data = sample_linear(bbbnet.fc2)    
        self.fc3.weight.data, self.fc3.bias.data = sample_linear(bbbnet.fc3)    
        

    def forward(self, x):
        '''
        Forward method follow the order of BayesianLeNet
        '''
        x = self.conv1(x)
        x = self.act1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.act2(x)
        x = self.pool2(x)

        x = x.view(-1, 5 * 5 * 16)
        x = self.fc1(x)
        x = self.act3(x)
        x = self.fc2(x)
        x = self.act4(x)
        x = self.fc3(x)
        return x

class AlexNet(nn.Module):
    
    def __init__(self, outputs, inputs, layer_type='lrt', activation_type='softplus'):
        '''
        Base AlexNet model that matches the architecture of BayesianAlexNet with randomly 
        initialized weights
        '''
        super(AlexNet, self).__init__()
        
        # initialization follows the BBBAlexNet initialization, changing
        # BBBLinear and BBBConv2D layers to nn.Linear and nn.Conv2D
        
        if activation_type == 'softplus':
            self.act = nn.Softplus
        elif activation_type == 'relu':
            self.act = nn.ReLU
        else:
            raise ValueError("Only softplus or relu supported")

        self.conv1 = nn.Conv2d(inputs, 64, 11, stride=4, padding=5, bias=True)
        self.act1 = self.act()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(64, 192, 5, padding=2, bias=True)
        self.act2 = self.act()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(192,384,3, padding=1, bias=True)
        self.act3 = self.act()

        self.conv4 = nn.Conv2d(384,256,3, padding=1, bias=True)
        self.act4 = self.act()
        
        self.conv5 = nn.Conv2d(256, 128, 3, padding=1, bias=True)
        self.act5 = self.act()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # call x.view explicitly in forward
        self.classifier = nn.Linear(1 * 1 * 128, outputs, bias=True)

    def sample(self, bbbnet):
        ### store activation function used by BNN, only relu and softplus  currently supported
        self.act1 = bbbnet.act()
        self.act2 = bbbnet.act()
        self.act3 = bbbnet.act()
        self.act4 = bbbnet.act()

        ### maxpool
        self.pool1 = nn.MaxPool2d(kernel_size=bbbnet.pool1.kernel_size, stride=bbbnet.pool1.stride)
        self.pool2 = nn.MaxPool2d(kernel_size=bbbnet.pool2.kernel_size, stride=bbbnet.pool2.stride)
        self.pool3 = nn.MaxPool2d(kernel_size=bbbnet.pool3.kernel_size, stride=bbbnet.pool3.stride)

        ### Create Convolution layers
        self.conv1 = nn.Conv2d(bbbnet.conv1.in_channels, bbbnet.conv1.out_channels, bbbnet.conv1.kernel_size,
                                stride=bbbnet.conv1.stride, padding=bbbnet.conv1.padding, dilation=bbbnet.conv1.dilation,
                                groups=bbbnet.conv1.groups)
        
        self.conv2 = nn.Conv2d(bbbnet.conv2.in_channels, bbbnet.conv2.out_channels, bbbnet.conv2.kernel_size,
                        stride=bbbnet.conv2.stride, padding=bbbnet.conv2.padding, dilation=bbbnet.conv2.dilation,
                        groups=bbbnet.conv2.groups)

        self.conv3 = nn.Conv2d(bbbnet.conv3.in_channels, bbbnet.conv3.out_channels, bbbnet.conv3.kernel_size,
                                stride=bbbnet.conv3.stride, padding=bbbnet.conv3.padding, dilation=bbbnet.conv3.dilation,
                                groups=bbbnet.conv3.groups)
        
        self.conv4 = nn.Conv2d(bbbnet.conv4.in_channels, bbbnet.conv4.out_channels, bbbnet.conv4.kernel_size,
                        stride=bbbnet.conv4.stride, padding=bbbnet.conv4.padding, dilation=bbbnet.conv4.dilation,
                        groups=bbbnet.conv4.groups)

        self.conv5 = nn.Conv2d(bbbnet.conv5.in_channels, bbbnet.conv5.out_channels, bbbnet.conv5.kernel_size,
                        stride=bbbnet.conv5.stride, padding=bbbnet.conv5.padding, dilation=bbbnet.conv5.dilation,
                        groups=bbbnet.conv5.groups)

        # Sample convolutional layers
        self.conv1.weight.data, self.conv1.bias.data = sample_conv2d(bbbnet.conv1)
        self.conv2.weight.data, self.conv2.bias.data = sample_conv2d(bbbnet.conv2)
        self.conv3.weight.data, self.conv3.bias.data = sample_conv2d(bbbnet.conv3)
        self.conv4.weight.data, self.conv4.bias.data = sample_conv2d(bbbnet.conv4)
        self.conv5.weight.data, self.conv5.bias.data = sample_conv2d(bbbnet.conv5)

        ### Create Linear Layers
        self.classifier = nn.Linear(bbbnet.classifier.in_features, bbbnet.classifier.out_features, bbbnet.classifier.use_bias)
        self.classifier.weight.data, self.classifier.bias.data = sample_linear(bbbnet.classifier)            

    def forward(self, x):

        x = self.conv1(x)
        x = self.act1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.act2(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.act3(x)
        
        x = self.conv4(x)
        x = self.act4(x)

        x = self.conv5(x)
        x = self.act5(x)
        x = self.pool3(x)

        x = x.view(-1, 1 * 1 * 128)
        x = self.classifier(x)
        return x
    
# follows the procedure for sampling in the forward methods of BBBConv and 
# BBBLinear forward to create a fixed set of weights to use for the sampled model
def sample_conv2d(bbb_layer):
    # CUDA settings
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    conv_W_mu = bbb_layer.W_mu
    conv_W_rho = bbb_layer.W_rho
    conv_W_eps = torch.empty(conv_W_mu.size()).normal_(0,1).to(device)
    conv_W_sigma = torch.log1p(torch.exp(conv_W_rho))
    conv_weight = conv_W_mu + conv_W_eps * conv_W_sigma
    if bbb_layer.use_bias:
        conv_bias_mu = bbb_layer.bias_mu
        conv_bias_rho = bbb_layer.bias_rho
        conv_bias_eps = torch.empty(conv_bias_mu.size()).normal_(0,1).to(device)
        conv_bias_sigma = torch.log1p(torch.exp(conv_bias_rho))
        conv_bias = conv_bias_mu + conv_bias_eps * conv_bias_sigma
    else:
        conv_bias = None
    return conv_weight.data, conv_bias.data

def sample_linear(bbb_layer):
    # CUDA settings
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    fc_W_mu = bbb_layer.W_mu
    fc_W_rho = bbb_layer.W_rho
    fc_W_eps = torch.empty(fc_W_mu.size()).normal_(0,1).to(device)
    fc_W_sigma = torch.log1p(torch.exp(fc_W_rho))
    fc_weight = fc_W_mu + fc_W_eps * fc_W_sigma
    if bbb_layer.use_bias:
        fc_bias_mu = bbb_layer.bias_mu
        fc_bias_rho = bbb_layer.bias_rho
        fc_bias_eps = torch.empty(fc_bias_mu.size()).normal_(0,1).to(device)
        fc_bias_sigma = torch.log1p(torch.exp(fc_bias_rho))
        fc_bias = fc_bias_mu + fc_bias_eps * fc_bias_sigma
    else:
        fc_bias = None

    return fc_weight.data, fc_bias.data