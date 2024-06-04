import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor


def create_model(name, num_classes=10):
    model_dic = {
        'vgg13': vgg13_modified,
        'vgg16': vgg16_modified,
        'resnet18': resnet18_modified,
        'resnet34': resnet34_modified,
    }


    def handler(func, *args, num_classes=10):
        return func(*args, num_classes=num_classes)

    net = handler(model_dic[name], num_classes=num_classes)

    return net




def vgg13_modified(num_classes=10):
    net = models.vgg13_bn()
    net.avgpool = nn.AdaptiveAvgPool2d((1,1))
    net.classifier = nn.Linear(512, num_classes)
    return net

def vgg16_modified(num_classes=10):
    net = models.vgg16_bn()
    net.avgpool = nn.AdaptiveAvgPool2d((1,1))
    net.classifier = nn.Linear(512, num_classes)
    return net


def resnet18_modified(num_classes=10):
    net = models.resnet18()
    net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    net.maxpool = nn.Identity()
    net.fc = nn.Linear(512, num_classes)
    return net

def resnet34_modified(num_classes=10):
    net = models.resnet34()
    net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    net.maxpool = torch.nn.Identity()
    net.fc = nn.Linear(512, num_classes)
    return net


def partialprop(model, net, x, inlayer, layers):
    if model=='vgg13':
        logits, features = vgg13_partial_prop(net, x, inlayer, layers)
    elif model=='vgg16':
        logits, features = vgg16_partial_prop(net, x, inlayer, layers)
    elif model=='resnet18':
        logits, features = resnet18_partial_prop(net, x, inlayer, layers)
    elif model=='resnet34':
        logits, features = resnet34_partial_prop(net, x, inlayer, layers)

    return logits, features


def vgg13_partial_prop(net, x, inlayer, layers):
    features = []
    out = x

    if inlayer <= 0:
        if 0 in layers: features.append(out.clone().detach())
        out = net.module.features[1](net.module.features[0](out))
        if '0_bn' in layers: features.append(out.clone().detach())
        out = net.module.features[2](out)

    if inlayer <= 1:
        if 1 in layers: features.append(out.clone().detach())
        out = net.module.features[4](net.module.features[3](out))
        if '1_bn' in layers: features.append(out.clone().detach())
        out = net.module.features[6](net.module.features[5](out))


    if inlayer <= 2:
        if 2 in layers: features.append(out.clone().detach())
        out = net.module.features[8](net.module.features[7](out))
        if '2_bn' in layers: features.append(out.clone().detach())
        out = net.module.features[9](out)

    if inlayer <= 3:
        if 3 in layers: features.append(out.clone().detach())
        out = net.module.features[11](net.module.features[10](out))
        if '3_bn' in layers: features.append(out.clone().detach())
        out = net.module.features[13](net.module.features[12](out))


    if inlayer <= 4:
        if 4 in layers: features.append(out.clone().detach())
        out = net.module.features[15](net.module.features[14](out))
        if '4_bn' in layers: features.append(out.clone().detach())
        out = net.module.features[16](out)

    if inlayer <= 5:
        if 5 in layers: features.append(out.clone().detach())
        out = net.module.features[18](net.module.features[17](out))
        if '5_bn' in layers: features.append(out.clone().detach())
        out = net.module.features[20](net.module.features[19](out))


    if inlayer <= 6:
        if 6 in layers: features.append(out.clone().detach())
        out = net.module.features[22](net.module.features[21](out))
        if '6_bn' in layers: features.append(out.clone().detach())
        out = net.module.features[23](out)

    if inlayer <= 7:
        if 7 in layers: features.append(out.clone().detach())
        out = net.module.features[25](net.module.features[24](out))
        if '7_bn' in layers: features.append(out.clone().detach())
        out = net.module.features[27](net.module.features[26](out))


    if inlayer <= 8:
        if 8 in layers: features.append(out.clone().detach())
        out = net.module.features[29](net.module.features[28](out))
        if '8_bn' in layers: features.append(out.clone().detach())
        out = net.module.features[30](out)

    if inlayer <= 9:
        if 9 in layers: features.append(out.clone().detach())
        out = net.module.features[32](net.module.features[31](out))
        if '9_bn' in layers: features.append(out.clone().detach())
        out = net.module.features[34](net.module.features[33](out))

        out = net.module.avgpool(out)
        out = torch.flatten(out, 1)


    if inlayer <= 10:
        if 10 in layers: features.append(out.clone().detach())
        out = net.module.classifier(out)


    return out, features



def vgg16_partial_prop(net, x, inlayer, layers):
    features = []
    out = x

    if inlayer <= 0:
        if 0 in layers: features.append(out.clone().detach())
        out = net.module.features[1](net.module.features[0](out))
        if '0_bn' in layers: features.append(out.clone().detach())
        out = net.module.features[2](out)

    if inlayer <= 1:
        if 1 in layers: features.append(out.clone().detach())
        out = net.module.features[4](net.module.features[3](out))
        if '1_bn' in layers: features.append(out.clone().detach())
        out = net.module.features[6](net.module.features[5](out))


    if inlayer <= 2:
        if 2 in layers: features.append(out.clone().detach())
        out = net.module.features[8](net.module.features[7](out))
        if '2_bn' in layers: features.append(out.clone().detach())
        out = net.module.features[9](out)

    if inlayer <= 3:
        if 3 in layers: features.append(out.clone().detach())
        out = net.module.features[11](net.module.features[10](out))
        if '3_bn' in layers: features.append(out.clone().detach())
        out = net.module.features[13](net.module.features[12](out))


    if inlayer <= 4:
        if 4 in layers: features.append(out.clone().detach())
        out = net.module.features[15](net.module.features[14](out))
        if '4_bn' in layers: features.append(out.clone().detach())
        out = net.module.features[16](out)

    if inlayer <= 5:
        if 5 in layers: features.append(out.clone().detach())
        out = net.module.features[18](net.module.features[17](out))
        if '5_bn' in layers: features.append(out.clone().detach())
        out = net.module.features[19](out)

    if inlayer <= 6:
        if 6 in layers: features.append(out.clone().detach())
        out = net.module.features[21](net.module.features[20](out))
        if '6_bn' in layers: features.append(out.clone().detach())
        out = net.module.features[23](net.module.features[22](out))


    if inlayer <= 7:
        if 7 in layers: features.append(out.clone().detach())
        out = net.module.features[25](net.module.features[24](out))
        if '7_bn' in layers: features.append(out.clone().detach())
        out = net.module.features[26](out)

    if inlayer <= 8:
        if 8 in layers: features.append(out.clone().detach())
        out = net.module.features[28](net.module.features[27](out))
        if '8_bn' in layers: features.append(out.clone().detach())
        out = net.module.features[29](out)

    if inlayer <= 9:
        if 9 in layers: features.append(out.clone().detach())
        out = net.module.features[31](net.module.features[30](out))
        if '9_bn' in layers: features.append(out.clone().detach())
        out = net.module.features[33](net.module.features[32](out))


    if inlayer <= 10:
        if 10 in layers: features.append(out.clone().detach())
        out = net.module.features[35](net.module.features[34](out))
        if '10_bn' in layers: features.append(out.clone().detach())
        out = net.module.features[36](out)

    if inlayer <= 11:
        if 11 in layers: features.append(out.clone().detach())
        out = net.module.features[38](net.module.features[37](out))
        if '11_bn' in layers: features.append(out.clone().detach())
        out = net.module.features[39](out)

    if inlayer <= 12:
        if 12 in layers: features.append(out.clone().detach())
        out = net.module.features[41](net.module.features[40](out))
        if '12_bn' in layers: features.append(out.clone().detach())
        out = net.module.features[43](net.module.features[42](out))

        out = net.module.avgpool(out)
        out = torch.flatten(out, 1)


    if inlayer <= 13:
        if 13 in layers: features.append(out.clone().detach())
        out = net.module.classifier(out)


    return out, features


                              

def resnet18_partial_prop(net, x, inlayer, layers):
    features = []
    out = x

    if inlayer <= 0:
        if 0 in layers: features.append(out.clone().detach())
        out = net.module.bn1(net.module.conv1(out))
        if '0_bn' in layers: features.append(out.clone().detach())
        out = net.module.relu(out)


    if inlayer <= 1:
        out0 = out
        if 1 in layers: features.append(out.clone().detach())
        out = net.module.layer1[0].bn1(net.module.layer1[0].conv1(out))
        if '1_bn' in layers: features.append(out.clone().detach())
        out = net.module.layer1[0].relu(out)
        if 2 in layers: features.append(out.clone().detach())
        out = net.module.layer1[0].bn2(net.module.layer1[0].conv2(out))
        if '2_bn' in layers: features.append(out.clone().detach())
        out = net.module.layer1[0].relu(out+out0)

    if inlayer <= 3:
        out0 = out
        if 3 in layers: features.append(out.clone().detach())
        out = net.module.layer1[1].bn1(net.module.layer1[1].conv1(out))
        if '3_bn' in layers: features.append(out.clone().detach())
        out = net.module.layer1[1].relu(out)
        if 4 in layers: features.append(out.clone().detach())
        out = net.module.layer1[1].bn2(net.module.layer1[1].conv2(out))
        if '4_bn' in layers: features.append(out.clone().detach())
        out = net.module.layer1[1].relu(out+out0)


    if inlayer <= 5:
        out0 = net.module.layer2[0].downsample(out)
        if 5 in layers: features.append(out.clone().detach())
        out = net.module.layer2[0].bn1(net.module.layer2[0].conv1(out))
        if '5_bn' in layers: features.append(out.clone().detach())
        out = net.module.layer2[0].relu(out)
        if 6 in layers: features.append(out.clone().detach())
        out = net.module.layer2[0].bn2(net.module.layer2[0].conv2(out))
        if '6_bn' in layers: features.append(out.clone().detach())
        out = net.module.layer2[0].relu(out+out0)

    if inlayer <= 7:
        out0 = out
        if 7 in layers: features.append(out.clone().detach())
        out = net.module.layer2[1].bn1(net.module.layer2[1].conv1(out))
        if '7_bn' in layers: features.append(out.clone().detach())
        out = net.module.layer2[1].relu(out)
        if 8 in layers: features.append(out.clone().detach())
        out = net.module.layer2[1].bn2(net.module.layer2[1].conv2(out))
        if '8_bn' in layers: features.append(out.clone().detach())
        out = net.module.layer2[1].relu(out+out0)


    if inlayer <= 9:
        out0 = net.module.layer3[0].downsample(out)
        if 9 in layers: features.append(out.clone().detach())
        out = net.module.layer3[0].bn1(net.module.layer3[0].conv1(out))
        if '9_bn' in layers: features.append(out.clone().detach())
        out = net.module.layer3[0].relu(out)
        if 10 in layers: features.append(out.clone().detach())
        out = net.module.layer3[0].bn2(net.module.layer3[0].conv2(out))
        if '10_bn' in layers: features.append(out.clone().detach())
        out = net.module.layer3[0].relu(out+out0)

    if inlayer <= 11:
        out0 = out
        if 11 in layers: features.append(out.clone().detach())
        out = net.module.layer3[1].bn1(net.module.layer3[1].conv1(out))
        if '11_bn' in layers: features.append(out.clone().detach())
        out = net.module.layer3[1].relu(out)
        if 12 in layers: features.append(out.clone().detach())
        out = net.module.layer3[1].bn2(net.module.layer3[1].conv2(out))
        if '12_bn' in layers: features.append(out.clone().detach())
        out = net.module.layer3[1].relu(out+out0)


    if inlayer <= 13:
        out0 = net.module.layer4[0].downsample(out)
        if 13 in layers: features.append(out.clone().detach())
        out = net.module.layer4[0].bn1(net.module.layer4[0].conv1(out))
        if '13_bn' in layers: features.append(out.clone().detach())
        out = net.module.layer4[0].relu(out)
        if 14 in layers: features.append(out.clone().detach())
        out = net.module.layer4[0].bn2(net.module.layer4[0].conv2(out))
        if '14_bn' in layers: features.append(out.clone().detach())
        out = net.module.layer4[0].relu(out+out0)

    if inlayer <= 15:
        out0 = out
        if 15 in layers: features.append(out.clone().detach())
        out = net.module.layer4[1].bn1(net.module.layer4[1].conv1(out))
        if '15_bn' in layers: features.append(out.clone().detach())
        out = net.module.layer4[1].relu(out)
        if 16 in layers: features.append(out.clone().detach())
        out = net.module.layer4[1].bn2(net.module.layer4[1].conv2(out))
        if '16_bn' in layers: features.append(out.clone().detach())
        out = net.module.layer4[1].relu(out+out0)

        out = net.module.avgpool(out)
        out = torch.flatten(out, 1)


    if inlayer <= 17:
        if 17 in layers: features.append(out.clone().detach())
        out = net.module.fc(out)


    return out, features



def resnet34_partial_prop(net, x, inlayer, layers):
    features = []
    out = x

    if inlayer <= 0:
        if 0 in layers: features.append(x.clone().detach())
        out = net.module.bn1(net.module.conv1(x))
        if '0_bn' in layers: features.append(out.clone().detach())
        out = net.module.relu(out)



    if inlayer <= 1:
        out0 = out
        if 1 in layers: features.append(out.clone().detach())
        out = net.module.layer1[0].bn1(net.module.layer1[0].conv1(out))
        if '1_bn' in layers: features.append(out.clone().detach())
        out = net.module.layer1[0].relu(out)
        if 2 in layers: features.append(out.clone().detach())
        out = net.module.layer1[0].bn2(net.module.layer1[0].conv2(out))
        if '2_bn' in layers: features.append(out.clone().detach())
        out = net.module.layer1[0].relu(out+out0)

    if inlayer <= 3:
        out0 = out
        if 3 in layers: features.append(out.clone().detach())
        out = net.module.layer1[1].bn1(net.module.layer1[1].conv1(out))
        if '3_bn' in layers: features.append(out.clone().detach())
        out = net.module.layer1[1].relu(out)
        if 4 in layers: features.append(out.clone().detach())
        out = net.module.layer1[1].bn2(net.module.layer1[1].conv2(out))
        if '4_bn' in layers: features.append(out.clone().detach())
        out = net.module.layer1[1].relu(out+out0)

    if inlayer <= 5:
        out0 = out
        if 5 in layers: features.append(out.clone().detach())
        out = net.module.layer1[2].bn1(net.module.layer1[2].conv1(out))
        if '5_bn' in layers: features.append(out.clone().detach())
        out = net.module.layer1[2].relu(out)
        if 6 in layers: features.append(out.clone().detach())
        out = net.module.layer1[2].bn2(net.module.layer1[2].conv2(out))
        if '6_bn' in layers: features.append(out.clone().detach())
        out = net.module.layer1[2].relu(out+out0)


    if inlayer <= 7:
        out0 = net.module.layer2[0].downsample(out)
        if 7 in layers: features.append(out.clone().detach())
        out = net.module.layer2[0].bn1(net.module.layer2[0].conv1(out))
        if '7_bn' in layers: features.append(out.clone().detach())
        out = net.module.layer2[0].relu(out)
        if 8 in layers: features.append(out.clone().detach())
        out = net.module.layer2[0].bn2(net.module.layer2[0].conv2(out))
        if '8_bn' in layers: features.append(out.clone().detach())
        out = net.module.layer2[0].relu(out+out0)

    if inlayer <= 9:
        out0 = out
        if 9 in layers: features.append(out.clone().detach())
        out = net.module.layer2[1].bn1(net.module.layer2[1].conv1(out))
        if '9_bn' in layers: features.append(out.clone().detach())
        out = net.module.layer2[1].relu(out)
        if 10 in layers: features.append(out.clone().detach())
        out = net.module.layer2[1].bn2(net.module.layer2[1].conv2(out))
        if '10_bn' in layers: features.append(out.clone().detach())
        out = net.module.layer2[1].relu(out+out0)

    if inlayer <= 11:
        out0 = out
        if 11 in layers: features.append(out.clone().detach())
        out = net.module.layer2[2].bn1(net.module.layer2[2].conv1(out))
        if '11_bn' in layers: features.append(out.clone().detach())
        out = net.module.layer2[2].relu(out)
        if 12 in layers: features.append(out.clone().detach())
        out = net.module.layer2[2].bn2(net.module.layer2[2].conv2(out))
        if '12_bn' in layers: features.append(out.clone().detach())
        out = net.module.layer2[2].relu(out+out0)

    if inlayer <= 13:
        out0 = out
        if 13 in layers: features.append(out.clone().detach())
        out = net.module.layer2[3].bn1(net.module.layer2[3].conv1(out))
        if '13_bn' in layers: features.append(out.clone().detach())
        out = net.module.layer2[3].relu(out)
        if 14 in layers: features.append(out.clone().detach())
        out = net.module.layer2[3].bn2(net.module.layer2[3].conv2(out))
        if '14_bn' in layers: features.append(out.clone().detach())
        out = net.module.layer2[3].relu(out+out0)


    if inlayer <= 15:
        out0 = net.module.layer3[0].downsample(out)
        if 15 in layers: features.append(out.clone().detach())
        out = net.module.layer3[0].bn1(net.module.layer3[0].conv1(out))
        if '15_bn' in layers: features.append(out.clone().detach())
        out = net.module.layer3[0].relu(out)
        if 16 in layers: features.append(out.clone().detach())
        out = net.module.layer3[0].bn2(net.module.layer3[0].conv2(out))
        if '16_bn' in layers: features.append(out.clone().detach())
        out = net.module.layer3[0].relu(out+out0)

    if inlayer <= 17:
        out0 = out
        if 17 in layers: features.append(out.clone().detach())
        out = net.module.layer3[1].bn1(net.module.layer3[1].conv1(out))
        if '17_bn' in layers: features.append(out.clone().detach())
        out = net.module.layer3[1].relu(out)
        if 18 in layers: features.append(out.clone().detach())
        out = net.module.layer3[1].bn2(net.module.layer3[1].conv2(out))
        if '18_bn' in layers: features.append(out.clone().detach())
        out = net.module.layer3[1].relu(out+out0)

    if inlayer <= 19:
        out0 = out
        if 19 in layers: features.append(out.clone().detach())
        out = net.module.layer3[2].bn1(net.module.layer3[2].conv1(out))
        if '19_bn' in layers: features.append(out.clone().detach())
        out = net.module.layer3[2].relu(out)
        if 20 in layers: features.append(out.clone().detach())
        out = net.module.layer3[2].bn2(net.module.layer3[2].conv2(out))
        if '20_bn' in layers: features.append(out.clone().detach())
        out = net.module.layer3[2].relu(out+out0)

    if inlayer <= 21:
        out0 = out
        if 21 in layers: features.append(out.clone().detach())
        out = net.module.layer3[3].bn1(net.module.layer3[3].conv1(out))
        if '21_bn' in layers: features.append(out.clone().detach())
        out = net.module.layer3[3].relu(out)
        if 22 in layers: features.append(out.clone().detach())
        out = net.module.layer3[3].bn2(net.module.layer3[3].conv2(out))
        if '22_bn' in layers: features.append(out.clone().detach())
        out = net.module.layer3[3].relu(out+out0)

    if inlayer <= 23:
        out0 = out
        if 23 in layers: features.append(out.clone().detach())
        out = net.module.layer3[4].bn1(net.module.layer3[4].conv1(out))
        if '23_bn' in layers: features.append(out.clone().detach())
        out = net.module.layer3[4].relu(out)
        if 24 in layers: features.append(out.clone().detach())
        out = net.module.layer3[4].bn2(net.module.layer3[4].conv2(out))
        if '24_bn' in layers: features.append(out.clone().detach())
        out = net.module.layer3[4].relu(out+out0)

    if inlayer <= 25:
        out0 = out
        if 25 in layers: features.append(out.clone().detach())
        out = net.module.layer3[5].bn1(net.module.layer3[5].conv1(out))
        if '25_bn' in layers: features.append(out.clone().detach())
        out = net.module.layer3[5].relu(out)
        if 26 in layers: features.append(out.clone().detach())
        out = net.module.layer3[5].bn2(net.module.layer3[5].conv2(out))
        if '26_bn' in layers: features.append(out.clone().detach())
        out = net.module.layer3[5].relu(out+out0)


    if inlayer <= 27:
        out0 = net.module.layer4[0].downsample(out)
        if 27 in layers: features.append(out.clone().detach())
        out = net.module.layer4[0].bn1(net.module.layer4[0].conv1(out))
        if '27_bn' in layers: features.append(out.clone().detach())
        out = net.module.layer4[0].relu(out)
        if 28 in layers: features.append(out.clone().detach())
        out = net.module.layer4[0].bn2(net.module.layer4[0].conv2(out))
        if '28_bn' in layers: features.append(out.clone().detach())
        out = net.module.layer4[0].relu(out+out0)

    if inlayer <= 29:
        out0 = out
        if 29 in layers: features.append(out.clone().detach())
        out = net.module.layer4[1].bn1(net.module.layer4[1].conv1(out))
        if '29_bn' in layers: features.append(out.clone().detach())
        out = net.module.layer4[1].relu(out)
        if 30 in layers: features.append(out.clone().detach())
        out = net.module.layer4[1].bn2(net.module.layer4[1].conv2(out))
        if '30_bn' in layers: features.append(out.clone().detach())
        out = net.module.layer4[1].relu(out+out0)

    if inlayer <= 31:
        out0 = out
        if 31 in layers: features.append(out.clone().detach())
        out = net.module.layer4[2].bn1(net.module.layer4[2].conv1(out))
        if '31_bn' in layers: features.append(out.clone().detach())
        out = net.module.layer4[2].relu(out)
        if 32 in layers: features.append(out.clone().detach())
        out = net.module.layer4[2].bn2(net.module.layer4[2].conv2(out))
        if '32_bn' in layers: features.append(out.clone().detach())
        out = net.module.layer4[2].relu(out+out0)

        out = net.module.avgpool(out)
        out = torch.flatten(out, 1)


    if inlayer <= 33:
        if 33 in layers: features.append(out.clone().detach())
        out = net.module.fc(out)


    return out, features
