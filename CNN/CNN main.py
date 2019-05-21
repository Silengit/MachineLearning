import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

# Hyper Parameters
EPOCH = 1  # train the training data n times
BATCH_SIZE = 50

LR = 0.001  # learning rate
DOWNLOAD_MNIST = False
TRANSFORM = transforms.ToTensor()

trainset = torchvision.datasets.MNIST(root='./data', 
                                      train=True,
                                      download=DOWNLOAD_MNIST, 
                                      transform=TRANSFORM)
trainloader = Data.DataLoader(dataset=trainset, 
                              batch_size=BATCH_SIZE,
                              shuffle=True, 
                              num_workers=2)

testset = torchvision.datasets.MNIST(root='./data', 
                                     train=False,
                                     download=DOWNLOAD_MNIST,
                                     transform=TRANSFORM)
with torch.no_grad():
    test_x = torch.unsqueeze(testset.data, dim=1).type(torch.FloatTensor)[:2000]/255.
test_y = testset.targets[:2000]

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
                nn.Conv2d(          # (1, 28, 28)
                        in_channels=1,
                        out_channels=16,
                        kernel_size=5,
                        stride=1,
                        padding=2 # padding = (kernel_size-1)/2
                        ),          # (1, 28, 28) -> (16, 28, 28)
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2)) # (16, 28, 28) -> (16, 14, 14)
        self.conv2 = nn.Sequential(
                nn.Conv2d(
                        in_channels=16,
                        out_channels=32,
                        kernel_size=5,
                        stride=1,
                        padding=2 # padding = (kernel_size-1)/2
                        ),          # (16, 14, 14) -> (32, 14, 14)
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2)) # (32, 14, 14) -> (32, 7, 7)
        self.out = nn.Linear(32 * 7 * 7, 10) # Linear(in_features, out_features, bias=True)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x) # (batch, 32, 7, 7)
        x = x.view(x.size(0), -1) # (batch, 32 * 7 * 7)
        # the size -1 is inferred from other dimensions
        output = self.out(x) # linear function of input
        return output

def select_optimizer(cnn, alg='ADAM', lr=LR):
    if alg == 'Momentum':
        return torch.optim.SGD(cnn.parameters(), lr=lr, momentum=0.9)
    elif alg == 'SGD':
        return torch.optim.SGD(cnn.parameters(), lr=lr)
    else:
        return torch.optim.Adam(cnn.parameters(), lr=lr)
        

def draw_ag(x, y, alg='Adam'):
    if alg == 'Momentum':
        plt.plot(x, y, color='g', label='Momentum')
    elif alg == 'SGD':
        plt.plot(x, y, color='r', label='SGD')
    else:
        plt.plot(x, y, color='b', label='Adam')
        
def draw_lr(x, y, lr):
    if lr == 1e-4:
        plt.plot(x, y, color='g', label='lr=0.0001')
    elif lr == 1e-3:
        plt.plot(x, y, color='r', label='lr=0.001')
    elif lr == 1e-2:
        plt.plot(x, y, color='b', label='1r=0.01')
    else:
        plt.plot(x, y, color='c', label='1r=0.1')

def draw_epoch(x, y):
    x = np.multiply(np.array(x),1/len(trainloader))
    plt.plot(x, y, color='b')

def draw_loss(x, y1, y2):
    plt.plot(x, y1, color='g', label='Train loss')
    plt.plot(x, y2, color='r', label='Test loss')

def process(cnn, optimizer, loss_func, alg='Adam', lr=LR):
    x_axis = []
    y_axis = []

    for epoch in range(EPOCH):
        for step, (x, y) in enumerate(trainloader):        
            output = cnn(x)
            loss = loss_func(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step() 
            
            if step % 50 == 0:
                test_output = cnn(test_x)
                pred_y = torch.max(test_output, 1)[1]   # most possible num
                accuracy = torch.sum(torch.eq(test_y, pred_y)).numpy() / test_y.size(0)
                test_loss = loss_func(test_output, test_y)
                print('Epoch: ', epoch, 
                      '| train loss: %.4f' % loss.data,
                      '| test loss: %.4f' % test_loss.data,
                      '| test accuracy: %.4f' % accuracy,)
                x_axis.append(step)
                y_axis.append(accuracy)
    
    draw_ag(x_axis, y_axis, alg)
#    draw_lr(x_axis, y_axis, lr)
#    draw_epoch(x_axis, y_axis)

     
def main():   
    cnn = CNN()
    loss_func = nn.CrossEntropyLoss()
    optimizer = select_optimizer(cnn)
    process(cnn, optimizer, loss_func)
    
    plt.xlabel('Step')
    plt.ylabel('Accuracy')
    plt.legend(loc="lower right")
    plt.show()
    
if __name__ == '__main__':
    main()