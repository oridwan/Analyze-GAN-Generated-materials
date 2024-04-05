import torch.nn as nn
import torch.nn.functional as F
import torch


class Generator(nn.Module):
    def __init__(self,opt):
        super(Generator, self).__init__()
        input_dim = opt.latent_dim + 8 + 8 + 12 +1
        self.input_dim = opt.input_dim # 512+28+1

        self.l1 = nn.Sequential(nn.Linear(input_dim, 128*28),nn.ReLU(True))
        self.map1 = nn.Sequential(nn.ConvTranspose2d(128,256,(1,3),stride = 1,padding=0),nn.BatchNorm2d(256,0.8),nn.ReLU(True)) #(28,3)
        self.map2 = nn.Sequential(nn.ConvTranspose2d(256,512,(1,1),stride = 1,padding=0),nn.BatchNorm2d(512,0.8),nn.ReLU(True)) #(28,3)
        self.map3 = nn.Sequential(nn.ConvTranspose2d(512,256,(1,1),stride = 1,padding=0),nn.BatchNorm2d(256,0.8),nn.ReLU(True)) #(28,3)
        self.map4 = nn.Sequential(nn.ConvTranspose2d(256,1,(1,1),stride=1,padding=0)) #(28,3)
        self.cellmap = nn.Sequential(nn.Linear(84,30),nn.BatchNorm1d(30),nn.ReLU(True),nn.Linear(30,6),nn.Sigmoid())

        self.sigmoid = nn.Sigmoid()

    def forward(self, noise,c1,c2,c3,c4):
        """
        noise: random seeds (32)
        c1: c_mg, (32, 8)
        c2: c_mn, (32, 8)
        c3: c_O, (32, 12)
        c4: number of atoms: (32) bounded in (0, 1) to represent number between 1 and 28
        """
        gen_input = torch.cat((noise,c4,c1,c2,c3), -1); print('gen_input', gen_input.shape) #(32, 541)
        h = self.l1(gen_input); print('l1 layer in generator', h.shape) # (32, 128*28)
        h = h.view(h.shape[0], 128, 28, 1); print('Convert it to pixel data', h.shape) # (32, 128, 28, 1)
        h = self.map1(h); print("Conv1 layer in generator", h.shape) # (32, 256, 28, 3) channels 1 -> 3
        h = self.map2(h); print("Conv2 layer in generator", h.shape) # (32, 512, 28, 3)
        h = self.map3(h); print("Conv3 layer in generator", h.shape) # (32, 256, 28, 3)
        h = self.map4(h); print("Conv4 layer in generator", h.shape) # (32, 1, 28, 3)

        h_flatten = h.view(h.shape[0],-1); print("flatten Conv4 layer in generator", h_flatten.shape) # (32, 84)
        pos = self.sigmoid(h); print("Apply activation to get pos", pos.shape) # (32, 1, 28, 3)
        cell = self.cellmap(h_flatten); print("Get Cell from position", cell.shape) # (32, 6)
        cell = cell.view(cell.shape[0],1,2,3); print('Final generator output', torch.cat((cell,pos),dim =2).shape)
        return torch.cat((cell,pos),dim =2) # (32, 1, 30, 3) i.e., 32 batch crystal structures



class Discriminator(nn.Module):
    def __init__(self,opt):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(nn.Conv2d(in_channels = 1, out_channels = 512, kernel_size = (1,3), stride = 1, padding = 0),
                                   nn.LeakyReLU(0.2, inplace=True),nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = (1,1), stride = 1, padding = 0),                                                                             nn.LeakyReLU(0.2,inplace=True),nn.Conv2d(in_channels = 512, out_channels = 256, kernel_size= (1,1), stride = 1, padding = 0),                                                                               nn.LeakyReLU(0.2,inplace=True))

        self.avgpool_mg = nn.AvgPool2d(kernel_size = (8,1))
        self.avgpool_mn = nn.AvgPool2d(kernel_size = (8,1))
        self.avgpool_o = nn.AvgPool2d(kernel_size = (12,1))

        self.feature_layer = nn.Sequential(nn.Linear(1280, 1000), nn.LeakyReLU(0.2, inplace =True), nn.Linear(1000,200),nn.LeakyReLU(0.2, inplace = True))
        self.output = nn.Sequential(nn.Linear(200,10))

    def forward(self, x):
        """
        receive the input image and check it is true or fake image

        x: (32, 1, 30, 3)
        """
        #print(x.shape)
        B = x.shape[0]
        output = self.model(x); print("model layer in Discriminator", output.shape) # (32, 1, 30, 3) -> (32, 256, 30, 1)
        output_c = output[:,:,:2,:]; print('extract cell', output_c.shape) # (32, 256, 2, 1)
        output_mg = output[:,:,2:10,:]                               # (32, 258, 8, 1)
        output_mn = output[:,:,10:18,:]                              # (32, 258, 8, 1)
        output_o = output[:,:,18:,:]                                 # (32, 258, 12, 1)

        output_mg = self.avgpool_mg(output_mg); print("2D average pooling on Mg", output_mg.shape) # (32, 256, 1, 1)
        output_mn = self.avgpool_mn(output_mn); print("2D average pooling on Mn", output_mn.shape) # (32, 256, 1, 1)
        output_o = self.avgpool_o(output_o);    print("2D average pooling on O", output_o.shape)   # (32, 256, 1, 1)

        output_all = torch.cat((output_c,output_mg,output_mn,output_o),dim=-2)
        output_all = output_all.view(B, -1); print("final output", output_all.shape) # (32, 1280)

        feature = self.feature_layer(output_all); print("Feature layer in Discriminator", feature.shape) # (32, 200)
        print("Output layer in Discriminator", self.output(feature).shape) # (32, 10)
        return feature, self.output(feature) # (32, 200) for features; (32, 10) for output


class QHead_(nn.Module):
    def __init__(self,opt):
        super(QHead_,self).__init__()
        self.model_mg = nn.Sequential(nn.Conv2d(in_channels = 1, out_channels = 512, kernel_size = (1,3), stride = 1, padding = 0),
                                      nn.BatchNorm2d(512,0.8),nn.LeakyReLU(0.2,inplace=True),
                                      nn.Conv2d(in_channels = 512, out_channels = 256, kernel_size = (1,1), stride = 1, padding = 0),
                                      nn.BatchNorm2d(256,0.8),nn.LeakyReLU(0.2,inplace=True),
                                      nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size= (1,1), stride = 1, padding = 0),
                                      nn.BatchNorm2d(256,0.8),nn.LeakyReLU(0.2,inplace=True),                                                                                                                                                     nn.Conv2d(in_channels = 256, out_channels = 2, kernel_size = (1,1), stride =1, padding =0))

        self.model_mn = nn.Sequential(nn.Conv2d(in_channels = 1, out_channels = 512, kernel_size = (1,3), stride = 1, padding = 0),
                                      nn.BatchNorm2d(512,0.8),nn.LeakyReLU(0.2,inplace=True),
                                      nn.Conv2d(in_channels = 512, out_channels = 256, kernel_size = (1,1), stride = 1, padding = 0),
                                      nn.BatchNorm2d(256,0.8),nn.LeakyReLU(0.2,inplace=True),
                                      nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size= (1,1), stride = 1, padding = 0),
                                      nn.BatchNorm2d(256,0.8),nn.LeakyReLU(0.2,inplace=True),
                                      nn.Conv2d(in_channels = 256, out_channels = 2, kernel_size = (1,1), stride =1, padding =0))

        self.model_o = nn.Sequential(nn.Conv2d(in_channels = 1, out_channels = 512, kernel_size = (1,3), stride = 1, padding = 0),
                                     nn.BatchNorm2d(512,0.8),nn.LeakyReLU(0.2,inplace=True),
                                     nn.Conv2d(in_channels = 512, out_channels = 256, kernel_size = (1,1), stride = 1, padding = 0),
                                     nn.BatchNorm2d(256,0.8),nn.LeakyReLU(0.2,inplace=True),
                                     nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size= (1,1), stride = 1, padding = 0),
                                     nn.BatchNorm2d(256,0.8),nn.LeakyReLU(0.2,inplace=True),
                                     nn.Conv2d(in_channels = 256, out_channels = 2, kernel_size = (1,1), stride =1, padding =0))

        self.model_cell = nn.Sequential(nn.Conv2d(in_channels = 1, out_channels = 64, kernel_size = (1,3), stride= 1, padding = 0),
                                        nn.BatchNorm2d(64,0.8),nn.LeakyReLU(0.2,inplace=True),
                                        nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (1,1), stride = 1, padding = 0),
                                        nn.BatchNorm2d(64,0.8),nn.LeakyReLU(0.2,inplace=True))

        self.softmax = nn.Softmax2d()
        self.label_mg_layer = nn.Sequential(nn.Linear(16,300),nn.BatchNorm1d(300,0.8),nn.LeakyReLU(0.2,inplace=True),
                                            nn.Linear(300,100),nn.BatchNorm1d(100,0.8),nn.LeakyReLU(0.2,inplace=True),nn.Linear(100,8),nn.Softmax())
        self.label_mn_layer = nn.Sequential(nn.Linear(16,300),nn.BatchNorm1d(300,0.8),nn.LeakyReLU(0.2,inplace=True),
                                            nn.Linear(300,100),nn.BatchNorm1d(100,0.8),nn.LeakyReLU(0.2,inplace=True),nn.Linear(100,8),nn.Softmax())
        self.label_o_layer = nn.Sequential(nn.Linear(24,300),nn.BatchNorm1d(300,0.8),nn.LeakyReLU(0.2,inplace=True),
                                           nn.Linear(300,100),nn.BatchNorm1d(100,0.8),nn.LeakyReLU(0.2,inplace=True),nn.Linear(100,12),nn.Softmax())
        self.label_c_layer = nn.Sequential(nn.Linear(128,100),nn.BatchNorm1d(100,0.8),nn.LeakyReLU(0.2,inplace=True),nn.Linear(100,50),nn.BatchNorm1d(50,0.8),nn.LeakyReLU(),nn.Linear(50,1),nn.Sigmoid())

    def forward(self, image):
        """
        Take the input image to check if the composition is correct

        image: (32, 1, 30, 3)
        """
        print("Input of classifier", image.shape)
        cell = image[:,:,:2,:]
        mg = image[:,:,2:10,:]
        mn = image[:,:,10:18,:]
        o = image[:,:,18:,:]

        cell_output = self.model_cell(cell); print("model_cell in classifier", cell_output.shape) # (32, 64, 2, 1)
        mg_output = self.model_mg(mg); print("model_mg in classifier", mg_output.shape) # (32, 2, 8, 1)
        mn_output = self.model_mn(mn); print("model_mn in classifier", mn_output.shape) # (32, 2, 8, 1)
        o_output = self.model_o(o); print("model_o in classifier", o_output.shape) # (32, 2, 12, 1)

        cell_output_f = torch.flatten(cell_output,start_dim=1)
        mg_output_f = torch.flatten(mg_output,start_dim=1)
        mn_output_f = torch.flatten(mn_output,start_dim=1)
        o_output_f = torch.flatten(o_output,start_dim=1)

        mg_output_sm = self.softmax(mg_output); print('final output of mg from classifier', mg_output_sm.shape) #(32, 2, 8, 1)
        mn_output_sm = self.softmax(mn_output); print('final output of mn from classifier', mn_output_sm.shape) #(32, 2, 8, 1)
        o_output_sm = self.softmax(o_output); print('final output of o from classifier', o_output_sm.shape) #(32, 2, 12, 1)

        cell_label = self.label_c_layer(cell_output_f); print("final cell label from classifier", cell_label.shape) #(32, 1)
        mg_cat = self.label_mg_layer(mg_output_f); print("final mg_cat from classifier", mg_cat.shape) # (32, 8)
        mn_cat = self.label_mn_layer(mn_output_f); print("final mn_cat from classifier", mn_cat.shape) # (32, 8)
        o_cat = self.label_o_layer(o_output_f); print("final o_cat from classifier", o_cat.shape) # (32, 12)

        return mg_output_sm,mn_output_sm,o_output_sm, mg_cat,mn_cat,o_cat,cell_label

#if __name__ == '__main__':
#    pass



