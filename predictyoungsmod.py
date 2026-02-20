import numpy as np
import h5py
import torch
from torch import nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset, random_split

dtype = torch.float32
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

porosity_dataset_filename = 'porositydataset_2000runs.hdf5'
##load the dataset
d = h5py.File(porosity_dataset_filename,'a')
#load original images
og_images = np.zeros(d['Original Images'].shape)
og_images[:,:,:] = d['Original Images']
#load deformed images
deformed_images = np.zeros(d['Deformed Images'].shape)
deformed_images[:,:,:] = d['Deformed Images']
#load stress-strain data
data = np.zeros(d['Data'].shape)
data[:,:,:] = d['Data']
#separate stresses/strains, stresses in MPa, strain is unitless
SMises = data[:,0,:] #von Mises stress
S11 = data[:,1,:] #stress in 11 direction
S22 = data[:,2,:] #stress in 22 direction
SP = data[:,3,:] #pressure stress
strain = data[:,4,:]*0.006 #strain (have to multiply by 0.006 to get value)
#load youngs moduli, units are GPa
E = np.zeros(d['Youngs Modulus'].shape)
E[:] = d['Youngs Modulus']

class PorousMaterialImageToYoungsModulusDataset(Dataset):
    def __init__(self, hdf5_porosity_path, img_transform=None):
        self.img_transform = img_transform
        ##load the dataset
        self.d = h5py.File(hdf5_porosity_path,'a')
        #load original images
        self.og_images = np.zeros(self.d['Original Images'].shape)
        self.og_images[:,:,:] = self.d['Original Images']
        #load deformed images
        self.deformed_images = np.zeros(self.d['Deformed Images'].shape)
        self.deformed_images[:,:,:] = self.d['Deformed Images']
        #load stress-strain data
        self.ss_data = np.zeros(self.d['Data'].shape)
        self.ss_data[:,:,:] = self.d['Data']
        #separate stresses/strains, stresses in MPa, strain is unitless
        self.SMises = self.ss_data[:,0,:] #von Mises stress
        self.S11 = self.ss_data[:,1,:] #stress in 11 direction
        self.S22 = self.ss_data[:,2,:] #stress in 22 direction
        self.SP = self.ss_data[:,3,:] #pressure stress
        self.strain = self.ss_data[:,4,:]*0.006 #strain (have to multiply by 0.006 to get value)
        #load youngs moduli, units are GPa
        self.E = np.zeros(self.d['Youngs Modulus'].shape)
        self.E[:] = self.d['Youngs Modulus']
        
    def __len__(self):
        return self.E.size

    def __getitem__(self, idx):
        og_image_idx = self.og_images[idx,:,:,:].astype('uint8')
        deformed_image_idx = self.deformed_images[idx,:,:,:].astype('uint8')
        if (self.img_transform is not None):
            og_image_idx = self.img_transform(og_image_idx)
            deformed_image_idx = self.img_transform(deformed_image_idx)
        SMises_idx = self.SMises[idx,:]
        S11_idx = self.S11[idx,:]
        S22_idx = self.S22[idx,:]
        SP_idx = self.SP[idx,:]
        strain_idx = self.strain[idx,:]
        E_idx = self.E[idx]
        # return og_image_idx, SMises_idx, S11_idx, S22_idx, SP_idx, SP_idx, strain_idx, E_idx
        # Have to decide what exactly to return here, depends on training task
        # For now will have og_image_idx be the feature and E_idx be the label
        image = og_image_idx
        label = E_idx
        return image, label

# Load porous dataset into memory
porous_dataset = PorousMaterialImageToYoungsModulusDataset(porosity_dataset_filename, img_transform=transforms.Compose([transforms.ToPILImage(), transforms.Resize(224), transforms.ToTensor()]))

# Split into train and test
test_prop = 0.2
test_size = int(test_prop*len(porous_dataset))
train_size = len(porous_dataset) - test_size
porous_train_dataset, porous_test_dataset = random_split(porous_dataset, (train_size, test_size)) 

# Split train into train and val
val_prop = 0.2
val_size = int(val_prop*len(porous_train_dataset))
train_size = len(porous_train_dataset) - val_size
porous_train_dataset, porous_val_dataset = random_split(porous_train_dataset, (train_size, val_size)) 

batch_size = 32
train_loader = DataLoader(porous_train_dataset, batch_size=batch_size, shuffle=True)

# Batch size doesn't really matter for testing
test_loader = DataLoader(porous_test_dataset)
val_loader = DataLoader(porous_val_dataset)

class PorousMaterialImageToYoungsModulusNet(nn.Module):
    def __init__(self):
        super().__init__()
        squeeze_net_model = nn.Sequential(*list(models.squeezenet1_1(pretrained=False).children()))
        self.model = torch.nn.Sequential(
            nn.Conv2d(3,32,5),
            nn.Flatten(),
            nn.Linear(1548800, 1)
            )

    def forward(self, porous_material_image):
        scores = self.model(porous_material_image)
        return torch.flatten(scores, start_dim=1, end_dim=1)

def get_PorousMaterialImageToYoungsModulusNet_accuracy(model, evaluation_dataloader, criterion):
    model.eval()
    model = model.to(device=device)
    total_MSE = 0
    num_vals = 0
    with torch.no_grad():
        for eval_val in evaluation_dataloader:
            feat, label, = eval_val
            feat = feat.to(dtype=dtype, device=device)
            label = label.to(dtype=dtype, device=device)

            pred_label = torch.flatten(model(feat))
            total_MSE += criterion(pred_label, label).cpu().item()
            num_vals += 1
    avg_MSE = total_MSE / num_vals
    return total_MSE, avg_MSE 

model = PorousMaterialImageToYoungsModulusNet()

learning_rate = 1e-3
nepochs = 100 
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_avg_MSE_vals = []
val_avg_MSE_vals = []
# date_str = datetime.datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
# model_save_name = 'model_{0}.pt'.format(date_str)
# model_save_path = '{0}{1}'.format(model_checkpoint_save_path, model_save_name)
# model_training_info_name = 'model_{0}_training_info.pkl'.format(date_str)
# model_training_info_name_path = '{0}{1}'.format(model_checkpoint_save_path, model_training_info_name)
best_val_avg_MSE = 1000  
model = model.to(device=device)
for e in range(nepochs):
    model.train() # Will put model in training mode
    print("Begin epoch {0}".format(e))
    for batch in train_loader:
        train_features, train_labels = batch
        train_features = train_features.to(dtype=dtype, device=device)
        train_labels = train_labels.to(dtype=dtype, device=device)

        pred_labels = torch.flatten(model(train_features))
        loss = criterion(pred_labels, train_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    _, current_train_avg_MSE = get_PorousMaterialImageToYoungsModulusNet_accuracy(model, train_loader, criterion) # Will put model in eval mode
    _, current_val_avg_MSE = get_PorousMaterialImageToYoungsModulusNet_accuracy(model, val_loader, criterion) # Will put model in eval mode
    train_avg_MSE_vals.append(current_train_avg_MSE)
    val_avg_MSE_vals.append(current_val_avg_MSE)
    # if current_val_avg_MSE < best_val_avg_MSE:
    #     torch.save({
    #         'epoch':e,
    #         'model_state_dict':model.state_dict(),
    #         'optimizer_state_dict':optimizer.state_dict(),
    #         'avg_train_loss':current_train_avg_MSE,
    #         'avg_val_loss':current_val_avg_MSE
    #     }, model_save_path)
    #     best_val_avg_MSE = current_val_avg_MSE
    print("Epoch {0}: train avg MSE={1}, val avg MSE={2}\n\n".format(e, current_train_avg_MSE, current_val_avg_MSE))
# with open(model_training_info_name_path, 'wb') as f:
#     pickle.dump((train_avg_MSE_vals, val_avg_MSE_vals),f)
