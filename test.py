import os
import torch
import torch.utils.data as data
import data_preprocess
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def test_main():
    print('Load Testing Data ...')
    path2data = "./test"
    test_path = os.listdir(path2data)
    video_paths = []
    labels = [None for i in range(len(test_path))]
    for name in test_path:
        vedio = os.path.join(path2data, name)
        video_paths.append(vedio)

    """ DATASET parameter setting """
    IMG_SIZE = 112
    MAX_FRAME_LENGTH = 10
    TEST_BATCH_SIZE = 8
    
    test_transformer = transforms.Compose([
        transforms.ToPILImage(),
		transforms.RandomResizedCrop(IMG_SIZE,scale=(0.5,1.0)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        # transforms.Normalize(mean, std),
    ])
    
    test_aug = test_transformer
    test_data = data_preprocess.VideoDataset(video_paths, labels, MAX_FRAME_LENGTH, IMG_SIZE, test_aug)

    test_data_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_model = torch.load("model.pth").to(device)
    
    print("Begin testing...")
    with torch.no_grad():
        test_model.eval()
        valid_loss = 0
        correct = 0
        bs = test_data_loader.batch_size
        result = []
        for i, (data, id) in enumerate(test_data_loader):
            data = data.to(device)
            output = test_model(data)
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            arr = pred.data.cpu().numpy()
            for j in range(pred.size()[0]):
                file_name = test_data[i*bs+j][1].split('/')[-1]
                result.append((file_name,pred[j].cpu().numpy()[0]))

    with open('test_result.csv', 'w') as f:
        f.write('name,label\n')
        for data in result:
            f.write(data[0] + ',' + str(data[1]) + '\n')
        print('Done.')

if __name__ == "__main__":
    test_main()



