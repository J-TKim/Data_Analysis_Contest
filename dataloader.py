import scipy
import imageio
from skimage.transform import resize
from glob import glob
import numpy as np

class DataLoader():
    def __init__(self, img_res=(64, 64)):
        self.img_res = img_res
        
    def load_data(self, batch_size=1, is_testing=False):
        path = glob('../datasets/1')
        
        batch_images = np.random.choice(path, size=batch_size)
        
        imgs = []
        for img_path in batch_images:
            img = self.imread(img_path)
            if not is_testing:
                img = resize(img, self.img_res)
                
                if np.random.random() > 0.5:
                    img = np.fliplr(img)
            
            else:
                img = resize(img, self.img_res)
            imgs.append(img)
        
        imgs = np.array(imgs) / 127.5 - 1.
        
        return imgs
    
    def load_batch(self, batch_size=1, is_testing=False):
        path_A = glob('../datasets/1')
        
        self.n_batches = int(len(path_A) / batch_size)
        total_samples = self.n_batches * batch_size
        
        # Sample n_batches* batch_size from each path list so that model sees all
        
        path_A = np.random.choice(path_A, total_samples, replace=False)
        
        for i in range(self.n_batches-1):
            batch_A = path_A[i * batch_size:(i+1)*batch_size]
            imgs_A = []
            for img_A in batch_A:
                img_A = self.imread(img_A)
                
                img_A = resize(img_A, self.img_res)
                
                if not is_testing and np.random.random() > 0.5:
                    img_A = np.fliplr(img_A)
                    
                imgs_A.append(img_A)
                
            imgs_A = np.array(imgs_A) / 127.5 - 1
            
            yield imgs_A
    def imread(self, path):
        return imageio.imread(path, pilmode="RGB").astype(np.float)