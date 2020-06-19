#!/usr/bin/env python
# coding: utf-8

# In[20]:


from minisom import MiniSom  
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as im


# In[21]:


def vectorize(tumor,step):
    
    vectors = []
    m,n = np.shape(tumor)
    for i in np.arange(0,m,step):
        for j in np.arange(0,n,step):
            block = tumor[i:i+step,j:j+step].reshape(1,256)
            vectors.append(block)
            
    v = np.array(vectors).reshape(4096,256)
    return v


# In[22]:


def compress(BV,it,shape,step,lat1,lat2):
    som = MiniSom(lat1, lat2, 256)
    som.random_weights_init(BV)
    starting_weights = som.get_weights().copy()  # saving the starting weights
    som.train_random(BV, it)
    qnt = som.quantization(BV)
    
    [m,n] = shape
    image = np.zeros((m,n))
    
    blocks = [q.reshape(16,16) for q in qnt]
    count = 0
    
    for i in np.arange(0,m,step):
        for j in np.arange(0,n,step):
            image[i:i+step,j:j+step] = blocks[count]
            count += 1
            
    return image


# In[23]:

print("Example input: T1.pgm. Only takes .pgm at present")
user_image = input("Enter image filename:")
print("1-50 units. Larger lattice size means less compression")
lattice = int(input("Enter Latice Size: "))
B1 = im.imread(user_image)


# In[6]:


BV1 = vectorize(B1,16)


# In[26]:


B1_compressed_10 = compress(BV1,1000,(1024,1024),16,lattice,lattice)

plt.imshow(B1_compressed_10);
plt.savefig("Image_compressed.png")
plt.show();

plt.imshow(B1);
plt.title("uncompressed image")
plt.show();

