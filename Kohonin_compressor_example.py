#!/usr/bin/env python
# coding: utf-8

# In[1]:


from minisom import MiniSom
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as im


# In[2]:


def vectorize(tumor,step):

    vectors = []
    m,n = np.shape(tumor)
    for i in np.arange(0,m,step):
        for j in np.arange(0,n,step):
            block = tumor[i:i+step,j:j+step].reshape(1,256)
            vectors.append(block)

    v = np.array(vectors).reshape(4096,256)
    return v


# In[3]:


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


# In[4]:


B1 = im.imread("T1.pgm")
M1 = im.imread("T23.pgm")
print(np.shape(B1))

# In[6]:


BV1 = vectorize(B1,16)
MV1 = vectorize(M1,16)


# In[7]:


B1_compressed_10 = compress(BV1,1000,(1024,1024),16,10,10)
M1_compressed_10 = compress(MV1,1000,(1024,1024),16,10,10)

print("The following was done using a 10 by 10 lattice")

plt.imshow(B1_compressed_10);
plt.title("Compressed version of benign tumor 1")
plt.show();

plt.imshow(B1);
plt.title("uncompressed version of benign tumor 1")
plt.show();

plt.imshow(M1_compressed_10);
plt.title("Compressed version of malignant tumor 1")
plt.show();

plt.imshow(M1);
plt.title("uncompressed version of malignant tumor 1")
plt.show();


# In[8]:


B1_compressed_25 = compress(BV1,1000,(1024,1024),16,25,25)

M1_compressed_25 = compress(MV1,1000,(1024,1024),16,25,25)

print("The following was done using a 25 by 25 lattice")

plt.imshow(B1_compressed_25);
plt.title("Compressed version of benign tumor 1")
plt.show();

plt.imshow(B1);
plt.title("uncompressed version of benign tumor 1")
plt.show();


plt.imshow(M1_compressed_25);
plt.title("Compressed version of malignant tumor 1")
plt.show();

plt.imshow(M1);
plt.title("uncompressed version of malignant tumor 1")
plt.show();


# In[ ]:
