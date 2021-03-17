#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
from matplotlib import pyplot as plt
from IPython.display import Image
import numpy as np
import pandas as pd
from PIL import ImageColor
import seaborn as sns


# ## Paths

# In[2]:


image_path = "imagen.jpg"
raster_path = "raster.pgm"
output_path = "result.png"


# In[3]:


img = Image(filename=image_path)
img


# In[4]:


imagen = cv2.imread(image_path,-1)
imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
raster = cv2.imread(raster_path,-1)


# ## Show Raster

# In[5]:


sns.color_palette("hls", 38)


# In[6]:


color_list = sns.color_palette("hls", 38).as_hex()
colors = []
for color in color_list:
    rgb_color = ImageColor.getrgb(color)
    colors.append(rgb_color)


# In[7]:


array_of_values = [] #podria usar range
for i in range(1,raster.max()+1):
    array_of_values.append(i)


# In[8]:


RGB_im = cv2.cvtColor(raster, cv2.COLOR_BGR2RGB)
for value in array_of_values:
    mask = np.isin(raster, value)
    mask = mask*1
    mask = mask.astype(np.uint8)
    r_mask = mask*colors[value-1][0]
    g_mask = mask*colors[value-1][1]
    b_mask = mask*colors[value-1][2]
    RGB_im[:, :, 0] += r_mask
    RGB_im[:, :, 1] += g_mask
    RGB_im[:, :, 2] += b_mask

# for fila,row in enumerate(raster):
#     for col,value in enumerate(row):
#         imagen[fila][col] = colors[value-1]
            
plt.imshow(RGB_im)
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
plt.axis('off')
plt.show()


# In[9]:


values_to_keep = range(5,39)#range(12,len(array_of_values)) #o usar una list


# In[10]:


imagen = cv2.imread(image_path,-1)
bgra = cv2.cvtColor(imagen, cv2.COLOR_BGR2BGRA)
# Then assign the mask to the last channel of the image
bgra[:, :, 3] = 128


# In[11]:


mask = np.isin(raster, values_to_keep)
m = mask*1
m = m*127
m = m.astype(np.uint8)
bgra[:, :, 3] += m
cv2.imwrite(output_path, bgra)


# In[12]:


img = Image(filename=output_path)
img


# In[13]:


# plt.imshow(RGB_im)
# # plt.title('raster')
# fig = plt.gcf()
# fig.set_size_inches(18.5, 10.5)
# plt.axis('off')
# plt.show()


# In[14]:


# plt.imshow(rgba)
# fig = plt.gcf()
# fig.set_size_inches(18.5, 10.5)
# plt.axis('off')
# plt.show()


# In[ ]:





# In[15]:


# import random 
# import matplotlib.pyplot as plt 
# def generate_colors(n): 
#     rgb_values = [] 
#     hex_values = [] 
#     r = int(random.random() * 256) 
#     g = int(random.random() * 256) 
#     b = int(random.random() * 256) 
#     step = 256 / n 
#     for _ in range(n): 
#         r += step 
#         g += step 
#         b += step 
#         r = int(r) % 256 
#         g = int(g) % 256 
#         b = int(b) % 256 
#         r_hex = hex(r)[2:] 
#         g_hex = hex(g)[2:] 
#         b_hex = hex(b)[2:] 
#         hex_values.append('#' + r_hex + g_hex + b_hex) 
#         rgb_values.append((r,g,b)) 
#     return rgb_values, hex_values 



# # generate values and print them 
# rgb_values, hex_values = generate_colors(len(array_of_values)) 
# print (rgb_values, hex_values) 

# color_count = 10 
# plt.show() 


# In[ ]:





# In[ ]:





# In[ ]:




