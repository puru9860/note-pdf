import json
import cv2 
import numpy as np


#read json file from word detection
result = open('detect/result.json')
labels_json =json.load(result)

labels = []

for line in labels_json[0]['objects']:
    cord = line['relative_coordinates'] 
    box=[round(cord['center_x']*1024),round(cord['center_y']*1024),round(cord['width']*1024),round(cord['height']*1024)]
    labels.append(box)
labels = np.array(labels)

# sort vertically 
vertical_sort=np.sort(labels.view('i8,i8,i8,i8'), order=['f1'], axis=0).view(np.int)

# make vertical group
sorted=[]
count = 1
gap =25
temp=[]
for i in range(vertical_sort.shape[0]):
    
    if vertical_sort[i][1]<=count*gap:
        temp.append(vertical_sort[i].tolist())
        
    else:
        if temp!= []:
            sorted.append(temp)
            temp=[]
        while count*gap <= vertical_sort[i][1]:
            count += 1
        temp.append(vertical_sort[i].tolist())        
sorted.append(temp)


# sort horizontally
sorted_horizontally=[]
for i in range(len(sorted)):
    if len(sorted[i]) >1 :
        temp = np.array(sorted[i])
        temp=np.sort(temp.view('i8,i8,i8,i8'), order=['f0'], axis=0).view(np.int) 
        for j in range(temp.shape[0]):
          sorted_horizontally.append(temp[j])
    else:
        sorted_horizontally.append(sorted[i][0])

sorted_horizontally = np.array(sorted_horizontally,dtype='object')

bounding_box=[]
for index in range(labels.shape[0]):
    box=sorted_horizontally[index]
    left = int(float(box[0]) - float(box[2]) / 2.);
    right = int(float(box[0]) + float(box[2])/ 2.);
    top = int(float(box[1]) - float(box[3]) / 2.);
    bot = int(float(box[1]) + float(box[3]) / 2.);
    bounding_box.append([top,right,bot,left])
bounding_box = np.array(bounding_box)

#open image for cropping
img_path = 'detect/image.jpg'
img = cv2.imread(img_path, 1)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
image = np.array(img)

#height and width of image
height = image.shape[0]
width = image.shape[1]

#crop words
word_list = open("detect/test.txt", "w")  


for index in range(labels.shape[0]):
  # top = word[0]
  # right = word[1]
  # bot = word[2]
  # left = word[3]
  box=labels[index]
  left = int(float(box[0]) - float(box[2]) / 2.);
  right = int(float(box[0]) + float(box[2])/ 2.);
  top = int(float(box[1]) - float(box[3]) / 2.);
  bot = int(float(box[1]) + float(box[3]) / 2.);
  img_c = image[top:bot, left:right]
  pad = np.array([0,0,0,0])
  #padding array in order top,bot,left,right
  if img_c.shape[0] < 64:
      padding = 64-img_c.shape[0]
      if padding%2 != 0:
          pad[0] = int(padding//2)
          pad[1] = padding-pad[0]
      else:
          pad[0] = pad[1] = int(padding/2)
      
  if img_c.shape[1] < 256:
      padding = 256 - img_c.shape[1]
      if padding%2 != 0:
          pad[2] = int(padding//2)
          pad[3] = padding-pad[2]
      else:
          pad[2] = pad[3] = int(padding/2)
  img_c = cv2.copyMakeBorder(img_c, pad[0], pad[1] , pad[2], pad[3], cv2.BORDER_CONSTANT,None,[255,255,255])
  if img_c.shape !=(64,256):
      img_c = cv2.resize(img_c,(256,64),cv2.INTER_CUBIC)
  assert img_c.shape ==(64,256)
  cropped_save = "detect/cropped/"+str(labels_json[0]['objects'][index]['relative_coordinates']['center_x'])+'_'+str(labels_json[0]['objects'][index]['relative_coordinates']['center_y'])+'_'+str(labels_json[0]['objects'][index]['relative_coordinates']['width'])+'_'+str(labels_json[0]['objects'][index]['relative_coordinates']['height'])+'.jpg'

  cv2.imwrite(cropped_save,img_c) 
  word_list.write(cropped_save)
  word_list.write("\n")

  
word_list.close()

