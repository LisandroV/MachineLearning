import sys
import numpy as np
from numpy import asarray
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale as escalar
from time import time, strftime,gmtime
from sklearn.model_selection import train_test_split
from PIL import Image, ImageOps
from jinja2 import Template

#function to read images path from input file
def get_image_list():
    if (len(sys.argv)<2):
      raise Exception("Missing argument file with images!")
    file = open(sys.argv[1], 'r')
    images = file.read().splitlines()
    return images

#function to make all the images the same size
def scale_image(im,width,height):
    old_size = im.size  # old_size[0] is in (width, height) format

    if(old_size[0]>width):
        im = im.resize((width,old_size[1]))
        old_size = im.size

    if(old_size[1]>height):
        im = im.resize((old_size[0],height))
        old_size = im.size

    delta_w = width - old_size[0]
    delta_h = height - old_size[1]
    padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
    new_im = ImageOps.expand(im, padding,fill=(255,255,255))#fills spaces with white pixels
    return new_im.convert('L')

start_time = time() #start counter

print("Loading data...")
images = []
files = get_image_list()

for f in files:
    im = Image.open(f)
    im = scale_image(im,64,32)
    img = np.asarray(im)
    images.append(np.reshape(img,64*32))

X_train = np.array(images)
train_names = files

data = escalar(X_train)

n_samples, n_features = data.shape
num_clusters = 72

estimator = KMeans(init='k-means++', n_clusters=num_clusters, n_init=8)
print("Training...")
estimator.fit(data)
training_time = time() - start_time
string_time = strftime("%M:%S", gmtime(training_time))
print("Total time",string_time)

pred = estimator.predict(data)

classes = np.unique(pred)
template_images = dict.fromkeys(classes, [])#dict with clusteres images for the template
output_images = dict.fromkeys(classes, "")#dict with clustered image names for output file
for i in range(len(pred)):
    template_images[pred[i]]= template_images[pred[i]] + [train_names[i]]
    output_images[pred[i]] = output_images[pred[i]] + " " + train_names[i].split("/")[-1:][0]

templ = open('template.html', 'r')
string_template = templ.read().replace('\n', '').replace('    ', '')

t = Template(string_template)
html = t.render(cluster=template_images,time=string_time,n=len(pred))
html_out = open("output.html", 'w')
html_out.write(html)


names_clustered_file = open('output.txt', 'w')
for key in output_images:
    names_clustered_file.write(output_images[key] + "\n")
names_clustered_file.close()
