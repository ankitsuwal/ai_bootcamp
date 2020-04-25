# url to image
import numpy as np 
import urllib.request
import cv2
from skimage import io



class UrlToImage():
    """docstring for UrlToImage"""
    def __init__(self, urls):
        super(UrlToImage, self).__init__()
        self.urls = urls
##### ***** ##### ***** ##### ***** ##### ***** 
# Method : 1 
##### ***** ##### ***** ##### ***** ##### *****  
    def url_to_img(self):
        for url in self.urls:
            resp = urllib.request.urlopen(url)
            image = np.asarray(bytearray(resp.read()), dtype='uint8')
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            cv2.imshow("Image", image)
            cv2.waitKey(0)
        
##### ***** ##### ***** ##### ***** ##### ***** 
# Method : 2
##### ***** ##### ***** ##### ***** ##### *****  
    def scikit_urltoimg(self):
        for url in self.urls:
            print("downloading %s" % (url))
            image = io.imread(url)
            cv2.imshow("Incorrect", image)
            cv2.imshow("Correct", cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            cv2.waitKey(0) 


if __name__ == '__main__':
    urls = ["https://images.dog.ceo/breeds/mastiff-tibetan/n02108551_660.jpg",
            "https://images.unsplash.com/photo-1491604612772-6853927639ef?ixlib=rb-1.2.1&auto=format&fit=crop&w=500&q=60",
            "https://images.unsplash.com/photo-1518914781460-a3ada465edec?ixlib=rb-1.2.1&auto=format&fit=crop&w=500&q=60"]
    obj = UrlToImage(urls)
    # obj.url_to_img()
    obj.scikit_urltoimg()
        