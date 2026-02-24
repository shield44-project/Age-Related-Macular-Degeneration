#########TEMPO########

import pydicom as pd
import cv2

'''
ds=pd.dcmread(Path)

ino={"info": ds.modality,"size":ds.pixel_array.shape}

print(ino)
'''

#CONVERTION TO PNG
img=pd.dcmread('med.dcm').pixel_array
cv2.imwrite("out.png",img)

# FIX FOR TRANSFER SYNTAX ERROR
pydicom.dcmread(path, force=True)

# FIX FOR NO_PIXEL_ARRAY
if hasattr(ds, "pixel_array"):






