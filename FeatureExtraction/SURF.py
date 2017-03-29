# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 13:22:22 2017

@author: Lalit Singh
"""

import mahotas as mh
from mahotas.features import surf
image = mh.imread('sexy.jpg',as_gray=True)
print("The first surf descriptor : %s" %surf.surf(image)[0])
print("Extracted %s SURF descriptor "%len(surf.surf(image)))