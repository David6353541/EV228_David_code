import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from extract_variable import variable

os.chdir(r'C:\\Users\\zheng\\Desktop\\EV228')

tempdataA1=pd.read_csv("C:\\Users\\zheng\\Desktop\\EV228\\AYW00090001_temp_195702-202508.csv")
tempdataB1=pd.read_csv("C:\\Users\\zheng\\Desktop\\EV228\\BR038014410_temp_189601-202508.csv")
tempdataU1=pd.read_csv("C:\\Users\\zheng\\Desktop\\EV228\\USW00093009_temp_190801-202508.csv")
tempdataR1=pd.read_csv("C:\\Users\\zheng\\Desktop\\EV228\\ROE00108901_temp_188001-202508.csv")
tempdataR2=pd.read_csv("C:\\Users\\zheng\\Desktop\\EV228\\RSM00021432_temp_193601-202508.csv")
tempdataS1=pd.read_csv("C:\\Users\\zheng\\Desktop\\EV228\\SG000061641_temp_189906-202508.csv")
tempdataI1=pd.read_csv("C:\\Users\\zheng\\Desktop\\EV228\\IN020100400_temp_189101-202508.csv")
tempdataK1=pd.read_csv("C:\\Users\\zheng\\Desktop\\EV228\\KSM00047108_temp_190710-202508.csv")
"""
remove invalid value
"""
tempdataA1=tempdataA1.mask(tempdataA1==999.9)
tempdataB1=tempdataB1.mask(tempdataB1==999.9)
tempdataU1=tempdataU1.mask(tempdataU1==999.9)
tempdataR1=tempdataR1.mask(tempdataR1==999.9)
tempdataR2=tempdataR2.mask(tempdataR2==999.9)
tempdataS1=tempdataS1.mask(tempdataS1==999.9)
tempdataI1=tempdataI1.mask(tempdataI1==999.9)
tempdataK1=tempdataK1.mask(tempdataK1==999.9)

"""
Apply the function on datasets
"""
tempdataA1_JAN=variable("C:\\Users\\zheng\\Desktop\\EV228\\AYW00090001_temp_195702-202508.csv",'JAN')
print(tempdataA1_JAN)
tempdataB1_MAY=variable("C:\\Users\\zheng\\Desktop\\EV228\\BR038014410_temp_189601-202508.csv",'MAY')
print(tempdataB1_MAY)
tempdataU1_JUN=variable("C:\\Users\\zheng\\Desktop\\EV228\\USW00093009_temp_190801-202508.csv",'JUN')
print(tempdataU1_JUN)
tempdataR1_APR=variable("C:\\Users\\zheng\\Desktop\\EV228\\ROE00108901_temp_188001-202508.csv",'APR')
print(tempdataR1_APR)
tempdataR2_OCT=variable("C:\\Users\\zheng\\Desktop\\EV228\\RSM00021432_temp_193601-202508.csv",'OCT')
print(tempdataR2_OCT)
tempdataS1_NOV=variable("C:\\Users\\zheng\\Desktop\\EV228\\SG000061641_temp_189906-202508.csv",'NOV')
print(tempdataS1_NOV)
tempdataI1_SEP=variable("C:\\Users\\zheng\\Desktop\\EV228\\IN020100400_temp_189101-202508.csv",'SEP')
print(tempdataI1_SEP)
tempdataK1_metANN=variable("C:\\Users\\zheng\\Desktop\\EV228\\KSM00047108_temp_190710-202508.csv",'metANN')
print(tempdataK1_metANN)




