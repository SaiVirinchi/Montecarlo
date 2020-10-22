
from scipy import random as sciran
import random
from scipy import stats
import scipy
import numpy as np
import matplotlib.pyplot as plt



def free_path():
    min = 0
    max = 1
    xrand = sciran.uniform(min,max)
    #value of parameter
    k = 1/sigma_t

    s = -k*np.log(xrand)
    return s

def collision():
    ab = sciran.uniform(0,1)
    if ab <= sigma_a/sigma_t:
      return True
    else:
      return False

def new_angle():
    ang = sciran.uniform(0,pi)
    return ang

def update_location(rnew,rold,anew):
    rfresh = rold + rnew*np.cos(anew)
    return rfresh

def boundary(rfresh):
    if rfresh > half_length:
      return 5
    elif rfresh < 0:
      return 4

def cycle():


  #SOURCE
  freepath = free_path()
  angle = 0


  position = 0
  global right
  global left
  global absorb


  position = update_location(freepath,position,angle)
  #print(position)
  if boundary(position) == 4:
    left = left + 1
    #print("out",left)
    return
  elif boundary(position) == 5:
    right = right + 1
    #print("right")
    return
  if collision():
    absorb = absorb + 1
    #print("absorb",absorb)
    return


  #TRANSITION
  while True:
    freepath = free_path()
    angle = new_angle()
    position = update_location(freepath,position,angle)
    #print("lol", j)
    #print(position)
    if boundary(position) == 4:
      left = left + 1
      #print("out2",left)
      return
    elif boundary(position) == 5:
      right = right + 1
      #print("out3",right)
      return
    if collision():
      absorb = absorb + 1
      #print("absorb2",absorb)
      return

def simul():

  for i in range(N):
    cycle()
    #print("Interation:",i)

  #print("left: ",left)
  #print("right: ",right)
  #print("absorb: ", absorb)
  global Data
  Data = np.append(Data,[[left,right,absorb]],0)

def plott(M):
  param = stats.norm.fit(M) # distribution fitting

  # now, param[0] and param[1] are the mean and
  # the standard deviation of the fitted distribution
  #x = linspace(-5,5,100)
  domain = np.linspace(np.min(M),np.max(M))
  x = domain
  # fitted distribution
  pdf_fitted = stats.norm.pdf(domain,loc=param[0],scale=param[1])
  # original distribution
  pdf = stats.norm.pdf(x)

  plt.title('Normal distribution')
  plt.plot(x,pdf_fitted,)
  plt.hist(M,domain,ec='black',density=1,alpha=.3)
  plt.show()

#global variables
sigma_t = 3 #Σt
sigma_a = 1 #Σa ratio
interation = 1000
half_length = 0.5

pi = np.pi

N = interation

Data = [[0,0,0]]



for j in range(1000):
  right = 0
  left = 0
  absorb = 0
  simul()

Data = np.delete(Data,0,0)

for w in range(3):
  plott(Data[:,w])
