import pickle
import sys
import time 

def tic():
  return time.time()
def toc(tstart, nm=""):
  print('%s took: %s sec.\n' % (nm,(time.time() - tstart)))

def read_data(fname):
  d = []
  with open(fname, 'rb') as f:
    if sys.version_info[0] < 3:
      d = pickle.load(f)
    else:
      d = pickle.load(f, encoding='latin1')  # needed for python 3
  return d

def load(fname=None,dataset="1"):
  if fname is not None:
    cfile = fname + "/cam/cam" + dataset + ".p"
    ifile = fname + "/imu/imuRaw" + dataset + ".p"
    vfile = fname + "/vicon/viconRot" + dataset + ".p"
  else:
    cfile = "../data/cam/cam" + dataset + ".p"
    ifile = "../data/imu/imuRaw" + dataset + ".p"
    vfile = "../data/vicon/viconRot" + dataset + ".p"

  ts = tic()
  camd = read_data(cfile)
  imud = read_data(ifile)
  vicd = read_data(vfile)
  toc(ts,"Data import")

  return (camd,imud,vicd)





