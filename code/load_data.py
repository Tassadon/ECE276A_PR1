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

def load(fname=None,dataset="1",camera_data=False,testing=False):
  """
  Returns: (cam,imud,vicd)
  """
  if fname is not None:
    if camera_data:
      cfile = fname + "/cam/cam" + dataset + ".p"
    ifile = fname + "/imu/imuRaw" + dataset + ".p"
    if not testing:
      vfile = fname + "/vicon/viconRot" + dataset + ".p"
  else:
    if camera_data:
      cfile = "../data/cam/cam" + dataset + ".p"
    ifile = "../data/imu/imuRaw" + dataset + ".p"
    if not testing:
      vfile = "../data/vicon/viconRot" + dataset + ".p"

  ts = tic()
  if camera_data:
    camd = read_data(cfile)
  else:
    camd = None
  imud = read_data(ifile)
  if not testing:
    vicd = read_data(vfile)
  else:
    vicd = None
  toc(ts,"Data import")

  return (camd,imud,vicd)