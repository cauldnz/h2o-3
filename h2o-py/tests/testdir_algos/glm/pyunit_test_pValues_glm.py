from __future__ import print_function
from builtins import range
import sys
sys.path.insert(1,"../../../")
import h2o
from tests import pyunit_utils
import random
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
import os
import pickle
import time
import numpy

sys.path.extend([ '/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages' ])

import rpy2
from rpy2.robjects.packages import importr
import rpy2.robjects as ro
import subprocess


"""
Kevin's method to generate synthetic datasets.  I am going to make sure it can generate different types like
integer, floating point, enum and others.  Basically, it will generate the w and b first for the whole training
dataset.  Next, it will generate the predictors x and generate the corresponding y using the formula:
y = wTx+b+e where T is transpose, e is the small Guassian noised added to generate the response.

The output format of the data will be:
each row will be the predictor x1 plus y1,...

The output for weight file will be:
b w

"""
def write_syn_floating_point_dataset(SEED,csvTrainingname,csvWeightname,rowCount, colCount, dataType, \
                                     maxPValue, minPValue, maxWValue, minWValue, noiseStd):

  random.seed(SEED)
  dsf = open(csvTrainingname, "w")
  wsf = open(csvWeightname,"w")

  # first generate the random weight
  weight = []
  for i in range(colCount+1):
    if (dataType == 1):
      weight.append(random.randint(minWValue,maxWValue))
    elif (dataType == 2):
      weight.append(random.uniform(minPValue,maxPValue))

  rowDataCsv = ",".join(map(str,weight))
  wsf.write(rowDataCsv+"\n")
  wsf.close()

  for i in range(rowCount):
    rowData = []
    for j in range(colCount):
      if (dataType == 1):
        ri = random.randint(minPValue, maxPValue)
      elif (dataType == 2):
        ri =random.uniform(minPValue, maxPValue)
      rowData.append(ri)

      # generate and append the noise to the response variable y
    rowData.append(generate_response(weight,rowData,noiseStd))

    rowDataCsv = ",".join(map(str,rowData))
    dsf.write(rowDataCsv + "\n")

  dsf.close()

def generate_response(weight,rowData,noiseStd):

  response_y = weight[0]
  for ind in range(len(rowData)):
    response_y += weight[ind+1]*rowData[ind]

  response_y += random.gauss(0, noiseStd)

  return response_y

def copy_files(old_name, new_name):
  cmd = 'cp '+old_name+' '+new_name
  subprocess.call(cmd,shell=True)





class Test4PValues:
  SEED = 0    # denote random seed used in this test.

  max_col_count = 1000   # randomly generate the train/test row and column counts
  # set the column count to be a function of max_row_count
  max_col_count_ratio = 100   # set row count to be multiples of col_count

  maxPValue = 100  # predictor values are set to be random from -100 to 100
  minPValue = -100

  maxWValue = 100     # weight and bias values max and min values
  minWValue = -100

  noise_std = 0.01         # noise variance in random data generation

  train_row_count = 0
  train_col_count = 0

  number_experiment = 100 # number of experiments to run.  To loop forever, set it to be -1
  number_failed_tolerated = 10    # number of failed experiments allowed before quitting

  data_type = 2       # dealing with real numbers here

  current_dir = os.path.dirname(os.path.realpath(__file__)) # directory where we are running
  random_seed_pickle_file = os.path.join(current_dir,"glm_p_values_failed.pickle")
  training_filename = "training_set"
  weight_filename = "weight"
  training_data_file = os.path.join(current_dir,training_filename+".csv")
  weight_data_file = os.path.join(current_dir, weight_filename+".csv")

  failed_test_random_seed_info = {"Number_of_tests_performed":0,"Random_seed_list":[],"maxPValue":[], \
                                  "minPValue":[],"maxWValue":[],"minWValue":[],"train_row_count":[],"train_col_count":[], \
                                  "data_type":[],"noise_std":[]}

  test_failed_R = 0 # 1 if test failed and 0 otherwise
  test_failed_Py = 0

  ignored_eps = 1e-15  # if p-values < than this value, no comparison is performed
  allowed_diff = 1e-5 # value of p-values difference allowed between theoretical and h2o p-values.


  def __init__(self):
    self.setUp()


  def setUp(self):

    self.noise_std = random.uniform(0,pow((self.maxPValue - self.minPValue),2)/12)

    # load and add to failed test seed info and others
    if (os.path.isfile(self.random_seed_pickle_file)):
      with open(self.random_seed_pickle_file,'rb') as sf:
        self.failed_test_random_seed_info = pickle.load(sf)

    self.train_col_count = random.randint(1,self.max_col_count)
    self.train_row_count = round(self.train_col_count * random.uniform(1.1,self.max_col_count_ratio))

  def tearDown(self):
    #print "In tearDown"
    # need to add saving seed when test failed
    self.failed_test_random_seed_info["Number_of_tests_performed"] += 1

    if (self.test_failed_Py or self.test_failed_R):
      self.failed_test_random_seed_info["Random_seed_list"].append(self.SEED)
      self.failed_test_random_seed_info["maxPValue"].append(self.maxPValue)
      self.failed_test_random_seed_info["minPValue"].append(self.minPValue)
      self.failed_test_random_seed_info["maxWValue"].append(self.maxWValue)
      self.failed_test_random_seed_info["minWValue"].append(self.minWValue)
      self.failed_test_random_seed_info["train_row_count"].append(self.train_row_count)
      self.failed_test_random_seed_info["train_col_count"].append(self.train_col_count)
      self.failed_test_random_seed_info["data_type"].append(self.data_type)
      self.failed_test_random_seed_info["noise_std"].append(self.noise_std)

      # mv the training_set.csv and weight.csv to one added with current time stamp in ms
      timeStamp = time.time()*1000; # current time in ms

      extraStr = ''
      # move the training and weight file to different places so that they won't get overwritten
      if self.test_failed_Py:
        extraStr += 'Py'

      if self.test_failed_R:
        extraStr += 'R'

      training_filename = os.path.join(self.current_dir,self.training_filename+"_"+extraStr+"_"+str(timeStamp)+".csv")
      copy_files(self.training_data_file,training_filename)

      weight_filename = os.path.join(self.current_dir,self.weight_filename+"_"+extraStr+"_"+str(timeStamp)+".csv")
      copy_files(self.weight_data_file, weight_filename)

    with open(self.random_seed_pickle_file,'wb') as sf:
      pickle.dump(self.failed_test_random_seed_info, sf)


    # exit with error
    # if self.test_failed:
    #   sys.exit(1)

  def printPValues(self, startString,numpArray):
    print(startString)
    print(numpArray)


  def comparePValues(self,pValue_theory,pValue_h2o,pValue_h2oR):
    # compare p-values from R, theory and h2o glm
    for ind in range(self.train_col_count+1):
      if not ((pValue_theory[ind] <self.ignored_eps) and (pValue_h2o[ind] < self.ignored_eps) and (pValue_h2oR[ind] < self.ignored_eps)):  # p-values not too small, perform comparison

        if not (type(pValue_h2o[ind]) == 'str'):
          compare_val_h2o_Py = abs(pValue_theory[ind]-pValue_h2o[ind])   # look at p-value differences between theory and h2o
          if (compare_val_h2o_Py > self.allowed_diff):
            self.test_failed_Py = 1
        else:
          print(pValue_h2o[ind])

        compare_val_h2o_R = abs(pValue_theory[ind] - pValue_h2oR[ind])



        if (compare_val_h2o_R > self.allowed_diff):
          self.test_failed_R = 1

  def test_GLM_p_values(self):
    # print self.current_dir
     #print self.SEED

    self.SEED = round(time.time())  # current time in second

     # generate training set data
#    self.train_col_count = 1
#    self.train_row_count = 20
#    self.noise_std=0.01

     #  generate test set data
    write_syn_floating_point_dataset(self.SEED,self.training_data_file,self.weight_data_file, \
                                      self.train_row_count, self.train_col_count, self.data_type, \
                                      self.maxPValue, self.minPValue, self.maxWValue, self.minWValue,self.noise_std)

     # call R commands and get the SESquare variable from R space into a numpy array
    ro.r("source('~/h2o-3/h2o-py/tests/testdir_algos/glm/testGLMinR.R', echo=FALSE)")
    pValue_theory = numpy.array(ro.r['pvalue_theory'])
    pValue_R = numpy.array(ro.r['pvalue_R'])
    pValue_h2oR = numpy.array(ro.r['pvalue_h2oR'])
    pValue_h2oR_standard = numpy.array(ro.r['pvalue_h2oR_standardized'])
    pValue_R_standard = numpy.array(ro.r['pvalue_R_standardized'])
    pValue_theory_standard = numpy.array(ro.r['pvalue_theory_standardized'])

     # call H2O and get the glm and the p-values
    training_data = h2o.import_file(pyunit_utils.locate("h2o-py/tests/testdir_algos/glm/training_set.csv"))

    Y = self.train_col_count
    X = list(range(self.train_col_count))

    pValue_h2o = self.trainGlm_get_pvalues(False,training_data,X,Y)
    pValue_h2o_standard = self.trainGlm_get_pvalues(True,training_data,X,Y)

    # for debugging
    # self.printPValues("P-values from H2O python: ",pValue_h2o)
    # self.printPValues("P-values from H2O R: ",pValue_h2oR)
    # self.printPValues("P-values from R: ",pValue_R)
    # self.printPValues("P-values from theory: ", pValue_theory)
    #
    # self.printPValues("P-values from H2O python standardized: ",pValue_h2o_standard)
    # self.printPValues("P-values from H2O R standardized: ",pValue_h2oR_standard)
    # self.printPValues("P-values from R standardized: ",pValue_R_standard)
    # self.printPValues("P-values from theory standardized: ", pValue_theory_standard)

    # compare p-values from R, theory and h2o glm
    self.comparePValues(pValue_theory,pValue_h2o,pValue_h2oR)
    self.comparePValues(pValue_theory_standard, pValue_h2o_standard, pValue_h2oR_standard)

  def trainGlm_get_pvalues(self,myStandardize,training_data,X,Y):

    model = H2OGeneralizedLinearEstimator(family="gaussian", Lambda=0, compute_p_values=True, standardize=myStandardize)
    model.train(x=X,y=Y, training_frame=training_data)

    coeff_pvalues = model._model_json["output"]["coefficients_table"].cell_values   # list of lists
    pValue_h2o = []

    for ind in range(self.train_col_count+1):
      if myStandardize:
        pValue_h2o.append(coeff_pvalues[ind][-2])
      else:
        pValue_h2o.append(coeff_pvalues[ind][-1])

    return pValue_h2o

def test_glm_pValues():
  numTest = 100000000000

  count = 0
  while count < numTest:
    count = count+1
    myTest = Test4PValues()
    myTest.test_GLM_p_values()
    myTest.tearDown()


if __name__ == "__main__":
  pyunit_utils.standalone_test(test_glm_pValues)
else:
  test_glm_pValues()
