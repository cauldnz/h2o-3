import unittest
import sys
import random
import os
from time import time
import pickle
import rpy2
from rpy2.robjects.packages import importr
import rpy2.robjects as ro
import numpy

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
def write_syn_floating_point_dataset(SEED,csvTrainingname,csvWeightname,rowCount, colCount, dataType,\
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
class Basic(unittest.TestCase):
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
    training_data_file = os.path.join(current_dir,"training_set.csv")
    weight_data_file = os.path.join(current_dir, "weight.csv")

    failed_test_random_seed_info = {"Number_of_tests_performed":0,"Random_seed_list":[],"maxPValue":[],\
                                    "minPValue":[],"maxWValue":[],"minWValue":[],"train_row_count":[],"train_col_count":[],\
                                    "data_type":[],"noise_std":[]}

    test_failed = 0 # 1 if test failed and 0 otherwise

    def setUp(self):

        self.noise_std = random.uniform(0,pow((self.maxPValue - self.minPValue),2)/12)

        # load and add to failed test seed info and others
        if (os.path.isfile(self.random_seed_pickle_file)):
            with open(self.random_seed_pickle_file,'rb') as sf:
                self.failed_test_random_seed_info = pickle.load(sf)

        self.train_col_count = random.randint(1,self.max_col_count)
        self.train_row_count = round(self.train_col_count * random.uniform(1.1,self.max_col_count_ratio))

    def tearDown(self):
        print "In tearDown\n"
        # need to add saving seed when test failed
        self.failed_test_random_seed_info["Number_of_tests_performed"] += 1

        if self.test_failed:
            self.failed_test_random_seed_info["Random_seed_list"].append(self.SEED)
            self.failed_test_random_seed_info["maxPValue"].append(self.maxPValue)
            self.failed_test_random_seed_info["minPValue"].append(self.minPValue)
            self.failed_test_random_seed_info["maxWValue"].appen(self.maxWValue)
            self.failed_test_random_seed_info["minWValue"].append(self.minWValue)
            self.failed_test_random_seed_info["train_row_count"].append(self.train_row_count)
            self.failed_test_random_seed_info["train_col_round"].append(self.train_col_count)
            self.failed_test_random_seed_info["data_type"].append(self.data_type)
            self.failed_test_random_seed_info["noise_std"].append(self.noise_std)

        with open(self.random_seed_pickle_file,'wb') as sf:
            pickle.dump(self.failed_test_random_seed_info, sf)



    # @classmethod
    # def setUpClass(cls):
    #     print ""

    # @classmethod
    # def tearDownClass(cls):
    #     print "WTF"


    def test_GLM_p_values(self):
        print self.current_dir
        print self.SEED

        self.SEED = round(time())

        # generate training set data
        self.train_col_count = 3
        self.train_row_count = 20
        self.noise_std=0.01

        #  generate test set data
        write_syn_floating_point_dataset(self.SEED,self.training_data_file,self.weight_data_file,\
                                         self.train_row_count, self.train_col_count, self.data_type,\
                                         self.maxPValue, self.minPValue, self.maxWValue, self.minWValue,self.noise_std)

        # call R commands and get the SESquare variable from R space into a numpy array
        ro.r("source('~/h2o-3/h2o-py/dynamic_tests/testdir_algos/glm/testGLMinR.R', echo=FALSE)")
        pVal = numpy.array(ro.r['pVal'])
        rPVal = numpy.array(ro.r['R_glm_p_values'])

        # call H2O and get the glm and the p-values




if __name__ == '__main__':
    unittest.main()
