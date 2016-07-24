##########################################################################################################################    
## MACHINE LEARNING SECTION
##########################################################################################################################    
from data import OcrData
from cifar import Cifar
    #
    ####################################################################
    ## 1- GENERATE MODEL TO PREDICT WHETHER AN OBJECT CONTAINS TEXT OR NOT
    ####################################################################
    #
    # CREATES AN INSTANCE OF THE CLASS LOADING THE OCR DATA 
data = OcrData('E:/1_MachineLearning/ImageTextRecognition-master/ocr-config.py')
    #
    # GENERATES A UNIQUE DATA SET MERGING NON-TEXT WITH TEXT IMAGES
data.merge_with_cifar()
    #
    # PERFORMS GRID SEARCH CROSS VALIDATION GETTING BEST MODEL OUT OF PASSED PARAMETERS
data.perform_grid_search_cv('linearsvc-hog')
    #
    # TAKES THE PARAMETERS LINKED TO BEST MODEL AND RE-TRAINS THE MODEL ON THE WHOLE TRAIN SET
data.generate_best_hog_model()
    #
    # TAKES THE JUST GENERATED MODEL AND EVALUATES IT ON TRAIN SET
data.evaluate('E:/1_MachineLearning/ImageTextRecognition-master/linearsvc-hog-fulltrain2-90.pickle')


    ####################################################################
    ## 2- GENERATE MODEL TO CLASSIFY SINGLE CHARACTERS
    ####################################################################
    #
    # CREATES AN INSTANCE OF THE CLASS LOADING THE OCR DATA 
#data = OcrData('E:/1_MachineLearning/ImageTextRecognition-master/ocr-config.py')
    #
    # PERFORMS GRID SEARCH CROSS VALIDATION GETTING BEST MODEL OUT OF PASSED PARAMETERS
#data.perform_grid_search_cv('linearsvc-hog')
    #
    # TAKES THE PARAMETERS LINKED TO BEST MODEL AND RE-TRAINS THE MODEL ON THE WHOLE TRAIN SET
#data.generate_best_hog_model()
    #
    # TAKES THE JUST GENERATED MODEL AND EVALUATES IT ON TRAIN SET
#data.evaluate('E:/1_MachineLearning/ImageTextRecognition-master/linearsvc-hog-fulltrain36-90.pickle')