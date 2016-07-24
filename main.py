from userimageski import UserData

if __name__ == '__main__':
   
    ##### the following code includes all the steps to get from a raw image to a prediction.
    ##### the working code is the uncommented one. 
    ##### the two pickle models which are passed as argument to the select_text_among_candidates
    ##### and classify_text methods are obviously the result of a previously implemented pipeline.
    ##### just for the purpose of clearness below the code is provided. 
    ##### I want to emphasize that the commented code is the one necessary to get the models trained.
    
    # creates instance of class and loads image    
    
    user = UserData('lao.jpg')
    # plots preprocessed image
    user.plot_preprocessed_image()
    print 'preprocessed image!!!'
    # detects objects in preprocessed image
    candidates = user.get_text_candidates()
    print 'get_text_candidates!!!'
    # plots objects detected
    user.plot_to_check(candidates, 'Total Objects Detected')
    print 'objects detected!!!'
    # selects objects containing text
    maybe_text = user.select_text_among_candidates('E:/1_MachineLearning/ImageTextRecognition-master/linearsvc-hog-fulltrain2-90.pickle')
    print 'select text!!!'
    # plots objects after text detection
    user.plot_to_check(maybe_text, 'Objects Containing Text Detected')
    print 'obj contains text detected!!!'
    # classifies single characters
    classified = user.classify_text('E:/1_MachineLearning/ImageTextRecognition-master/linearsvc-hog-fulltrain36-90.pickle')
    print 'classify text'
    # plots letters after classification 
    user.plot_to_check(classified, 'Single Character Recognition')
    print 'single character recognition'
    # plots the realigned text
    user.realign_text()
  
