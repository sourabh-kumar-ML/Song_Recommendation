# Song_Recommendation
Recommending songs based on the mood of user.

Task : Recommending music to user according to the mood detected.

Data : Fer 2013

Description of files:
  
  1.calm.cav,happy.csv,sad.csv,energetic.csv ==> files containing names of music according to genere. These songs as classified based on multiple features.
  2.Facial.py contains various deep learning models.
        
        1.CNN
        2.VGG16
        3.Resnet 
        4.Xception
        5.Conv_LSTM
  3.haarcascade_frontalface_default.xml is used in opencv to detect faces.
  4.training.ipynb uses models described in Facial.py and uses best model
  5.tinkter_GUI.py is graphical interface which combines model and user for preditive analysis.
