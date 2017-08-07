## Package Requirements
This script should be run on Python 3.5+ and was tested in Spyder IDE (with [Anaconda] distribution). Assuming pre-installation of these following packages (using *pip install* command or similar methods):

  - NumPy
  - Pandas
  - SciPy
  - Scikit-learn
  - Scikit-multilearn (optional, for experiment)

## Detailed Instructions

This project was divided into two smaller parts with different approaches and implementations to the main problem. They are to be run separately and produce several distinguished results (please refer to code comments for more details). Here are some basic steps and correspondent outputs for the programs:

1. Include **yvr-weather** (for weather data) and **katkam-scaled** (for weather images) folder in your running directory.
2. Run **rescale_image.py** to resize images to half the original size (to reduce runtime while keeping accuracy). This will output a new folder called **katkam-rescaled** in the same directory that contains new images.
3. Execute **predict\_weather\_pdnguyen.py** to produce some output as format shown on screen (only useful information).
4. (hey Jag add something here, for your parts)

## Further Notes

In the **predict\_weather\_pdnguyen.py**, there are three major parts which carry out different tasks (**Part I** for reading input, **Part II** for processing data and **Part III** for training model). They are to be run consecutively once the program is executed. Code commenting conventions are followed: ### to mark major parts, ## to mark main points in each part and # to mark explanations and notes. 

Single lines with # mark in front are debugging code and should not be worried about. Execution time might vary between 30-40s so please be patient and wait for it to finish running! 

(Jag add something in here too, if you want)



[anaconda]: https://www.continuum.io/anaconda-overview

















