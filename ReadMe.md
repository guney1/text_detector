# Steps to run (Mac M1)

* Change directory to where project files exist
  * `cd ./text_detector`

* Generate environment (Python 3.10, install venv with brew)
  * `python3 -m venv text_detection_env`

* Install requirements
  * `pip install -r ./requirements.txt`

## Run

* Activate the environment
  * `source text_detection_env/bin/activate`

* Run it by giving all the flags in main file
  * `python main.py -id test_folder -ujp example_urls.json -tjp example_target_brands.json -p 1`

# How to make it work in Linux

* Setup virtualenv

* From `requirements.txt`, remove below lines before executing `pip install -r ./requirements.txt`
    - tensorboard==2.11.2
    - tensorboard-data-server==0.6.1
    - tensorboard-plugin-wit==1.8.1
    - tensorflow-estimator==2.11.0
    - tensorflow-macos==2.11.0

* Execute `pip install -r ./requirements.txt`

* Execute `pip install tensorflow==2.11.*`