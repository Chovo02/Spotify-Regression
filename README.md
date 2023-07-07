<div align="center" id="top"> 
  <img src="./.github/app.gif" alt="Spotify Regression" />

  &#xa0;

  <!-- <a href="https://github.com/ChovoITS/Spotify-Regression">Demo</a> -->
</div>

<h1 align="center">Spotify Regression</h1>

<p align="center">
  <img alt="Github top language" src="https://img.shields.io/github/languages/top/{{ChovoITS}}/Spotify-Regression?color=56BEB8">

  <img alt="Github language count" src="https://img.shields.io/github/languages/count/{{ChovoITS}}/Spotify-Regression?color=56BEB8">

  <img alt="Repository size" src="https://img.shields.io/github/repo-size/{{ChovoITS}}/Spotify-Regression?color=56BEB8">

  <img alt="License" src="https://img.shields.io/github/license/{{ChovoITS}}/Spotify-Regression?color=56BEB8">

  <!-- <img alt="Github issues" src="https://img.shields.io/github/issues/{{YOUR_GITHUB_USERNAME}}/spotify-regression?color=56BEB8" /> -->

  <!-- <img alt="Github forks" src="https://img.shields.io/github/forks/{{YOUR_GITHUB_USERNAME}}/spotify-regression?color=56BEB8" /> -->

  <!-- <img alt="Github stars" src="https://img.shields.io/github/stars/{{YOUR_GITHUB_USERNAME}}/spotify-regression?color=56BEB8" /> -->
</p>

<!-- Status -->

<!-- <h4 align="center"> 
	🚧  Spotify Regression 🚀 Under construction...  🚧
</h4> 

<hr> -->

<p align="center">
  <a href="#dart-about">About</a> &#xa0; | &#xa0; 
  <a href="#sparkles-features">Features</a> &#xa0; | &#xa0;
  <a href="#rocket-technologies">Technologies</a> &#xa0; | &#xa0;
  <a href="#white_check_mark-requirements">Requirements</a> &#xa0; | &#xa0;
  <a href="#checkered_flag-starting">Starting</a> &#xa0; | &#xa0;
  <a href="#memo-license">License</a> &#xa0; | &#xa0;
  <a href="https://github.com/{{YOUR_GITHUB_USERNAME}}" target="_blank">Author</a>
</p>

<br>

## :dart: About ##

Spotify-Regression is a project made for a machine learning exam, the goal is, given a dataset containing certain informations about songs from spotify, to create a model that can predict the popularity of a song.

Not only you will be able to train the model, you will also be able to save it locally on your machine and use it to predict the popularity of a song that you choose by providing the link.

Feel free to contribute :smile:

## :sparkles: Features ##

:heavy_check_mark: Model Selection;\
:heavy_check_mark: Export a model;\
:heavy_check_mark: Try Pre Trained Model;

## :rocket: Technologies ##

The following tools were used in this project:

- [Optuna](https://optuna.org/)
- [Python](https://www.python.org/)
- [scikit-learn](https://scikit-learn.org/stable/)

## :white_check_mark: Requirements ##

Before starting :checkered_flag:, you need to have python and all the dependencies installed.

## :checkered_flag: Starting ##

```bash
# Clone this project
$ git clone https://github.com/{{ChovoITS}}/Spotify-Regression

# Navigate to the project
$ cd Spotify-Regression

#Install the required dependencies
# for pip
$ pip install -r requirements.txt

# for yarn
$ yarn install

# for conda
$ conda install --file requirements.txt

# after installing the dependencies, you have to create a .env file in the root of the project 

$ touch .env

```

# Populating the .env file
## The .env file is used to store the environment variables, which are used to connect to the database and to the spotify api

Required variables:

MY_SQL_CONNECTION = "your mysql connection" used for the model selection and later for export and import of the models

CLIENT_ID_LIST = "your spotify client id" can be more than 1, separated by a comma

CLIENT_SECRET_LIST = "your spotify secret id" can be more than 1, separated by a comma

JSON_PATH = "path to the json file used as cache for the transformers" if left empty, the default path will be used (data\\feats.json)

# Run the project
## Model Selection

This module will start a model selection process, it will try different models and different hyperparameters to find the best model for the dataset

by running the following command you will be able to choose between 7 different models, 2 different transformers methods and 3 different scores

The models will be saved on the database specified in the .env file

For the first run it will probably take a lot of time, as the cache will be empty and you'll need to retrieve all the informations, but once the cache is filled, the process will take a couple of seconds

If you want to see the progress bar you can set the verbosity to 1 as a parameter for the transformer


```bash
python .\ml-model\Model_Selection.py --help
```
This command will show you the help page that will explain how to pass all the parameters to the script

Note: if want the verbose output, you need to add it as parameter to the trasformer

## Export Model

This module will export take from the database the best parameters of a specified study and it will export it as a pickle file, making it usable by the Test_Model module

The model will be saved in the folder "data" with the name formed as "{model_name}_{with or without feat}_{score}"
```bash
python .\ml-model\Export_Model.py --help
```
This command will show you the help page for the export model script, wich will explain how to pass all the parameters to the script

## Try Pre Trained Model

This module will take a model exported with the Export_Model module and will allow you to test it by providing a link to a spotify song

It will then predict the popularity of the song and will show you the result

```bash
python .\ml-model\Test_Model.py --help
```
This command will show you the help page for the test model script, wich will explain how to pass all the parameters to the script

## :memo: License ##

This project is under license from MIT. For more details, see the [LICENSE](LICENSE.md) file.


Made with :heart: by <a href="https://github.com/{{ChovoITS}}" target="_blank">{{Matteo Civita, Mattia Rossini, Davide Soltys, Luca Pugno, Niccolò Ballabio}}</a>

&#xa0;

<a href="#top">Back to top</a>
