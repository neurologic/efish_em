# efish_em
This repo  includes custom notebooks and scripts that are created for processing and analyzing data in the electric fish ELL connectome project 

## eCREST
Connectome Reconstruction and Exploration Simple Tool, or CREST, is a simple GUI tool (written and hosted by Alex Shapson-Coe, @ashaponscoe) that enables users to (1) proofread biological objects and (2) identify individual network pathways, connections and cell types of interest, in the Neuroglancer interface.

eCREST is written in Python and makes extensive use of the [Neuroglancer Python API](https://connectomics.readthedocs.io/en/latest/external/neuroglancer.html).

eCREST was forked from [CREST](https://github.com/ashapsoncoe/CREST) and modified into a CLI-based interface for utilizing CREST methods.  Additional methods were added to meet the needs of this specific project.


# PC (windows)  

(getting things running on a Mac will be very similar except, after downloading and installing the Anaconda Navigator .exe for mac, you would do all of the command prompts through the native Mac Terminal app rather than through Anaconda PowerShell Prompt -- though both should work, it is easier to use Terminal)

## Installing eCREST

### Anaconda

Make sure that Anaconda is installed. If not already installed, you can get the individual distribution [here](https://www.anaconda.com/products/distribution).

### Create a local environment for using eCREST tools

1. Launch **Anaconda Navigator**
2. Launch **Powershell Prompt** from the main Navigator GUI. 
		> A *command* window will pop up (the line ending with ```>``` is where you will enter the commands in the following steps). 
3. In the Powershell Prompt screen, run the command ```conda create --name ell```. To "run a command", type the command (exactly as it is written) after the ```>``` on the last line of the Prompt screen -- then press the **Enter** key on your keyboard (Note that the computer mouse does not help you navigate these text commands... use the arrow keys to edit). 
	> You can name the environment anything you want... just replace "*ell*" with the name you want (and use your name in place of "ell" for all following steps).  
4. type "Y" and hit enter if prompted to do so (unless you have a reason to say "N")
	> Repeat this step after any of the "run command" steps as prompted.
5. run the command ```conda activate ell```
	> the beginning of the Powershell Prompt command line should now start with ```(ell)``` instead of ```(base)```
6. Run the following command lines in order: 
	- ```conda install -c anaconda git```
	- ```pip install neuroglancer igraph``` 
		> NOTE: if you want even more functionality and access to analysis notebooks, include cloud-volume and igneous: ```pip install neuroglancer igraph cloud-volume igneous ```
	- ```conda install scipy matplotlib seaborn```
	- ```conda install -c conda-forge cairocffi pycairo google-cloud-storage```

	
	conda install -c anaconda git
pip install neuroglancer igraph
conda install scipy matplotlib pandas seaborn
conda install -c conda-forge cairocffi pycairo google-cloud-storage tqdm

### Clone this repository to your computer

1. In the Powershell Prompt that you have been using, run the command ```cd <path-to-where you want the repository>```.  
2. Run the command ```git clone https://github.com/neurologic/eCREST.git```

#### Keeping things up-to-date

In the future, you can run ```git pull``` from within the eCREST directory to make sure you have the latest version of scripts from the repo. However, you will first need to stash with ```git stash``` (and delete stash if want: ```git stash clear```)

## Running jupyter lab notebooks 

1. Launch **Anaconda Navigator**
2. Activate the **ell** environment from the Navigator main window by selecting it from the dropdown menu of environments.  
3. Launch **Powershell Prompt** from the main Navigator GUI. 
4. "**Change Directory**" to the cloned eCREST repository (from step 7 of the install... the path now includes eCREST directory itself).
5. run the command ```jupyter lab```


## Running eCREST from the command line 
(we don't currently use this method, but it was the main use case implemented with CREST)

### Basic Steps to Launch

1. Launch **Anaconda Navigator**
2. Activate the **ell** environment from the Navigator main window by selecting it from the dropdown menu of environments.  
3. Launch **Powershell Prompt** from the main Navigator GUI. 
4. "**Change Directory**" to the cloned eCREST repository (from step 7 of the install... the path now includes eCREST directory itself).
5. run the command ```python eCREST_stable.py```

> Note that if your data files are on a different drive than your main computer drive (where Anaconda is installed and run), then you will need to do steps 2 and 3 in opposite order (with a "cd" command in between) and activate the environment from the Prompt with the command ```conda activate ell```.

### Possible Errors and Solutions

<details><summary>ImportError: cannot import name 'COMMON_SAFE_ASCII_CHARACTERS' from 'charset_normalizer.constant'</summary>
	For example, this error might happen when you try to launch eCREST.py or load a cell from file once it is running.  
	**Solution** to try:
	```conda install -c anaconda chardet```
</details>

# Mac

## Install and Run

Most steps are the same. The only thing different is that, once Anaconda has been installed, you do not need to use Anaconda Navigator (though I suppose you can). 
Instead, use the native "Terminal" application on your mac computer (comes installed on all macs). After installation, Anaconda's python installation should be the default that is called from the Terminal. "Git" is also likely already installed on your mac... so you can probably skip the ```conda install git``` step in the setup.

