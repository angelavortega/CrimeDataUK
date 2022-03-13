# CrimeDataUK
This experiment was made to demonstrate how data can be significantly influenced by variables describing human characteristics, like ethnicity or gender, suggesting to have caution at the moment of implementing and using a model, due to the controversial results that could give. A machine learning model of a decision tree regressor was implemented on two datasets of the UK Government with information of all people that were represented on the Criminal Justice System of England and Wales. Both original datasets are inside of the folder "data".

## Setting up the project
---
### **Step 1) Create a virtual enviroment.**
On a Terminal, install the package for python virtual enviroment using pip.
```bash
python -m pip install virtualenv
```

Set up the virtual enviroment in the folder named **pyvirtual**
```bash
python -m virtualenv venv
```
Activate the virtual env in the terminal to be used for execution:

- *on Linux*
```bash
source venv/bin/activate
```
* *on windows*
```bash
venv/bin/activate
```

### **Step 2) Install all the required dependencies.**
Once the virtual environment is activated, proceed to install all the required packages. pip offers a way to automatically install all the dependencies used with a requirements file.

To install the requirements, follow:
```bash
python -m pip install -r requirements.txt
```

If a new dependency is added to the code, an update to the requirements is required. To do so follow:
```bash
python -m pip freeze > requirements.txt
```

## Running the project
---
In line 13 of the file, there is a section, "Changeable variables", where you have the option to change variables that will influence the models' performance and results. 
To run the project, you need to execute the decision tree file:
```bash
python decision_Tree_Regresion.py
```
## Good to know
---
1. In the folder "dataValues" one can find CSV files containing how data is seen after processing it, and also txt files containing all dummy values that were used for the creation of the columns in the processed data.

2. In the folder "input" there is an xlsx file where the introduced independent variables for Observed and Self-Identified datasets can be modified to change the outcome of the machine learning predictions. All predictions will be stored in the "prediction" folder.

3. Results of the model will be stored in three different files. The "results.txt" will contain both information of the Observed and Self-Identified datasets, including information of all the metrics and the independent variables that were used. The file "Model_Results_(Ob or Self).txt" will contain information just of the specific dataset that was chosen, including information of the metrics, independent variables, the chosen dependent variable, test size, and random state. 

## Resources
---
References to the datasets, python libraries, and books that helped to the development of this project can be found here
-Ministry of Justice of UK. (2013, November 14). Statistics on Race and the Criminal Justice System 2012. Statistics on Race and the Criminal Justice System 2012. Retrieved February 5, 2022, from https://www.gov.uk/government/statistics/statistics-on-race-and-the-criminal-justice-system-2012
-Pandas. (2022). Pandas - Get Dummies. Pandas. Retrieved February 2, 2022, from https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html
-Scikit Learn. (2021). 1.10. Decision Trees — scikit-learn 1.0.2 documentation. Scikit-learn. Retrieved February 6, 2022, from https://scikit-learn.org/stable/modules/tree.html#tree
-Berk, R. A. (2017). Statistical Learning from a Regression Perspective (2nd ed.). Springer.
-Ertel, W. (2017). Introduction to Artificial Intelligence (2nd ed.). Springer.
-Harrell, F. J. E. (2015). Regression Modeling Strategies (2nd ed.). Springer.
-Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning. Springer.
-Heumann, C., & Schomaker, M. (2016). Introduction to Statistics and Data Analysis. Springer.
-Igual, L., & Seguí, S. (2017). Introduction to Data Science. Springer.
-James, G., Witten, D., Hastie, T., & Tibshirani, R. (2017). An Introduction to Statistical Learning (8th ed.). Springer.
