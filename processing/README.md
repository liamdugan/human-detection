# RoFT Data Processing
The python files in this folder contain the code that we used to process the data from our database as well
as filter out generations that we deemed not worthy to be included in the final dataset

We are only releasing the final version of the dataset for privacy reasons so neither of these scripts are runnable. However, we include them with our code release for full transparency.

## Database Processing
The `process_database_dump.py` file takes in the raw database JSON file and cleans it up into a dataframe to aid in filtering. It also processes column values and renames column headers so that they make more sense and are easier to read.

In addition, this file also attaches the survey responses to the relevant participants and removes all participants who did not explicitly agree to be a part of our study. 

## Filtering
The `filter_data.py` file takes in the processed data and (as discussed in Appendix C) filters for the following things

1. We filter out all annotations where the player gradually started to guess the same value repeatedly (i.e. all spans of 5 or more annotations with the exact same value)
2. We filter out extra "all-human" examples from the Recipes dataset due to a mistake on our part. This ensures that the distribution across sentence boundaries is relatively equal (as one would expect). 
3. Finally we filter out all annotations from annotator 4334 as we know they used Javascript exploits instead of completing the game as intended

Again, similar to `process_database_dump.py` this script is included for the sake of transparency and is not runnable.