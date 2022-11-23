# wellytrees

## About
Code and environment for script developed as part of Wellington Urban Tree project at Manaaki Whenua/Landcare Research. 

The code takes in two input data sets:
1. Polygon layer of individual tree crowns across Wellington City, New Zealand produced by a deep learning model. 
2. Point layer of individual records of trees managed by Wellington City Council

This script has been developed to link tree point records with tree crown polygon records and assess the certainty in the relationship. It does this by calculating the following indicators to assess confidence in the relationship of the tree point record with the associated tree crown record, then combining the indicators into a single certainty score for each tree record.
- Within crown, or not. 
- Number of points within crown. 
- Position within crown
- Distance to nearest crown (if not within one)

The main goal was to associate the information held within tree point records (such as species and age) with tree crowns identified from remote imagery, dependent on the certainty of the relationship. Such data could then be used as training data in the future. Information held within tree crown records (such as height, obtained through remote sensing techniques) can also be used to update the tree point records. 

## Outputs
The script can be used and adapted to produce the following outputs:
- Report in the form of a HTML table, containing data from both tree point and crown records plus additional contextual imagery
- Various excel file outputs where desired
- Updated version of input tree dataset with added information from tree crown dataset for each tree where certainty threshold met. 

The script has multiple parts where the user can make specifications based on individual requirements e.g subsetting trees by species, age, or area, choosing how many records to include in output record, deciding on certainty threshold to use, adjusting weights for final certainty score etc. 

## Example
Example of the HTML table generated, containing info from both input datasets plus calculated confidence indicators, overall certainty score, and visualisation. 
![image](https://user-images.githubusercontent.com/115961095/203456313-a6c3c93d-11b6-4903-9256-d47c0086add6.png)
