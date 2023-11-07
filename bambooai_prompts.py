default_example = """
import pandas as pd

# Identify the dataframe `df`
# df has already been defined and populated with the required data

# Call the `describe()` method on `df`
df_description = df.describe()

# Print the output of the `describe()` method
print(df_description)
"""

code_generator_user_df = """
You have been presented with a pandas dataframe named `df`.
The dataframe df has already been defined and populated with the required data.
The result of `print(df.head(1))` is:
{}.
Return the python code that accomplishes the following tasks: {}.
Approach each task from the list in isolation, advancing to the next only upon its successful resolution. 
Strictly adhere to the prescribed instructions to avoid oversights and ensure an accurate solution.
For context, here is the output of the previous task: {}.
Always include the import statements at the top of the code.
Always include print statements to output the results of your code.
Always use the backticks to enclose the code.

Example Output:
```python
{}
```
"""

