import pandas as pd
from zipfile import ZipFile
from typing import Dict, List
import random
import os
from tempfile import TemporaryDirectory

def create_submission(output_file_path : str, submission_dictionary : Dict[int, str], base_keys : List[int]) -> None:
    """Function that validates the submission data types and schema and zip it to be ready from submission
    
    Parameters
    ----------
    output_file_path : str 
        The locaiton and file name you want to save the zip file at, ex : "/home/user/submission_123.zip"
    submission_dictionary : dict[int, str]
        dictionary of int keys (example_id) and string values (summary)
    base_keys: list[int]
        list of keys of the original unlabeled validation set

    
    Returns
    -------
    None
    """
    #assertions
    assert all(isinstance(i, int) for i in submission_dictionary.keys()), "Make sure example_ids elements (key of submission_dictionary) are of type int"
    assert all(isinstance(i, str) for i in submission_dictionary.values()), "Make sure summary elements (value of submission_dictionary) are of type str"
    assert all(isinstance(i, int) for i in base_keys), "Make sure base_keys elements is of type int"
    
    diff_sub = set(submission_dictionary.keys()) - set(base_keys)
    diff_base = set(base_keys) - set(submission_dictionary.keys())
    
    assert len(diff_sub) == 0, f"Keys {diff_sub} is in submission but not in base_keys"
    assert len(diff_base) == 0, f"Keys {diff_base} is in base_keys but not in submission"
    
    #saving
    final_submission = pd.DataFrame(submission_dictionary.items(), columns=['example_id', 'summary'])
    
    if final_submission.example_id.dtype != 'int64' :
        final_submission.example_id = final_submission.example_id.astype(int)
    
    assert len(final_submission[final_submission.summary.isna()]) == 0, f"summaries with the example_id = {final_submission[final_submission.summary.isna()].example_id.values.tolist()} is NaN"
    assert len(final_submission[final_submission.example_id.isna()]) == 0, f"example_ids with the following index = {final_submission[final_submission.example_id.isna()].index.tolist()} is NaN"
    
    with TemporaryDirectory(dir=".") as tmpdirname:
        os.chdir(tmpdirname)
        jsonl_name = "predictions.jsonl"
        final_submission.to_json(jsonl_name, lines=True, orient='records', force_ascii=False)
        with ZipFile(output_file_path, "w") as zip_file:
            zip_file.write(filename = jsonl_name)
            print(f"Submission of {jsonl_name} as .zip saved at {output_file_path}")
        os.chdir("..")