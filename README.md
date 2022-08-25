# meaning_error_rate

Word Error Rate (WER) is the primary quality metric for all speech providers. Engineers use it to make almost every research decision and customers use it to evaluate vendors. But it has many flaws…

Who cares about an extra um, a missing hyphen, or an alternative spelling. Or how about a difference in written numerical formatting (1,000 vs 1000 or eleven vs 11)? These all increase WER but do not affect meaning in the slightest.

How about omitting the word ‘suspected’ before murderer? Or predicting a wrong word that changes the sentiment? An ideal metric would take these minor and serious errors into account and score a system appropriately.

That is where Meaning Error Rate (MER) comes in - it considers each utterance as a “chunk of meaning” and tries to take into account the severity of changes in meaning when comparing the recognised transcript to the reference. It uses the power of large language models combined with chain of thought reasoning and majority voting (alos known as self consistency).

## How to run
```
# Install environment
mkdir ~/env
virtualenv --python python3.8 ~/env/venv_mer
source ~/env/venv_mer/bin/activate
make deps

# Export OpenAI API Key
export OPENAI_API_KEY=<api_key>

# Ensure unittests run
make unittest

# Test with dbls
python -m mer.run --ref_dbl unittests/data/ref.dbl --rec_dbl unittests/data/rec.dbl --prompt_config_path ./config/prompt.json

# Test on testset
python -m mer.test --test_json ./config/test.json --prompt_config_path ./config/prompt.json --output_json ./results.json

# Fair speech dataprep
Download FairSpeech csv from https://github.com/stanford-policylab/asr-disparities/blob/master/input/CORAAL_transcripts.csv
python3 -m dataprep.csv_2_json --csv_path <path_to_csv> --json_out_path ./CORAAL_transcripts.json
```
