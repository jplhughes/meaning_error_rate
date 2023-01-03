# Meaning Error Rate

Meaning Error Rate (MER) is a metric that aims to evaluate the quality of a speech recognition system based on how well it preserves the meaning of the original audio. MER aims to improve upon Word Error Rate (WER), which is the primary quality metric used by speech providers, by taking into account the severity of changes in meaning when comparing the recognized transcript to the reference. To do this, MER uses large language models, chain of thought reasoning, and majority voting. See the blog on the [Future of WER](https://www.speechmatics.com/company/articles-and-news/the-future-of-word-error-rate) for more details.

This repository allows you to easily curate a prompt for meaning error rate that includes examples for minor, standard and serious errors. This method employs few-shot learning, allowing the OpenAI's GPT-3 to understand the task and its structure from a small number of examples. It also uses chain of thought reasoning to improve the model's predictions and provide insight into its decision-making process. Once the response from the OpenAI API is recieved, it automatically extracts the severity of error and reason for the input utterance of interest. This can then be collated across many utterances and the MER can be calculated.


## Usage

To use MER, you can install the necessary environment and dependencies using the following commands:

```
mkdir ~/env
virtualenv --python python3.8 ~/env/venv_mer
source ~/env/venv_mer/bin/activate
make deps
```

You will also need to export your OpenAI API key:
```
export OPENAI_API_KEY=<api_key>
```

You can then run unit tests to ensure that everything is working correctly:

```
make unittest
```

To test MER on a specific dataset, you can use the command below. This uses a json file that contains the reference and recognised transcript path. It can also contain the severity of error and reasons so you can get the ground truth MER compared to the LLM generated MER.

```
python -m mer.test \
  --test_json ./config/test.json \
  --prompt_config_path ./config/prompt.json \
  --output_json ./results.json
```

Alternatively, you can provide "dbl" files that list the text files in them. This can be a quick solution especially if you don't have the human labelled severity of errors and reasons along with each utterance.
```
python -m mer.run \
  --ref_dbl unittests/data/ref.dbl \
  --rec_dbl unittests/data/rec.dbl \
  --prompt_config_path ./config/prompt.json \
  --output_json ./results.json
```

Note: that you need the reference and recognised transcript for each utterance in your testset in order to calculate the MER, just like you do for WER. You can prepare you data in simple dbl files for reference and recognised or in a json format as above. Please see the unittests to understand the differences.

You can also use MER to prepare data from the FairSpeech dataset. To do this, you will need to download the dataset as a CSV file from the Stanford Policy Lab GitHub repository, and then use the following command to convert it to JSON format:
```
python3 -m dataprep.csv_2_json --csv_path <path_to_csv> --json_out_path ./CORAAL_transcripts.json
```
