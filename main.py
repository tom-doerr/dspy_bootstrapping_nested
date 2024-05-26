#!/usr/bin/env python3

import re
import os
import sys
os.environ["DSP_CACHEDIR"] = 'local_cache'
# dspy_cachebool = False
dspy_cachebool = True
os.environ["DSP_CACHEBOOL"] = str(dspy_cachebool)

import dspy
from dspy.predict import Retry
from dspy.datasets import HotPotQA
from dspy.teleprompt import BootstrapFewShotWithRandomSearch
from dsp.utils import deduplicate
from dspy.evaluate.evaluate import Evaluate
from dspy.primitives.assertions import assert_transform_module, backtrack_handler
import numpy as np
import random
from tqdm import tqdm

# set random seed
random.seed(0)

lm = dspy.HFClientVLLM(model="microsoft/Phi-3-medium-128k-instruct", port=38242, url="http://localhost", max_tokens=800)
dspy.settings.configure(lm=lm, trace=[], temperature=0.7)

CSV_OUTPUT_DIR = 'parameter_study'
os.makedirs(CSV_OUTPUT_DIR, exist_ok=True)

def generate_dataset():
    # for i in range(150):
    # tqdm
    dataset = []
    for i in tqdm(range(150)):
        random_number = random.randint(0, 1000000000000000000000000000)
        location = random.choice(['LA', 'New York', 'Berlin', 'Munich', 'Hongkong', 'Tokyo', 'London', 'Paris', 'Singapore', 'Sydney'])
        business_sector = random.choice(['tech', 'finance', 'health', 'education', 'entertainment', 'food', 'travel', 'real estate', 'fashion', 'sports'])
        business_stage = random.choice(['early stage', 'growth stage', 'mature stage', 'decline stage'])
        business_size = random.choice(['small', 'medium', 'large'])
        business_age = random.choice(['new', 'established', 'old'])
        generated_text = lm(f'Random input: {random_number} Location: {location} Business sector: {business_sector} Business stage: {business_stage} Business size: {business_size} Business age: {business_age}\n\nGenerate a very long company description for a company with the abouve attributes. Generated text: ', max_tokens=1000)[0]
        print("generated_text:", generated_text)
        dataset.append(dspy.Example(company_description=generated_text).with_inputs('company_description'))

    return dataset

generate_dataset()


class GenerateMail(dspy.Signature):
    """Generate an engaging email that effectively makes it clear to the recepient why they specifically should buy a new company office."""
    company_description = dspy.InputField(desc="Company description")
    mail = dspy.OutputField(desc="Generated email")


class AssessMail(dspy.Signature):
    """Assess the quality of a email along the specified dimension."""
    assessed_text = dspy.InputField()
    assessment_question = dspy.InputField()
    assessment_answer = dspy.OutputField(desc="Yes or No")

def is_assessment_yes(assessment_answer):
    """Check if the first word of the assessment answer is 'yes'."""
    return assessment_answer.split()[0].lower() == 'yes'

class Emailer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_mail = dspy.ChainOfThought(GenerateMail)

    def forward(self, company_description):
        print("company_description:", company_description)
        generation_output = self.generate_mail(company_description=company_description)
        generated_mail = generation_output.mail
        generated_mail = generated_mail.split('---')[0]

        return dspy.Prediction(mail=generated_mail)


def get_sum_true_false(logprobs):
    true_strs = ["true", "True", "0"]
    false_strs = ["false", "False", "1"]
    true_sum = 0
    false_sum = 0
    for logprob_str in logprobs['top_logprobs'][0]:
        if logprob_str in true_strs:
            true_sum += np.exp(logprobs['top_logprobs'][0][logprob_str])
        elif logprob_str in false_strs:
            false_sum += np.exp(logprobs['top_logprobs'][0][logprob_str])

    return true_sum, false_sum


def get_logprob_score(prompt):
    response = lm(prompt, logprobs=5, max_tokens=2)
    true_sum, false_sum = get_sum_true_false(response[0]['logprobs'])
    score = true_sum / (true_sum + false_sum + 1e-6)
    return score


def great_mail_metric(gold, pred, trace=None, return_individual_scores=False):
    prompts = {
            'good_mail': f'Email:\n{pred.mail}\n\nDoes the assessed text make for a self-contained, engaging email? Answer false if it is not a great mail.\nanswer = {{"great_mail_bool": ',
            'professional': f'Email:\n{pred.mail}\n\nDoes the assessed email sound professional? Answer false if it is not professional sounding.\nanswer = {{"professional_email_bool": ',
            'faithful': f'Email:\n{pred.mail}\n\nIs the assessed text grounded in the context? Say false if it includes significant facts not in the context.\nanswer = {{"faithful_bool": ',
            }

    scores = {}
    for prompt_key in prompts:
        prompt = prompts[prompt_key]
        score = get_logprob_score(prompt)
        scores[prompt_key] = score
        print(f'{prompt_key}: {score}')

    avg_score = sum(scores.values()) / len(scores)
    scores['avg_score'] = avg_score
    print("avg_score:", avg_score)
    if return_individual_scores:
        return scores
    else:
        return avg_score



TRAIN_SIZE = int(2**7)
DEV_SIZE_0 = int(2**2)
DEV_SIZE_1 = int(2**4)
# TRAIN_SIZE = int(2**10)
# DEV_SIZE_0 = int(2**2)
# DEV_SIZE_1 = int(2**4)
dataset = generate_dataset()
random.shuffle(dataset)

def run_optimization(evaluate=True):
    num_candidate_programs = 6
    max_bootstrapped_demos = 4
    emailer = assert_transform_module(Emailer().map_named_predictors(Retry), backtrack_handler)
    nesting_scores = []
    if evaluate:
        trainset = dataset[:TRAIN_SIZE]
        devset_0 = dataset[TRAIN_SIZE:TRAIN_SIZE+DEV_SIZE_0]
        devset_1 = dataset[TRAIN_SIZE+DEV_SIZE_0:TRAIN_SIZE+DEV_SIZE_0+DEV_SIZE_1]
        evaluate = Evaluate(metric=great_mail_metric, devset=devset_1, num_threads=32, display_progress=True, display_table=5)
        score_start = evaluate(emailer)
        print("score_start:", score_start)
        nesting_scores.append({"nesting_level": -1, "score": score_start})

    compiled_with_assertions_mailer = None
    num_nesting_levels = 20
    for nesting_level in range(num_nesting_levels):
        print("nesting_level:", nesting_level)
        random.shuffle(dataset)
        trainset = dataset[:TRAIN_SIZE]
        devset_0 = dataset[TRAIN_SIZE:TRAIN_SIZE+DEV_SIZE_0]
        devset_1 = dataset[TRAIN_SIZE+DEV_SIZE_0:TRAIN_SIZE+DEV_SIZE_0+DEV_SIZE_1]
        teleprompter = BootstrapFewShotWithRandomSearch(metric = great_mail_metric, max_bootstrapped_demos=max_bootstrapped_demos, num_candidate_programs=num_candidate_programs, num_threads=32, metric_threshold=None)
        compiled_with_assertions_mailer = teleprompter.compile(student=emailer, trainset=trainset, valset=devset_0, teacher=compiled_with_assertions_mailer)
        if evaluate:
            score = evaluate(compiled_with_assertions_mailer)
            print("score_start:", score_start)
            print("score:", score)
            nesting_scores.append({"nesting_level": nesting_level, "score": score})
        print('=== Nesting Scores ===')
        for nesting_score in nesting_scores:
            print(nesting_score)

    return compiled_with_assertions_mailer


def main():
    EVALUATE = True
    mailer_pipeline = run_optimization(evaluate=EVALUATE)

if __name__ == '__main__':
    main()
