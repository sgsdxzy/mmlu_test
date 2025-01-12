import argparse
import openai
import os
import numpy as np
import pandas as pd
import time

choices = ["A", "B", "C", "D"]


def softmax(x):
    z = x - max(x)
    numerator = np.exp(z)
    denominator = np.sum(numerator)
    softmax = numerator / denominator
    return softmax


def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s


def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
    prompt += "\nAnswer:"
    messages = [
        {
            "role": "user",
            "content": prompt,
        },
    ]
    if include_answer:
        messages.append(
            {
                "role": "assistant",
                "content": df.iloc[idx, k + 1],
            }
        )
    return messages


def gen_messages(train_df, subject, k=-1):
    prompt = (
        "The following are multiple choice questions (with answers) about{}.".format(
            format_subject(subject)
        )
    )
    messages = [
        {
            "role": "system",
            "content": prompt,
        },
    ]
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        messages += format_example(train_df, i)
    return messages


def eval(args, subject, model, dev_df, test_df):
    cors = []
    all_probs = []
    preds = []
    answers = choices[: test_df.shape[1] - 2]

    for i in range(test_df.shape[0]):
        # get prompt and make sure it fits
        k = args.ntrain
        message_end = format_example(test_df, i, include_answer=False)
        train_message = gen_messages(dev_df, subject, k)
        messages = train_message + message_end

        # while crop(prompt) != prompt:
        #     k -= 1
        #     train_prompt = gen_prompt(dev_df, subject, k)
        #     prompt = train_prompt + prompt_end

        label = test_df.iloc[i, test_df.shape[1] - 1]

        while True:
            try:
                c = openai.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=1,
                    logprobs=True,
                    top_logprobs=10,
                    temperature=0,
                )
                break
            except Exception as e:
                print(e)
                time.sleep(1)
                continue

        lprobs = []
        for ans in answers:
            try:
                lprobs.append(c.choices[0].logprobs.top_logprobs[-1][ans])
            except:
                lprobs.append(-100)
        pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(lprobs)]
        probs = softmax(np.array(lprobs))

        cor = pred == label
        cors.append(cor)
        all_probs.append(probs)
        preds.append(pred)

    acc = np.mean(cors)
    cors = np.array(cors)

    all_probs = np.array(all_probs)
    print("Average accuracy {:.3f} - {}".format(acc, subject))

    return cors, acc, all_probs, preds


def main(args):
    model = args.model
    subjects = sorted(
        [
            f.split("_test.csv")[0]
            for f in os.listdir(os.path.join(args.data_dir, "test"))
            if "_test.csv" in f
        ]
    )
    subjects = subjects[args.begin:args.end]

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    if not os.path.exists(os.path.join(args.save_dir, "results_{}".format(model))):
        os.mkdir(os.path.join(args.save_dir, "results_{}".format(model)))

    print(subjects)
    print(args)

    print(model)
    all_cors = []

    for subject in subjects:
        dev_df = pd.read_csv(
            os.path.join(args.data_dir, "dev", subject + "_dev.csv"), header=None
        )[: args.ntrain]
        test_df = pd.read_csv(
            os.path.join(args.data_dir, "test", subject + "_test.csv"), header=None
        )

        cors, acc, probs, preds = eval(args, subject, model, dev_df, test_df)
        all_cors.append(cors)

        test_df["{}_pred".format(model)] = preds
        test_df["{}_correct".format(model)] = cors
        for j in range(probs.shape[1]):
            choice = choices[j]
            test_df["{}_choice{}_probs".format(model, choice)] = probs[:, j]
        test_df.to_csv(
            os.path.join(
                args.save_dir, "results_{}".format(model), "{}.csv".format(subject)
            ),
            index=None,
        )

    weighted_acc = np.mean(np.concatenate(all_cors))
    print("Average accuracy: {:.3f}".format(weighted_acc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--data_dir", "-d", type=str, default="data")
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    parser.add_argument("--model", "-m", type=str)
    parser.add_argument("--begin", "-b", type=int, default=None)
    parser.add_argument("--end", "-e", type=int, default=None)
    args = parser.parse_args()
    main(args)
