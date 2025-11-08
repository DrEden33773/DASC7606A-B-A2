import evaluate
import numpy as np


def postprocess_text(eval_preds, tokenizer):
    """
    Postprocess the text.

    NOTE: You are free to change this function if needed.
    """
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    print(f"Decoded Preds: {decoded_preds[:10]}, Decoded Labels: {decoded_labels[:10]}")

    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]

    return decoded_preds, decoded_labels


def compute_metrics(eval_preds, tokenizer):
    """
    Compute the metrics.

    NOTE: You are NOT allowed to change this function.
    """
    decoded_preds, decoded_labels = postprocess_text(eval_preds, tokenizer)
    metric = evaluate.load("sacrebleu")
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    return {"bleu": result["score"]}
